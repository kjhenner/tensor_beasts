"""PPO on a mostly-empty grid.

The thing worth testing here is not the clipped surrogate, which is textbook.
It is the masking. Almost every cell in this world is empty, so a loss term that
leaks into empty cells is still small, still finite, still decreasing, and still
wrong: it divides the gradient by the grid area instead of by the population, so
the effective learning rate tracks population density. Several tests below feed
deliberate garbage into non-acting cells and assert nothing downstream moves.

These use tiny synthetic grids rather than a real world, so they run in
milliseconds. They say nothing about the ecology and are not meant to.
"""

import math

import pytest
import torch

from tensor_beasts.rl.networks import build_network
from tensor_beasts.rl.ppo import (
    PPO,
    PPOConfig,
    explained_variance,
    iter_minibatches_with_value,
    masked_mean,
)
from tensor_beasts.rl.rollout import Rollout

STEPS = 6
CHANNELS = 4
HEIGHT = WIDTH = 8


def make_network(seed: int = 0):
    torch.manual_seed(seed)
    return build_network("conv", CHANNELS, hidden_channels=8, depth=2)


def make_rollout(network, seed: int = 0, density: float = 0.1) -> Rollout:
    """A rollout whose log-probs and values come from ``network`` itself.

    The observation is stored the way ``RolloutBuffer`` stores it, in float16,
    and the network is shown the float16 round trip, so the log-probabilities
    recorded here are bit-for-bit what the update will recompute. That is what
    makes the ratio-is-one check meaningful rather than approximate.
    """
    generator = torch.Generator().manual_seed(seed)
    observation = torch.randn(STEPS, CHANNELS, HEIGHT, WIDTH, generator=generator)
    observation = observation.to(torch.float16).float()
    acted = torch.rand(STEPS, HEIGHT, WIDTH, generator=generator) < density

    with torch.no_grad():
        logits, value = network(observation)
        log_probs = torch.log_softmax(logits, dim=1)
        flat = log_probs.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        action = torch.multinomial(flat.exp(), 1, generator=generator).reshape(
            STEPS, HEIGHT, WIDTH
        )
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)

    reward = torch.rand(STEPS, HEIGHT, WIDTH, generator=generator) * acted
    advantage = torch.randn(STEPS, HEIGHT, WIDTH, generator=generator) * acted
    ret = value + advantage

    return Rollout(
        observation=observation.to(torch.float16),
        acted=acted,
        action=action,
        log_prob=log_prob,
        value=value,
        reward=reward,
        done=torch.zeros(STEPS, HEIGHT, WIDTH, dtype=torch.bool),
        successor=torch.zeros(STEPS, HEIGHT, WIDTH, dtype=torch.long),
        advantage=advantage,
        ret=ret,
    )


def single_batch(rollout: Rollout):
    """The whole rollout as one minibatch, in collection order."""
    return next(iter_minibatches_with_value(rollout, rollout.steps, shuffle=False))


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------
def test_masked_mean_matches_mean_over_selected_entries():
    values = torch.randn(4, 5)
    mask = torch.rand(4, 5) < 0.5
    if not mask.any():
        mask[0, 0] = True
    assert torch.allclose(masked_mean(values, mask.float()), values[mask].mean())


def test_masked_mean_ignores_everything_outside_the_mask():
    values = torch.randn(4, 5)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[0, :] = True
    before = masked_mean(values, mask.float())
    values = values.clone()
    values[1:] = 1e6
    assert torch.allclose(before, masked_mean(values, mask.float()))


def test_masked_mean_of_empty_mask_is_zero_not_nan():
    values = torch.randn(4, 5)
    result = masked_mean(values, torch.zeros(4, 5))
    assert float(result) == 0.0
    assert torch.isfinite(result)


def test_losses_ignore_non_acting_cells():
    """Garbage in empty cells must not reach a single diagnostic."""
    network = make_network()
    rollout = make_rollout(network)
    ppo = PPO(PPOConfig(minibatch_steps=STEPS))

    observation, acted, action, log_prob, value, advantage, ret = single_batch(rollout)
    _, clean = ppo._losses(
        network, observation, acted, action, log_prob, value, advantage, ret
    )

    empty = ~acted
    dirty_advantage = advantage.clone()
    dirty_advantage[empty] = 1e6
    dirty_ret = ret.clone()
    dirty_ret[empty] = -1e6
    dirty_log_prob = log_prob.clone()
    dirty_log_prob[empty] = -50.0
    dirty_action = action.clone()
    dirty_action[empty] = (dirty_action[empty] + 1) % 5

    _, dirty = ppo._losses(
        network, observation, acted, dirty_action, dirty_log_prob, value,
        dirty_advantage, dirty_ret,
    )
    for key in clean:
        assert clean[key] == pytest.approx(dirty[key], abs=1e-6), key


def test_losses_are_finite():
    network = make_network()
    rollout = make_rollout(network)
    ppo = PPO(PPOConfig(minibatch_steps=STEPS))
    loss = ppo.minibatch_loss(network, single_batch(rollout))
    assert torch.isfinite(loss)


def test_update_with_no_agents_anywhere_is_a_no_op():
    """A total population crash must not produce NaN weights."""
    network = make_network()
    rollout = make_rollout(network)
    rollout.acted = torch.zeros_like(rollout.acted)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    before = [p.clone() for p in network.parameters()]

    PPO(PPOConfig(minibatch_steps=STEPS)).update(network, rollout, optimizer)

    for parameter, original in zip(network.parameters(), before):
        assert torch.equal(parameter, original)


# ----------------------------------------------------------------------
# The standard PPO correctness check
# ----------------------------------------------------------------------
def test_ratio_is_exactly_one_on_the_first_epoch():
    """Before any gradient step the new policy is the old policy.

    If this drifts, either the stored log-probabilities came from a different
    observation than the update sees (the float16 round trip is the usual
    culprit) or the action gather has the axes crossed.
    """
    network = make_network()
    rollout = make_rollout(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    diagnostics = PPO(
        PPOConfig(epochs=1, minibatch_steps=STEPS)
    ).update(network, rollout, optimizer)

    assert diagnostics["ratio_max_deviation"] == 0.0
    assert diagnostics["approx_kl"] == 0.0
    assert diagnostics["clip_fraction"] == 0.0


def test_clip_fraction_and_kl_stay_in_range_after_real_updates():
    network = make_network()
    rollout = make_rollout(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)

    diagnostics = PPO(
        PPOConfig(epochs=4, minibatch_steps=2)
    ).update(network, rollout, optimizer)

    assert 0.0 <= diagnostics["clip_fraction"] <= 1.0
    # Schulman's k3 estimator is non-negative by construction; a negative value
    # means the ratio and the log ratio disagree, i.e. a masking bug.
    assert diagnostics["approx_kl"] >= 0.0
    # Four epochs at 3e-4 should not move the policy anywhere near the clip.
    assert diagnostics["approx_kl"] < 0.1
    assert diagnostics["clip_fraction"] < 0.5
    assert 0.0 <= diagnostics["entropy"] <= math.log(5) + 1e-6


def test_update_reduces_loss_on_a_fixed_batch():
    """With the batch held fixed, PPO is just optimization. It should descend."""
    network = make_network()
    rollout = make_rollout(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    ppo = PPO(PPOConfig(epochs=8, minibatch_steps=STEPS, entropy_coef=0.0, clip_range=10.0))

    batch = single_batch(rollout)
    with torch.no_grad():
        before = float(ppo.minibatch_loss(network, batch))
    ppo.update(network, rollout, optimizer)
    with torch.no_grad():
        after = float(ppo.minibatch_loss(network, batch))

    assert after < before


def test_target_kl_stops_the_epoch_loop_early():
    network = make_network()
    rollout = make_rollout(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.1)
    diagnostics = PPO(
        PPOConfig(epochs=10, minibatch_steps=STEPS, target_kl=1e-6)
    ).update(network, rollout, optimizer)
    assert diagnostics["epochs_run"] < 10


def test_value_clipping_runs_and_changes_the_value_loss():
    network = make_network()
    rollout = make_rollout(network)
    batch = single_batch(rollout)

    unclipped = PPO(PPOConfig(minibatch_steps=STEPS))
    clipped = PPO(PPOConfig(minibatch_steps=STEPS, value_clip_range=1e-4))
    observation, acted, action, log_prob, value, advantage, ret = batch
    # Stand the recorded value away from the current prediction, the way it
    # would be part way through an update. With them equal, clipping is a no-op
    # by construction and the test would prove nothing.
    stale_value = value - 1.0
    _, a = unclipped._losses(
        network, observation, acted, action, log_prob, stale_value, advantage, ret
    )
    _, b = clipped._losses(
        network, observation, acted, action, log_prob, stale_value, advantage, ret
    )

    assert torch.isfinite(torch.tensor(b["value_loss"]))
    # The clipped branch is pinned near the stale value, and max() takes the
    # worse of the two, so clipping can only raise the loss.
    assert b["value_loss"] > a["value_loss"]


def test_update_without_advantages_is_an_error():
    network = make_network()
    rollout = make_rollout(network)
    rollout.advantage = None
    optimizer = torch.optim.Adam(network.parameters())
    with pytest.raises(ValueError, match="compute_gae"):
        PPO().update(network, rollout, optimizer)


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def test_explained_variance_is_one_for_a_perfect_critic():
    ret = torch.randn(4, 4)
    mask = torch.ones(4, 4, dtype=torch.bool)
    assert explained_variance(ret.clone(), ret, mask) == pytest.approx(1.0)


def test_explained_variance_is_zero_for_a_constant_critic():
    ret = torch.randn(50)
    mask = torch.ones(50, dtype=torch.bool)
    constant = torch.full_like(ret, float(ret.mean()))
    assert explained_variance(constant, ret, mask) == pytest.approx(0.0, abs=1e-5)


def test_explained_variance_uses_acting_cells_only():
    ret = torch.randn(50)
    mask = torch.zeros(50, dtype=torch.bool)
    mask[:20] = True
    good = ret.clone()
    good[20:] = 1e3  # nonsense where nobody acted
    assert explained_variance(good, ret, mask) == pytest.approx(1.0)


def test_minibatch_iteration_covers_every_timestep_once():
    network = make_network()
    rollout = make_rollout(network)
    seen = 0
    for batch in iter_minibatches_with_value(rollout, 4):
        seen += batch[0].shape[0]
        assert len(batch) == 7
    assert seen == STEPS
