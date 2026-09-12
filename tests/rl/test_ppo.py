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

    observation, acted, action, log_prob, value, advantage, ret, _rule, _scores = single_batch(rollout)
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
        assert clean[key] == pytest.approx(dirty[key], abs=1e-6, nan_ok=True), key


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
    observation, acted, action, log_prob, value, advantage, ret, _rule, _scores = batch
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
        assert len(batch) == 9
    assert seen == STEPS


# ---------------------------------------------------------------------------
# Return normalization
# ---------------------------------------------------------------------------


def test_value_normalization_keeps_the_policy_gradient_alive():
    """The measurement that motivated rl/normalization.py, as a regression test.

    Without normalization the squared-error value loss is two orders of
    magnitude larger than the policy loss, so it is essentially the entire
    gradient norm, and global clipping then scales the policy's share down with
    it. The symptom is a policy that never moves while the loss curve looks
    busy, which is exactly the kind of failure that wastes days.
    """
    import torch

    from tensor_beasts.rl.normalization import ValueNormalizer
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import masked_mean

    torch.manual_seed(0)
    channels, size, steps = 6, 8, 4
    network = build_network("linear", channels)
    observation = torch.randn(steps, channels, size, size)
    acted = torch.rand(steps, size, size) < 0.3
    action = torch.randint(0, 5, (steps, size, size))
    advantage = torch.randn(steps, size, size)
    # Returns on the scale survival reward actually produces: large and smooth.
    returns = torch.full((steps, size, size), 95.0) + torch.randn(steps, size, size)

    def gradient_norms(normalizer):
        logits, value = network(observation)
        log_probs = torch.log_softmax(logits, dim=1)
        chosen = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        policy_loss = -masked_mean(chosen * advantage, acted)
        value_loss = masked_mean((value - normalizer.normalize(returns)) ** 2, acted)

        def norm(loss):
            network.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            return sum(
                float(p.grad.pow(2).sum()) for p in network.parameters() if p.grad is not None
            ) ** 0.5

        return norm(policy_loss), norm(0.5 * value_loss)

    off = ValueNormalizer(enabled=False)
    on = ValueNormalizer(enabled=True)
    on.update(returns, acted)

    policy_off, value_off = gradient_norms(off)
    policy_on, value_on = gradient_norms(on)

    assert value_off > 50 * policy_off, "the unnormalized value term should dwarf the policy term"
    assert value_on < 10 * policy_on, "normalization should bring them onto comparable scales"


def test_normalizer_round_trips():
    import torch

    from tensor_beasts.rl.normalization import ValueNormalizer

    normalizer = ValueNormalizer(enabled=True)
    values = torch.randn(500) * 12.0 + 90.0
    normalizer.update(values)

    normalized = normalizer.normalize(values)
    assert abs(float(normalized.mean())) < 0.05
    assert abs(float(normalized.std()) - 1.0) < 0.05
    assert torch.allclose(normalizer.denormalize(normalized), values, atol=1e-3)


def test_disabled_normalizer_is_the_identity():
    import torch

    from tensor_beasts.rl.normalization import ValueNormalizer

    normalizer = ValueNormalizer(enabled=False)
    values = torch.randn(50)
    normalizer.update(values)
    assert torch.equal(normalizer.normalize(values), values)
    assert torch.equal(normalizer.denormalize(values), values)


def test_running_stats_match_a_single_pass():
    import torch

    from tensor_beasts.rl.normalization import RunningMeanStd

    torch.manual_seed(0)
    chunks = [torch.randn(300) * 3.0 + 7.0 for _ in range(8)]
    stats = RunningMeanStd()
    for chunk in chunks:
        stats.update(chunk)

    everything = torch.cat(chunks)
    assert abs(stats.mean - float(everything.mean())) < 1e-3
    assert abs(stats.std - float(everything.std(unbiased=False))) < 1e-3



# ---------------------------------------------------------------------------
# Imitation of the rule-based policy, cross-faded out as conformance rises
# ---------------------------------------------------------------------------


def _rollout_with_rule_actions(seed=0, steps=6, size=8, channels=6):
    """A synthetic rollout whose rule action is a fixed function of the
    observation, so a policy can actually learn to agree with it."""
    import torch
    from tensor_beasts.rl.rollout import Rollout

    g = torch.Generator().manual_seed(seed)
    observation = torch.rand(steps, channels, size, size, generator=g)
    acted = torch.rand(steps, size, size, generator=g) < 0.5
    # Rule: the argmax over the first five channels. Learnable by a 1x1 conv.
    rule_action = observation[:, :5].argmax(dim=1)
    flat = torch.arange(size * size).reshape(size, size)
    rollout = Rollout(
        observation=observation.half(),
        acted=acted,
        action=torch.randint(0, 5, (steps, size, size), generator=g),
        log_prob=torch.full((steps, size, size), -1.6),
        value=torch.zeros(steps, size, size),
        reward=torch.zeros(steps, size, size),
        done=torch.zeros(steps, size, size, dtype=torch.bool),
        successor=flat.expand(steps, size, size).clone(),
        advantage=torch.zeros(steps, size, size),
        ret=torch.zeros(steps, size, size),
        rule_action=rule_action,
    )
    return rollout


def test_imitation_weight_cross_fades_to_zero_at_target_conformance():
    from tensor_beasts.rl.ppo import PPO, PPOConfig

    ppo = PPO(PPOConfig(imitation_coef=2.0, imitation_target_conformance=0.8))
    ppo.conformance = 0.0
    assert ppo.imitation_weight() == pytest.approx(2.0), "full strength from a random start"
    ppo.conformance = 0.4
    assert ppo.imitation_weight() == pytest.approx(1.0), "halfway to target, half strength"
    ppo.conformance = 0.8
    assert ppo.imitation_weight() == 0.0, "at target the rules let go"
    ppo.conformance = 0.95
    assert ppo.imitation_weight() == 0.0, "and never pull backwards"

    assert PPO(PPOConfig(imitation_coef=0.0)).imitation_weight() == 0.0, "off means off"


def test_imitation_pulls_the_policy_toward_the_rules_and_conformance_rises():
    """On a batch with zero advantages the only learning signal is imitation,
    so agreement with the rule action must climb and the weight must fall."""
    import torch
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig

    torch.manual_seed(0)
    rollout = _rollout_with_rule_actions()
    network = build_network("linear", 6)
    ppo = PPO(PPOConfig(imitation_coef=1.0, imitation_target_conformance=0.9,
                        learning_rate=0.1, epochs=1, minibatch_steps=6,
                        entropy_coef=0.0, value_coef=0.0))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.1)

    first = ppo.update(network, rollout, optimizer)
    for _ in range(40):
        last = ppo.update(network, rollout, optimizer)

    assert last["conformance"] > first["conformance"] + 0.3
    assert last["imitation_loss"] < first["imitation_loss"]
    assert last["imitation_weight"] < first["imitation_weight"], "the cross-fade is fading"


def test_conformance_is_measured_even_when_imitation_is_off():
    """So the log shows how far the policy drifts from the rules regardless."""
    import torch
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig

    rollout = _rollout_with_rule_actions()
    ppo = PPO(PPOConfig(imitation_coef=0.0, epochs=1, minibatch_steps=6))
    result = ppo.update(network := build_network("linear", 6), rollout,
                        torch.optim.Adam(network.parameters()))
    assert 0.0 <= result["conformance"] <= 1.0
    assert result["imitation_weight"] == 0.0


def test_imitation_term_is_masked_to_acting_cells():
    import torch
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig, iter_minibatches_with_value

    rollout = _rollout_with_rule_actions()
    network = build_network("linear", 6)
    ppo = PPO(PPOConfig(imitation_coef=1.0))
    batch = next(iter_minibatches_with_value(rollout, rollout.steps, shuffle=False))
    before = float(ppo.minibatch_loss(network, batch))

    # Scramble the rule action everywhere nobody is standing.
    scrambled = list(batch)
    rule = scrambled[7].clone()
    rule[~batch[1]] = (rule[~batch[1]] + 1) % 5
    scrambled[7] = rule
    after = float(ppo.minibatch_loss(network, tuple(scrambled)))
    assert before == pytest.approx(after)


# ---------------------------------------------------------------------------
# Soft distillation toward the rule's scoring regime
# ---------------------------------------------------------------------------


def _rollout_with_rule_scores(seed=0, steps=6, size=8, channels=6, gap=1.0):
    """Rule scores that are a fixed function of the observation, with a
    controllable gap between best and second-best so tests can build both
    confident cells and near-ties."""
    import torch
    from tensor_beasts.rl.rollout import Rollout

    rollout = _rollout_with_rule_actions(seed=seed, steps=steps, size=size, channels=channels)
    obs = rollout.observation.float()
    # Scores: the first five channels, sharpened so that the best beats the
    # rest by roughly `gap` on the softmax temperature's scale.
    scores = obs[:, :5] * gap
    rollout.rule_scores = scores
    rollout.rule_action = scores.argmax(dim=1)
    return rollout


def test_soft_distillation_raises_scoring_conformance():
    import torch
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig

    torch.manual_seed(0)
    rollout = _rollout_with_rule_scores(gap=1.0)
    network = build_network("linear", 6)
    ppo = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=0.1, imitation_target_conformance=0.99,
                        learning_rate=0.1, epochs=1, minibatch_steps=6, entropy_coef=0.0, value_coef=0.0))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.1)
    first = ppo.update(network, rollout, optimizer)
    for _ in range(40):
        last = ppo.update(network, rollout, optimizer)
    assert first["conformance"] < 0.3, "a near-uniform policy should start near zero progress"
    assert last["conformance"] > first["conformance"] + 0.3
    assert last["conformance"] > 0.7
    assert last["imitation_loss"] < first["imitation_loss"] * 0.5
    assert 0.0 <= last["conformance"] <= 1.0 + 1e-6


def test_near_ties_cost_almost_nothing_to_disagree_with():
    """The reason for soft targets. A uniform policy pays the full KL against a
    confident rule but next to nothing against a rule that cannot decide."""
    import torch
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig, iter_minibatches_with_value

    torch.manual_seed(0)
    network = build_network("linear", 6)
    with torch.no_grad():
        network.policy_head.weight.zero_()
        network.policy_head.bias.zero_()  # exactly uniform policy

    ppo = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=0.1, entropy_coef=0.0, value_coef=0.0))

    def imitation_loss(gap):
        rollout = _rollout_with_rule_scores(gap=gap)
        batch = next(iter_minibatches_with_value(rollout, rollout.steps, shuffle=False))
        _, diagnostics = ppo._losses(network, *batch)
        return diagnostics["imitation_loss"]

    confident = imitation_loss(gap=5.0)
    near_tie = imitation_loss(gap=0.01)
    assert near_tie < 0.1 * confident, f"near-tie loss {near_tie:.4f} vs confident {confident:.4f}"


def test_soft_conformance_ceiling_is_one_even_where_the_rule_is_unsure():
    """A policy that exactly matches the rule's distribution must score 1,
    including at near-ties where the raw overlap would be far below 1."""
    import torch
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig, iter_minibatches_with_value

    rollout = _rollout_with_rule_scores(gap=0.05)  # near-ties everywhere
    temperature = 0.1
    ppo = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=temperature))
    network = build_network("linear", 6)
    with torch.no_grad():
        # logits = scores / temperature reproduces the target exactly.
        network.policy_head.weight.zero_()
        network.policy_head.bias.zero_()
        for a in range(5):
            network.policy_head.weight[a, a] = 0.05 / temperature  # scores = obs * gap(0.05)
    batch = next(iter_minibatches_with_value(rollout, rollout.steps, shuffle=False))
    _, diagnostics = ppo._losses(network, *batch)
    assert diagnostics["conformance"] == pytest.approx(1.0, abs=1e-3)
    assert diagnostics["imitation_loss"] == pytest.approx(0.0, abs=1e-4)


def test_zero_temperature_falls_back_to_hard_imitation():
    import torch
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig, iter_minibatches_with_value

    rollout = _rollout_with_rule_scores()
    network = build_network("linear", 6)
    batch = next(iter_minibatches_with_value(rollout, rollout.steps, shuffle=False))
    soft = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=0.1))
    hard = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=0.0))
    _, d_soft = soft._losses(network, *batch)
    _, d_hard = hard._losses(network, *batch)
    assert d_hard["conformance"] == pytest.approx(d_hard["argmax_agreement"])
    assert d_soft["imitation_loss"] != pytest.approx(d_hard["imitation_loss"])


def test_soft_imitation_is_masked_to_acting_cells():
    import torch
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig, iter_minibatches_with_value

    rollout = _rollout_with_rule_scores()
    network = build_network("linear", 6)
    ppo = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=0.1))
    batch = list(next(iter_minibatches_with_value(rollout, rollout.steps, shuffle=False)))
    before = float(ppo.minibatch_loss(network, tuple(batch)))
    scores = batch[8].clone()
    idle = (~batch[1]).unsqueeze(1).expand_as(scores)  # per-step idle cells, all five actions
    scores[idle] = torch.randn(int(idle.sum())) * 10
    batch[8] = scores
    after = float(ppo.minibatch_loss(network, tuple(batch)))
    assert before == pytest.approx(after)



def test_uniform_policy_has_zero_soft_conformance():
    """The calibration bug this guards against: the raw overlap of a uniform
    policy with a diffuse target is already high, and a cross-fade keyed on it
    released the anchor at chance-level agreement."""
    import torch
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig, iter_minibatches_with_value

    rollout = _rollout_with_rule_scores(gap=0.5)
    network = build_network("linear", 6)
    with torch.no_grad():
        network.policy_head.weight.zero_()
        network.policy_head.bias.zero_()
    ppo = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=0.1))
    batch = next(iter_minibatches_with_value(rollout, rollout.steps, shuffle=False))
    _, diagnostics = ppo._losses(network, *batch)
    assert diagnostics["conformance"] == pytest.approx(0.0, abs=1e-4)



def test_imitation_floor_keeps_a_permanent_pull():
    """Releasing fully at the target was seen to oscillate: the RL gradient
    lowers conformance, the anchor re-engages, repeat. A floor keeps a light
    pull on however far conformance climbs, and zero preserves the old rule."""
    from tensor_beasts.rl.ppo import PPO, PPOConfig

    floored = PPO(PPOConfig(imitation_coef=2.0, imitation_target_conformance=0.8, imitation_floor=0.25))
    for conformance in (0.0, 0.5, 0.8, 0.95):
        floored.conformance = conformance
        assert floored.imitation_weight() >= 0.5
    floored.conformance = 0.0
    assert floored.imitation_weight() == pytest.approx(2.0), "the floor never raises the weight above full strength"

    released = PPO(PPOConfig(imitation_coef=2.0, imitation_target_conformance=0.8, imitation_floor=0.0))
    released.conformance = 0.9
    assert released.imitation_weight() == 0.0
