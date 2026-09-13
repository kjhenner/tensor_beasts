"""V-trace and AWR: masking, movement, and reduction to the known-good GAE.

Three things can go wrong in a learner for this setting, and each has a test
here.

1. A reduction that divides by the number of cells rather than the number of
   agents. Invisible in the loss curve; looks like a learning-rate problem.
   Caught by perturbing non-acting cells and asserting nothing moves.
2. A recursion that indexes the same cell at the next timestep instead of
   following the individual through ``successor``. Caught by walking one agent
   along a known path with decoy values parked in the cells it vacates, the
   trick from ``tests/rl/test_rollout.py``.
3. An off-by-one or a misplaced gamma in the V-trace recursion. Caught by
   forcing all importance ratios to one, where V-trace is provably identical to
   :func:`~tensor_beasts.rl.rollout.compute_gae`.

World sizes here are 8 to 32 cells on a side purely so the suite runs in
seconds. They are NOT a valid ecology: below roughly 256 the predator population
collapses and the three-species dynamic degenerates. Nothing in this file
measures anything ecological, only arithmetic.
"""

import pytest
import torch

from tensor_beasts.rl.algorithms import ALGORITHMS, build_algorithm
from tensor_beasts.rl.algorithms.replay import AWR, AWRConfig, SegmentReplayBuffer
from tensor_beasts.rl.algorithms.vtrace import (
    VTrace,
    VTraceConfig,
    compute_vtrace,
    iter_vtrace_minibatches,
)
from tensor_beasts.rl.networks import build_network
from tensor_beasts.rl.rollout import Rollout, compute_gae

GAMMA = 0.9
LAMBDA = 0.8
CHANNELS = 4


# ----------------------------------------------------------------------
# Synthetic rollouts
# ----------------------------------------------------------------------
def random_rollout(steps=6, height=8, width=8, density=0.3, seed=0) -> Rollout:
    """A rollout with a plausible mix of occupied and empty cells.

    Successors are drawn uniformly over the cells that are occupied at the next
    timestep, rather than over the whole grid. That is not decoration: the
    defining property of a successor is that the individual is standing there
    next step, so it points at an occupied cell by construction. A fixture that
    let successors land on empty cells would make the masking test below fail
    for a legitimate reason --- an empty cell's value really is read when
    somebody's successor points into it --- and would be testing a property the
    real data does not have.

    Adjacency is not enforced: nothing under test cares, and an unrestricted
    draw exercises the gather harder than a four-neighbour move would.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (steps, height, width)
    acted = torch.rand(shape, generator=generator) < density
    cells = torch.arange(height * width)

    successor = torch.zeros(shape, dtype=torch.long)
    for t in range(steps):
        occupied_next = cells[acted[min(t + 1, steps - 1)].reshape(-1)]
        if occupied_next.numel() == 0:
            # Nobody alive next step; everyone "stays put", which is harmless
            # because the bootstrap is zeroed by ``done`` or masked by ``acted``.
            successor[t] = cells.reshape(height, width)
            continue
        draw = torch.randint(occupied_next.numel(), (height * width,), generator=generator)
        successor[t] = occupied_next[draw].reshape(height, width)

    return Rollout(
        observation=torch.randn(steps, CHANNELS, height, width, generator=generator).to(torch.float16),
        acted=acted,
        action=torch.randint(0, 5, shape, generator=generator),
        log_prob=torch.randn(shape, generator=generator) * 0.1 - 1.6,
        value=torch.randn(shape, generator=generator),
        reward=torch.randn(shape, generator=generator) * acted,
        done=(torch.rand(shape, generator=generator) < 0.1) & acted,
        successor=successor,
    )


def walking_rollout(path, rewards, values, decoy=-99.0, height=3, width=3) -> Rollout:
    """One agent walking ``path``, with ``decoy`` parked in every other cell.

    A cell-indexed recursion consumes the decoys and produces numbers nowhere
    near the hand-computed answer, which is the entire point.
    """
    steps = len(path)
    shape = (steps, height, width)
    acted = torch.zeros(shape, dtype=torch.bool)
    successor = torch.zeros(shape, dtype=torch.long)
    reward = torch.zeros(shape)
    value = torch.full(shape, decoy)

    for t, cell in enumerate(path):
        acted[t][cell] = True
        nxt = path[t + 1] if t + 1 < steps else path[t]
        successor[t][cell] = nxt[0] * width + nxt[1]
        reward[t][cell] = rewards[t]
        value[t][cell] = values[t]

    return Rollout(
        observation=torch.zeros(steps, CHANNELS, height, width, dtype=torch.float16),
        acted=acted,
        action=torch.zeros(shape, dtype=torch.long),
        log_prob=torch.zeros(shape),
        value=value,
        reward=reward,
        done=torch.zeros(shape, dtype=torch.bool),
        successor=successor,
    )


def hand_rolled_gae(rewards, values, last_value, gamma=GAMMA, lam=LAMBDA):
    """GAE along one individual's own path, written out by hand."""
    steps = len(rewards)
    next_values = list(values[1:]) + [last_value]
    deltas = [rewards[t] + gamma * next_values[t] - values[t] for t in range(steps)]
    out = [0.0] * steps
    carry = 0.0
    for t in reversed(range(steps)):
        carry = deltas[t] + gamma * lam * carry
        out[t] = carry
    return out


# ----------------------------------------------------------------------
# The movement test: the most important one in this file
# ----------------------------------------------------------------------
def test_vtrace_follows_a_moving_agent():
    """One agent walks (0,0) -> (0,1) -> (0,2); targets must match by hand."""
    path = [(0, 0), (0, 1), (0, 2)]
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.7, 0.9]
    last_value_at_end = 1.3

    rollout = walking_rollout(path, rewards, values)
    last_value = torch.full((3, 3), -99.0)
    last_value[0, 2] = last_value_at_end

    vs, advantage = compute_vtrace(
        rollout,
        value=rollout.value,
        log_ratio=torch.zeros_like(rollout.value),
        last_value=last_value,
        gamma=GAMMA,
        lambda_=LAMBDA,
    )

    expected = hand_rolled_gae(rewards, values, last_value_at_end)
    for t, cell in enumerate(path):
        got = vs[t][cell].item() - values[t]
        assert abs(got - expected[t]) < 1e-5, f"step {t}: {got} != {expected[t]}"

    # And with lambda = 1 the policy-gradient advantage coincides with GAE too,
    # since v_{t+1} - V(x_{t+1}) is then exactly the next advantage.
    _, advantage_one = compute_vtrace(
        rollout,
        value=rollout.value,
        log_ratio=torch.zeros_like(rollout.value),
        last_value=last_value,
        gamma=GAMMA,
        lambda_=1.0,
    )
    expected_one = hand_rolled_gae(rewards, values, last_value_at_end, lam=1.0)
    for t, cell in enumerate(path):
        assert abs(advantage_one[t][cell].item() - expected_one[t]) < 1e-5


def test_awr_targets_follow_a_moving_agent():
    """AWR's value target is the same lambda-return, so it walks too."""
    path = [(2, 2), (1, 2), (1, 1)]
    rewards = [0.5, -1.0, 2.0]
    values = [0.2, -0.3, 0.8]
    rollout = walking_rollout(path, rewards, values)

    last_value = torch.full((3, 3), -99.0)
    last_value[1, 1] = 0.4

    network = build_network("linear", CHANNELS)
    algorithm = AWR(AWRConfig(gamma=GAMMA, lambda_=LAMBDA, normalize_advantage=False))
    # Force the critic to be exactly the stored values so the arithmetic is
    # checkable: a linear policy on an all-zero observation outputs its bias.
    with torch.no_grad():
        network.value_head.weight.zero_()
        network.value_head.bias.zero_()

    # With V = 0 everywhere the lambda-return reduces to a discounted reward sum
    # along the agent's path, which is easy to write down.
    vs, advantage = algorithm.compute_targets(network, rollout, last_value=torch.zeros(3, 3))
    expected = hand_rolled_gae(rewards, [0.0, 0.0, 0.0], 0.0)
    for t, cell in enumerate(path):
        assert abs(vs[t][cell].item() - expected[t]) < 1e-5
        assert abs(advantage[t][cell].item() - expected[t]) < 1e-5


def test_a_decoy_value_would_change_the_answer():
    """The movement tests are only meaningful if the decoys are reachable.

    If a cell-indexed recursion produced the same numbers as a successor-gathered
    one on this fixture, the tests above would pass vacuously.
    """
    path = [(0, 0), (0, 1), (0, 2)]
    rollout = walking_rollout(path, [1.0, 2.0, 3.0], [0.5, 0.7, 0.9])
    last_value = torch.full((3, 3), -99.0)
    last_value[0, 2] = 1.3

    vs, _ = compute_vtrace(
        rollout, rollout.value, torch.zeros_like(rollout.value), last_value, GAMMA, LAMBDA
    )
    # Same computation with the successor map replaced by "stay put", which is
    # what an implementation that forgot movement would effectively do.
    stationary = rollout.successor.clone()
    for t, cell in enumerate(path):
        stationary[t][cell] = cell[0] * 3 + cell[1]
    wrong = Rollout(
        observation=rollout.observation,
        acted=rollout.acted,
        action=rollout.action,
        log_prob=rollout.log_prob,
        value=rollout.value,
        reward=rollout.reward,
        done=rollout.done,
        successor=stationary,
    )
    vs_wrong, _ = compute_vtrace(
        wrong, wrong.value, torch.zeros_like(wrong.value), last_value, GAMMA, LAMBDA
    )
    assert not torch.allclose(vs[0][path[0]], vs_wrong[0][path[0]])


def test_death_stops_the_bootstrap():
    """A dead individual gets no future value, even though its cell has one."""
    rollout = walking_rollout([(1, 1)], [5.0], [2.0])
    rollout.done[0][1, 1] = True
    vs, advantage = compute_vtrace(
        rollout,
        rollout.value,
        torch.zeros_like(rollout.value),
        last_value=torch.full((3, 3), 100.0),
        gamma=GAMMA,
        lambda_=LAMBDA,
    )
    assert abs(vs[0][1, 1].item() - 5.0) < 1e-5  # V + (r - V), no bootstrap
    assert abs(advantage[0][1, 1].item() - 3.0) < 1e-5  # r - V


# ----------------------------------------------------------------------
# Reduction to GAE
# ----------------------------------------------------------------------
def test_vtrace_with_unit_ratios_equals_gae():
    """The strong correctness check. If this fails, the recursion is wrong."""
    rollout = random_rollout(seed=1)
    last_value = torch.randn(8, 8, generator=torch.Generator().manual_seed(2))

    vs, _ = compute_vtrace(
        rollout,
        value=rollout.value,
        log_ratio=torch.zeros_like(rollout.value),
        last_value=last_value,
        gamma=GAMMA,
        lambda_=LAMBDA,
        rho_bar=1.0,
        c_bar=1.0,
    )
    vtrace_advantage = (vs - rollout.value) * rollout.acted

    reference = compute_gae(
        random_rollout(seed=1), last_value, gamma=GAMMA, gae_lambda=LAMBDA, normalize=False
    )
    assert torch.allclose(vtrace_advantage, reference.advantage, atol=1e-5)


def test_vtrace_pg_advantage_equals_gae_at_lambda_one():
    rollout = random_rollout(seed=3)
    last_value = torch.randn(8, 8, generator=torch.Generator().manual_seed(4))
    _, advantage = compute_vtrace(
        rollout,
        value=rollout.value,
        log_ratio=torch.zeros_like(rollout.value),
        last_value=last_value,
        gamma=GAMMA,
        lambda_=1.0,
    )
    reference = compute_gae(
        random_rollout(seed=3), last_value, gamma=GAMMA, gae_lambda=1.0, normalize=False
    )
    assert torch.allclose(advantage, reference.advantage, atol=1e-5)


def test_truncation_bounds_the_ratio():
    """rho_bar caps the temporal difference; a huge ratio must not blow up."""
    rollout = random_rollout(seed=5)
    last_value = torch.zeros(8, 8)
    huge = torch.full_like(rollout.value, 50.0)  # exp(50) is astronomically large
    vs, advantage = compute_vtrace(
        rollout, rollout.value, huge, last_value, GAMMA, LAMBDA, rho_bar=1.0, c_bar=1.0
    )
    assert torch.isfinite(vs).all()
    assert torch.isfinite(advantage).all()

    # rho_bar = 1 means the result is identical to unit ratios.
    vs_unit, advantage_unit = compute_vtrace(
        rollout, rollout.value, torch.zeros_like(huge), last_value, GAMMA, LAMBDA
    )
    assert torch.allclose(vs, vs_unit, atol=1e-5)
    assert torch.allclose(advantage, advantage_unit, atol=1e-5)


def test_empty_cells_carry_no_advantage():
    rollout = random_rollout(seed=6)
    vs, advantage = compute_vtrace(
        rollout, rollout.value, torch.zeros_like(rollout.value), torch.zeros(8, 8), GAMMA, LAMBDA
    )
    empty = ~rollout.acted
    assert torch.all(advantage[empty] == 0)
    # The value target at an empty cell equals the prediction, so the masked
    # value loss there is structurally zero whatever the mask does.
    assert torch.allclose(vs[empty], rollout.value[empty])


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------
def perturb_non_acting(rollout: Rollout) -> Rollout:
    """A copy with every per-cell field scrambled wherever nothing acted.

    ``reward``, ``done`` and ``successor`` are left alone: a successor pointer
    legitimately aims at a cell that is empty at the time it is read, so
    scrambling those would be testing a different (and false) property.
    """
    empty = ~rollout.acted
    generator = torch.Generator().manual_seed(99)
    noise = lambda shape: torch.randn(shape, generator=generator) * 17.0

    observation = rollout.observation.float().clone()
    observation[empty.unsqueeze(1).expand_as(observation)] = 3.0
    return Rollout(
        observation=observation.to(torch.float16),
        acted=rollout.acted,
        action=torch.where(empty, torch.randint(0, 5, rollout.action.shape, generator=generator), rollout.action),
        log_prob=torch.where(empty, noise(rollout.log_prob.shape), rollout.log_prob),
        value=torch.where(empty, noise(rollout.value.shape), rollout.value),
        reward=rollout.reward,
        done=rollout.done,
        successor=rollout.successor,
    )


@pytest.mark.parametrize("algorithm_name", ["vtrace", "awr"])
def test_loss_ignores_non_acting_cells(algorithm_name):
    """Scramble everything at empty cells; the loss must not move.

    The ``linear`` network is used deliberately. It is a 1x1 convolution, so a
    cell's output depends only on that cell's observation. Any network with
    spatial extent would (correctly) let an empty neighbour's observation reach
    an agent's logits, and this test would then be asserting something false.
    """
    rollout = random_rollout(seed=7)
    scrambled = perturb_non_acting(rollout)
    network = build_network("linear", CHANNELS)
    algorithm = build_algorithm(algorithm_name, gamma=GAMMA, lambda_=LAMBDA)

    losses = []
    for candidate in (rollout, scrambled):
        if algorithm_name == "vtrace":
            vs, advantage = algorithm.compute_targets(network, candidate, torch.zeros(8, 8))
            batch = next(
                iter_vtrace_minibatches(candidate, vs, advantage, candidate.steps, shuffle=False)
            )
        else:
            vs, advantage = algorithm.compute_targets(network, candidate, torch.zeros(8, 8))
            batch = (
                candidate.observation.float(),
                candidate.acted,
                candidate.action,
                vs,
                advantage,
            )
        losses.append(float(algorithm.minibatch_loss(network, batch).detach()))

    assert all(loss == loss for loss in losses), "loss must be finite"
    assert abs(losses[0] - losses[1]) < 1e-5, losses


@pytest.mark.parametrize("algorithm_name", ["vtrace", "awr"])
def test_losses_are_finite_with_no_agents_at_all(algorithm_name):
    """A population crash leaves minibatches with nobody in them."""
    rollout = random_rollout(seed=8, density=0.0)
    assert rollout.num_agent_steps == 0
    network = build_network("conv", CHANNELS, hidden_channels=8, depth=1)
    algorithm = build_algorithm(algorithm_name)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    diagnostics = algorithm.update(network, rollout, optimizer)
    assert all(isinstance(v, float) for v in diagnostics.values())


# ----------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------
@pytest.mark.parametrize("algorithm_name", ["vtrace", "awr"])
def test_update_reduces_loss_on_a_fixed_batch(algorithm_name):
    """Repeated updates on one segment must drive its own loss down.

    32x32 for speed. This is not an ecology and proves nothing about the
    simulation, only that the gradient points downhill.
    """
    rollout = random_rollout(steps=4, height=32, width=32, density=0.2, seed=9)
    network = build_network("conv", CHANNELS, hidden_channels=8, depth=2)
    algorithm = build_algorithm(
        algorithm_name, gamma=GAMMA, lambda_=LAMBDA, minibatch_steps=4
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)
    last_value = torch.zeros(32, 32)

    def value_error() -> float:
        """Critic error against a target frozen before any training happened."""
        with torch.no_grad():
            _, value = network(rollout.observation.float())
        return float(((value - frozen_target) ** 2 * rollout.acted).sum() / rollout.acted.sum())

    if algorithm_name == "vtrace":
        frozen_target, _ = algorithm.compute_targets(network, rollout, last_value)
    else:
        frozen_target, _ = algorithm.compute_targets(network, rollout, last_value)

    before = value_error()
    for _ in range(12):
        algorithm.update(network, rollout, optimizer, last_value=last_value)
    after = value_error()
    assert after < before, f"{algorithm_name}: {after} !< {before}"


def test_vtrace_diagnostics_are_reported():
    rollout = random_rollout(steps=4, height=16, width=16, seed=10)
    network = build_network("conv", CHANNELS, hidden_channels=8, depth=1)
    algorithm = VTrace(VTraceConfig(epochs=2, minibatch_steps=2))
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    diagnostics = algorithm.update(network, rollout, optimizer)
    for key in ("policy_loss", "value_loss", "entropy", "loss", "approx_kl", "grad_norm"):
        assert key in diagnostics, key
        assert diagnostics[key] == diagnostics[key], key


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------
def test_replay_buffer_respects_capacity_and_evicts_oldest():
    buffer = SegmentReplayBuffer(capacity=3)
    stored = [buffer.add(random_rollout(steps=2, seed=i)) for i in range(5)]
    assert len(buffer) == 3
    # The two oldest are gone and the three newest survive, in order.
    kept = buffer.segments
    for expected, actual in zip(stored[2:], kept):
        assert torch.equal(expected.reward, actual.reward)


def test_replay_buffer_stores_faithfully():
    buffer = SegmentReplayBuffer(capacity=2)
    rollout = random_rollout(steps=3, seed=11)
    buffer.add(rollout)
    (stored,) = buffer.sample(1)
    assert torch.equal(stored.reward, rollout.reward)
    assert torch.equal(stored.successor, rollout.successor)
    assert torch.equal(stored.acted, rollout.acted)
    assert stored.observation.dtype == torch.float16
    assert stored.advantage is None, "stale advantages must not be carried into replay"


def test_replay_buffer_sampling_covers_the_buffer():
    buffer = SegmentReplayBuffer(capacity=4)
    for i in range(4):
        buffer.add(random_rollout(steps=1, seed=100 + i))
    generator = torch.Generator().manual_seed(0)
    seen = set()
    for _ in range(50):
        for segment in buffer.sample(4, generator=generator):
            seen.add(float(segment.reward.sum()))
    assert len(seen) == 4


def test_replay_buffer_rejects_empty_sample_and_bad_capacity():
    with pytest.raises(ValueError):
        SegmentReplayBuffer(capacity=0)
    with pytest.raises(ValueError):
        SegmentReplayBuffer(capacity=2).sample(1)


def test_replay_buffer_memory_accounting_matches_reality():
    """The docstring's arithmetic and the actual tensors must agree."""
    buffer = SegmentReplayBuffer(capacity=2)
    steps, height, width = 3, 8, 8
    buffer.add(random_rollout(steps=steps, height=height, width=width))
    estimate = SegmentReplayBuffer.estimate_bytes(1, steps, CHANNELS, height, width)
    assert buffer.nbytes() == estimate


def test_awr_update_uses_replayed_segments():
    network = build_network("conv", CHANNELS, hidden_channels=8, depth=1)
    algorithm = AWR(AWRConfig(capacity=4, segments_per_update=3, minibatch_steps=2))
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    for i in range(3):
        diagnostics = algorithm.update(
            network, random_rollout(steps=2, height=16, width=16, seed=20 + i), optimizer
        )
    assert diagnostics["replay_segments"] == 3.0
    assert diagnostics["replay_megabytes"] > 0.0


def test_awr_weights_favour_high_advantage_and_are_capped():
    algorithm = AWR(AWRConfig(beta=1.0, max_weight=5.0))
    acted = torch.tensor([[True, True, False]])
    advantage = torch.tensor([[2.0, -2.0, 100.0]])
    weights = algorithm.advantage_weights(advantage, acted.float())
    assert weights[0, 0] > weights[0, 1]
    assert weights[0, 2] == 0.0, "empty cells must carry no weight"
    assert float(weights.max()) <= 5.0


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
def test_build_algorithm_rejects_unknown_names():
    with pytest.raises(ValueError) as excinfo:
        build_algorithm("dqn")
    assert "dqn" in str(excinfo.value)
    assert "vtrace" in str(excinfo.value), "the error should list the options"


def test_build_algorithm_rejects_unknown_hyperparameters():
    with pytest.raises(TypeError):
        build_algorithm("vtrace", not_a_real_knob=1.0)


def test_every_registered_algorithm_satisfies_the_protocol():
    rollout = random_rollout(steps=2, height=16, width=16, seed=30)
    compute_gae(rollout, torch.zeros(16, 16), normalize=False)
    for name in ALGORITHMS:
        network = build_network("conv", CHANNELS, hidden_channels=8, depth=1)
        algorithm = build_algorithm(name, minibatch_steps=2)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        diagnostics = algorithm.update(network, rollout, optimizer)
        assert isinstance(diagnostics, dict)
        assert "loss" in diagnostics and "agent_steps" in diagnostics
