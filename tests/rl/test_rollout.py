"""Advantage estimation must follow individuals, not cells.

The failure this guards against is silent and plausible-looking: bootstrap each
cell's value from the same cell at the next timestep. That is correct only if
nothing moves. Here agents move every step, so it would credit an individual
with the future of whichever animal wandered into the cell it left.
"""

import torch

from tensor_beasts.rl.multiagent import AgentBatch
from tensor_beasts.rl.rollout import RolloutBuffer, compute_gae, iter_minibatches


HEIGHT = WIDTH = 3
GAMMA = 0.9
LAMBDA = 0.8


def flat(y: int, x: int) -> int:
    return y * WIDTH + x


def make_batch(acted_cell, action=0, reward=0.0, done=False, successor_cell=None, reproduced=False):
    """One agent at ``acted_cell``, everything else empty."""
    acted = torch.zeros(HEIGHT, WIDTH, dtype=torch.bool)
    acted[acted_cell] = True

    successor = torch.zeros(HEIGHT, WIDTH, dtype=torch.long)
    successor[acted_cell] = flat(*(successor_cell if successor_cell else acted_cell))

    reward_grid = torch.zeros(HEIGHT, WIDTH)
    reward_grid[acted_cell] = reward

    done_grid = torch.zeros(HEIGHT, WIDTH, dtype=torch.bool)
    done_grid[acted_cell] = done

    reproduced_grid = torch.zeros(HEIGHT, WIDTH, dtype=torch.bool)
    reproduced_grid[acted_cell] = reproduced

    action_grid = torch.zeros(HEIGHT, WIDTH, dtype=torch.long)
    action_grid[acted_cell] = action

    return AgentBatch(
        observation=torch.zeros(2, HEIGHT, WIDTH),
        acted=acted,
        action=action_grid,
        reward=reward_grid,
        done=done_grid,
        successor=successor,
        reproduced=reproduced_grid,
    )


def value_grid(cell, value):
    grid = torch.zeros(HEIGHT, WIDTH)
    grid[cell] = value
    return grid


def test_gae_follows_a_moving_agent():
    """One agent walks (0,0) -> (0,1) -> (0,2), and we hand-compute GAE.

    Crucially, a decoy value is parked in the cell the agent vacates. If the
    recursion bootstrapped from the same cell rather than the successor, it
    would pick the decoy up and the numbers would not match.
    """
    path = [(0, 0), (0, 1), (0, 2)]
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.7, 0.9]
    last_value_at_end = 1.3

    buffer = RolloutBuffer(steps=3)
    for t in range(3):
        successor = path[t + 1] if t + 1 < len(path) else (0, 2)
        batch = make_batch(path[t], reward=rewards[t], successor_cell=successor)

        # Value for the agent's cell, plus a decoy in every other cell that a
        # cell-indexed recursion would wrongly consume.
        grid = torch.full((HEIGHT, WIDTH), -99.0)
        grid[path[t]] = values[t]
        buffer.add(batch, log_prob=torch.zeros(HEIGHT, WIDTH), value=grid)

    rollout = buffer.build()
    last_value = torch.full((HEIGHT, WIDTH), -99.0)
    last_value[0, 2] = last_value_at_end

    compute_gae(rollout, last_value, gamma=GAMMA, gae_lambda=LAMBDA, normalize=False)

    # Hand-rolled GAE along the individual's own path.
    next_values = [values[1], values[2], last_value_at_end]
    deltas = [rewards[t] + GAMMA * next_values[t] - values[t] for t in range(3)]
    expected = [0.0, 0.0, 0.0]
    carry = 0.0
    for t in reversed(range(3)):
        carry = deltas[t] + GAMMA * LAMBDA * carry
        expected[t] = carry

    for t in range(3):
        got = rollout.advantage[t][path[t]].item()
        assert abs(got - expected[t]) < 1e-5, f"step {t}: {got} != {expected[t]}"


def test_death_stops_the_bootstrap():
    """A dead individual gets no future value, even though its cell has one."""
    buffer = RolloutBuffer(steps=1)
    batch = make_batch((1, 1), reward=5.0, done=True, successor_cell=(1, 1))
    buffer.add(batch, log_prob=torch.zeros(HEIGHT, WIDTH), value=value_grid((1, 1), 2.0))
    rollout = buffer.build()

    # A large value sits in the very cell it died in.
    compute_gae(rollout, torch.full((HEIGHT, WIDTH), 100.0), gamma=GAMMA, gae_lambda=LAMBDA, normalize=False)

    # advantage = reward - value, with no bootstrap at all.
    assert abs(rollout.advantage[0][1, 1].item() - (5.0 - 2.0)) < 1e-5


def test_empty_cells_carry_no_advantage():
    buffer = RolloutBuffer(steps=2)
    for _ in range(2):
        batch = make_batch((2, 2), reward=1.0, successor_cell=(2, 2))
        buffer.add(batch, log_prob=torch.zeros(HEIGHT, WIDTH), value=torch.ones(HEIGHT, WIDTH))
    rollout = buffer.build()
    compute_gae(rollout, torch.ones(HEIGHT, WIDTH), gamma=GAMMA, gae_lambda=LAMBDA, normalize=False)

    for t in range(2):
        empty = ~rollout.acted[t]
        assert torch.all(rollout.advantage[t][empty] == 0)


def test_returns_are_advantage_plus_value():
    buffer = RolloutBuffer(steps=2)
    for _ in range(2):
        batch = make_batch((0, 0), reward=1.0, successor_cell=(0, 0))
        buffer.add(batch, log_prob=torch.zeros(HEIGHT, WIDTH), value=value_grid((0, 0), 0.4))
    rollout = buffer.build()
    compute_gae(rollout, value_grid((0, 0), 0.4), gamma=GAMMA, gae_lambda=LAMBDA, normalize=False)
    assert torch.allclose(rollout.ret, rollout.advantage + rollout.value)


def test_normalization_uses_only_acting_cells():
    buffer = RolloutBuffer(steps=4)
    for t in range(4):
        batch = make_batch((0, 0), reward=float(t), successor_cell=(0, 0))
        buffer.add(batch, log_prob=torch.zeros(HEIGHT, WIDTH), value=value_grid((0, 0), 0.1 * t))
    rollout = buffer.build()
    compute_gae(rollout, value_grid((0, 0), 0.0), gamma=GAMMA, gae_lambda=LAMBDA, normalize=True)

    acting = rollout.advantage[rollout.acted]
    assert abs(acting.mean().item()) < 1e-5
    assert abs(acting.std(unbiased=False).item() - 1.0) < 1e-4
    # Non-acting cells stay exactly zero rather than picking up the shift.
    assert torch.all(rollout.advantage[~rollout.acted] == 0)


def test_minibatches_cover_every_step_once():
    buffer = RolloutBuffer(steps=6)
    for _ in range(6):
        batch = make_batch((0, 0), reward=1.0, successor_cell=(0, 0))
        buffer.add(batch, log_prob=torch.zeros(HEIGHT, WIDTH), value=torch.zeros(HEIGHT, WIDTH))
    rollout = buffer.build()
    compute_gae(rollout, torch.zeros(HEIGHT, WIDTH), normalize=False)

    seen = 0
    for observation, acted, action, log_prob, advantage, ret, value in iter_minibatches(rollout, minibatch_steps=4):
        assert observation.dtype == torch.float32, "observations are stored as half and must come back as float"
        seen += observation.shape[0]
    assert seen == 6


def test_last_value_is_recorded_for_boundary_bootstrapping():
    """Algorithms need the post-segment value; reusing the last step's own value
    is a step stale and biases every segment boundary the same way."""
    buffer = RolloutBuffer(steps=2)
    for _ in range(2):
        batch = make_batch((1, 1), reward=1.0, successor_cell=(1, 1))
        buffer.add(batch, log_prob=torch.zeros(HEIGHT, WIDTH), value=torch.zeros(HEIGHT, WIDTH))
    rollout = buffer.build()
    last_value = value_grid((1, 1), 7.0)
    compute_gae(rollout, last_value, normalize=False)
    assert rollout.last_value is not None
    assert torch.equal(rollout.last_value, last_value)
