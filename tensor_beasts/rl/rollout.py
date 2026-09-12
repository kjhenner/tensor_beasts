"""Rollout storage and advantage estimation for moving agents.

The interesting part is :func:`compute_gae`. Generalized advantage estimation
assumes you can follow an agent from one timestep to the next, but here agents
move, so the individual that acted from cell ``p`` at time ``t`` is somewhere
else at ``t + 1``. The usual grid-shaped recursion would silently bootstrap each
cell's value from whatever individual happened to be standing there next, which
is a different animal.

The fix is small: carry the successor map recorded by the simulation and gather
through it at every step of the backward recursion. The standard GAE recursion

    delta_t = r_t + gamma * V_{t+1} - V_t
    A_t     = delta_t + gamma * lambda * A_{t+1}

becomes, per individual,

    delta_t[p] = r_t[p] + gamma * V_{t+1}[succ_t[p]] - V_t[p]
    A_t[p]     = delta_t[p] + gamma * lambda * A_{t+1}[succ_t[p]]

which is one gather per timestep and stays fully vectorized. Individuals that
died have ``done`` set, which zeroes both bootstrap terms, so the fact that
another animal may have walked into their cell never matters.
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch

from tensor_beasts.rl.multiagent import AgentBatch


@dataclass
class Rollout:
    """A fixed-length segment of world time, in grid layout.

    Leading dimension is time. Every other field is indexed by the cell an
    individual acted from, so entries line up across fields.

    Memory is the reason observations are stored in half precision: a segment
    costs ``steps * channels * height * width * 2`` bytes for observations
    alone, which at 128 steps of 14 channels on a 256x256 world is about 235 MB
    and on a 512x512 world is about 940 MB. Shorten the segment or shrink the
    world rather than being surprised.
    """

    observation: torch.Tensor  # (T, C, H, W) float16
    acted: torch.Tensor  # (T, H, W) bool
    action: torch.Tensor  # (T, H, W) int64
    log_prob: torch.Tensor  # (T, H, W) float32
    value: torch.Tensor  # (T, H, W) float32
    reward: torch.Tensor  # (T, H, W) float32
    done: torch.Tensor  # (T, H, W) bool
    successor: torch.Tensor  # (T, H, W) int64
    advantage: Optional[torch.Tensor] = None  # (T, H, W) float32
    ret: Optional[torch.Tensor] = None  # (T, H, W) float32

    @property
    def steps(self) -> int:
        return self.observation.shape[0]

    @property
    def num_agent_steps(self) -> int:
        return int(self.acted.sum())


class RolloutBuffer:
    """Collects grid-shaped transitions for a fixed number of world steps."""

    def __init__(self, steps: int):
        self.steps = steps
        self._observation: List[torch.Tensor] = []
        self._acted: List[torch.Tensor] = []
        self._action: List[torch.Tensor] = []
        self._log_prob: List[torch.Tensor] = []
        self._value: List[torch.Tensor] = []
        self._reward: List[torch.Tensor] = []
        self._done: List[torch.Tensor] = []
        self._successor: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self._observation)

    def add(
        self,
        batch: AgentBatch,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._observation.append(batch.observation.detach().to(torch.float16))
        self._acted.append(batch.acted.detach())
        self._action.append(batch.action.detach())
        self._log_prob.append(log_prob.detach())
        self._value.append(value.detach())
        self._reward.append(batch.reward.detach())
        self._done.append(batch.done.detach())
        self._successor.append(batch.successor.detach())

    def build(self) -> Rollout:
        return Rollout(
            observation=torch.stack(self._observation),
            acted=torch.stack(self._acted),
            action=torch.stack(self._action),
            log_prob=torch.stack(self._log_prob),
            value=torch.stack(self._value),
            reward=torch.stack(self._reward),
            done=torch.stack(self._done),
            successor=torch.stack(self._successor),
        )

    def clear(self) -> None:
        for store in (
            self._observation,
            self._acted,
            self._action,
            self._log_prob,
            self._value,
            self._reward,
            self._done,
            self._successor,
        ):
            store.clear()


def compute_gae(
    rollout: Rollout,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize: bool = True,
) -> Rollout:
    """Generalized advantage estimation that follows each individual as it moves.

    Args:
        rollout: Collected segment. Modified in place and returned.
        last_value: (H, W) value estimate for the state after the final step,
            used to bootstrap individuals still alive at the segment boundary.
        gamma: Discount.
        gae_lambda: GAE trace decay.
        normalize: Whether to standardize advantages over the acting cells only.
            Normalizing over the whole grid would be dominated by empty cells.

    Returns:
        The same rollout with ``advantage`` and ``ret`` populated.
    """
    steps = rollout.steps
    height, width = rollout.reward.shape[-2:]

    advantage = torch.zeros_like(rollout.reward)
    next_advantage = torch.zeros(height * width, device=rollout.reward.device)
    next_value = last_value.reshape(-1)

    for t in reversed(range(steps)):
        successor = rollout.successor[t].reshape(-1)
        alive = (~rollout.done[t]).reshape(-1).float()

        # Follow each individual to wherever it actually ended up.
        bootstrap_value = next_value[successor] * alive
        bootstrap_advantage = next_advantage[successor] * alive

        value_t = rollout.value[t].reshape(-1)
        delta = rollout.reward[t].reshape(-1) + gamma * bootstrap_value - value_t
        advantage_t = delta + gamma * gae_lambda * bootstrap_advantage

        # Cells with no individual carry no signal onward.
        acted = rollout.acted[t].reshape(-1)
        advantage_t = advantage_t * acted

        advantage[t] = advantage_t.reshape(height, width)
        next_advantage = advantage_t
        next_value = value_t

    rollout.advantage = advantage
    rollout.ret = advantage + rollout.value

    if normalize:
        mask = rollout.acted
        if bool(mask.any()):
            values = rollout.advantage[mask]
            centered = rollout.advantage - values.mean()
            scaled = centered / (values.std(unbiased=False) + 1e-8)
            rollout.advantage = torch.where(mask, scaled, torch.zeros_like(scaled))

    return rollout


def iter_minibatches(
    rollout: Rollout,
    minibatch_steps: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Iterator[Tuple[torch.Tensor, ...]]:
    """Yield minibatches of whole timesteps.

    Minibatching is over timesteps rather than individuals because the policy is
    fully convolutional: one forward pass covers every individual on the grid at
    once, so splitting the agents within a step would mean recomputing the same
    convolution repeatedly. Each yielded element is a stack of complete grids,
    and the loss masks out cells where nothing acted.

    Yields:
        (observation, acted, action, log_prob, advantage, ret) tuples.
    """
    order = torch.randperm(rollout.steps, generator=generator) if shuffle else torch.arange(rollout.steps)
    for start in range(0, rollout.steps, minibatch_steps):
        index = order[start : start + minibatch_steps]
        yield (
            rollout.observation[index].float(),
            rollout.acted[index],
            rollout.action[index],
            rollout.log_prob[index],
            rollout.advantage[index],
            rollout.ret[index],
        )
