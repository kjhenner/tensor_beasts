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
    # (T, H, W) float32. When the policy has a metabolic head this is the JOINT
    # log-probability, direction plus metabolic level, because the two heads
    # are independent categoricals per cell and the PPO ratio is taken over the
    # joint action. With no metabolic head it is the direction log-prob alone.
    log_prob: torch.Tensor
    value: torch.Tensor  # (T, H, W) float32
    reward: torch.Tensor  # (T, H, W) float32
    done: torch.Tensor  # (T, H, W) bool
    successor: torch.Tensor  # (T, H, W) int64
    advantage: Optional[torch.Tensor] = None  # (T, H, W) float32
    ret: Optional[torch.Tensor] = None  # (T, H, W) float32
    # (H, W) value of the state after the final step, set by compute_gae.
    # Algorithms need this to bootstrap individuals that are still alive when
    # the segment ends. Without it they have to reuse the last step's own value,
    # which is a step stale and biases every segment boundary the same way.
    last_value: Optional[torch.Tensor] = None
    # (T, H, W) int64, the rule-based policy's direction at each cell, or None
    # if the environment did not supply it. Used by the imitation term.
    rule_action: Optional[torch.Tensor] = None
    # (T, 5, H, W) float32, the rule's per-action scores, or None.
    rule_scores: Optional[torch.Tensor] = None
    # (T, H, W) bool, individuals that divided this step. Recurrent training
    # needs it to route an inherited memory copy to the offspring.
    reproduced: Optional[torch.Tensor] = None
    # (T, H, W) int64, the metabolic level each individual chose, or None when
    # the policy has no metabolic head.
    metabolic_action: Optional[torch.Tensor] = None
    # (T, H, W) int64, the rule's own metabolic rate as the nearest level, or
    # None. The anchor target for the metabolic head.
    rule_metabolic_level: Optional[torch.Tensor] = None

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
        self._rule_action: List[torch.Tensor] = []
        self._rule_scores: List[torch.Tensor] = []
        self._reproduced: List[torch.Tensor] = []
        self._metabolic_action: List[torch.Tensor] = []
        self._rule_metabolic_level: List[torch.Tensor] = []

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
        if batch.rule_action is not None:
            self._rule_action.append(batch.rule_action.detach())
        if batch.rule_scores is not None:
            self._rule_scores.append(batch.rule_scores.detach().to(torch.float16))
        self._reproduced.append(batch.reproduced.detach())
        if batch.metabolic_action is not None:
            self._metabolic_action.append(batch.metabolic_action.detach())
        if batch.rule_metabolic_level is not None:
            self._rule_metabolic_level.append(batch.rule_metabolic_level.detach())

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
            rule_action=(
                torch.stack(self._rule_action)
                if len(self._rule_action) == len(self._successor)
                else None
            ),
            rule_scores=(
                torch.stack(self._rule_scores)
                if len(self._rule_scores) == len(self._successor)
                else None
            ),
            metabolic_action=(
                torch.stack(self._metabolic_action)
                if len(self._metabolic_action) == len(self._successor)
                else None
            ),
            rule_metabolic_level=(
                torch.stack(self._rule_metabolic_level)
                if len(self._rule_metabolic_level) == len(self._successor)
                else None
            ),
            reproduced=(
                torch.stack(self._reproduced)
                if len(self._reproduced) == len(self._successor)
                else None
            ),
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
            self._rule_action,
            self._rule_scores,
            self._metabolic_action,
            self._rule_metabolic_level,
            self._reproduced,
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
    grid = rollout.reward.shape[1:]
    height, width = grid[-2:]
    # Everything is flattened to (worlds, H * W) rather than to (-1), and the
    # gather runs along the last axis. That is what keeps a batched world's
    # individuals inside their own world: successor indices are per world, so a
    # bare `.reshape(-1)` over (B, H, W) would index a B * H * W buffer with
    # per-world indices and quietly bootstrap world 2's animals from world 0.
    # With one world this is (1, H * W) and arithmetically identical to before.
    cells = height * width
    worlds = int(torch.tensor(grid).prod().item() // cells)

    def per_world(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(worlds, cells)

    advantage = torch.zeros_like(rollout.reward)
    next_advantage = torch.zeros(worlds, cells, device=rollout.reward.device)
    next_value = per_world(last_value)

    for t in reversed(range(steps)):
        successor = per_world(rollout.successor[t])
        alive = per_world(~rollout.done[t]).float()

        # Follow each individual to wherever it actually ended up, within its
        # own world.
        bootstrap_value = next_value.gather(1, successor) * alive
        bootstrap_advantage = next_advantage.gather(1, successor) * alive

        value_t = per_world(rollout.value[t])
        delta = per_world(rollout.reward[t]) + gamma * bootstrap_value - value_t
        advantage_t = delta + gamma * gae_lambda * bootstrap_advantage

        # Cells with no individual carry no signal onward.
        acted = per_world(rollout.acted[t])
        advantage_t = advantage_t * acted

        advantage[t] = advantage_t.reshape(grid)
        next_advantage = advantage_t
        next_value = value_t

    rollout.advantage = advantage
    rollout.ret = advantage + rollout.value
    rollout.last_value = last_value

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

    The collection-time ``value`` is included because a clipped value loss needs
    it and it cannot be recovered from the rest: ``ret`` was formed as
    ``advantage + value`` *before* ``compute_gae`` standardized the advantages
    in place, so ``ret - advantage`` is no longer the original value.

    Yields:
        (observation, acted, action, log_prob, advantage, ret, value) tuples.
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
            rollout.value[index],
        )
