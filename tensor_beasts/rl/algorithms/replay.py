"""Advantage-weighted regression over a replay of grid-shaped segments.

Which replay method, and why this one
-------------------------------------

The motivation for replay here is the same as for V-trace: the simulation is
the expensive part. A 512x512 world advances at roughly nine steps per second
while a gradient step over the same grid costs milliseconds, so the ratio of
"gradient steps we could afford" to "transitions we can collect" is enormous.
Replay is the bluntest way to spend that ratio: every stored transition gets
used many times instead of once.

Three candidates were on the table. **Discrete SAC** and **Munchausen DQN** both
need a Q-function over the five actions, plus a target network, and in SAC's
case twin critics. Every network in
:mod:`tensor_beasts.rl.networks` exposes exactly one interface --- ``(logits,
value)`` --- with a scalar value head, so either method would mean either new
architectures or reinterpreting the policy logits as Q-values, which is a
misuse of a head that was deliberately initialized near-uniform for a policy.

**Advantage-weighted regression** (Peng et al. 2019) needs precisely the
interface that already exists. Its critic is a state-value function, fit by
regression to lambda-returns; its actor is fit by weighted maximum likelihood,
with weights ``exp(A / beta)``. No target network, no second critic, no
importance ratios, and no new head. It is also the most forgiving of stale data
of the three: the weights are a monotone function of an advantage estimate, so
a somewhat wrong advantage degrades the update smoothly instead of
bootstrapping a Q-function into divergence. In an ecology whose dynamics drift
under the policy, that robustness is worth more than the sample efficiency SAC
would buy.

The cost is that AWR is an approximate policy improvement rather than an exact
one: it never explicitly maximizes value, it just reweights behaviour toward
what looked good. If the buffer's data is much worse than the current policy,
AWR will hold the policy back. The FIFO buffer keeps the data recent for exactly
this reason.

Movement
--------

Value targets come from :func:`~tensor_beasts.rl.algorithms.vtrace.compute_vtrace`
with unit importance ratios, which makes them lambda-returns and inherits its
successor-gathering: the next-step value for an individual is read from the cell
that individual moved *into*, never from the cell it left. That is the only
place in this module where a temporal index appears, which is intentional --- one
implementation of the movement-aware recursion, tested once.

Everything else is masked to ``acted`` through
:func:`~tensor_beasts.rl.ppo.masked_mean`, as in the rest of the package.
"""

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from tensor_beasts.rl.ppo import PPO, _Accumulator, explained_variance, masked_mean
from tensor_beasts.rl.rollout import Rollout
from tensor_beasts.rl.algorithms.vtrace import compute_vtrace, normalize_masked

# Bytes per cell per step for everything that is not the observation:
# action (int64) + successor (int64) = 16, log_prob + value + reward (float32)
# = 12, acted + done (bool) = 2.
_BYTES_PER_CELL_STEP_WITHOUT_OBSERVATION = 30


class SegmentReplayBuffer:
    """FIFO replay over whole :class:`Rollout` segments, in grid layout.

    Segments rather than individual transitions, for two reasons. The value
    target is a multi-step return that has to follow individuals through the
    ``successor`` map, which only exists inside a contiguous segment; and the
    policy is convolutional, so a single forward pass over a whole grid is the
    natural unit of work anyway. Storing loose transitions would forfeit both.

    Memory, in the accounting style of :class:`~tensor_beasts.rl.rollout.Rollout`
    ----------------------------------------------------------------------------

    One stored segment costs ``steps * height * width * (2 * channels + 30)``
    bytes: observations are float16, everything else adds up to the 30 bytes per
    cell per step tabulated above. With 14 channels that is 58 bytes per cell
    per step, so a 64-step segment costs

        * 256x256: about 243 MB
        * 512x512: about 973 MB

    and the buffer holds ``capacity`` of them. A capacity of 8 at 256x256 is
    therefore about 1.9 GB, and the same capacity at 512x512 is about 7.8 GB,
    which will not fit on most accelerators. Two levers: pass ``device="cpu"``
    to keep the buffer in host memory and pay a transfer per sample, or shorten
    the segment. :meth:`nbytes` reports the actual figure so it can be logged
    rather than guessed at.

    Args:
        capacity: Maximum number of segments retained. Oldest is evicted first.
        device: Where to hold stored segments. ``None`` keeps them wherever they
            arrived; ``"cpu"`` is the usual choice when the learner is on an
            accelerator.
    """

    def __init__(self, capacity: int, device: Optional[str] = None):
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.capacity = capacity
        self.device = device
        self._segments: Deque[Rollout] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._segments)

    @property
    def segments(self) -> List[Rollout]:
        return list(self._segments)

    @property
    def total_steps(self) -> int:
        return sum(segment.steps for segment in self._segments)

    @property
    def total_agent_steps(self) -> int:
        return sum(segment.num_agent_steps for segment in self._segments)

    def add(self, rollout: Rollout) -> Rollout:
        """Store a segment, evicting the oldest once at capacity.

        The advantage and return fields are dropped: they were computed under
        whatever policy collected the segment and would be stale on the first
        replay. AWR recomputes them from ``value`` every time.

        Returns:
            The stored copy, which is the caller's tensors moved to
            ``self.device`` (or the caller's own tensors if no device was set).
        """
        stored = self._store(rollout)
        self._segments.append(stored)
        return stored

    def _store(self, rollout: Rollout) -> Rollout:
        def move(tensor: torch.Tensor) -> torch.Tensor:
            detached = tensor.detach()
            return detached.to(self.device) if self.device is not None else detached

        return Rollout(
            observation=move(rollout.observation),
            acted=move(rollout.acted),
            action=move(rollout.action),
            log_prob=move(rollout.log_prob),
            value=move(rollout.value),
            reward=move(rollout.reward),
            done=move(rollout.done),
            successor=move(rollout.successor),
        )

    def sample(
        self,
        count: int,
        generator: Optional[torch.Generator] = None,
        device: Optional[str] = None,
    ) -> List[Rollout]:
        """Sample ``count`` segments uniformly, with replacement.

        With replacement because the buffer is usually small in segment count
        (a handful of big grids, not a million transitions) and sampling without
        replacement would cap the number of gradient steps per update at the
        buffer size.

        Args:
            count: How many segments to return.
            generator: Optional RNG, for reproducible tests.
            device: Optional device to move the sampled segments onto.
        """
        if not self._segments:
            raise ValueError("Cannot sample from an empty replay buffer.")
        index = torch.randint(
            len(self._segments), (count,), generator=generator, device="cpu"
        )
        sampled = [self._segments[int(i)] for i in index]
        if device is None:
            return sampled
        return [self._to_device(segment, device) for segment in sampled]

    @staticmethod
    def _to_device(rollout: Rollout, device: str) -> Rollout:
        return Rollout(
            observation=rollout.observation.to(device),
            acted=rollout.acted.to(device),
            action=rollout.action.to(device),
            log_prob=rollout.log_prob.to(device),
            value=rollout.value.to(device),
            reward=rollout.reward.to(device),
            done=rollout.done.to(device),
            successor=rollout.successor.to(device),
        )

    def nbytes(self) -> int:
        """Bytes actually held, summed over every stored tensor."""
        total = 0
        for segment in self._segments:
            for tensor in (
                segment.observation,
                segment.acted,
                segment.action,
                segment.log_prob,
                segment.value,
                segment.reward,
                segment.done,
                segment.successor,
            ):
                total += tensor.numel() * tensor.element_size()
        return total

    @staticmethod
    def estimate_bytes(capacity: int, steps: int, channels: int, height: int, width: int) -> int:
        """Bytes a full buffer of this shape will occupy. For sizing a run."""
        per_cell_step = 2 * channels + _BYTES_PER_CELL_STEP_WITHOUT_OBSERVATION
        return capacity * steps * height * width * per_cell_step

    def clear(self) -> None:
        self._segments.clear()


@dataclass
class AWRConfig:
    """Hyperparameters for :class:`AWR`.

    Attributes:
        learning_rate: Unused by ``update`` (the optimizer is supplied); carried
            so a config file can describe a run in one place.
        beta: Temperature on the advantage weights, ``exp(A / beta)``. Small
            beta makes the update nearly greedy toward the best-looking replayed
            actions; large beta makes it nearly behaviour cloning.
        max_weight: Cap on the advantage weight. Without it a single lucky
            individual can dominate a whole minibatch, which is exactly the
            failure mode AWR is known for.
        gamma: Discount.
        lambda_: Trace decay for the lambda-return value target.
        entropy_coef: Weight on the masked entropy bonus. Defaults to zero:
            AWR's weighting already keeps the policy near the data, and an
            entropy term fights it. Turn it up if the policy collapses.
        value_coef: Weight on the value loss.
        capacity: Segments retained by the replay buffer.
        segments_per_update: Segments sampled per call to ``update``. This is
            the replay ratio knob: with a capacity of 8 and 4 segments per
            update, each collected segment is revisited several times before it
            is evicted.
        value_epochs: Passes over each sampled segment spent on the critic
            before the actor sees it. The actor's weights are a function of the
            critic, so a critic that has not caught up produces meaningless
            weights; one extra pass is cheap insurance.
        minibatch_steps: Timesteps per minibatch.
        max_grad_norm: Global gradient clipping.
        normalize_advantage: Standardize advantages over acting cells before
            exponentiating. Strongly recommended: it makes ``beta`` mean the
            same thing regardless of the reward scale of the moment.
        buffer_device: Where to hold the replay buffer. ``"cpu"`` for large
            worlds.
    """

    learning_rate: float = 3e-4
    beta: float = 1.0
    max_weight: float = 20.0
    gamma: float = 0.99
    lambda_: float = 0.95
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    capacity: int = 8
    segments_per_update: int = 4
    value_epochs: int = 1
    minibatch_steps: int = 16
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    buffer_device: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class AWR:
    """Advantage-weighted regression with a segment replay. Implements ``Algorithm``.

    ``update`` stores the incoming segment, then takes gradient steps on
    segments sampled from the buffer, so a single call trains on both fresh and
    stale data. The buffer is owned by the algorithm rather than the trainer
    because the trainer's contract passes one segment at a time and nothing
    else.
    """

    def __init__(self, config: Optional[AWRConfig] = None):
        self.config = config or AWRConfig()
        self.buffer = SegmentReplayBuffer(self.config.capacity, device=self.config.buffer_device)

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    def compute_targets(
        self,
        network: nn.Module,
        rollout: Rollout,
        last_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(value target, advantage) for a replayed segment, under the current critic.

        Unit importance ratios turn :func:`compute_vtrace` into the lambda-return
        recursion, which is what AWR wants: its advantage is
        ``R_lambda - V(s)``, with no correction for the fact that the actions
        were chosen by an older policy. AWR deliberately does not correct for
        that --- the stale actions are the regression targets.
        """
        config = self.config
        log_prob, value = self._evaluate_segment(network, rollout)
        if last_value is None:
            # Same one-step-stale bootstrap as V-trace; see its ``compute_targets``.
            last_value = value[-1]

        vs, _ = compute_vtrace(
            rollout,
            value=value,
            log_ratio=torch.zeros_like(value),
            last_value=last_value,
            gamma=config.gamma,
            lambda_=config.lambda_,
            rho_bar=1.0,
            c_bar=1.0,
        )
        advantage = (vs - value) * rollout.acted
        if config.normalize_advantage:
            advantage = normalize_masked(advantage, rollout.acted)
        return vs, advantage

    def _evaluate_segment(
        self, network: nn.Module, rollout: Rollout
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.config.minibatch_steps
        log_probs, values = [], []
        with torch.no_grad():
            for start in range(0, rollout.steps, chunk):
                stop = min(start + chunk, rollout.steps)
                log_prob, _, value = PPO.evaluate_actions(
                    network,
                    rollout.observation[start:stop].float(),
                    rollout.action[start:stop],
                )
                log_probs.append(log_prob)
                values.append(value)
        return torch.cat(log_probs), torch.cat(values)

    def advantage_weights(self, advantage: torch.Tensor, acted: torch.Tensor) -> torch.Tensor:
        """``clamp(exp(A / beta), max=max_weight)``, zero where nothing acted.

        Zeroing rather than leaving ``exp(0) = 1`` at empty cells matters even
        though the loss is masked: the weights are also reported as diagnostics,
        and a mean weight computed over 65,536 cells of which 3,000 are real is
        a number about the grid, not about the agents.
        """
        config = self.config
        # Clamped before the exponential as well as after: a standardized
        # advantage divided by a small beta overflows float32 long before the
        # cap would have bitten, and inf * 0 at an empty cell is NaN.
        scaled = torch.clamp(advantage / config.beta, max=math.log(config.max_weight))
        weights = torch.clamp(torch.exp(scaled), max=config.max_weight)
        return weights * acted

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def _losses(
        self,
        network: nn.Module,
        observation: torch.Tensor,
        acted: torch.Tensor,
        action: torch.Tensor,
        vs: torch.Tensor,
        advantage: torch.Tensor,
        actor: bool = True,
    ):
        config = self.config
        log_prob, entropy, value = PPO.evaluate_actions(network, observation, action)
        mask = acted.float()

        value_loss = masked_mean((value - vs) ** 2, mask)
        entropy_mean = masked_mean(entropy, mask)

        if actor:
            weights = self.advantage_weights(advantage, mask)
            policy_loss = -masked_mean(log_prob * weights, mask)
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
            weight_mean = float(masked_mean(weights, mask).detach())
            weight_max = float((weights * mask).max()) if bool(acted.any()) else 0.0
        else:
            policy_loss = torch.zeros((), device=value.device)
            loss = config.value_coef * value_loss
            weight_mean = float("nan")
            weight_max = float("nan")

        diagnostics = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "loss": float(loss.detach()),
            "awr_weight_mean": weight_mean,
            "awr_weight_max": weight_max,
        }
        return loss, diagnostics

    def minibatch_loss(self, network: nn.Module, batch) -> torch.Tensor:
        """Loss for one :func:`iter_awr_minibatches` tuple. For tests."""
        observation, acted, action, vs, advantage = batch
        loss, _ = self._losses(network, observation, acted, action, vs, advantage)
        return loss

    # ------------------------------------------------------------------
    # The update
    # ------------------------------------------------------------------
    def update(
        self,
        network: nn.Module,
        rollout: Rollout,
        optimizer: torch.optim.Optimizer,
        generator: Optional[torch.Generator] = None,
        last_value: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Store the segment, then train on segments drawn from the buffer.

        Args:
            network: Shared actor-critic. Updated in place.
            rollout: Freshly collected segment. Added to the replay buffer;
                its ``advantage`` and ``ret`` are ignored.
            optimizer: Optimizer over ``network.parameters()``.
            generator: Optional RNG for sampling and shuffling.
            last_value: Optional (H, W) bootstrap for the *fresh* segment only.
                Replayed segments cannot use it --- they ended long ago --- so
                they always take the stale-bootstrap path.

        Returns:
            Diagnostics averaged over minibatches, weighted by agent-steps, plus
            replay bookkeeping (buffer size, replayed steps, memory).
        """
        config = self.config
        self.buffer.add(rollout)

        device = str(rollout.reward.device)
        # The fresh segment is always trained on, and always first: it is the
        # only one whose boundary bootstrap the caller can supply, and it is the
        # closest thing available to on-policy data. The rest are replayed.
        replayed = self.buffer.sample(
            max(config.segments_per_update - 1, 0), generator=generator, device=device
        ) if config.segments_per_update > 1 else []
        schedule = [(rollout, last_value)] + [(segment, None) for segment in replayed]

        accumulator = _Accumulator()
        last_vs = None
        last_segment = None

        for segment, bootstrap in schedule:
            vs, advantage = self.compute_targets(network, segment, bootstrap)
            last_vs, last_segment = vs, segment

            # Critic first: the actor's weights are a function of it.
            for epoch in range(config.value_epochs + 1):
                actor = epoch == config.value_epochs
                for batch in iter_awr_minibatches(
                    segment, vs, advantage, config.minibatch_steps, generator=generator
                ):
                    acted = batch[1]
                    agent_steps = float(acted.sum())
                    if agent_steps == 0:
                        continue

                    loss, diagnostics = self._losses(network, *batch, actor=actor)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        network.parameters(), config.max_grad_norm
                    )
                    optimizer.step()

                    if actor:
                        diagnostics["grad_norm"] = float(grad_norm)
                        accumulator.add(diagnostics, agent_steps)

        result = accumulator.mean()
        if last_vs is not None and last_segment is not None:
            result["explained_variance"] = explained_variance(
                last_segment.value, last_vs, last_segment.acted
            )
        result["replay_segments"] = float(len(self.buffer))
        result["replay_steps"] = float(self.buffer.total_steps)
        result["replay_megabytes"] = self.buffer.nbytes() / 1e6
        result["agent_steps"] = float(rollout.num_agent_steps)
        return result


def iter_awr_minibatches(
    rollout: Rollout,
    vs: torch.Tensor,
    advantage: torch.Tensor,
    minibatch_steps: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Iterator[Tuple[torch.Tensor, ...]]:
    """Minibatches of whole timesteps with externally computed AWR targets.

    Yields:
        (observation, acted, action, vs, advantage).
    """
    if shuffle:
        order = torch.randperm(rollout.steps, generator=generator)
    else:
        order = torch.arange(rollout.steps)
    for start in range(0, rollout.steps, minibatch_steps):
        index = order[start : start + minibatch_steps]
        yield (
            rollout.observation[index].float(),
            rollout.acted[index],
            rollout.action[index],
            vs[index],
            advantage[index],
        )
