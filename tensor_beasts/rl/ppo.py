"""Proximal policy optimization for a grid of individuals sharing one policy.

Why this is not just a copy of a standard PPO implementation
------------------------------------------------------------

Every tensor in this setting is grid shaped, ``(T, H, W)``, and almost every
cell is empty. On a 256x256 world a healthy herbivore population is a few
thousand individuals, so fewer than five percent of cells hold an agent. A
standard ``loss.mean()`` therefore divides by 65,536 instead of by the number
of agents, and the effective learning rate then scales with population density:
gradients shrink during a bust and grow during a boom. That looks exactly like
a mysterious learning-rate problem and is invisible in the loss curve, because
the loss is small for the same reason the gradient is.

So every reduction in this module goes through :func:`masked_mean` over the
``acted`` mask, and nothing else. The one place that is easy to get wrong is the
entropy bonus, which is defined at every cell whether or not anybody is standing
there; masking it is what keeps the entropy number interpretable as
"entropy of a real agent's action distribution".

Slotting in other algorithms
----------------------------

:class:`PPO` implements :class:`Algorithm`, which is the whole contract::

    diagnostics = algorithm.update(network, rollout, optimizer)

A V-trace learner or a value-based method needs the same inputs (a
:class:`~tensor_beasts.rl.rollout.Rollout` and a network with the
``(logits, value)`` interface) and returns the same kind of diagnostics dict, so
the trainer does not need to know which one it is holding. The masking helpers
here are deliberately free functions for the same reason.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Iterator, Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor_beasts.rl.normalization import ValueNormalizer
from tensor_beasts.rl.rollout import Rollout


def iter_minibatches_with_value(
    rollout: Rollout,
    minibatch_steps: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Iterator[Tuple[torch.Tensor, ...]]:
    """``rollout.iter_minibatches`` plus the value recorded at collection time.

    The shared iterator yields six tensors and leaves out ``rollout.value``. The
    clipped value loss needs it, and it cannot be recovered from what is
    yielded: ``ret`` was formed as ``advantage + value`` *before* ``compute_gae``
    standardized the advantages in place, so ``ret - advantage`` is no longer
    the original value. Rather than reach into the rollout module, this repeats
    its four-line slicing with the extra field. If ``iter_minibatches`` ever
    grows a ``value`` output, delete this and use it.

    Yields:
        (observation, acted, action, log_prob, value, advantage, ret).
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
            rollout.log_prob[index],
            rollout.value[index],
            rollout.advantage[index],
            rollout.ret[index],
        )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the True entries of ``mask``.

    Returns a zero that still carries gradient structure when the mask is empty,
    which happens for real: a minibatch of timesteps can contain no living
    herbivore at all during a population crash.
    """
    count = mask.sum()
    if count == 0:
        return (values * 0.0).sum()
    return (values * mask).sum() / count


def explained_variance(value: torch.Tensor, ret: torch.Tensor, mask: torch.Tensor) -> float:
    """1 - Var(ret - value) / Var(ret), over acting cells only.

    Zero means the critic is no better than predicting the mean return; one
    means it is perfect; negative means it is worse than the mean.
    """
    if mask.sum() < 2:
        return float("nan")
    target = ret[mask]
    predicted = value[mask]
    variance = target.var(unbiased=False)
    if variance == 0:
        return float("nan")
    return float(1.0 - (target - predicted).var(unbiased=False) / variance)


@dataclass
class PPOConfig:
    """Hyperparameters. Defaults are the usual PPO starting point.

    Attributes:
        learning_rate: Adam step size.
        clip_range: Policy ratio clip epsilon.
        value_clip_range: If set, the value function is clipped to this
            distance from the value recorded at collection time. Off by default
            because value clipping helps mainly when returns are large and
            non-stationary; here it is worth trying precisely because the
            ecology is non-stationary, so it is exposed as an option.
        entropy_coef: Weight on the entropy bonus.
        value_coef: Weight on the value loss.
        epochs: Passes over each collected segment.
        minibatch_steps: Timesteps per minibatch. Minibatching is over whole
            timesteps because the policy is convolutional; see
            :func:`~tensor_beasts.rl.rollout.iter_minibatches`.
        max_grad_norm: Global gradient clipping.
        gamma: Discount, used by the trainer when it calls compute_gae.
        gae_lambda: GAE trace decay, likewise.
        normalize_advantage: Standardize advantages over acting cells. Applied
            by compute_gae over the whole segment; this flag is passed through.
        target_kl: If set, stop the epoch loop early once approximate KL exceeds
            it. A safety valve for the non-stationarity, not a tuning knob.
    """

    learning_rate: float = 3e-4
    clip_range: float = 0.2
    value_clip_range: Optional[float] = None
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 4
    minibatch_steps: int = 16
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    target_kl: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class Algorithm(Protocol):
    """The contract a learner has to satisfy to be usable by the trainer."""

    def update(
        self,
        network: nn.Module,
        rollout: Rollout,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        ...


@dataclass
class _Accumulator:
    """Sums diagnostics weighted by the number of agent-steps behind them.

    Averaging minibatch diagnostics unweighted would give a timestep with three
    surviving herbivores the same say as one with three thousand.
    """

    totals: Dict[str, float] = field(default_factory=dict)
    weight: float = 0.0

    def add(self, values: Dict[str, float], weight: float) -> None:
        if weight <= 0:
            return
        self.weight += weight
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + value * weight

    def mean(self) -> Dict[str, float]:
        if self.weight == 0:
            return {key: float("nan") for key in self.totals}
        return {key: value / self.weight for key, value in self.totals.items()}


class PPO:
    """Clipped-surrogate PPO with every loss term masked to acting cells."""

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        value_normalizer: Optional[ValueNormalizer] = None,
    ):
        self.config = config or PPOConfig()
        # Without normalization the value term is the entire gradient norm and
        # global clipping leaves the policy with a thousandth of its intended
        # step. See tensor_beasts/rl/normalization.py for the measurement.
        self.value_normalizer = value_normalizer or ValueNormalizer(enabled=False)

    # ------------------------------------------------------------------
    # Evaluation of a batch of grids under the current policy
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_actions(
        network: nn.Module,
        observation: torch.Tensor,
        action: torch.Tensor,
    ):
        """Return (log_prob, entropy, value), each grid shaped ``(B, H, W)``.

        Written by hand rather than through ``torch.distributions.Categorical``
        because that would need a permute to put the action axis last on a
        four-dimensional tensor, which is both a copy of the largest tensor in
        the loop and an easy place to transpose height and width by accident.
        """
        logits, value = network(observation)
        log_probs = F.log_softmax(logits, dim=1)
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1)
        return log_prob, entropy, value

    # ------------------------------------------------------------------
    # One minibatch
    # ------------------------------------------------------------------
    def _losses(
        self,
        network: nn.Module,
        observation: torch.Tensor,
        acted: torch.Tensor,
        action: torch.Tensor,
        old_log_prob: torch.Tensor,
        old_value: torch.Tensor,
        advantage: torch.Tensor,
        ret: torch.Tensor,
    ):
        config = self.config
        log_prob, entropy, value = self.evaluate_actions(network, observation, action)

        mask = acted.float()
        log_ratio = (log_prob - old_log_prob) * mask
        ratio = torch.exp(log_ratio)

        surrogate = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantage
        policy_loss = -masked_mean(torch.min(surrogate, clipped), mask)

        # The network predicts normalized values, so the target and the
        # collection-time value are brought onto the same scale before the loss.
        # `value` is left exactly as the network produced it.
        target = self.value_normalizer.normalize(ret)
        old_value_scaled = self.value_normalizer.normalize(old_value)

        if config.value_clip_range is None:
            value_loss = masked_mean((value - target) ** 2, mask)
        else:
            clipped_value = old_value_scaled + torch.clamp(
                value - old_value_scaled, -config.value_clip_range, config.value_clip_range
            )
            value_loss = masked_mean(
                torch.max((value - target) ** 2, (clipped_value - target) ** 2), mask
            )

        entropy_mean = masked_mean(entropy, mask)
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean

        with torch.no_grad():
            # Schulman's k3 estimator: low variance and always non-negative.
            approx_kl = masked_mean((ratio - 1.0) - log_ratio, mask)
            clip_fraction = masked_mean(
                ((ratio - 1.0).abs() > config.clip_range).float(), mask
            )
            ratio_deviation = ((ratio - 1.0).abs() * mask).max() if mask.any() else torch.zeros(())

        diagnostics = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "loss": float(loss.detach()),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
            "ratio_max_deviation": float(ratio_deviation),
        }
        return loss, diagnostics

    def minibatch_loss(self, network: nn.Module, batch) -> torch.Tensor:
        """Loss for one :func:`iter_minibatches_with_value` tuple. For tests."""
        observation, acted, action, old_log_prob, old_value, advantage, ret = batch
        loss, _ = self._losses(
            network, observation, acted, action, old_log_prob, old_value, advantage, ret
        )
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
    ) -> Dict[str, float]:
        """Run the PPO epochs over one collected segment.

        Args:
            network: Shared actor-critic. Updated in place.
            rollout: Segment with ``advantage`` and ``ret`` already populated by
                :func:`~tensor_beasts.rl.rollout.compute_gae`.
            optimizer: Optimizer over ``network.parameters()``.
            generator: Optional RNG for minibatch shuffling, for reproducibility.

        Returns:
            Diagnostics averaged over minibatches, weighted by agent-steps:
            policy_loss, value_loss, entropy, loss, approx_kl, clip_fraction,
            ratio_max_deviation, explained_variance, grad_norm, epochs_run.
        """
        if rollout.advantage is None or rollout.ret is None:
            raise ValueError(
                "Rollout has no advantages. Call compute_gae before update()."
            )

        # Fold this segment's returns into the running scale before the epochs
        # run, so target and old value are normalized by the same statistics.
        self.value_normalizer.update(rollout.ret, rollout.acted)

        config = self.config
        accumulator = _Accumulator()
        epochs_run = 0

        for epoch in range(config.epochs):
            epochs_run = epoch + 1
            epoch_kl = _Accumulator()

            for batch in iter_minibatches_with_value(
                rollout, config.minibatch_steps, shuffle=True, generator=generator
            ):
                observation, acted, action, old_log_prob, old_value, advantage, ret = batch
                agent_steps = float(acted.sum())
                if agent_steps == 0:
                    # No individuals in these timesteps. Nothing to learn from,
                    # and every masked mean would be a structural zero.
                    continue

                loss, diagnostics = self._losses(
                    network, observation, acted, action, old_log_prob, old_value, advantage, ret
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    network.parameters(), config.max_grad_norm
                )
                optimizer.step()

                diagnostics["grad_norm"] = float(grad_norm)
                accumulator.add(diagnostics, agent_steps)
                epoch_kl.add({"approx_kl": diagnostics["approx_kl"]}, agent_steps)

            if config.target_kl is not None:
                kl = epoch_kl.mean().get("approx_kl", 0.0)
                if kl == kl and kl > config.target_kl:  # not NaN and over budget
                    break

        result = accumulator.mean()
        result["explained_variance"] = explained_variance(
            rollout.value, rollout.ret, rollout.acted
        )
        result["epochs_run"] = float(epochs_run)
        result["agent_steps"] = float(rollout.num_agent_steps)
        return result
