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

Two levers
----------

When the network carries a metabolic head (see
:class:`~tensor_beasts.rl.networks.ActorCritic`) the per-cell policy is the
product of a categorical over directions and a Gaussian over a continuous
throttle. The joint log-probability is the sum of the two, the entropy bonus
is the sum of the two entropies, and the clipped ratio is taken over the
joint, so ``rollout.log_prob`` must hold the joint log-prob at collection
time.

The rule-based policy and the learner
-------------------------------------

The rules are the learner's initialisation and its yardstick, nothing more.
:func:`rule_distillation` and :func:`throttle_distillation` are the losses
pretraining fits both heads with; during RL the same functions run without
gradient so the log shows how far the policy has drifted from the rules
(``conformance``, ``argmax_agreement``, ``metabolic_error``), and nothing
pulls it back. An anchor that kept pulling during RL, in three versions,
was what kept every predator run alive and what hid that RL was degrading
the policy (planning/HISTORY.md). It was removed rather than tuned.

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
from typing import Dict, Iterator, Optional, Protocol, Tuple, List

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor_beasts.rl.normalization import ValueNormalizer
from tensor_beasts.rl.recurrent import propagate_memory
from tensor_beasts.rl.rollout import Rollout

NUM_ACTIONS = 5


# Bounds on the metabolic head's learned log standard deviation. The floor stops
# the Gaussian collapsing to a delta, which makes its log-probability explode
# and the PPO ratio with it; the ceiling stops it widening to cover the whole
# range, which is the other degenerate solution and costs nothing to rule out.
MIN_LOG_STD = -4.0
MAX_LOG_STD = 1.0

# The scale pretraining judges throttle disagreement at, in normalised
# throttle units, so a tenth of the basal-to-max range costs half a nat.
# Fixed rather than the policy's own learned std: under the learned std the
# pull on the mean scaled as 1/std^2 while the same term shrank the std, so
# the fit strengthened itself until the throttle was pinned.
# The policy's std is exploration only, set by its initialisation and PPO.
METABOLIC_ANCHOR_STD = 0.1


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
        (observation, acted, action, log_prob, value, advantage, ret, rule_action,
        rule_scores, metabolic_unit, rule_metabolic_unit), the last four None
        if the rollout does not carry them.
    """
    if shuffle:
        order = torch.randperm(rollout.steps, generator=generator)
    else:
        order = torch.arange(rollout.steps)

    # With several worlds a stored field is (T, B, ..., H, W), and a convolution
    # wants one batch axis. Time and worlds are folded together here, at the
    # minibatch boundary, so every loss downstream keeps seeing exactly the
    # (batch, ..., H, W) it always did. Folding is correct because the losses
    # are per cell and masked by `acted`: a timestep and a world are both just
    # independent samples once the advantages have been computed, and those were
    # computed per world, before this.
    worlds = rollout.observation.shape[1] if rollout.observation.dim() == 5 else 0

    def fold(tensor):
        """(T, B, ..., H, W) -> (T * B, ..., H, W); unchanged without worlds."""
        if tensor is None or not worlds:
            return tensor
        return tensor.reshape(tensor.shape[0] * worlds, *tensor.shape[2:])

    for start in range(0, rollout.steps, minibatch_steps):
        index = order[start : start + minibatch_steps]
        yield (
            fold(rollout.observation[index].float()),
            fold(rollout.acted[index]),
            fold(rollout.action[index]),
            fold(rollout.log_prob[index]),
            fold(rollout.value[index]),
            fold(rollout.advantage[index]),
            fold(rollout.ret[index]),
            fold(rollout.rule_action[index]) if rollout.rule_action is not None else None,
            fold(rollout.rule_scores[index].float()) if rollout.rule_scores is not None else None,
            fold(rollout.metabolic_unit[index]) if rollout.metabolic_unit is not None else None,
            fold(rollout.rule_metabolic_unit[index]) if rollout.rule_metabolic_unit is not None else None,
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


def rule_distillation(
    log_probs: torch.Tensor,
    rule_action: torch.Tensor,
    rule_scores: Optional[torch.Tensor],
    temperature: float,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """How far a direction policy is from the rule, as a loss and two readings.

    Args:
        log_probs: (B, 5, H, W) log-probabilities over directions.
        rule_action: (B, H, W) the rule's direction.
        rule_scores: (B, 5, H, W) the rule's per-direction scores, or None.
        temperature: Softmax temperature over the scores. Positive with
            scores present is soft distillation toward softmax(scores / T);
            otherwise hard cross-entropy to the rule's direction.
        mask: (B, H, W) acting cells.

    Returns:
        (loss, conformance, argmax_agreement), each a scalar tensor. Loss is
        the masked mean KL (soft) or negative log-likelihood (hard).
        Conformance is progress from a uniform policy (0) to an exact match
        (1) under the soft target, or argmax agreement under the hard one.

    Soft, because the rule's decisions are knife-edge: its absolute score
    gaps have a median near 0.01, so hard argmax imitation spends its
    gradient on coin-flips. At a temperature of that order a typical
    decision is a mild preference, a clear one is sharp, and a near-tie is
    near-uniform and costs nothing to disagree with.
    """
    with torch.no_grad():
        argmax_agreement = masked_mean((log_probs.argmax(dim=1) == rule_action).float(), mask)
    if rule_scores is not None and temperature > 0.0:
        target = F.softmax(rule_scores / temperature, dim=1)
        kl = (target * (torch.log(target + 1e-12) - log_probs)).sum(dim=1)
        loss = masked_mean(kl, mask)
        with torch.no_grad():
            # Progress from a uniform policy toward an exact match. The raw
            # Bhattacharyya overlap with a diffuse target is already about
            # 0.77 for a uniform policy, so it is rescaled: uniform sits at 0,
            # an exact match at 1, comparable with argmax agreement.
            overlap = (target * log_probs.exp()).sqrt().sum(dim=1)
            uniform_overlap = (target / NUM_ACTIONS).sqrt().sum(dim=1)
            progress = (overlap - uniform_overlap) / (1.0 - uniform_overlap).clamp(min=1e-6)
            conformance = masked_mean(progress.clamp(-1.0, 1.0), mask)
    else:
        rule_log_prob = log_probs.gather(1, rule_action.unsqueeze(1)).squeeze(1)
        loss = -masked_mean(rule_log_prob, mask)
        conformance = argmax_agreement
    return loss, conformance, argmax_agreement


def throttle_distillation(
    mean: torch.Tensor, rule_unit: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """How far a throttle head's mean is from the rule's throttle.

    Returns (loss, error): the masked Gaussian negative log-likelihood of the
    rule's unit around the mean at METABOLIC_ANCHOR_STD, up to a constant,
    and the masked mean absolute distance in normalised units, so 0.1 is a
    tenth of the range from basal to maximum. Fits the mean only; the
    policy's std is exploration and is not the rule's to set.
    """
    rule_unit = rule_unit.float()
    loss = masked_mean(0.5 * ((rule_unit - mean) / METABOLIC_ANCHOR_STD) ** 2, mask)
    with torch.no_grad():
        error = masked_mean((mean - rule_unit).abs(), mask)
    return loss, error


def masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Population standard deviation of ``values`` over the True entries of ``mask``."""
    mean = masked_mean(values, mask)
    return masked_mean((values - mean) ** 2, mask).clamp(min=0.0).sqrt()


def masked_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of ``a`` and ``b`` over the True entries of ``mask``.

    Zero when either side is constant, which is the honest reading: a flat
    throttle is not correlated with anything.
    """
    a_std = masked_std(a, mask)
    b_std = masked_std(b, mask)
    if float(a_std) <= 1e-6 or float(b_std) <= 1e-6:
        return torch.zeros((), device=a.device)
    a_centred = a - masked_mean(a, mask)
    b_centred = b - masked_mean(b, mask)
    return masked_mean(a_centred * b_centred, mask) / (a_std * b_std)


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
    # Zero: thousands of sampled individuals explore already, and in the runs
    # that collapsed the direction entropy rose on its own as the policy
    # diffused. A bonus for that is a bonus for the failure.
    entropy_coef: float = 0.0
    # Pretraining only: the temperature of the soft target softmax(rule scores
    # / T) the direction head is distilled toward. See rule_distillation for
    # why it is soft and why 0.01. Zero falls back to hard imitation.
    imitation_temperature: float = 0.01
    # Recurrent training of the memory write. Zero is off: the memory read at
    # each step is the stored one and the write is a fixed function of the
    # observation. Positive N replays each segment in
    # time order, feeds every step's recomputed write to the next step's read
    # through the successor map, and backpropagates through windows of N
    # steps, detaching at window boundaries. That is what lets the gradient at
    # a decision reach the earlier step that wrote what it read.
    recurrent_window: int = 0
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
    def _heads(
        network: nn.Module,
        observation: torch.Tensor,
        action: torch.Tensor,
        metabolic_unit: Optional[torch.Tensor] = None,
    ):
        """One forward pass, every per-cell quantity the losses need.

        Returns a dict with ``logits``, ``log_probs`` (direction, (B,5,H,W)),
        ``log_prob`` (joint, (B,H,W)), ``entropy`` (joint), ``value``, and when
        the network has a metabolic head also ``metabolic_mean``,
        ``metabolic_log_std``, ``metabolic_log_prob`` and ``metabolic_entropy``.

        Written by hand rather than through ``torch.distributions.Categorical``
        because that would need a permute to put the action axis last on a
        four-dimensional tensor, which is both a copy of the largest tensor in
        the loop and an easy place to transpose height and width by accident.
        """
        has_head = bool(getattr(network, "has_metabolic_head", False))
        if has_head:
            out = network.forward_all(observation)
            logits, value = out["logits"], out["value"]
        else:
            # forward_all also surfaces the memory write when the network has one.
            out = network.forward_all(observation) if hasattr(network, "forward_all") else {}
            logits, value = (out["logits"], out["value"]) if out else network(observation)
        log_probs = F.log_softmax(logits, dim=1)
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1)
        result = {
            "logits": logits,
            "log_probs": log_probs,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }
        if has_head:
            if metabolic_unit is None:
                raise ValueError(
                    "The network has a metabolic head but the rollout carries no "
                    "metabolic_unit. Build the environment with metabolic=True, "
                    "as the network was."
                )
            # The throttle is a continuous rate, so its policy is a Gaussian
            # over the normalised unit rather than a categorical over bins.
            # Discrete levels were the earlier design and were a modelling
            # error: they threw away resolution and made "how many bins" a
            # parameter that says nothing about the ecology.
            mean = out["metabolic_mean"]
            log_std = out["metabolic_log_std"].clamp(MIN_LOG_STD, MAX_LOG_STD)
            std = log_std.exp()
            metabolic_log_prob = (
                -0.5 * ((metabolic_unit - mean) / std) ** 2
                - log_std
                - 0.5 * math.log(2 * math.pi)
            )
            # Differential entropy of a Gaussian, one scalar broadcast per cell.
            metabolic_entropy = (log_std + 0.5 * math.log(2 * math.pi * math.e)).expand_as(mean)
            # Independent heads: the joint log-prob and entropy are sums.
            result["log_prob"] = log_prob + metabolic_log_prob
            result["entropy"] = entropy + metabolic_entropy
            result["metabolic_mean"] = mean
            result["metabolic_log_std"] = log_std
            result["metabolic_log_prob"] = metabolic_log_prob
            result["metabolic_entropy"] = metabolic_entropy
        if "memory" in out:
            result["memory"] = out["memory"]
        return result

    @staticmethod
    def evaluate_actions(
        network: nn.Module,
        observation: torch.Tensor,
        action: torch.Tensor,
        metabolic_unit: Optional[torch.Tensor] = None,
    ):
        """Return (log_prob, entropy, value), each grid shaped ``(B, H, W)``.

        With a metabolic head on the network, ``log_prob`` and ``entropy`` are
        the joint quantities over both levers; see the module docstring.
        """
        heads = PPO._heads(network, observation, action, metabolic_unit)
        return heads["log_prob"], heads["entropy"], heads["value"]

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
        rule_action: Optional[torch.Tensor] = None,
        rule_scores: Optional[torch.Tensor] = None,
        metabolic_unit: Optional[torch.Tensor] = None,
        rule_metabolic_unit: Optional[torch.Tensor] = None,
        heads: Optional[dict] = None,
    ):
        config = self.config
        if heads is None:
            heads = self._heads(network, observation, action, metabolic_unit)
        logits, value = heads["logits"], heads["value"]
        log_probs = heads["log_probs"]
        # Joint over both levers when the network has a metabolic head; the
        # stored old_log_prob is the joint too, so the ratio is well defined.
        log_prob = heads["log_prob"]
        entropy = heads["entropy"]

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

        # Distance from the rules, logged and nothing else: how far the
        # policy has drifted from its initialisation, and whether its throttle
        # runs hot or cold and varies across the field at all.
        conformance = float("nan")
        argmax_agreement = float("nan")
        if rule_action is not None:
            with torch.no_grad():
                _, conformance_t, agreement_t = rule_distillation(
                    log_probs, rule_action, rule_scores, config.imitation_temperature, mask
                )
                conformance = float(conformance_t)
                argmax_agreement = float(agreement_t)
        metabolic_error = float("nan")
        metabolic_unit_mean = float("nan")
        metabolic_head_mean = float("nan")
        metabolic_head_spread = float("nan")
        metabolic_rule_mean = float("nan")
        metabolic_rule_corr = float("nan")
        metabolic_entropy = float("nan")
        metabolic_std = float("nan")
        if "metabolic_mean" in heads:
            mean = heads["metabolic_mean"]
            with torch.no_grad():
                metabolic_unit_mean = float(masked_mean(metabolic_unit.float(), mask))
                metabolic_entropy = float(masked_mean(heads["metabolic_entropy"], mask))
                metabolic_std = float(heads["metabolic_log_std"].exp())
                # The head's mean, as distinct from the sampled unit: with a
                # wide Gaussian clamped to [0, 1] the two differ by a lot, and
                # it was the sampled one that ran hot in the first sweep.
                metabolic_head_mean = float(masked_mean(mean, mask))
                # How much the throttle varies across acting individuals. A
                # throttle that has learned nothing is flat; one that moves
                # with the hunt has spread.
                metabolic_head_spread = float(masked_std(mean, mask))
                if rule_metabolic_unit is not None:
                    rule_unit = rule_metabolic_unit.float()
                    _, error_t = throttle_distillation(mean, rule_unit, mask)
                    metabolic_error = float(error_t)
                    metabolic_rule_mean = float(masked_mean(rule_unit, mask))
                    # Does the policy sprint where the rule would? Zero says
                    # the policy's variation, if any, is not the rule's.
                    metabolic_rule_corr = float(masked_corr(mean, rule_unit, mask))

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
            "conformance": conformance,
            "argmax_agreement": argmax_agreement,
            # Mean absolute error between the policy's throttle and the
            # rule's, in normalised units, so 0.1 is a tenth of the range from
            # basal to maximum. Replaces an argmax agreement, which a
            # continuous action has no equivalent of.
            "metabolic_error": metabolic_error,
            "metabolic_unit_mean": metabolic_unit_mean,
            "metabolic_head_mean": metabolic_head_mean,
            "metabolic_head_spread": metabolic_head_spread,
            "metabolic_rule_mean": metabolic_rule_mean,
            "metabolic_rule_corr": metabolic_rule_corr,
            "metabolic_entropy": metabolic_entropy,
            "metabolic_std": metabolic_std,
        }
        return loss, diagnostics

    def minibatch_loss(self, network: nn.Module, batch) -> torch.Tensor:
        """Loss for one :func:`iter_minibatches_with_value` tuple. For tests."""
        loss, _ = self._losses(network, *batch)
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
        memory_size = int(getattr(network, "memory_size", 0) or 0)
        if self.config.recurrent_window > 0 and memory_size > 0:
            return self.update_recurrent(network, rollout, optimizer, memory_size)

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
                acted = batch[1]
                agent_steps = float(acted.sum())
                if agent_steps == 0:
                    # No individuals in these timesteps. Nothing to learn from,
                    # and every masked mean would be a structural zero.
                    continue

                loss, diagnostics = self._losses(network, *batch)

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


    # ------------------------------------------------------------------
    # Recurrent update: backpropagation through the individual
    # ------------------------------------------------------------------
    def update_recurrent(
        self,
        network: nn.Module,
        rollout: Rollout,
        optimizer: torch.optim.Optimizer,
        memory_size: int,
    ) -> Dict[str, float]:
        """Replay the segment in time order, routing each step's recomputed
        memory write into the next step's read, and backpropagate through
        windows of ``recurrent_window`` steps.

        The memory channels are the last ``memory_size`` channels of the
        observation. At the first step of the segment they are the stored ones;
        from then on they are the network's own recomputed write from the
        previous step, carried through the successor map by
        :func:`propagate_memory`, so the loss at a decision differentiates back
        into the write it depended on. Every other channel comes from storage.

        PPO's ratio still uses the behaviour policy's log-probability from
        collection time. With unchanged weights the recomputed reads equal the
        stored ones up to float16 storage rounding, so the first epoch's ratio
        is one, the standard check.
        """
        if rollout.reproduced is None:
            raise ValueError("Recurrent training needs rollout.reproduced; collect with a current RolloutBuffer.")

        self.value_normalizer.update(rollout.ret, rollout.acted)
        config = self.config
        window = int(config.recurrent_window)
        steps = rollout.steps
        accumulator = _Accumulator()
        # Gradient norms arrive once per window, not once per step. Feeding them
        # into the same accumulator added the window's whole weight to the
        # shared denominator and halved every other diagnostic. They get
        # their own.
        gradient_accumulator = _Accumulator()
        epochs_run = 0
        write_magnitudes: List[float] = []

        def part(tensor, t):
            return None if tensor is None else tensor[t : t + 1]

        for epoch in range(config.epochs):
            epochs_run = epoch + 1
            epoch_kl = _Accumulator()
            read_prev: Optional[torch.Tensor] = None
            pending_loss = None
            pending_weight = 0.0

            for t in range(steps):
                observation = rollout.observation[t].float()
                if read_prev is not None:
                    observation = torch.cat([observation[:-memory_size], read_prev], dim=0)
                observation = observation.unsqueeze(0)

                acted = rollout.acted[t : t + 1]
                agent_steps = float(acted.sum())
                heads = self._heads(network, observation, rollout.action[t : t + 1], part(rollout.metabolic_unit, t))
                write = heads["memory"][0]
                write_magnitudes.append(float(write.detach().abs().mean()))
                read_prev = propagate_memory(
                    write, rollout.successor[t], rollout.acted[t], rollout.reproduced[t], rollout.done[t]
                )

                if agent_steps > 0:
                    loss, diagnostics = self._losses(
                        network, observation, acted, rollout.action[t : t + 1],
                        rollout.log_prob[t : t + 1], rollout.value[t : t + 1],
                        rollout.advantage[t : t + 1], rollout.ret[t : t + 1],
                        part(rollout.rule_action, t), part(rollout.rule_scores, t),
                        part(rollout.metabolic_unit, t), part(rollout.rule_metabolic_unit, t),
                        heads=heads,
                    )
                    # Weight each step by its agent-steps so a window's loss is the
                    # same agent-weighted mean the minibatch update uses.
                    pending_loss = loss * agent_steps if pending_loss is None else pending_loss + loss * agent_steps
                    pending_weight += agent_steps
                    accumulator.add(diagnostics, agent_steps)
                    epoch_kl.add({"approx_kl": diagnostics["approx_kl"]}, agent_steps)

                if (t + 1) % window == 0 or t == steps - 1:
                    if pending_loss is not None and pending_weight > 0:
                        optimizer.zero_grad(set_to_none=True)
                        (pending_loss / pending_weight).backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
                        optimizer.step()
                        gradient_accumulator.add({"grad_norm": float(grad_norm)}, pending_weight)
                    pending_loss = None
                    pending_weight = 0.0
                    # Truncate: the next window starts from a constant read.
                    read_prev = read_prev.detach()

            if config.target_kl is not None:
                kl = epoch_kl.mean().get("approx_kl", 0.0)
                if kl == kl and kl > config.target_kl:
                    break

        result = accumulator.mean()
        result.update(gradient_accumulator.mean())
        result["explained_variance"] = explained_variance(rollout.value, rollout.ret, rollout.acted)
        result["epochs_run"] = float(epochs_run)
        result["agent_steps"] = float(rollout.num_agent_steps)
        result["recurrent_window"] = float(window)
        result["memory_write_abs_mean"] = sum(write_magnitudes) / max(len(write_magnitudes), 1)
        return result
