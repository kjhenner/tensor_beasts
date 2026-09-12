"""V-trace (IMPALA) for a grid of individuals sharing one policy.

Why V-trace here
----------------

The simulation is the bottleneck, not the data. A 512x512 world steps at roughly
nine steps per second, while a gradient step over the same grid is milliseconds.
Any method that can only take one pass over fresh on-policy data is therefore
leaving almost all of the available compute on the floor. V-trace buys the right
to keep learning from a segment after the policy that collected it has moved on:
the importance ratios ``pi(a|s) / mu(a|s)`` correct the value target, and the
truncations ``rho_bar`` and ``c_bar`` bound the variance that correction would
otherwise introduce.

The recursion, with movement
----------------------------

Textbook V-trace (Espeholt et al. 2018) is

    delta_t V = rho_t (r_t + gamma V(x_{t+1}) - V(x_t))
    v_t       = V(x_t) + delta_t V + gamma c_t (v_{t+1} - V(x_{t+1}))

and every ``x_{t+1}`` in it means "the state this same individual is in next".
Here individuals move, so ``V(x_{t+1})`` is *not* the value of the cell they
acted from at the next timestep; that cell may well be occupied by a different
animal. Exactly as in :func:`~tensor_beasts.rl.rollout.compute_gae`, both
next-step terms are gathered through ``rollout.successor``:

    V(x_{t+1})        -> value[t + 1].flatten()[succ_t] * alive_t
    v_{t+1} - V(x_{t+1}) -> carry.flatten()[succ_t] * alive_t

``alive_t`` is ``~done_t``, which zeroes the bootstrap for individuals that died
during the step, so whoever walks into the vacated cell is never credited.

Relation to GAE
---------------

Folding a trace decay into the clipped ratio, ``c_t = lambda * min(c_bar, ratio)``,
makes V-trace a strict generalization of GAE: with all ratios equal to one,

    v_t - V(x_t) = delta_t + gamma * lambda * (v_{t+1} - V(x_{t+1}))

which is the GAE recursion verbatim. ``tests/rl/test_algorithms.py`` asserts
that equality against :func:`compute_gae` numerically, and it is the check to
run first if the recursion is ever touched.

Masking
-------

Same rule as everywhere else in this package: most cells are empty, so every
reduction goes through :func:`~tensor_beasts.rl.ppo.masked_mean` over ``acted``.
The carry is masked too, otherwise an empty cell's junk delta would be gathered
by an individual whose successor happens to point at it.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from tensor_beasts.rl.ppo import PPO, _Accumulator, explained_variance, masked_mean
from tensor_beasts.rl.rollout import Rollout

# exp() of anything much larger than this is inf, and inf * 0 is NaN, which
# would then spread through the backward recursion. Ratios this extreme are
# clipped by rho_bar/c_bar anyway, so the clamp changes no useful number.
MAX_LOG_RATIO = 20.0


def compute_vtrace(
    rollout: Rollout,
    value: torch.Tensor,
    log_ratio: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V-trace targets and policy-gradient advantages, following each individual.

    Args:
        rollout: Collected segment. Read only; nothing is modified.
        value: (T, H, W) value estimates under the *learner's current* policy.
            Not ``rollout.value``, which is the behaviour policy's estimate from
            collection time; passing that in is what turns this back into GAE.
        log_ratio: (T, H, W) ``log pi(a|s) - log mu(a|s)``, current policy over
            behaviour policy. Zero everywhere reduces this to GAE.
        last_value: (H, W) value of the state after the final step, used to
            bootstrap individuals still alive at the segment boundary.
        gamma: Discount.
        lambda_: Trace decay, folded into ``c`` so lambda=1 recovers plain
            V-trace and ratios of one recover GAE at this lambda.
        rho_bar: Truncation on the ratio in the temporal difference. Bounds the
            fixed point: the value function converges to V^pi only at rho_bar
            large, and to something between V^mu and V^pi below that.
        c_bar: Truncation on the trace ratio. Bounds the variance of the
            multi-step product; it does not change the fixed point.

    Returns:
        ``(vs, advantage)``, both (T, H, W). ``vs`` is the value target,
        ``advantage`` the ``rho``-weighted policy gradient advantage. Both are
        zero-signal at non-acting cells: ``vs`` equals ``value`` there, so the
        value loss contributes nothing, and ``advantage`` is exactly zero.
    """
    steps = rollout.steps
    height, width = rollout.reward.shape[-2:]
    device = rollout.reward.device

    acted_mask = rollout.acted.reshape(steps, -1).float()
    # Empty cells get ratio exactly 1; their log_ratio is meaningless and could
    # be anything, including values that overflow exp().
    clamped = torch.clamp(log_ratio.reshape(steps, -1), -MAX_LOG_RATIO, MAX_LOG_RATIO)
    ratio = torch.exp(clamped * acted_mask)
    rho = torch.clamp(ratio, max=rho_bar)
    c = lambda_ * torch.clamp(ratio, max=c_bar)

    vs = torch.zeros_like(rollout.reward)
    advantage = torch.zeros_like(rollout.reward)

    next_value = last_value.reshape(-1)
    carry = torch.zeros(height * width, device=device)  # v_{t+1} - V(x_{t+1})

    for t in reversed(range(steps)):
        successor = rollout.successor[t].reshape(-1)
        alive = (~rollout.done[t]).reshape(-1).float()
        acted = acted_mask[t]

        # Follow the individual to wherever it actually went.
        bootstrap_value = next_value[successor] * alive
        bootstrap_carry = carry[successor] * alive

        value_t = value[t].reshape(-1)
        reward_t = rollout.reward[t].reshape(-1)

        delta = rho[t] * (reward_t + gamma * bootstrap_value - value_t)
        vs_minus_value = delta + gamma * c[t] * bootstrap_carry

        # The policy gradient advantage uses the *corrected* next-step target,
        # v_{t+1}, which is bootstrap_value + bootstrap_carry by construction.
        advantage_t = rho[t] * (
            reward_t + gamma * (bootstrap_value + bootstrap_carry) - value_t
        )

        vs_minus_value = vs_minus_value * acted
        advantage_t = advantage_t * acted

        vs[t] = (value_t + vs_minus_value).reshape(height, width)
        advantage[t] = advantage_t.reshape(height, width)

        carry = vs_minus_value
        next_value = value_t

    return vs, advantage


@dataclass
class VTraceConfig:
    """Hyperparameters for :class:`VTrace`.

    Attributes:
        learning_rate: Unused by ``update`` (the optimizer is supplied), kept so
            a config file can describe a run in one place, as ``PPOConfig`` does.
        rho_bar: Temporal-difference ratio truncation. One is the IMPALA default.
        c_bar: Trace ratio truncation. One is the IMPALA default.
        gamma: Discount.
        lambda_: Trace decay folded into ``c``. 0.95 matches the PPO default;
            1.0 gives textbook V-trace.
        entropy_coef: Weight on the (masked) entropy bonus.
        value_coef: Weight on the value loss.
        epochs: Passes over each collected segment. More than one is the whole
            point of using V-trace: the correction is what makes the second and
            later passes legitimate rather than silently biased.
        minibatch_steps: Timesteps per minibatch, as in PPO.
        max_grad_norm: Global gradient clipping.
        normalize_advantage: Standardize advantages over acting cells only.
    """

    learning_rate: float = 3e-4
    rho_bar: float = 1.0
    c_bar: float = 1.0
    gamma: float = 0.99
    lambda_: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 4
    minibatch_steps: int = 16
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class VTrace:
    """IMPALA-style off-policy actor-critic. Implements ``Algorithm``.

    Unlike PPO this ignores ``rollout.advantage`` and ``rollout.ret``: the whole
    point is to recompute the targets under the learner's current policy at the
    start of every epoch, so that a segment stays usable as the policy drifts
    away from the one that collected it.
    """

    def __init__(self, config: Optional[VTraceConfig] = None):
        self.config = config or VTraceConfig()

    # ------------------------------------------------------------------
    # Current-policy pass over a whole segment
    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate_segment(
        network: nn.Module,
        rollout: Rollout,
        chunk_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(log_prob, value) for every stored action under the current policy.

        Chunked because the observation is the largest tensor in the system and
        a whole segment at 512x512 does not want to be activated at once.
        """
        log_probs = []
        values = []
        with torch.no_grad():
            for start in range(0, rollout.steps, chunk_steps):
                stop = min(start + chunk_steps, rollout.steps)
                observation = rollout.observation[start:stop].float()
                action = rollout.action[start:stop]
                log_prob, _, value = PPO.evaluate_actions(network, observation, action)
                log_probs.append(log_prob)
                values.append(value)
        return torch.cat(log_probs), torch.cat(values)

    def compute_targets(
        self,
        network: nn.Module,
        rollout: Rollout,
        last_value: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        log_prob, value = self._evaluate_segment(network, rollout, config.minibatch_steps)

        if last_value is None:
            # The observation after the segment's final step was never stored,
            # so the exact bootstrap is not available here. Reusing the final
            # step's own value grid is a one-step-stale estimate of it, which is
            # a far smaller error than bootstrapping from zero. Pass an explicit
            # ``last_value`` (the trainer has one) whenever you can.
            last_value = value[-1]

        vs, advantage = compute_vtrace(
            rollout,
            value=value,
            log_ratio=log_prob - rollout.log_prob,
            last_value=last_value,
            gamma=config.gamma,
            lambda_=config.lambda_,
            rho_bar=config.rho_bar,
            c_bar=config.c_bar,
        )

        if config.normalize_advantage:
            advantage = normalize_masked(advantage, rollout.acted)
        return vs, advantage

    # ------------------------------------------------------------------
    # One minibatch
    # ------------------------------------------------------------------
    def _losses(
        self,
        network: nn.Module,
        observation: torch.Tensor,
        acted: torch.Tensor,
        action: torch.Tensor,
        behaviour_log_prob: torch.Tensor,
        vs: torch.Tensor,
        advantage: torch.Tensor,
    ):
        config = self.config
        log_prob, entropy, value = PPO.evaluate_actions(network, observation, action)

        mask = acted.float()

        # The rho weighting is already inside ``advantage``; this is the plain
        # score-function estimator against that corrected advantage.
        policy_loss = -masked_mean(log_prob * advantage, mask)
        value_loss = masked_mean((value - vs) ** 2, mask)
        entropy_mean = masked_mean(entropy, mask)
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean

        with torch.no_grad():
            log_ratio = (log_prob - behaviour_log_prob) * mask
            ratio = torch.exp(torch.clamp(log_ratio, -MAX_LOG_RATIO, MAX_LOG_RATIO))
            approx_kl = masked_mean((ratio - 1.0) - log_ratio, mask)
            clipped_fraction = masked_mean((ratio > config.rho_bar).float(), mask)

        diagnostics = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "loss": float(loss.detach()),
            "approx_kl": float(approx_kl),
            "rho_clipped_fraction": float(clipped_fraction),
        }
        return loss, diagnostics

    def minibatch_loss(self, network: nn.Module, batch) -> torch.Tensor:
        """Loss for one :func:`iter_vtrace_minibatches` tuple. For tests."""
        observation, acted, action, behaviour_log_prob, vs, advantage = batch
        loss, _ = self._losses(
            network, observation, acted, action, behaviour_log_prob, vs, advantage
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
        last_value: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Run the V-trace epochs over one collected segment.

        Args:
            network: Shared actor-critic. Updated in place.
            rollout: Segment. ``advantage`` and ``ret`` are ignored if present;
                targets are recomputed under the current policy each epoch.
            optimizer: Optimizer over ``network.parameters()``.
            generator: Optional RNG for minibatch shuffling.
            last_value: Optional (H, W) bootstrap for the segment boundary. See
                :meth:`compute_targets` for what happens when it is omitted.

        Returns:
            Diagnostics averaged over minibatches, weighted by agent-steps.
        """
        config = self.config
        accumulator = _Accumulator()
        vs = advantage = None

        for _ in range(config.epochs):
            # Recomputed per epoch: after a gradient step the learner is a
            # different policy, and the ratios have to reflect that.
            vs, advantage = self.compute_targets(network, rollout, last_value)

            for batch in iter_vtrace_minibatches(
                rollout, vs, advantage, config.minibatch_steps, generator=generator
            ):
                acted = batch[1]
                agent_steps = float(acted.sum())
                if agent_steps == 0:
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

        result = accumulator.mean()
        if vs is not None:
            result["explained_variance"] = explained_variance(
                rollout.value, vs, rollout.acted
            )
        result["epochs_run"] = float(config.epochs)
        result["agent_steps"] = float(rollout.num_agent_steps)
        return result


def normalize_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standardize over the True entries of ``mask``, leaving the rest at zero.

    Standardizing over the whole grid would be dominated by empty cells and
    would give every one of them a nonzero (and meaningless) advantage.
    """
    if not bool(mask.any()):
        return values
    selected = values[mask]
    centered = values - selected.mean()
    scaled = centered / (selected.std(unbiased=False) + 1e-8)
    return torch.where(mask, scaled, torch.zeros_like(scaled))


def iter_vtrace_minibatches(
    rollout: Rollout,
    vs: torch.Tensor,
    advantage: torch.Tensor,
    minibatch_steps: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Iterator[Tuple[torch.Tensor, ...]]:
    """Minibatches of whole timesteps with externally computed V-trace targets.

    Separate from ``rollout.iter_minibatches`` because the targets here are not
    stored on the rollout: they are recomputed every epoch and would be stale
    the moment they were written back.

    Yields:
        (observation, acted, action, behaviour_log_prob, vs, advantage).
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
            vs[index],
            advantage[index],
        )
