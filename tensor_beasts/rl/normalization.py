"""Return normalization, and why this setting needs it badly.

Measured on a real 32-step segment at 128x128 with the linear policy, before
this existed:

    value loss                       105.2
    gradient norm, value term only    14.02   (100.0% of the total)
    gradient norm, policy term only    0.027
    gradient norm, entropy term only   0.000

With ``max_grad_norm`` at the usual 0.5, the whole update is then rescaled by
0.036, so the policy gradient that actually reaches the weights is 0.00096
rather than 0.027. The policy is effectively frozen while the value head spends
the entire clipping budget learning to predict a number near 9.

This is not a quirk of one run. Survival reward is 1 per step and herbivores
survive about 99.4% of steps, so returns are large and smooth while advantages
are standardized to unit scale by construction. The squared-error value loss is
therefore always two to three orders of magnitude larger than the policy loss,
and global gradient clipping hands the whole budget to whichever term is
largest.

The fix is the standard one: keep a running estimate of the mean and standard
deviation of returns, have the network predict *normalized* values, and convert
back to real return units wherever a value is used as a value, which is
bootstrapping in the advantage recursion. The value loss then lives on the same
scale as the policy loss and neither starves the other.

Only the scale is learned from data; no gradient flows through the statistics.
"""

from dataclasses import dataclass
from typing import Optional

import torch


class RunningMeanStd:
    """Welford-style running mean and variance of a stream of scalars.

    The running accumulator is Python float64. Counts here are large, tens of
    thousands of agent-steps per segment and millions over a run, and a float32
    accumulator loses enough precision over that many updates to matter for the
    variance. Only each batch's three summary scalars cross over from the
    tensor, which also sidesteps Metal having no float64 type at all.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        # The batch reduction happens on whatever device the data is already on,
        # and only the three resulting scalars cross into Python, where they
        # accumulate at float64. Building a float64 tensor here instead would
        # fail outright under a Metal default device, which has no float64.
        batch = values.detach().reshape(-1)
        batch_mean = float(batch.mean())
        batch_var = float(batch.var(unbiased=False))
        batch_count = int(batch.numel())

        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta * delta * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return max(self.var, 1e-8) ** 0.5

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


@dataclass
class ValueNormalizer:
    """Maps between real returns and the normalized values the network predicts.

    Usage is three calls, in this order:

    * ``denormalize`` on the network's output, whenever a value is consumed as a
      value. That is the advantage recursion, which must run in real return
      units or the rewards and the bootstraps would be on different scales.
    * ``update`` with the returns from a finished segment.
    * ``normalize`` on the targets inside the loss.

    Setting ``enabled`` to False makes every method the identity, which keeps
    the ablation honest: the same code path runs either way.
    """

    enabled: bool = True
    stats: RunningMeanStd = None

    def __post_init__(self):
        if self.stats is None:
            self.stats = RunningMeanStd()

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return values
        return (values - self.stats.mean) / self.stats.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return values
        return values * self.stats.std + self.stats.mean

    def update(self, returns: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        if not self.enabled:
            return
        selected = returns[mask] if mask is not None else returns.reshape(-1)
        self.stats.update(selected)

    def state_dict(self) -> dict:
        return {"enabled": self.enabled, "stats": self.stats.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.enabled = state["enabled"]
        self.stats.load_state_dict(state["stats"])
