"""The biomass cap on metabolic rate, in one place.

An animal cannot burn biomass it does not carry, and it slows down before it
starves: the highest rate it can sustain rises from ``basal`` at the survival
threshold to ``max_rate`` at full biomass. That cap is physics rather than
policy, so it has to bind whoever sets the rate. The rule-based policy, the
evolvable parameterized policy, and an external (learned) override all go
through the functions here; if the cap ever needs to change it changes once.

The arithmetic is kept exactly as the rule-based policy originally wrote it,
in the same order, so the simulation's golden hashes do not move.
"""

import torch


def effective_max_metabolic_rate(
    biomass: torch.Tensor,
    basal_rate: float,
    max_rate: float,
    survival_threshold: float,
) -> torch.Tensor:
    """Highest rate an individual with this much biomass can run at.

    Linear from ``basal_rate`` at ``survival_threshold`` to ``max_rate`` at a
    biomass of 255, clamped at both ends.
    """
    biomass_range = 255.0 - survival_threshold
    biomass_above = (biomass.float() - survival_threshold).clamp(min=0)
    biomass_fraction = (biomass_above / biomass_range).clamp(0, 1)
    return basal_rate + (max_rate - basal_rate) * biomass_fraction


def clamp_metabolic_rate(
    rate: torch.Tensor,
    biomass: torch.Tensor,
    basal_rate: float,
    max_rate: float,
    survival_threshold: float,
) -> torch.Tensor:
    """Clamp a desired rate to ``[basal_rate, effective_max(biomass)]``.

    For an external override. The rule-based policies apply only the upper
    cap, because their rate is ``basal + gradient * sensitivity`` and can never
    fall below basal; a learned rate has no such guarantee, so the floor is
    enforced here too.
    """
    effective_max = effective_max_metabolic_rate(biomass, basal_rate, max_rate, survival_threshold)
    return torch.min(rate.float().clamp(min=float(basal_rate)), effective_max)
