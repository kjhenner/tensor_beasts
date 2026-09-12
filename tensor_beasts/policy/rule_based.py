"""
Rule-based policy that replicates the original Animal.update() behavior.

This policy is the default and should produce identical results to the
original hardcoded decision-making logic.
"""

from typing import Dict, Tuple, TYPE_CHECKING
import torch
from omegaconf import DictConfig

from tensor_beasts.policy.base import Action

if TYPE_CHECKING:
    from tensor_beasts.observations import Observation


class RuleBasedPolicy:
    """
    Rule-based policy matching original Animal.update() behavior.

    This policy replicates the exact decision-making logic from the original
    implementation, including:
    - Weighted perception combination for direction and gradient
    - Biomass-capped metabolic rate calculation
    - Energy-based movement probability

    Config parameters used:
    - navigation_weights: Dict[Tuple[str,str], float] - signed weights for direction
    - basal_rate: int - minimum metabolic rate
    - metabolic_sensitivity: float - gradient -> metabolic rate scaling
    - max_metabolic_rate: int - ceiling on metabolic rate
    - survival_threshold: int - biomass floor for max rate scaling
    - gradient_ema_alpha: float - EMA smoothing factor for gradient
    """

    def __init__(self, config: DictConfig):
        """
        Initialize policy from entity config.

        Args:
            config: Entity configuration (e.g., Herbivore config)
        """
        self.config = config

        # Extract and store relevant parameters
        self.navigation_weights: Dict[Tuple[str, str], float] = dict(config.navigation_weights)
        self.basal_rate: int = config.basal_rate
        self.metabolic_sensitivity: float = config.metabolic_sensitivity
        self.max_metabolic_rate: int = config.max_metabolic_rate
        self.survival_threshold: int = config.survival_threshold
        self.gradient_ema_alpha: float = config.gradient_ema_alpha

    def __call__(self, obs: 'Observation') -> Action:
        """
        Compute action from observation.

        Replicates the original Animal.update() decision logic:
        1. Process observation -> direction + gradient_strength
        2. Update gradient EMA from gradient_strength
        3. Compute metabolic rate from gradient_ema
        4. Compute movement probability from energy
        """
        # Step 1: Compute direction and gradient from weighted perception
        gradient_strength, move_direction = self._process_observation(obs)

        # Step 2: Update gradient EMA (only for alive cells)
        alpha = self.gradient_ema_alpha
        new_gradient_ema = torch.where(
            obs.alive_mask,
            alpha * gradient_strength + (1 - alpha) * obs.gradient_ema,
            obs.gradient_ema
        )

        # Step 3: Compute metabolic rate from updated gradient_ema
        metabolic_rate = self._compute_metabolic_rate(
            new_gradient_ema,
            obs.biomass
        )

        # Step 4: Compute movement probability from energy
        move_probability = obs.energy.float() / 255.0

        return Action(
            move_direction=move_direction,
            move_probability=move_probability,
            metabolic_rate=metabolic_rate,
            gradient_ema=new_gradient_ema,
        )

    def _process_observation(
        self,
        obs: 'Observation'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine weighted perception into gradient and direction.

        Replicates process_observation() logic:
        - Signed weights for direction calculation (positive=attract, negative=repel)
        - Absolute weights for gradient strength (all stimuli contribute)

        Returns:
            gradient_strength: (H, W) float - stimulus intensity for metabolism
            direction: (H, W) long - 0=stay, 1=up, 2=down, 3=left, 4=right
        """
        combined_directional = None
        combined_current = None
        abs_directional = None
        abs_current = None

        for key, weight in self.navigation_weights.items():
            if key not in obs.directional:
                continue

            dir_tensor = obs.directional[key]
            cur_tensor = obs.current[key]

            # Signed weighted values (for direction)
            weighted_dir = dir_tensor.float() * weight
            weighted_cur = cur_tensor.float() * weight

            # Absolute weighted values (for stimulus intensity)
            abs_weighted_dir = dir_tensor.float() * abs(weight)
            abs_weighted_cur = cur_tensor.float() * abs(weight)

            if combined_directional is None:
                combined_directional = weighted_dir
                combined_current = weighted_cur
                abs_directional = abs_weighted_dir
                abs_current = abs_weighted_cur
            else:
                combined_directional = combined_directional + weighted_dir
                combined_current = combined_current + weighted_cur
                abs_directional = abs_directional + abs_weighted_dir
                abs_current = abs_current + abs_weighted_cur

        # Handle case where no features are configured
        if combined_directional is None:
            h, w = obs.energy.shape
            device = obs.energy.device
            return (
                torch.zeros(h, w, device=device),
                torch.zeros(h, w, dtype=torch.long, device=device)
            )

        # Clamp signed values for direction calculation
        combined_directional = combined_directional.clamp(min=0)
        combined_current = combined_current.clamp(min=0)

        # Direction from signed (clamped) values
        direction = self._compute_direction(combined_current, combined_directional)

        # Gradient from absolute values
        gradient_strength = self._compute_gradient(abs_current, abs_directional)

        return gradient_strength, direction

    def _compute_direction(
        self,
        current: torch.Tensor,
        directional: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute movement direction: argmax over [stay, up, down, left, right].

        Uses random tie-breaking when multiple directions have equal value.
        """
        # Stack: [current, up, down, left, right] -> (5, H, W)
        stacked = torch.cat([current.unsqueeze(0), directional], dim=0)

        # Find max value at each position
        max_values = stacked.max(dim=0).values

        # Create mask of positions that equal max
        masks = (stacked == max_values.unsqueeze(0))

        # Random tie-breaking
        random_tiebreak = masks.float() * torch.rand_like(masks, dtype=torch.float32)
        # See note in observations.py: max().indices beats argmax() on dim 0.
        direction = random_tiebreak.max(dim=0).indices

        return direction

    def _compute_gradient(
        self,
        current: torch.Tensor,
        directional: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient strength: max_neighbor - current, clamped >= 0."""
        max_neighbor = directional.max(dim=0).values
        gradient = (max_neighbor - current).clamp(min=0)
        return gradient

    def _compute_metabolic_rate(
        self,
        gradient_ema: torch.Tensor,
        biomass: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute metabolic rate from gradient EMA and biomass.

        Biomass modulates maximum achievable rate:
        - At survival_threshold: can only do basal
        - At 255: can reach max_rate
        """
        basal = self.basal_rate
        sensitivity = self.metabolic_sensitivity
        max_rate = self.max_metabolic_rate
        threshold = self.survival_threshold

        # Biomass fraction above survival threshold
        biomass_range = 255.0 - threshold
        biomass_above = (biomass.float() - threshold).clamp(min=0)
        biomass_fraction = (biomass_above / biomass_range).clamp(0, 1)

        # Effective max rate based on biomass
        effective_max = basal + (max_rate - basal) * biomass_fraction

        # Rate from gradient
        rate = basal + gradient_ema * sensitivity

        # Clamp to biomass-limited maximum
        rate = torch.min(rate, effective_max)

        return rate

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return evolvable parameters as tensors."""
        keys = sorted(self.navigation_weights.keys())
        nav_weights = torch.tensor(
            [self.navigation_weights[k] for k in keys],
            dtype=torch.float32
        )

        return {
            "navigation_weights": nav_weights,
            "navigation_weight_keys": keys,
            "basal_rate": torch.tensor(float(self.basal_rate)),
            "metabolic_sensitivity": torch.tensor(self.metabolic_sensitivity),
            "max_metabolic_rate": torch.tensor(float(self.max_metabolic_rate)),
        }

    def set_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """Set evolvable parameters from tensors."""
        if "navigation_weights" in params:
            keys = params.get("navigation_weight_keys", sorted(self.navigation_weights.keys()))
            values = params["navigation_weights"].tolist()
            self.navigation_weights = dict(zip(keys, values))

        if "basal_rate" in params:
            self.basal_rate = int(params["basal_rate"].item())

        if "metabolic_sensitivity" in params:
            self.metabolic_sensitivity = float(params["metabolic_sensitivity"].item())

        if "max_metabolic_rate" in params:
            self.max_metabolic_rate = int(params["max_metabolic_rate"].item())
