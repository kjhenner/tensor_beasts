"""
Parameterized policy with evolvable parameters for genetic algorithm.

This policy stores parameters as tensors that can be mutated and crossed over
by the genetic algorithm system.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, TYPE_CHECKING
import torch
from omegaconf import DictConfig

from tensor_beasts.policy.base import Action

if TYPE_CHECKING:
    from tensor_beasts.observations import Observation


@dataclass
class ParameterBounds:
    """Bounds for a single evolvable parameter."""
    min_val: float
    max_val: float
    dtype: torch.dtype = torch.float32

    def clamp(self, value: torch.Tensor) -> torch.Tensor:
        """Clamp value to bounds."""
        return value.clamp(self.min_val, self.max_val)


# Default bounds for evolvable parameters
DEFAULT_BOUNDS = {
    "navigation_weights": ParameterBounds(-10.0, 10.0),
    "basal_rate": ParameterBounds(0.0, 10.0),
    "metabolic_sensitivity": ParameterBounds(0.0, 10.0),
    "max_metabolic_rate": ParameterBounds(1.0, 20.0),
    "reproduction_threshold": ParameterBounds(50.0, 250.0),
}


class ParameterizedPolicy:
    """
    Policy with evolvable parameters for genetic algorithm integration.

    Unlike RuleBasedPolicy which reads from config, this policy stores
    parameters as tensors that can be directly manipulated by mutation
    and crossover operations.

    Parameters:
    - navigation_weights: (N,) tensor of weights for each perceived feature
    - basal_rate: scalar - minimum metabolic rate
    - metabolic_sensitivity: scalar - gradient -> metabolic rate scaling
    - max_metabolic_rate: scalar - ceiling on metabolic rate
    - reproduction_threshold: scalar - biomass level to reproduce (optional)
    """

    def __init__(
        self,
        config: DictConfig,
        bounds: Optional[Dict[str, ParameterBounds]] = None,
    ):
        """
        Initialize policy from config.

        Args:
            config: Entity configuration with default parameter values
            bounds: Optional parameter bounds (defaults to DEFAULT_BOUNDS)
        """
        self.bounds = bounds or DEFAULT_BOUNDS

        # Extract navigation weight keys (sorted for consistency)
        nav_weights_dict = dict(config.navigation_weights)
        self.navigation_weight_keys: List[Tuple[str, str]] = sorted(nav_weights_dict.keys())

        # Store parameters as tensors
        self._navigation_weights = torch.tensor(
            [nav_weights_dict[k] for k in self.navigation_weight_keys],
            dtype=torch.float32
        )
        self._basal_rate = torch.tensor(float(config.basal_rate), dtype=torch.float32)
        self._metabolic_sensitivity = torch.tensor(
            float(config.metabolic_sensitivity), dtype=torch.float32
        )
        self._max_metabolic_rate = torch.tensor(
            float(config.max_metabolic_rate), dtype=torch.float32
        )
        self._survival_threshold = float(config.survival_threshold)
        self._gradient_ema_alpha = float(config.gradient_ema_alpha)

        # Optional: reproduction threshold (not always in config)
        if hasattr(config, 'reproduction_threshold'):
            self._reproduction_threshold = torch.tensor(
                float(config.reproduction_threshold), dtype=torch.float32
            )
        else:
            self._reproduction_threshold = torch.tensor(200.0, dtype=torch.float32)

    @property
    def navigation_weights(self) -> Dict[Tuple[str, str], float]:
        """Get navigation weights as dict (for compatibility)."""
        return dict(zip(
            self.navigation_weight_keys,
            self._navigation_weights.tolist()
        ))

    @property
    def basal_rate(self) -> float:
        return self._basal_rate.item()

    @property
    def metabolic_sensitivity(self) -> float:
        return self._metabolic_sensitivity.item()

    @property
    def max_metabolic_rate(self) -> float:
        return self._max_metabolic_rate.item()

    @property
    def reproduction_threshold(self) -> float:
        return self._reproduction_threshold.item()

    def __call__(self, obs: 'Observation') -> Action:
        """
        Compute all decisions from observation.

        Same logic as RuleBasedPolicy but using tensor parameters.
        """
        # Step 1: Compute direction and gradient from weighted perception
        gradient_strength, move_direction = self._process_observation(obs)

        # Step 2: Update gradient EMA (only for alive cells)
        alpha = self._gradient_ema_alpha
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
        """Combine weighted perception into gradient and direction."""
        combined_directional = None
        combined_current = None
        abs_directional = None
        abs_current = None

        # Use tensor weights directly
        for i, key in enumerate(self.navigation_weight_keys):
            if key not in obs.directional:
                continue

            weight = self._navigation_weights[i].item()
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
        """Compute movement direction with random tie-breaking."""
        stacked = torch.cat([current.unsqueeze(0), directional], dim=0)
        max_values = stacked.max(dim=0).values
        masks = (stacked == max_values.unsqueeze(0))
        random_tiebreak = masks.float() * torch.rand_like(masks, dtype=torch.float32)
        direction = torch.argmax(random_tiebreak, dim=0)
        return direction

    def _compute_gradient(
        self,
        current: torch.Tensor,
        directional: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient strength."""
        max_neighbor = directional.max(dim=0).values
        gradient = (max_neighbor - current).clamp(min=0)
        return gradient

    def _compute_metabolic_rate(
        self,
        gradient_ema: torch.Tensor,
        biomass: torch.Tensor
    ) -> torch.Tensor:
        """Compute metabolic rate from gradient EMA and biomass."""
        basal = self._basal_rate.item()
        sensitivity = self._metabolic_sensitivity.item()
        max_rate = self._max_metabolic_rate.item()
        threshold = self._survival_threshold

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
        """
        Return all evolvable parameters as tensors.

        Returns dict with cloned tensors (modifications don't affect policy).
        """
        return {
            "navigation_weights": self._navigation_weights.clone(),
            "navigation_weight_keys": self.navigation_weight_keys,
            "basal_rate": self._basal_rate.clone(),
            "metabolic_sensitivity": self._metabolic_sensitivity.clone(),
            "max_metabolic_rate": self._max_metabolic_rate.clone(),
            "reproduction_threshold": self._reproduction_threshold.clone(),
        }

    def set_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Set evolvable parameters from tensors.

        Applies bounds validation and clamping.
        """
        if "navigation_weights" in params:
            weights = params["navigation_weights"]
            if "navigation_weights" in self.bounds:
                weights = self.bounds["navigation_weights"].clamp(weights)
            self._navigation_weights = weights.clone()

            # Update keys if provided
            if "navigation_weight_keys" in params:
                self.navigation_weight_keys = params["navigation_weight_keys"]

        if "basal_rate" in params:
            val = params["basal_rate"]
            if "basal_rate" in self.bounds:
                val = self.bounds["basal_rate"].clamp(val)
            self._basal_rate = val.clone()

        if "metabolic_sensitivity" in params:
            val = params["metabolic_sensitivity"]
            if "metabolic_sensitivity" in self.bounds:
                val = self.bounds["metabolic_sensitivity"].clamp(val)
            self._metabolic_sensitivity = val.clone()

        if "max_metabolic_rate" in params:
            val = params["max_metabolic_rate"]
            if "max_metabolic_rate" in self.bounds:
                val = self.bounds["max_metabolic_rate"].clamp(val)
            self._max_metabolic_rate = val.clone()

        if "reproduction_threshold" in params:
            val = params["reproduction_threshold"]
            if "reproduction_threshold" in self.bounds:
                val = self.bounds["reproduction_threshold"].clamp(val)
            self._reproduction_threshold = val.clone()

    def mutate(
        self,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.1,
    ) -> None:
        """
        Apply random Gaussian mutation to parameters.

        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_scale: Standard deviation of mutation noise (relative to bounds range)
        """
        params = self.get_parameters()

        for name, tensor in params.items():
            if name == "navigation_weight_keys":
                continue  # Not a tensor

            if not isinstance(tensor, torch.Tensor):
                continue

            # Determine mutation mask
            mutation_mask = torch.rand_like(tensor) < mutation_rate

            if not mutation_mask.any():
                continue

            # Compute noise scale from bounds
            if name in self.bounds:
                bounds = self.bounds[name]
                scale = (bounds.max_val - bounds.min_val) * mutation_scale
            else:
                scale = mutation_scale

            # Apply Gaussian noise
            noise = torch.randn_like(tensor) * scale
            tensor = tensor + mutation_mask.float() * noise

            params[name] = tensor

        self.set_parameters(params)

    def crossover(self, other: 'ParameterizedPolicy') -> 'ParameterizedPolicy':
        """
        Create child policy via uniform crossover with another policy.

        Args:
            other: Other parent policy

        Returns:
            New policy with mixed parameters from both parents
        """
        # Get parameters from both parents
        params_a = self.get_parameters()
        params_b = other.get_parameters()

        child_params = {}
        for name in params_a:
            if name == "navigation_weight_keys":
                child_params[name] = params_a[name]
                continue

            tensor_a = params_a[name]
            tensor_b = params_b[name]

            if not isinstance(tensor_a, torch.Tensor):
                child_params[name] = tensor_a
                continue

            # Uniform crossover: randomly select from each parent
            mask = torch.rand_like(tensor_a) > 0.5
            child_tensor = torch.where(mask, tensor_a, tensor_b)
            child_params[name] = child_tensor

        # Create child with same structure, then set parameters
        # Need a dummy config - use self's navigation keys
        child = ParameterizedPolicy.__new__(ParameterizedPolicy)
        child.bounds = self.bounds
        child.navigation_weight_keys = self.navigation_weight_keys
        child._navigation_weights = params_a["navigation_weights"].clone()
        child._basal_rate = params_a["basal_rate"].clone()
        child._metabolic_sensitivity = params_a["metabolic_sensitivity"].clone()
        child._max_metabolic_rate = params_a["max_metabolic_rate"].clone()
        child._reproduction_threshold = params_a["reproduction_threshold"].clone()
        child._survival_threshold = self._survival_threshold
        child._gradient_ema_alpha = self._gradient_ema_alpha

        child.set_parameters(child_params)
        return child

    def clone(self) -> 'ParameterizedPolicy':
        """Create a copy of this policy."""
        child = ParameterizedPolicy.__new__(ParameterizedPolicy)
        child.bounds = self.bounds
        child.navigation_weight_keys = self.navigation_weight_keys.copy()
        child._navigation_weights = self._navigation_weights.clone()
        child._basal_rate = self._basal_rate.clone()
        child._metabolic_sensitivity = self._metabolic_sensitivity.clone()
        child._max_metabolic_rate = self._max_metabolic_rate.clone()
        child._reproduction_threshold = self._reproduction_threshold.clone()
        child._survival_threshold = self._survival_threshold
        child._gradient_ema_alpha = self._gradient_ema_alpha
        return child
