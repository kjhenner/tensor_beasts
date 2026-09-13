"""
Genome dataclass for storing evolvable parameters.

A genome represents a set of parameters that define an individual's behavior.
It can be serialized to/from tensors for efficient genetic operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch


@dataclass
class Genome:
    """
    Evolvable parameters for an animal.

    Parameters are stored as Python values but can be converted to/from
    tensors for mutation and crossover operations.
    """
    # Navigation weights: how much each sensory input attracts/repels
    navigation_weights: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Metabolic parameters
    basal_rate: float = 2.0
    metabolic_sensitivity: float = 2.0
    max_metabolic_rate: float = 6.0

    # Life cycle parameters
    reproduction_threshold: float = 200.0

    def to_tensor(self) -> torch.Tensor:
        """
        Flatten genome to a 1D tensor for mutation operations.

        Layout: [nav_weights..., basal_rate, metabolic_sensitivity,
                 max_metabolic_rate, reproduction_threshold]
        """
        # Sort navigation weights for consistent ordering
        nav_keys = sorted(self.navigation_weights.keys())
        nav_values = [self.navigation_weights[k] for k in nav_keys]

        values = nav_values + [
            self.basal_rate,
            self.metabolic_sensitivity,
            self.max_metabolic_rate,
            self.reproduction_threshold,
        ]
        return torch.tensor(values, dtype=torch.float32)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        nav_keys: List[Tuple[str, str]],
    ) -> 'Genome':
        """
        Reconstruct genome from tensor.

        Args:
            tensor: Flattened tensor from to_tensor()
            nav_keys: Navigation weight keys in sorted order
        """
        values = tensor.tolist()
        num_nav = len(nav_keys)

        nav_values = values[:num_nav]
        navigation_weights = dict(zip(nav_keys, nav_values))

        return cls(
            navigation_weights=navigation_weights,
            basal_rate=values[num_nav],
            metabolic_sensitivity=values[num_nav + 1],
            max_metabolic_rate=values[num_nav + 2],
            reproduction_threshold=values[num_nav + 3],
        )

    def copy(self) -> 'Genome':
        """Create a deep copy of this genome."""
        return Genome(
            navigation_weights=dict(self.navigation_weights),
            basal_rate=self.basal_rate,
            metabolic_sensitivity=self.metabolic_sensitivity,
            max_metabolic_rate=self.max_metabolic_rate,
            reproduction_threshold=self.reproduction_threshold,
        )

    def to_policy_params(self) -> Dict[str, torch.Tensor]:
        """
        Convert genome to policy parameter dict.

        Compatible with ParameterizedPolicy.set_parameters().
        """
        nav_keys = sorted(self.navigation_weights.keys())
        nav_values = [self.navigation_weights[k] for k in nav_keys]

        return {
            "navigation_weights": torch.tensor(nav_values, dtype=torch.float32),
            "navigation_weight_keys": nav_keys,
            "basal_rate": torch.tensor(self.basal_rate, dtype=torch.float32),
            "metabolic_sensitivity": torch.tensor(self.metabolic_sensitivity, dtype=torch.float32),
            "max_metabolic_rate": torch.tensor(self.max_metabolic_rate, dtype=torch.float32),
            "reproduction_threshold": torch.tensor(self.reproduction_threshold, dtype=torch.float32),
        }

    @classmethod
    def from_policy_params(cls, params: Dict[str, torch.Tensor]) -> 'Genome':
        """
        Create genome from policy parameter dict.

        Compatible with ParameterizedPolicy.get_parameters().
        """
        nav_keys = params.get("navigation_weight_keys", [])
        nav_values = params["navigation_weights"].tolist()
        navigation_weights = dict(zip(nav_keys, nav_values))

        return cls(
            navigation_weights=navigation_weights,
            basal_rate=params["basal_rate"].item(),
            metabolic_sensitivity=params["metabolic_sensitivity"].item(),
            max_metabolic_rate=params["max_metabolic_rate"].item(),
            reproduction_threshold=params["reproduction_threshold"].item(),
        )

    @classmethod
    def from_config(cls, config) -> 'Genome':
        """
        Create genome from entity config.

        Args:
            config: Entity configuration (e.g., Herbivore config)
        """
        return cls(
            navigation_weights=dict(config.navigation_weights),
            basal_rate=float(config.basal_rate),
            metabolic_sensitivity=float(config.metabolic_sensitivity),
            max_metabolic_rate=float(config.max_metabolic_rate),
            reproduction_threshold=float(config.reproduction_threshold),
        )
