"""
Base types for the policy system.

Following RL convention:
- Observation: Complete input to policy (from observations.py)
- Action: Output from policy
- AnimalPolicy: Protocol for policy implementations
"""

from dataclasses import dataclass
from typing import Dict, Protocol, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from tensor_beasts.observations import Observation


@dataclass
class Action:
    """
    Output from animal policy - all decisions for current step.

    All tensors are (H, W) matching the observation grid.
    """
    # === Movement ===
    move_direction: torch.Tensor    # (H, W) long: 0=stay, 1=up, 2=down, 3=left, 4=right
    move_probability: torch.Tensor  # (H, W) float [0, 1]

    # === Metabolism ===
    metabolic_rate: torch.Tensor    # (H, W) float: biomass to burn this step

    # === State Updates ===
    gradient_ema: torch.Tensor      # (H, W) float: updated gradient EMA (to be written back)


class AnimalPolicy(Protocol):
    """
    Protocol defining the animal policy interface.

    Following RL convention:
    - Takes Observation (complete input)
    - Returns Action (complete output)

    All policy implementations must support:
    - __call__: Compute action from observation
    - get_parameters / set_parameters: For genetic algorithm integration
    """

    def __call__(self, obs: 'Observation') -> Action:
        """
        Compute action from observation.

        This is the main entry point - called once per update step.
        Must be fully vectorized (operates on entire grid).
        """
        ...

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Return evolvable parameters as tensors.

        Keys should match config parameter names where possible.
        Used by genetic algorithm for crossover/mutation.
        """
        ...

    def set_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Set evolvable parameters from tensors.

        Used by genetic algorithm for mutation/crossover.
        """
        ...
