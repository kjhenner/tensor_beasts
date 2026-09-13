"""
GeneticRegistry for managing per-slot genomes.

The registry maintains a collection of genomes, one per "slot". Each slot
represents a distinct genetic lineage that can evolve independently.
"""

import colorsys
from typing import Dict, List, Optional, Tuple
import torch

from tensor_beasts.genetic.genome import Genome
from tensor_beasts.policy.parameterized import ParameterBounds, DEFAULT_BOUNDS


def generate_slot_colors(
    num_slots: int,
    base_hue: float = 0.6,  # 0.6 = blue, 0.0 = red, 0.3 = green
    hue_spread: float = 0.08,  # How far to spread hues around base (tighter = more cohesive)
    saturation_range: Tuple[float, float] = (0.7, 1.0),
    value_range: Tuple[float, float] = (0.8, 1.0),
) -> torch.Tensor:
    """
    Generate distinct colors for each slot, varying around a base hue.

    Args:
        num_slots: Number of colors to generate
        base_hue: Center hue in [0, 1] (0.6=blue, 0.0=red, 0.3=green)
        hue_spread: How much to vary hue around base (±spread)
        saturation_range: (min, max) saturation values
        value_range: (min, max) brightness values

    Returns:
        (num_slots, 3) tensor of RGB colors in [0, 255]
    """
    colors = []
    for i in range(num_slots):
        # Spread hues evenly around base, alternating sides
        if num_slots > 1:
            # Alternate between positive and negative offsets
            offset = (i // 2 + 1) * hue_spread / ((num_slots + 1) // 2)
            if i % 2 == 1:
                offset = -offset
            hue = (base_hue + offset) % 1.0
        else:
            hue = base_hue

        # Vary saturation and value for additional distinction
        sat = saturation_range[0] + (i / max(num_slots - 1, 1)) * (saturation_range[1] - saturation_range[0])
        val = value_range[1] - (i / max(num_slots - 1, 1)) * (value_range[1] - value_range[0])

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])

    # Create on CPU first to avoid MPS bugs that affect subsequent tensor operations
    return torch.tensor(colors, dtype=torch.uint8, device='cpu').to(torch.get_default_device())


class GeneticRegistry:
    """
    Manages genomes for all slots of an entity type.

    Each slot has its own genome that can be independently mutated.
    The registry handles:
    - Storing genomes per slot
    - Applying mutations
    - Performing crossover between slots
    - Tracking slot populations for selection pressure
    """

    def __init__(
        self,
        num_slots: int,
        base_genome: Genome,
        bounds: Optional[Dict[str, ParameterBounds]] = None,
        base_hue: float = 0.6,  # Default blue; 0.0=red, 0.3=green
    ):
        """
        Initialize registry with base genome for all slots.

        Args:
            num_slots: Number of genetic slots to maintain
            base_genome: Initial genome (copied to all slots)
            bounds: Parameter bounds for clamping after mutation
            base_hue: Base hue for slot colors (0.6=blue, 0.0=red, 0.3=green)
        """
        self.num_slots = num_slots
        self.bounds = bounds or DEFAULT_BOUNDS
        self.genomes: List[Genome] = [base_genome.copy() for _ in range(num_slots)]

        # Track population per slot for fitness/selection
        self.populations: torch.Tensor = torch.zeros(num_slots, dtype=torch.int64)

        # Track which slots have ever been occupied (for diversity metrics)
        self.ever_occupied: torch.Tensor = torch.zeros(num_slots, dtype=torch.bool)

        # Generate distinct colors for each slot
        self.slot_colors: torch.Tensor = generate_slot_colors(num_slots, base_hue=base_hue)

        # Navigation keys (needed for tensor conversion)
        self._nav_keys = sorted(base_genome.navigation_weights.keys())

    def get_genome(self, slot_id: int) -> Genome:
        """Get genome for a specific slot."""
        return self.genomes[slot_id]

    def set_genome(self, slot_id: int, genome: Genome) -> None:
        """Set genome for a specific slot."""
        self.genomes[slot_id] = genome

    def mutate_slot(
        self,
        slot_id: int,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.1,
    ) -> None:
        """
        Apply random mutation to a slot's genome.

        Args:
            slot_id: Slot to mutate
            mutation_rate: Probability of mutating each parameter
            mutation_scale: Std dev of mutation noise (relative to bounds)
        """
        genome = self.genomes[slot_id]
        tensor = genome.to_tensor()

        # Mutation mask
        mutation_mask = torch.rand_like(tensor) < mutation_rate

        if not mutation_mask.any():
            return

        # Apply Gaussian noise
        noise = torch.randn_like(tensor) * mutation_scale

        # Scale noise by parameter bounds
        # Navigation weights are first, then other params
        num_nav = len(self._nav_keys)
        scales = torch.ones_like(tensor)

        if "navigation_weights" in self.bounds:
            b = self.bounds["navigation_weights"]
            scales[:num_nav] = (b.max_val - b.min_val)
        if "basal_rate" in self.bounds:
            b = self.bounds["basal_rate"]
            scales[num_nav] = (b.max_val - b.min_val)
        if "metabolic_sensitivity" in self.bounds:
            b = self.bounds["metabolic_sensitivity"]
            scales[num_nav + 1] = (b.max_val - b.min_val)
        if "max_metabolic_rate" in self.bounds:
            b = self.bounds["max_metabolic_rate"]
            scales[num_nav + 2] = (b.max_val - b.min_val)
        if "reproduction_threshold" in self.bounds:
            b = self.bounds["reproduction_threshold"]
            scales[num_nav + 3] = (b.max_val - b.min_val)

        noise = noise * scales

        # Apply mutation
        tensor = tensor + mutation_mask.float() * noise

        # Clamp to bounds
        tensor = self._clamp_tensor(tensor)

        # Update genome
        self.genomes[slot_id] = Genome.from_tensor(tensor, self._nav_keys)

    def _clamp_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clamp tensor values to parameter bounds."""
        num_nav = len(self._nav_keys)

        if "navigation_weights" in self.bounds:
            b = self.bounds["navigation_weights"]
            tensor[:num_nav] = tensor[:num_nav].clamp(b.min_val, b.max_val)
        if "basal_rate" in self.bounds:
            b = self.bounds["basal_rate"]
            tensor[num_nav] = tensor[num_nav].clamp(b.min_val, b.max_val)
        if "metabolic_sensitivity" in self.bounds:
            b = self.bounds["metabolic_sensitivity"]
            tensor[num_nav + 1] = tensor[num_nav + 1].clamp(b.min_val, b.max_val)
        if "max_metabolic_rate" in self.bounds:
            b = self.bounds["max_metabolic_rate"]
            tensor[num_nav + 2] = tensor[num_nav + 2].clamp(b.min_val, b.max_val)
        if "reproduction_threshold" in self.bounds:
            b = self.bounds["reproduction_threshold"]
            tensor[num_nav + 3] = tensor[num_nav + 3].clamp(b.min_val, b.max_val)

        return tensor

    def crossover(self, parent_a: int, parent_b: int) -> Genome:
        """
        Create child genome via uniform crossover.

        Args:
            parent_a: First parent slot ID
            parent_b: Second parent slot ID

        Returns:
            New genome with mixed parameters from both parents
        """
        tensor_a = self.genomes[parent_a].to_tensor()
        tensor_b = self.genomes[parent_b].to_tensor()

        # Uniform crossover
        mask = torch.rand_like(tensor_a) > 0.5
        child_tensor = torch.where(mask, tensor_a, tensor_b)

        return Genome.from_tensor(child_tensor, self._nav_keys)

    def update_populations(
        self,
        slot_ids: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> None:
        """
        Update population counts per slot.

        Uses weighted bincount for O(H*W) instead of O(num_slots * H*W).

        Args:
            slot_ids: (H, W) tensor of slot IDs per cell
            alive_mask: (H, W) bool tensor of living entities
        """
        # Weighted bincount: weight=1 for alive, weight=0 for dead
        # This avoids expensive boolean indexing that allocates a new tensor
        flat_slots = slot_ids.flatten().long()
        weights = alive_mask.flatten().float()
        counts = torch.bincount(flat_slots, weights=weights, minlength=self.num_slots)
        self.populations = counts[:self.num_slots].to(self.populations.dtype)

        # Update ever_occupied mask
        self.ever_occupied |= (self.populations > 0)

    def get_empty_slots(self) -> torch.Tensor:
        """Get indices of slots with zero population."""
        return (self.populations == 0).nonzero(as_tuple=True)[0]

    def get_occupied_slots(self) -> torch.Tensor:
        """Get indices of slots with non-zero population."""
        return (self.populations > 0).nonzero(as_tuple=True)[0]

    def allocate_offspring_slot(
        self,
        parent_slot: int,
        mutation_probability: float = 0.1,
    ) -> Tuple[int, bool]:
        """
        Decide slot for offspring.

        With mutation_probability, offspring may be assigned to an empty slot
        and receive a mutated genome.

        Args:
            parent_slot: Parent's slot ID
            mutation_probability: Chance of slot change

        Returns:
            (slot_id, is_new_slot) - slot for offspring and whether it's new
        """
        empty_slots = self.get_empty_slots()

        # If no empty slots, inherit parent slot
        if len(empty_slots) == 0:
            return parent_slot, False

        # Check if mutation should occur
        if torch.rand(1).item() < mutation_probability:
            # Assign to random empty slot
            new_slot = empty_slots[torch.randint(len(empty_slots), (1,))].item()
            return new_slot, True

        return parent_slot, False

    def colonize_slot(
        self,
        slot_id: int,
        source_slot: int,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.1,
    ) -> None:
        """
        Colonize an empty slot from a source slot's genome.

        Copies the source genome with mutation.

        Args:
            slot_id: Slot to colonize
            source_slot: Slot to copy genome from
            mutation_rate: Mutation rate for new genome
            mutation_scale: Mutation scale for new genome
        """
        # Copy genome from source
        self.genomes[slot_id] = self.genomes[source_slot].copy()

        # Apply mutation
        self.mutate_slot(slot_id, mutation_rate, mutation_scale)

        self.ever_occupied[slot_id] = True

    def get_diversity_metrics(self) -> Dict[str, float]:
        """
        Compute diversity metrics across occupied slots.

        Returns dict with:
        - num_occupied: Number of slots with population > 0
        - parameter_variance: Average variance of parameters across slots
        """
        occupied = self.get_occupied_slots()

        if len(occupied) <= 1:
            return {
                "num_occupied": len(occupied),
                "parameter_variance": 0.0,
            }

        # Stack genomes as tensors
        tensors = torch.stack([
            self.genomes[slot].to_tensor()
            for slot in occupied.tolist()
        ])

        # Compute variance per parameter, then average
        variance = tensors.var(dim=0).mean().item()

        return {
            "num_occupied": len(occupied),
            "parameter_variance": variance,
        }

    def get_fitness_rankings(self, window: int = 1) -> torch.Tensor:
        """
        Rank slots by population (higher is better).

        Args:
            window: Not used currently (for future population history)

        Returns:
            Tensor of slot indices sorted by fitness (best first)
        """
        return torch.argsort(self.populations, descending=True)

    def get_status(self) -> Dict:
        """
        Get complete status for logging/monitoring.

        Returns:
            Dict with:
            - total_population: Total living entities
            - populations: List of population per slot
            - occupied_slots: List of slot indices with population > 0
            - diversity: Diversity metrics dict
            - genomes: Dict mapping slot_id -> genome parameters (for occupied slots)
        """
        occupied = self.get_occupied_slots().tolist()
        diversity = self.get_diversity_metrics()

        # Get genome parameters for occupied slots
        genomes = {}
        for slot in occupied:
            genome = self.genomes[slot]
            genomes[slot] = {
                "navigation_weights": dict(genome.navigation_weights),
                "basal_rate": genome.basal_rate,
                "metabolic_sensitivity": genome.metabolic_sensitivity,
                "max_metabolic_rate": genome.max_metabolic_rate,
                "reproduction_threshold": genome.reproduction_threshold,
            }

        return {
            "total_population": int(self.populations.sum().item()),
            "populations": self.populations.tolist(),
            "occupied_slots": occupied,
            "diversity": diversity,
            "genomes": genomes,
        }

    def log_status(self, entity_name: str = "", step: int = 0) -> None:
        """
        Log current genetic status to console.

        Args:
            entity_name: Name of entity type (for log prefix)
            step: Current simulation step
        """
        status = self.get_status()
        prefix = f"[{entity_name}]" if entity_name else "[Genetics]"

        print(f"\n{prefix} Step {step} - Population: {status['total_population']}")
        print(f"  Slots: {status['populations']}")
        print(f"  Occupied: {len(status['occupied_slots'])}/{self.num_slots}")
        print(f"  Diversity: {status['diversity']['parameter_variance']:.4f}")

        # Show genome details for occupied slots
        if status['occupied_slots']:
            print(f"  Genomes:")
            for slot in status['occupied_slots']:
                genome = status['genomes'][slot]
                pop = status['populations'][slot]
                nav = genome['navigation_weights']
                # Format navigation weights compactly
                nav_str = ", ".join(f"{k[0]}:{v:.2f}" for k, v in nav.items())
                print(f"    Slot {slot} (pop={pop}): [{nav_str}] "
                      f"basal={genome['basal_rate']:.1f} "
                      f"sens={genome['metabolic_sensitivity']:.2f}")
