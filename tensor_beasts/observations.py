from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch

from tensor_beasts.features.feature import Feature
from tensor_beasts.util import directional_kernel_set, torch_correlate_2d


# =============================================================================
# Sensorium: Raw sensory perception
# =============================================================================

def get_directional_values(
    matrix: torch.Tensor,
    kernel_size: int = 1,
) -> torch.Tensor:
    """
    Compute values in each of 4 directions.

    Args:
        matrix: (H, W) tensor of values (e.g., scent field)
        kernel_size: 1 = immediate neighbors (fast roll), >1 = wedge kernels (convolution)

    Returns:
        (4, H, W) tensor of values for [up, down, left, right]
    """
    if kernel_size == 1:
        # Fast path: simple roll for immediate neighbors
        up = torch.roll(matrix, shifts=1, dims=0)
        down = torch.roll(matrix, shifts=-1, dims=0)
        left = torch.roll(matrix, shifts=1, dims=1)
        right = torch.roll(matrix, shifts=-1, dims=1)

        # Zero boundaries to avoid wraparound
        up[-1, :] = 0
        down[0, :] = 0
        left[:, -1] = 0
        right[:, 0] = 0

        return torch.stack([up, down, left, right], dim=0)
    else:
        # Wedge kernels via convolution
        kernels = directional_kernel_set(kernel_size)
        return torch.stack([
            torch_correlate_2d(matrix.float(), kernels[d].float(), mode='constant', cval=0)
            for d in range(1, 5)
        ], dim=0)


def compute_gradient_and_direction(
    current: torch.Tensor,
    directional_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient strength and movement direction from directional values.

    Args:
        current: (H, W) current cell values
        directional_values: (4, H, W) from get_directional_values

    Returns:
        gradient: (H, W) max_neighbor - current, clamped >= 0
        direction: (H, W) indices 0=stay, 1=up, 2=down, 3=left, 4=right
    """
    # Stack: [current, up, down, left, right] -> (5, H, W)
    stacked = torch.cat([current.unsqueeze(0), directional_values], dim=0)

    # Gradient: best neighbor minus current (neighbors only, not current)
    max_neighbor = directional_values.max(dim=0).values
    gradient = (max_neighbor - current.float()).clamp(min=0)

    # Direction: argmax across all 5 (including stay)
    max_values = stacked.max(dim=0).values
    masks = (stacked == max_values.unsqueeze(0))

    # Random tie-breaking
    random_max_masks = masks * torch.rand_like(masks, dtype=torch.float32)
    direction = torch.argmax(random_max_masks, dim=0)

    return gradient, direction


# =============================================================================
# Observation: Complete sensory information available to an entity
# =============================================================================

@dataclass
class Observation:
    """
    Complete sensory information available to an entity.
    This is the "sensorium" - pure perception with no semantic meaning attached.

    The same observation can be used with different weight sets for different
    decisions (e.g., navigation vs. risk assessment).
    """
    # Spatial perception: (4, H, W) per feature [up, down, left, right]
    # Keys are tuples like ("entity", "feature")
    directional: Dict[str, torch.Tensor]

    # Current cell values: (H, W) per feature
    current: Dict[str, torch.Tensor]

    # Internal state
    energy: torch.Tensor
    biomass: torch.Tensor


def get_observation(
    td: 'TensorDict',
    perception: List[Tuple[str, int]],
    energy: torch.Tensor,
    biomass: torch.Tensor,
    log_compress: bool = True,
    log_scale: float = 1.0,
) -> Observation:
    """
    Gather sensory information into an Observation.

    Pure perception - no semantic meaning (food/predator) at this level.

    Args:
        td: TensorDict containing feature data
        perception: List of (feature_key, kernel_size) tuples specifying what to perceive
                   e.g., [(("plant", "scent"), 1), (("predator", "scent"), 3)]
        energy: Entity's energy tensor
        biomass: Entity's biomass tensor
        log_compress: If True, apply log1p compression to scent values for better
                      gradient detection across wide dynamic range
        log_scale: Scale factor applied before log compression

    Returns:
        Observation containing all sensory data
    """
    directional = {}
    current = {}

    for key, kernel_size in perception:
        feature_data = td.get(key).float()

        # Apply log compression for better gradient detection at low values
        # This makes small differences at low scent levels perceptible
        if log_compress:
            feature_data = torch.log1p(feature_data * log_scale)

        current[key] = feature_data
        directional[key] = get_directional_values(feature_data, kernel_size)

    return Observation(
        directional=directional,
        current=current,
        energy=energy,
        biomass=biomass,
    )


def process_observation(
    obs: Observation,
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collapse observation into gradient and direction using signed weights.

    This is the decision-making step: combine weighted features and
    compute the resulting gradient strength and movement direction.

    Args:
        obs: Observation from get_observation()
        weights: Signed weights per feature key (tuple).
                 Positive = attractive, negative = repulsive.
                 e.g., {("plant", "scent"): 1.0, ("predator", "scent"): -0.5}

    Returns:
        gradient: (H, W) strength of stimulus (absolute, for metabolism)
        direction: (H, W) indices 0=stay, 1=up, 2=down, 3=left, 4=right
    """
    combined_directional = None
    combined_current = None
    # Track absolute stimulus intensity separately (for metabolism)
    abs_directional = None
    abs_current = None

    for key, weight in weights.items():
        if key not in obs.directional:
            continue

        weighted_dir = obs.directional[key].float() * weight
        weighted_cur = obs.current[key].float() * weight

        # Absolute weighted values for stimulus intensity
        abs_weighted_dir = obs.directional[key].float() * abs(weight)
        abs_weighted_cur = obs.current[key].float() * abs(weight)

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
        # No features configured - return zeros
        h, w = obs.energy.shape
        device = obs.energy.device
        return torch.zeros(h, w, device=device), torch.zeros(h, w, dtype=torch.long, device=device)

    # Clamp signed values for direction calculation
    combined_directional = combined_directional.clamp(min=0)
    combined_current = combined_current.clamp(min=0)

    # Compute direction from signed (clamped) values
    _, direction = compute_gradient_and_direction(combined_current, combined_directional)

    # Compute gradient strength from absolute values (stimulus intensity)
    # This ensures both attraction and repulsion contribute to metabolic response
    gradient_strength, _ = compute_gradient_and_direction(abs_current, abs_directional)

    return gradient_strength, direction


# =============================================================================
# Legacy observation builder (for RL compatibility)
# =============================================================================

def _flatten_feature(feature: Feature) -> List[torch.Tensor]:
    data = feature.data
    if data.ndim == 2:
        return [data.unsqueeze(-1)]
    if data.ndim == 3:
        return [data]
    raise ValueError(f"Unsupported observable feature shape: {data.shape}")


def build_observation(features: Iterable[Feature]) -> torch.Tensor:
    channels: List[torch.Tensor] = []
    for feature in features:
        if "observable" not in (feature.tags or set()):
            continue
        channels.extend(_flatten_feature(feature))

    if not channels:
        raise ValueError("No observable features found for observation build.")

    stacked = torch.cat([c.to(torch.float32) for c in channels], dim=-1)
    return stacked
