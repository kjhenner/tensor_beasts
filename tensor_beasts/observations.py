from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import torch

from tensor_beasts.features.feature import Feature
from tensor_beasts.util import directional_kernel_set, torch_correlate_2d

if TYPE_CHECKING:
    from tensordict import TensorDict


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
    # max().indices rather than argmax(): argmax over a strided dim hits a slow
    # CPU kernel in torch and is ~30x slower here for identical results.
    direction = random_max_masks.max(dim=0).indices

    return gradient, direction


# =============================================================================
# Observation: Complete sensory information available to an entity
# =============================================================================

@dataclass
class Observation:
    """
    Complete observation available to an entity's policy.

    Following RL convention, this contains everything the policy needs
    to make a decision: sensory perception + internal state.

    All tensors are (H, W) unless otherwise noted.
    Keys in dicts are tuples: ("entity", "feature")
    """
    # === Sensory Perception ===
    # Directional values: (4, H, W) per feature [up, down, left, right]
    directional: Dict[Tuple[str, str], torch.Tensor]

    # Current cell values: (H, W) per feature
    current: Dict[Tuple[str, str], torch.Tensor]

    # === Internal State ===
    energy: torch.Tensor        # (H, W) uint8
    biomass: torch.Tensor       # (H, W) uint8
    gradient_ema: torch.Tensor  # (H, W) float32 - smoothed stimulus history

    # === Context ===
    alive_mask: torch.Tensor    # (H, W) bool - which cells have living entities
    step: Optional[int] = None  # Current simulation step


def get_observation(
    td: 'TensorDict',
    perception: List[Tuple[Tuple[str, str], int]],
    energy: torch.Tensor,
    biomass: torch.Tensor,
    gradient_ema: torch.Tensor,
    survival_threshold: int,
    log_compress: bool = True,
    log_scale: float = 1.0,
    step: Optional[int] = None,
) -> Observation:
    """
    Build complete observation for policy.

    Args:
        td: TensorDict containing feature data
        perception: List of (feature_key, kernel_size) tuples specifying what to perceive
                   e.g., [(("plant", "scent"), 1), (("predator", "scent"), 3)]
        energy: Entity's energy tensor (H, W) uint8
        biomass: Entity's biomass tensor (H, W) uint8
        gradient_ema: Smoothed stimulus history (H, W) float32
        survival_threshold: Biomass threshold for alive_mask
        log_compress: If True, apply log1p compression for better gradient detection
        log_scale: Scale factor applied before log compression
        step: Current simulation step (optional)

    Returns:
        Observation containing all data needed by policy
    """
    directional = {}
    current = {}

    for key, kernel_size in perception:
        feature_data = td.get(key).float()

        # Apply log compression for better gradient detection at low values
        if log_compress:
            feature_data = torch.log1p(feature_data * log_scale)

        current[key] = feature_data
        directional[key] = get_directional_values(feature_data, kernel_size)

    # Compute alive mask
    alive_mask = biomass >= survival_threshold

    return Observation(
        directional=directional,
        current=current,
        energy=energy,
        biomass=biomass,
        gradient_ema=gradient_ema,
        alive_mask=alive_mask,
        step=step,
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
