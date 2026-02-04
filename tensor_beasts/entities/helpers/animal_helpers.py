from typing import Dict, Optional, List, Callable, Tuple, Union

import torch

from tensor_beasts.util import (
    safe_sum, directional_kernel_set, torch_correlate_2d, safe_add, pad_matrix,
    get_direction_matrix, safe_sub
)


def get_direction_masks(
    directions: torch.Tensor,
    entity_energy: torch.Tensor,
    clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    clearance_kernel_size: Optional[int] = 5,
    move_mask: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    # If we get batched directions, we need to squeeze the batch dimension
    if len(directions.shape) == 3:
        directions = directions.squeeze(0)

    # Base condition: has energy and (if provided) passes move_mask
    can_move = entity_energy > 0
    if move_mask is not None:
        can_move = can_move & move_mask

    direction_masks = {d: ((directions == d) * can_move).type(torch.uint8) for d in range(1, 5)}

    if clearance_mask is not None:
        if isinstance(clearance_mask, list):
            clearance_mask = safe_sum(clearance_mask)
        else:
            clearance_mask = clearance_mask.clone()
        # Use torch.where for O(H*W) instead of boolean indexing which is O(population)
        entity_present = (entity_energy > 0).to(clearance_mask.dtype)
        clearance_mask = torch.maximum(clearance_mask, entity_present)
    else:
        clearance_mask = entity_energy > 0

    clearance_kernels = directional_kernel_set(clearance_kernel_size)
    for d in range(1, 5):
        direction_masks[d] *= ~(
            torch_correlate_2d(
                clearance_mask.type(torch.float32),
                clearance_kernels[d].type(torch.float32),
                mode='constant',
                cval=1
            ).detach().type(torch.bool))
    return direction_masks


def prepare_move(
    entity_energy: torch.Tensor,
    target_energy: Union[torch.Tensor, List[torch.Tensor]],
    target_energy_weights: List[float],
    opposite_energy: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    opposite_energy_weights: Optional[List[float]] = None,
    clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    clearance_kernel_size: Optional[int] = 5,
    move_mask: Optional[torch.Tensor] = None,
):
    if target_energy is not None:
        if target_energy_weights is None:
            target_energy_weights = [1] * len(target_energy)
        else:
            assert len(target_energy) == len(target_energy_weights)

        target_energy = safe_sum([(target * wt).type(torch.uint8) for target, wt in zip(target_energy, target_energy_weights)])

    if opposite_energy is not None:
        if opposite_energy_weights is None:
            opposite_energy_weights = [1] * len(opposite_energy)
        else:
            assert len(opposite_energy) == len(opposite_energy_weights)
        opposite_energy = safe_sum([(opposite * wt).type(torch.uint8) for opposite, wt in zip(opposite_energy, opposite_energy_weights)])
        target_energy = safe_sub(target_energy, opposite_energy, inplace=False)

    directions = get_direction_matrix(target_energy)

    direction_masks = get_direction_masks(directions, entity_energy, clearance_mask, clearance_kernel_size, move_mask)

    return direction_masks


def perform_move(
    entity_energy: torch.Tensor,
    direction_masks: dict,
    divide_threshold: Optional[int] = 250,
    divide_feature: Optional[torch.Tensor] = None,
    divide_fn_self: Optional[Callable] = lambda x: x // 2,
    divide_fn_offspring: Optional[Callable] = lambda x: x // 4,
    carried_features_self: Optional[List[torch.Tensor]] = None,
    carried_feature_fns_self: Optional[List[Callable]] = None,
    carried_features_offspring: Optional[List[torch.Tensor]] = None,
    carried_feature_fns_offspring: Optional[List[Callable]] = None,
    move_cost: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Perform movement and reproduction for entities.

    Args:
        divide_feature: Feature to check against divide_threshold for reproduction.
                       Defaults to entity_energy if not provided.

    Returns:
        move_origin_mask: Boolean tensor indicating which cells had entities that moved.
    """
    # CRITICAL: Validate threshold is within uint8 range.
    # PyTorch has a bug where comparing uint8 tensors to values 256-511 returns True
    # for ALL values, causing every entity to reproduce every step!
    if divide_threshold is not None and divide_threshold > 255:
        raise ValueError(
            f"divide_threshold={divide_threshold} exceeds uint8 max (255). "
            "This causes a PyTorch comparison bug where all values appear > threshold."
        )

    # Use divide_feature for reproduction check, default to entity_energy
    if divide_feature is None:
        divide_feature = entity_energy

    if carried_features_self is not None:
        if carried_feature_fns_self is None:
            carried_feature_fns_self = [lambda x: x] * len(carried_features_self)
        else:
            assert len(carried_features_self) == len(carried_feature_fns_self)

    if carried_features_offspring is not None:
        if carried_feature_fns_offspring is None:
            carried_feature_fns_offspring = [lambda x: x] * len(carried_features_offspring)
        else:
            assert len(carried_features_offspring) == len(carried_feature_fns_offspring)

    move_origin_mask = torch.sum(torch.stack(list(direction_masks.values())), dim=0).type(torch.bool)
    offspring_mask = move_origin_mask * (divide_feature > divide_threshold)
    vacated_mask = move_origin_mask * (divide_feature <= divide_threshold)

    for feature, fn in [(entity_energy, divide_fn_self)] + list(zip(carried_features_self or [], carried_feature_fns_self or [])):
        # Apply move_cost only to energy (first feature in the list)
        if move_cost is not None and feature is entity_energy:
            feature_after_cost = safe_sub(feature, move_cost, inplace=False)
        else:
            feature_after_cost = feature

        safe_add(feature, torch.sum(torch.stack([
            pad_matrix(
                torch.where(
                    ((direction_masks[d] * divide_feature) > divide_threshold).type(torch.bool),
                    direction_masks[d] * fn(feature_after_cost),  # Reproducing: apply fn after cost
                    direction_masks[d] * feature_after_cost       # Regular move: carry with cost applied
                ),
                d
            )
            for d in range(1, 5)
        ]), dim=0))

    # After this operation, each origin position where an offspring will be left will be adjusted by corresponding
    # feature functions
    for feature, fn in [(entity_energy, divide_fn_offspring)] + list(zip(carried_features_offspring or [], carried_feature_fns_offspring or [])):
        feature[:] = torch.where(
            offspring_mask,
            fn(feature),
            feature
        )

    # After this operation, each origin position where no offspring will be left will be zeroed
    for feature in [entity_energy] + (carried_features_self or []) + (carried_features_offspring or []):
        feature[:] *= ~vacated_mask

    return move_origin_mask


def move(
    primary_feature: torch.Tensor,
    target: Union[torch.Tensor, List[torch.Tensor]],
    target_weights: List[float],
    opposite_scent: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    opposite_scent_weights: Optional[List[float]] = None,
    clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    clearance_kernel_size: Optional[int] = 5,
    divide_threshold: Optional[int] = 250,
    divide_feature: Optional[torch.Tensor] = None,
    divide_fn_self: Optional[Callable] = lambda x: x // 2,
    divide_fn_offspring: Optional[Callable] = lambda x: x // 4,
    carried_features_self: Optional[List[str]] = None,
    carried_features_offspring: Optional[List[str]] = None,
    carried_feature_fns_self: Optional[List[Callable]] = None,
    carried_feature_fns_offspring: Optional[List[Callable]] = None,
    obstacle_mask: Optional[torch.Tensor] = None,
    agent_action: Optional[torch.Tensor] = None,
    move_mask: Optional[torch.Tensor] = None,
    move_cost: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Perform movement for entities based on target scent gradients.

    Args:
        divide_feature: Feature to check against divide_threshold for reproduction.
                       Defaults to primary_feature if not provided.

    Returns:
        move_origin_mask: Boolean tensor indicating which cells had entities that moved.
    """
    if agent_action is not None:
        # I.e. if we already have an action selected by the RL model.
        direction_masks = get_direction_masks(
            agent_action,
            primary_feature,
            obstacle_mask,
            5,
            move_mask
        )
    else:
        direction_masks = prepare_move(
            primary_feature,
            target,
            target_weights,
            opposite_scent,
            opposite_scent_weights,
            clearance_mask,
            clearance_kernel_size,
            move_mask
        )

    return perform_move(
        entity_energy=primary_feature,
        direction_masks=direction_masks,
        divide_threshold=divide_threshold,
        divide_feature=divide_feature,
        divide_fn_self=divide_fn_self,
        divide_fn_offspring=divide_fn_offspring,
        carried_features_self=carried_features_self,
        carried_feature_fns_self=carried_feature_fns_self,
        carried_features_offspring=carried_features_offspring,
        carried_feature_fns_offspring=carried_feature_fns_offspring,
        move_cost=move_cost,
    )
