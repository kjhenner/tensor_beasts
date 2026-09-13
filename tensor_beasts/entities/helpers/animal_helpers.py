from typing import Dict, Optional, List, Callable, Tuple, Union

import torch

from tensor_beasts.features.shared_features import ENERGY_MAX
from tensor_beasts.util import (
    directional_kernel_bank, torch_correlate_2d_bank, pad_matrix, get_direction_matrix
)


def get_direction_masks(
    directions: torch.Tensor,
    entity_energy: torch.Tensor,
    clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    clearance_kernel_size: Optional[int] = 5,
    move_mask: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    # If we get batched directions, we need to squeeze the batch dimension.
    #
    # PHASE 1 NOTE: this is the one spot that CANNOT be made rank-agnostic. It
    # dispatches on rank to strip the RL agent's leading singleton batch, so a
    # real (B, H, W) world batch is indistinguishable from it. Adding the batch
    # dimension will require the RL path to hand over directions already shaped
    # like the world instead of being normalized here. Left alone on purpose:
    # only the agent_action path in move() ever supplies a 3D tensor.
    if len(directions.shape) == 3:
        directions = directions.squeeze(0)

    # Base condition: has energy and (if provided) passes move_mask
    can_move = entity_energy > 0
    if move_mask is not None:
        can_move = can_move & move_mask

    direction_masks = {d: ((directions == d) * can_move).type(torch.uint8) for d in range(1, 5)}

    if clearance_mask is not None:
        if isinstance(clearance_mask, list):
            # Only "is anything here" matters downstream, so a plain sum does.
            clearance_mask = torch.stack(clearance_mask).sum(dim=0)
        else:
            clearance_mask = clearance_mask.clone()
        # Use torch.where for O(H*W) instead of boolean indexing which is O(population)
        entity_present = (entity_energy > 0).to(clearance_mask.dtype)
        clearance_mask = torch.maximum(clearance_mask, entity_present)
    else:
        clearance_mask = entity_energy > 0

    # One batched conv over all four directional kernels rather than four
    # separate convs, each of which re-pads the same input.
    blocked = torch_correlate_2d_bank(
        clearance_mask.type(torch.float32),
        directional_kernel_bank(clearance_kernel_size),
        cval=1,
    ).detach().type(torch.bool)
    # blocked is (..., 4, H, W): the kernel axis sits just before the spatial
    # axes, so index it from the end rather than positionally.
    for d in range(1, 5):
        direction_masks[d] *= ~blocked[..., d - 1, :, :]
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

        target_energy = sum(target.float() * wt for target, wt in zip(target_energy, target_energy_weights))

    if opposite_energy is not None:
        if opposite_energy_weights is None:
            opposite_energy_weights = [1] * len(opposite_energy)
        else:
            assert len(opposite_energy) == len(opposite_energy_weights)
        opposite_energy = sum(opposite.float() * wt for opposite, wt in zip(opposite_energy, opposite_energy_weights))
        target_energy = (target_energy - opposite_energy).clamp(min=0)

    directions = get_direction_matrix(target_energy)

    direction_masks = get_direction_masks(directions, entity_energy, clearance_mask, clearance_kernel_size, move_mask)

    return direction_masks


def perform_move(
    entity_energy: torch.Tensor,
    direction_masks: dict,
    divide_threshold: Optional[float] = 250,
    divide_feature: Optional[torch.Tensor] = None,
    divide_fn_self: Optional[Callable] = lambda x: x * 0.5,
    divide_fn_offspring: Optional[Callable] = lambda x: x * 0.25,
    carried_features_self: Optional[List[torch.Tensor]] = None,
    carried_feature_fns_self: Optional[List[Callable]] = None,
    carried_features_offspring: Optional[List[torch.Tensor]] = None,
    carried_feature_fns_offspring: Optional[List[Callable]] = None,
    move_cost: Optional[torch.Tensor] = None,
    energy_max: Optional[float] = ENERGY_MAX,
) -> torch.Tensor:
    """
    Perform movement and reproduction for entities.

    Args:
        divide_feature: Feature to check against divide_threshold for reproduction.
                       Defaults to entity_energy if not provided.
        energy_max: entity_energy saturates here when several movers land on
                    one cell. None disables the clamp. Carried features are
                    summed without a clamp (memory is signed, ids are labels);
                    a caller whose carried feature has a ceiling, such as
                    biomass, clamps it itself afterwards.

    Every carried_feature_fn and divide_fn MUST BE PURE. Each is applied once
    and its result is shared across all four directions, so a function that
    mutates its argument in place will corrupt the whole grid rather than the
    moving cells. Pass e.g. lambda x: x + 1, never lambda x: x.add_(1).

    Returns:
        move_origin_mask: Boolean tensor indicating which cells had entities that moved.
    """
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

    # Masks are 0/1 uint8, so bitwise or matches a stack-and-sum.
    move_origin_mask = (
        direction_masks[1] | direction_masks[2] | direction_masks[3] | direction_masks[4]
    ).type(torch.bool)

    # Who reproduces is decided once, from state as it stood on entry. It used
    # to be re-read per feature and per direction, which made the outcome depend
    # on the order of carried_features_self: divide_feature is normally biomass,
    # which is itself one of the carried features and is mutated by the loop
    # below, so features processed later saw a different reproduction decision
    # than features processed earlier.
    is_reproducing = divide_feature > divide_threshold
    offspring_mask = move_origin_mask * is_reproducing
    vacated_mask = move_origin_mask * ~is_reproducing

    # Per-direction reproduction, derived from the same single decision.
    reproducing_toward = {
        d: direction_masks[d].type(torch.bool) & is_reproducing for d in range(1, 5)
    }

    for feature, fn in [(entity_energy, divide_fn_self)] + list(zip(carried_features_self or [], carried_feature_fns_self or [])):
        # Apply move_cost only to energy (first feature in the list)
        if move_cost is not None and feature is entity_energy:
            feature_after_cost = (feature - move_cost).clamp(min=0)
        else:
            feature_after_cost = feature

        # fn is direction-independent, so it is applied once. It used to be
        # called once per direction, which silently multiplied the effect of any
        # caller that passed an in-place function.
        feature_if_reproducing = fn(feature_after_cost)

        # Accumulate instead of stacking four (H, W) tensors and reducing.
        arrivals = None
        for d in range(1, 5):
            carried = pad_matrix(
                torch.where(
                    reproducing_toward[d],
                    direction_masks[d] * feature_if_reproducing,  # Reproducing: apply fn after cost
                    direction_masks[d] * feature_after_cost       # Regular move: carry with cost applied
                ),
                d
            )
            arrivals = carried if arrivals is None else arrivals + carried

        feature += arrivals
        if feature is entity_energy and energy_max is not None:
            feature.clamp_(max=energy_max)

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
    divide_threshold: Optional[float] = 250,
    divide_feature: Optional[torch.Tensor] = None,
    divide_fn_self: Optional[Callable] = lambda x: x * 0.5,
    divide_fn_offspring: Optional[Callable] = lambda x: x * 0.25,
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
