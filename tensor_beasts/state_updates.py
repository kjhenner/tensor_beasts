from functools import lru_cache

import torch
from tensor_beasts.util import torch_correlate_2d as correlate_2d, torch_correlate_3d as correlate_3d, timing

from tensor_beasts.util import (
    directional_kernel_set, pad_matrix, safe_sub, safe_add, get_direction_matrix
)


DIRECTION_NAMES = {
    0: 'hold',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right'
}


@lru_cache
def generate_diffusion_kernel_bk():
    kernel = torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 3, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=torch.float32)
    return kernel / torch.sum(kernel)


def generate_diffusion_kernel():
    kernel = torch.tensor([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=torch.float32)
    return kernel / torch.sum(kernel)


@timing
def diffuse_scent(entity_energy, entity_scent):
    entity_scent[:] = correlate_3d(entity_scent.type(torch.float32), generate_diffusion_kernel().type(torch.float32)).type(torch.uint8)
    safe_sub(entity_scent, 1)
    entity_scent[:] = correlate_3d(entity_scent.type(torch.float32), generate_diffusion_kernel().type(torch.float32)).type(torch.uint8)
    safe_sub(entity_scent, 1)
    safe_add(entity_scent[:], entity_energy[:])


@timing
def move(
    entity_energy: torch.Tensor,
    divide_threshold: int,
    target_energy,
    opposite_energy=None,
    clearance_kernel_size=5,
    random_choices=None
):
    if random_choices is None:
        random_choices = torch.rand((*entity_energy.shape, 5), dtype=torch.float32)

    if opposite_energy is not None:
        target_energy = safe_sub(target_energy, opposite_energy, inplace=False)

    directions = get_direction_matrix(target_energy, random_choices=random_choices)

    # This is 1 where an entity is present and intends to move in that direction
    direction_masks = {d: ((directions == d) * (entity_energy > 0)).type(torch.uint8) for d in range(1, 5)}

    clearance_kernels = directional_kernel_set(clearance_kernel_size)
    for d in range(1, 5):
        direction_masks[d] *= ~(torch.tensor(
            correlate_2d((entity_energy > 0).type(torch.float32), clearance_kernels[d].type(torch.float32), mode='constant', cval=1)
        ).type(torch.bool))

    new_positions = torch.sum(torch.stack([
        pad_matrix(
            torch.where(
                ((direction_masks[d] * entity_energy) > divide_threshold).type(torch.bool),
                (direction_masks[d] * entity_energy) // 2,
                direction_masks[d] * entity_energy
            ),
            d
        )
        for d in range(1, 5)
    ]), dim=0)

    cleared_spaces = torch.sum(torch.stack(list(direction_masks.values())), dim=0)

    remaining_input = torch.where(
        entity_energy > divide_threshold,
        entity_energy // 4,
        entity_energy * ~cleared_spaces.type(torch.bool)
    )

    entity_energy[:] = new_positions + remaining_input

    return {
        "directions": directions,
    }


@timing
def eat(herbivore_energy, plant_energy, eat_max):
    eat_tensor = (herbivore_energy > 0).type(torch.uint8) * torch.min(
        torch.stack([plant_energy, torch.ones(plant_energy.shape, dtype=torch.uint8) * eat_max], dim=0),
        dim=0
    ).values
    safe_sub(plant_energy, eat_tensor)
    safe_add(herbivore_energy, eat_tensor // 2)


@timing
def germinate(seeds, plant_energy, germination_odds, rand_tensor):
    germination_rand = rand_tensor % germination_odds
    seed_germination = (
        seeds & ~(plant_energy > 0) & (germination_rand == 0)
    )
    safe_add(plant_energy, seed_germination)
    safe_sub(seeds, seed_germination)


@timing
def grow(plant_energy, plant_growth_odds, crowding, crowding_odds, rand_tensor):
    growth_rand = rand_tensor % plant_growth_odds
    growth = plant_energy <= growth_rand
    plant_crowding_mask = (rand_tensor % crowding_odds) >= crowding
    safe_add(plant_energy, (plant_energy > 0) * growth * plant_crowding_mask)