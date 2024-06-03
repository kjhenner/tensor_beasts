from functools import lru_cache

import numpy as np
from sympy.physics.quantum.identitysearch import scipy

from tensor_beasts.util_numpy import (
    directional_kernel_set, pad_matrix, safe_sub, safe_add, get_edge_mask,
    get_direction_matrix
)
from tensor_beasts.util_torch import safe_mult


DIRECTION_NAMES = {
    0: 'hold',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right'
}


@lru_cache
def generate_diffusion_kernel():
    kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 3, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ])
    return kernel / np.sum(kernel)


def diffuse_scent(entity_energy, entity_scent):
    entity_scent[:] = scipy.ndimage.correlate(entity_scent.astype(np.float32), generate_diffusion_kernel().astype(np.float32), mode='constant', cval=0).astype(np.uint8)
    safe_sub(entity_scent, 2)
    entity_scent[:] = scipy.ndimage.correlate(entity_scent.astype(np.float32), generate_diffusion_kernel().astype(np.float32), mode='constant', cval=0).astype(np.uint8)
    safe_sub(entity_scent, 1)
    safe_add(entity_scent[:], entity_energy[:])


def move(entity_energy, divide_threshold, target_energy, rand_array, intent_kernel_size=1, clearance_kernel_size=5):
    if intent_kernel_size == 1:
        # directions = np.argmax(
        #     # [target_energy] + [safe_add(pad_matrix(target_energy.copy(), d), rand_array[d, d]) for d in [2, 1, 4, 3]],
        #     # [target_energy] + [safe_add(pad_matrix(target_energy.copy(), d), rng.integers(0, 64, target_energy.shape, dtype=np.uint8)) for d in [2, 1, 4, 3]],
        #     [target_energy] + [rng.integers(0, 255, target_energy.shape, dtype=np.uint8) for d in [2, 1, 4, 3]],
        #     axis=0
        # )
        original = target_energy.copy()
        directions = get_direction_matrix(target_energy)
        assert np.all(original == target_energy)
    else:
        direction_kernels = directional_kernel_set(intent_kernel_size)
        directions = np.argmax(
            [target_energy] +
            [scipy.ndimage.correlate(target_energy.astype(np.float32) + (rand_array + d) % 2, direction_kernels[d].astype(np.float32) / 4, mode='constant', cval=0) for d in range(1, 5)],
            axis=0
        )

    for d in range(0, 5):
        print(f"Direction {DIRECTION_NAMES[d]}:")
        print(np.sum(directions == d))

    # Create boolean direction matrices
    # This is 1 where an entity is present and intends to move in that direction
    direction_masks = {d: ((directions == d) * (entity_energy > 0)).astype(np.uint8) for d in range(1, 5)}
    direction_masks_orig = {d: ((directions == d) * (entity_energy > 0)).astype(np.uint8) for d in range(0, 5)}

    clearance_kernels = directional_kernel_set(clearance_kernel_size)
    # Check clearance for each direction
    for d in range(1, 5):
        direction_masks[d] *= ~(scipy.ndimage.correlate((entity_energy > 0).astype(np.uint8), clearance_kernels[d], mode='constant', cval=1)).astype(bool)

    new_positions = np.sum(np.array([
        pad_matrix(
            np.where(
                direction_masks[d] * entity_energy > divide_threshold,
                (direction_masks[d] * entity_energy) // 2,
                direction_masks[d] * entity_energy
            ),
            d
        )
        for d in range(1, 5)
    ], dtype=np.uint8), axis=0)

    cleared_spaces = np.sum(np.array(list(direction_masks.values()), dtype=bool), axis=0)

    remaining_input = np.where(
        entity_energy > divide_threshold,
        entity_energy // 4,
        entity_energy * ~cleared_spaces.astype(bool)
    )

    # Sum everything together
    entity_energy[:] = new_positions + remaining_input

    return {
        "directions": directions,
    }


def eat(herbivore_energy, plant_energy, eat_max):
    eat_tensor = (herbivore_energy > 0).astype(np.uint8) * np.min([plant_energy, np.ones(plant_energy.shape, dtype=np.uint8) + eat_max], axis=0)
    safe_sub(plant_energy, eat_tensor)
    safe_add(herbivore_energy, eat_tensor // 2)


def germinate(seeds, plant_energy, germination_odds, rand_array):
    germination_rand = rand_array % germination_odds
    seed_germination = (
        seeds & ~(plant_energy > 0) & (germination_rand == 0)
    )
    safe_add(plant_energy, seed_germination)
    safe_sub(seeds, seed_germination)


def grow(plant_energy, plant_growth_odds, crowding, crowding_odds, rand_array):
    growth_rand = rand_array % plant_growth_odds
    growth = plant_energy <= growth_rand
    plant_crowding_mask = (rand_array % crowding_odds) >= crowding
    safe_add(plant_energy, (plant_energy > 0) * growth * plant_crowding_mask)
