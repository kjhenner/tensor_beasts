import numpy as np
from sympy.physics.quantum.identitysearch import scipy

from tensor_beasts.util_numpy import directional_kernel_set, pad_matrix, safe_sub, safe_add, get_edge_mask
from tensor_beasts.util_torch import safe_mult


DIRECTION_NAMES = {
    0: 'hold',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right'
}


def move(entity_energy, divide_threshold, food_energy, rand_array, large_kernel=False):
    direction_kernels = directional_kernel_set(11) if large_kernel else directional_kernel_set(5)
    small_kernels = directional_kernel_set(5)

    directions = np.argmax(
        [food_energy] +
        [scipy.ndimage.correlate(food_energy, direction_kernels[d], mode='constant', cval=0) + (rand_array % d) for d in range(1, 5)],
        axis=0
    )
    directions_orig = {d: (entity_energy > 0) * (directions == d).astype(np.uint8) for d in range(0, 5)}

    # Create boolean direction matrices
    # This is 1 where an entity is present and intends to move in that direction
    direction_masks = {d: ((directions == d) * (entity_energy > 0)).astype(np.uint8) for d in range(1, 5)}
    direction_masks_orig = {d: ((directions == d) * (entity_energy > 0)).astype(np.uint8) for d in range(0, 5)}

    # Check clearance for each direction
    for d in range(1, 5):
        direction_masks[d] *= ~(scipy.ndimage.correlate((entity_energy > 0).astype(np.uint8), small_kernels[d], mode='constant', cval=1)).astype(bool)

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
        "directions": direction_masks_orig
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
