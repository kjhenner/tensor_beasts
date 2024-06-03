import json
import numpy
import pygame
import numpy as np
from collections import defaultdict
import scipy.ndimage

from tensor_beasts.display_manager import DisplayManager
from tensor_beasts.state_updates import grow, germinate, move, eat, diffuse_scent
from tensor_beasts.util_numpy import directional_kernel_set, pad_matrix, safe_sub, safe_add






def main():
    rng = np.random.default_rng()

    plant_energy_idx = 1
    plant_scent_idx = 2
    seed_idx = 3
    herbivore_energy_idx = 4
    herbivore_momentum_idx = 5
    herbivore_scent_idx = 6
    predator_energy_idx = 7
    predator_momentum_idx = 8


    width = 512
    height = 512

    plant_init_odds = np.uint8(255)
    herbivore_init_odds = np.uint8(255)
    plant_growth_odds = np.uint8(255)
    plant_germination_odds = np.uint8(255)
    plant_crowding_odds = np.uint8(25)
    plant_seed_odds = np.uint8(255)
    herbivore_eat_max = np.uint8(16)
    predator_eat_max = np.uint8(128)

    clock = pygame.time.Clock()

    runtime_stats = {}

    screens = {
        'rgb': np.zeros((width, height, 3), dtype=np.uint8),
        'scent': np.zeros((width, height, 3), dtype=np.uint8)
    }

    display_manager = DisplayManager(width, height, screens)

    # Initialize a 1024x1024x8 tensor with zeros
    world_tensor = np.zeros((width, height, 8), dtype=np.uint8)
    plant_energy = world_tensor[:, :, plant_energy_idx]
    plant_scent = world_tensor[:, :, plant_scent_idx]
    herbivore_energy = world_tensor[:, :, herbivore_energy_idx]

    # Set the plant channel to 1 at random locations
    plant_energy[:] = (rng.integers(0, plant_init_odds, (width, height), dtype=np.uint8) == 0)

    herbivore_energy[:] = (
        rng.integers(0, herbivore_init_odds, (width, height), dtype=np.uint8) == 0
    ) * 255

    # world_tensor[:, :, predator_energy_idx] = (
    #     rng.integers(0, herbivore_init_odds, (width, height), dtype=np.uint8) == 0
    # ) * 255

    plant_crowding_kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 2, 1, 1],
        [1, 2, 0, 2, 1],
        [1, 1, 2, 1, 1],
        [0, 1, 1, 1, 0],
    ], dtype=np.uint8)

    rand_size = 512
    rand_arrays = rng.integers(0, 255, (rand_size, width, height), dtype=np.uint8)

    done = False
    step = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    display_manager.zoom_in()
                elif event.key == pygame.K_MINUS:
                    display_manager.zoom_out()
                elif event.key == pygame.K_n:
                    display_manager.next_screen()
                elif event.key == pygame.K_h:
                    herbivore_energy[:] = (
                      rng.integers(0, herbivore_init_odds, (width, height), dtype=np.uint8) == 0
                    ) * 255
        rand_array = rand_arrays[step % rand_size]

        plant_mask = numpy.array(plant_energy[:], dtype=bool)
        plant_crowding = scipy.ndimage.convolve(plant_mask, plant_crowding_kernel, mode='constant')

        grow(plant_energy, plant_growth_odds, plant_crowding, plant_crowding_odds, rand_array)
        world_tensor[:, :, seed_idx] |= plant_crowding > (rand_array % plant_seed_odds)

        germinate(world_tensor[:, :, seed_idx], plant_energy, plant_germination_odds, rand_array)

        diffuse_scent(plant_energy, plant_scent)

        move_data = move(herbivore_energy, 250, plant_scent, rand_array)
        safe_sub(herbivore_energy, 1)

        # move(world_tensor[:, :, predator_energy_idx], 250, herbivore_energy, rand_array, True)
        # safe_sub(world_tensor[:, :, predator_energy_idx], 1)

        eat(herbivore_energy, plant_energy, herbivore_eat_max)
        eat(world_tensor[:, :, predator_energy_idx], herbivore_energy, predator_eat_max)

        plant_rgb = world_tensor[:, :, (predator_energy_idx, plant_energy_idx, herbivore_energy_idx)]  # Extract RGB channels
        # seed_rgb = world_tensor[:, :, (seed_idx, 0, seed_idx)] * 255  # Extract RGB channels

        display_manager.set_screen('rgb', plant_rgb)
        display_manager.set_screen('scent', plant_scent)
        display_manager.update()

        runtime_stats['current_screen'] = display_manager.screen_names[display_manager.current_screen]
        runtime_stats['fps'] = clock.get_fps()
        runtime_stats['seed_count'] = float(np.sum(world_tensor[:, :, seed_idx]))
        runtime_stats['plant_mass'] = float(np.sum(plant_energy))
        runtime_stats['herbivore_mass'] = float(np.sum(herbivore_energy))
        runtime_stats['predator_mass'] = float(np.sum(world_tensor[:, :, predator_energy_idx]))
        runtime_stats['step'] = step

        print(json.dumps(runtime_stats, indent=4))

        clock.tick()
        step += 1


if __name__ == "__main__":
    main()
