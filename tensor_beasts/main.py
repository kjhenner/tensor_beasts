import json
import numpy
import pygame
import numpy as np
from collections import defaultdict
import scipy.ndimage

from tensor_beasts.display_manager import DisplayManager
from tensor_beasts.state_updates import grow, germinate, move, eat
from tensor_beasts.util_numpy import directional_kernel_set, pad_matrix, safe_sub, safe_add






def main():
    rng = np.random.default_rng()

    plant_energy_idx = 1
    seed_idx = 2
    herbivore_energy_idx = 3
    herbivore_momentum_idx = 4
    predator_energy_idx = 5
    predator_momentum_idx = 6


    width = 512
    height = 512

    plant_init_odds = np.uint8(255)
    herbivore_init_odds = np.uint8(255)
    plant_growth_odds = np.uint8(255)
    plant_germination_odds = np.uint8(255)
    plant_crowding_odds = np.uint8(25)
    plant_seed_odds = np.uint8(255)
    herbivore_eat_max = np.uint8(128)
    predator_eat_max = np.uint8(128)

    clock = pygame.time.Clock()

    runtime_stats = {}

    screens = {
        'rgb': np.zeros((width, height, 3), dtype=np.uint8),
        'move_hold': np.zeros((width, height, 3), dtype=np.uint8),
        'move_up': np.zeros((width, height, 3), dtype=np.uint8),
        'move_down': np.zeros((width, height, 3), dtype=np.uint8),
        'move_left': np.zeros((width, height, 3), dtype=np.uint8),
        'move_right': np.zeros((width, height, 3), dtype=np.uint8),
    }

    display_manager = DisplayManager(width, height, screens)

    # Initialize a 1024x1024x8 tensor with zeros
    world_tensor = np.zeros((width, height, 8), dtype=np.uint8)
    plant_energy = world_tensor[:, :, plant_energy_idx]

    # Set the plant channel to 1 at random locations
    plant_energy[:] = (rng.integers(0, plant_init_odds, (width, height), dtype=np.uint8) == 0)

    world_tensor[:, :, herbivore_energy_idx] = (
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
        rand_array = rand_arrays[step % rand_size]

        plant_mask = numpy.array(plant_energy[:], dtype=bool)
        plant_crowding = scipy.ndimage.convolve(plant_mask, plant_crowding_kernel, mode='constant')

        grow(plant_energy, plant_growth_odds, plant_crowding, plant_crowding_odds, rand_array)
        world_tensor[:, :, seed_idx] |= plant_crowding > (rand_array % plant_seed_odds)

        germinate(world_tensor[:, :, seed_idx], plant_energy, plant_germination_odds, rand_array)

        move_data = move(world_tensor[:, :, herbivore_energy_idx], 250, plant_energy, rand_array)
        # safe_sub(world_tensor[:, :, herbivore_energy_idx], 2)

        move(world_tensor[:, :, predator_energy_idx], 250, world_tensor[:, :, herbivore_energy_idx], rand_array, True)
        safe_sub(world_tensor[:, :, predator_energy_idx], 1)

        eat(world_tensor[:, :, herbivore_energy_idx], plant_energy, herbivore_eat_max)
        eat(world_tensor[:, :, predator_energy_idx], world_tensor[:, :, herbivore_energy_idx], predator_eat_max)

        plant_rgb = world_tensor[:, :, (predator_energy_idx, plant_energy_idx, herbivore_energy_idx)]  # Extract RGB channels
        # seed_rgb = world_tensor[:, :, (seed_idx, 0, seed_idx)] * 255  # Extract RGB channels

        display_manager.set_screen('rgb', plant_rgb)
        display_manager.set_screen('move_up', (move_data['directions'][1]).astype(np.uint8) * 255)
        display_manager.set_screen('move_down', (move_data['directions'][2]).astype(np.uint8) * 255)
        display_manager.set_screen('move_left', (move_data['directions'][3]).astype(np.uint8) * 255)
        display_manager.set_screen('move_right', (move_data['directions'][4]).astype(np.uint8) * 255)
        display_manager.set_screen('move_hold', (move_data['directions'][0]).astype(np.uint8) * 255)
        display_manager.update()

        runtime_stats['current_screen'] = display_manager.screen_names[display_manager.current_screen]
        runtime_stats['fps'] = clock.get_fps()
        runtime_stats['seed_count'] = float(np.sum(world_tensor[:, :, seed_idx]))
        runtime_stats['plant_mass'] = float(np.sum(plant_energy))
        runtime_stats['herbivore_mass'] = float(np.sum(world_tensor[:, :, herbivore_energy_idx]))
        runtime_stats['predator_mass'] = float(np.sum(world_tensor[:, :, predator_energy_idx]))
        runtime_stats['step'] = step

        print(json.dumps(runtime_stats, indent=4))

        clock.tick()
        step += 1


if __name__ == "__main__":
    main()
