import json
import torch
import pygame

from tensor_beasts.comparator import compare
from tensor_beasts.display_manager import DisplayManager
from tensor_beasts.state_updates_torch import grow, germinate, move, eat, diffuse_scent
from tensor_beasts.util_torch import directional_kernel_set, pad_matrix, safe_sub, safe_add, torch_correlate

from tensor_beasts.state_updates_numpy import move as move_numpy, safe_add as safe_add_numpy
import numpy as np

def main():
    torch.set_default_device(torch.device("mps"))

    rng = torch.Generator().manual_seed(0)

    predator_energy_idx = 7

    width, height = (512,) * 2

    world_tensor = torch.zeros((width, height, 10), dtype=torch.uint8)
    plant_energy = world_tensor[:, :, 1]
    plant_scent = world_tensor[:, :, 2]

    seed_energy = world_tensor[:, :, 3]

    herbivore_energy = world_tensor[:, :, 4]
    herbivore_scent = world_tensor[:, :, 6]
    predator_scent = world_tensor[:, :, 9]
    predator_energy = world_tensor[:, :, 7]

    rgb = world_tensor[:, :, [7, 1, 4]]

    plant_init_odds = 255
    herbivore_init_odds = 255
    plant_growth_odds = 255
    plant_germination_odds = 255
    plant_crowding_odds = 25
    plant_seed_odds = 255
    herbivore_eat_max = 16
    predator_eat_max = 128

    clock = pygame.time.Clock()

    screens = {
        'rgb': torch.zeros((width, height, 3), dtype=torch.uint8),
        'plant_scent': torch.zeros((width, height, 3), dtype=torch.uint8),
        'herbivore_scent': torch.zeros((width, height, 3), dtype=torch.uint8)
    }

    display_manager = DisplayManager(width, height, screens)


    plant_energy[:] = (torch.randint(0, plant_init_odds, (width, height), dtype=torch.uint8) == 0)

    herbivore_energy[:] = (
        (torch.randint(0, herbivore_init_odds, (width, height), dtype=torch.uint8) == 0)
    ) * 255

    world_tensor[:, :, predator_energy_idx] = (
        (torch.randint(0, herbivore_init_odds, (width, height), dtype=torch.uint8) == 0)
    ) * 255

    plant_crowding_kernel = torch.tensor([
        [0, 1, 1, 1, 0],
        [1, 1, 2, 1, 1],
        [1, 2, 0, 2, 1],
        [1, 1, 2, 1, 1],
        [0, 1, 1, 1, 0],
    ], dtype=torch.uint8)

    done, step = False, 0
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
                      torch.randint(0, herbivore_init_odds, (width, height), generator=rng, dtype=torch.uint8) == 0
                    ) * 255

        rand_array = torch.randint(0, 255, (width, height), dtype=torch.uint8)

        plant_mask = plant_energy.bool()
        plant_crowding = torch_correlate(plant_mask, plant_crowding_kernel, mode='constant')

        grow(plant_energy, plant_growth_odds, plant_crowding, plant_crowding_odds, rand_array)
        seed_energy |= plant_crowding > (rand_array % plant_seed_odds)

        germinate(seed_energy, plant_energy, plant_germination_odds, rand_array)

        diffuse_scent(plant_energy, plant_scent)
        diffuse_scent(herbivore_energy, herbivore_scent)
        diffuse_scent(predator_energy, predator_scent)

        move(herbivore_energy, 250, plant_scent, safe_add(herbivore_scent, predator_scent))

        safe_sub(herbivore_energy, 2)

        move(predator_energy, 250, herbivore_scent, predator_scent)
        safe_sub(predator_energy, 1)

        eat(herbivore_energy, plant_energy, herbivore_eat_max)
        eat(predator_energy, herbivore_energy, predator_eat_max)

        display_manager.set_screen('rgb', rgb)
        display_manager.set_screen('plant_scent', plant_scent)
        display_manager.set_screen('herbivore_scent', herbivore_scent)
        display_manager.update()

        runtime_stats = {
            'current_screen': display_manager.screen_names[display_manager.current_screen],
            'fps': clock.get_fps(),
            'seed_count': float(torch.sum(seed_energy)),
            'plant_mass': float(torch.sum(plant_energy)),
            'herbivore_mass': float(torch.sum(herbivore_energy)),
            'predator_mass': float(torch.sum(predator_energy)),
            'step': step
        }

        print(json.dumps(runtime_stats, indent=4))

        clock.tick()
        step += 1


if __name__ == "__main__":
    main()
