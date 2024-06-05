import json
import torch
import pygame

from tensor_beasts.display_manager import DisplayManager
from tensor_beasts.state_updates_torch import grow, germinate, move, eat, diffuse_scent
from tensor_beasts.util_torch import safe_sub, safe_add, torch_correlate_2d as correlate_2d, get_mean_execution_times


def main():
    torch.set_default_device(torch.device("mps"))

    width, height = (1024,) * 2

    world_tensor = torch.zeros((width, height, 10), dtype=torch.uint8)

    idx_iter = iter(range(10))

    predator_energy = world_tensor[:, :, next(idx_iter)]
    plant_energy = world_tensor[:, :, next(idx_iter)]
    herbivore_energy = world_tensor[:, :, next(idx_iter)]

    energy_group = world_tensor[:, :, :3]

    predator_scent = world_tensor[:, :, next(idx_iter)]
    plant_scent = world_tensor[:, :, next(idx_iter)]
    herbivore_scent = world_tensor[:, :, next(idx_iter)]

    scent_group = world_tensor[:, :, 3:6]

    seed_energy = world_tensor[:, :, next(idx_iter)]

    plant_init_odds = 255
    herbivore_init_odds = 255
    plant_growth_odds = 255
    plant_germination_odds = 255
    plant_crowding_odds = 25
    plant_seed_odds = 255
    herbivore_eat_max = 8
    predator_eat_max = 128

    clock = pygame.time.Clock()

    screens = {
        'energy_rgb': torch.zeros((width, height, 3), dtype=torch.uint8),
        'scent_rgb': torch.zeros((width, height, 3), dtype=torch.uint8),
        'plant_scent': torch.zeros((width, height, 3), dtype=torch.uint8),
        'herbivore_scent': torch.zeros((width, height, 3), dtype=torch.uint8),
        'predator_scent': torch.zeros((width, height, 3), dtype=torch.uint8)
    }

    display_manager = DisplayManager(width, height, screens)

    plant_energy[:] = (torch.randint(0, plant_init_odds, (width, height), dtype=torch.uint8) == 0)

    herbivore_energy[:] = (
        (torch.randint(0, herbivore_init_odds, (width, height), dtype=torch.uint8) == 0)
    ) * 255

    predator_energy[:] = (
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
                      torch.randint(0, herbivore_init_odds, (width, height), dtype=torch.uint8) == 0
                    ) * 255
                elif event.key == pygame.K_p:
                    predator_energy[:] = (
                        torch.randint(0, herbivore_init_odds, (width, height), dtype=torch.uint8) == 0
                    ) * 255

        rand_array = torch.randint(0, 255, (width, height), dtype=torch.uint8)

        plant_mask = plant_energy.bool()

        if step % 2 == 0:
            plant_crowding = correlate_2d(plant_mask, plant_crowding_kernel, mode='constant')
            grow(plant_energy, plant_growth_odds, plant_crowding, plant_crowding_odds, rand_array)
            seed_energy |= plant_crowding > (rand_array % plant_seed_odds)
            germinate(seed_energy, plant_energy, plant_germination_odds, rand_array)

        diffuse_scent(energy_group, scent_group)

        move(herbivore_energy, 250, plant_scent, safe_add(herbivore_scent, predator_scent, inplace=False))

        safe_sub(herbivore_energy, 2)

        move(predator_energy, 250, herbivore_scent, predator_scent)
        safe_sub(predator_energy, 1)

        eat(herbivore_energy, plant_energy, herbivore_eat_max)
        eat(predator_energy, herbivore_energy, predator_eat_max)

        display_manager.set_screen('energy_rgb', energy_group)
        display_manager.set_screen('scent_rgb', scent_group)
        display_manager.set_screen('plant_scent', plant_scent)
        display_manager.set_screen('herbivore_scent', herbivore_scent)
        display_manager.set_screen('predator_scent', predator_scent)
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

        runtime_stats.update(get_mean_execution_times())

        print(json.dumps(runtime_stats, indent=4))

        clock.tick()
        step += 1


if __name__ == "__main__":
    main()
