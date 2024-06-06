import json
import torch
import argparse
import pygame

from tensor_beasts.display_manager import DisplayManager
from tensor_beasts.util import get_mean_execution_times
from tensor_beasts.world import World


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tensor beasts simulation")
    parser.add_argument(
        "--size",
        type=int,
        help="The size of the world. (default: 768)",
        required=False,
        default=768
    )
    parser.add_argument(
        "--device",
        type=str,
        help="The device to use. (default: mps)",
        required=False,
        default="mps"
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    torch.set_default_device(torch.device(args.device))

    width, height = (args.size,) * 2

    config = {
        "entities": {
            "predator": {
                "features": [
                    {"name": "energy", "group": "energy"},
                    {"name": "scent", "group": "scent"}
                ]
            },
            "plant": {
                "features": [
                    {"name": "energy", "group": "energy"},
                    {"name": "scent", "group": "scent"}
                ]
            },
            "herbivore": {
                "features": [
                    {"name": "energy", "group": "energy"},
                    {"name": "scent", "group": "scent"}
                ]
            },
            "seed": {
                "features": [
                    {"name": "energy", "group": None}
                ]
            }
        }
    }

    scalars = {
        "plant_init_odds": 255,
        "herbivore_init_odds": 255,
        "plant_growth_odds": 255,
        "plant_germination_odds": 255,
        "plant_crowding_odds": 25,
        "plant_seed_odds": 255,
        "herbivore_eat_max": 8,
        "predator_eat_max": 128
    }

    world = World(size=args.size, config=config, scalars=scalars)

    clock = pygame.time.Clock()

    screens = {
        'energy_rgb': torch.zeros((width, height, 3), dtype=torch.uint8),
        'scent_rgb': torch.zeros((width, height, 3), dtype=torch.uint8),
        'plant_scent': torch.zeros((width, height, 3), dtype=torch.uint8),
        'herbivore_scent': torch.zeros((width, height, 3), dtype=torch.uint8),
        'predator_scent': torch.zeros((width, height, 3), dtype=torch.uint8)
    }

    display_manager = DisplayManager(width, height, screens)

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
                    world.herbivore.energy[:] = (
                      torch.randint(0, world.herbivore_init_odds, (width, height), dtype=torch.uint8) == 0
                    ) * 255
                elif event.key == pygame.K_p:
                    world.predator.energy[:] = (
                        torch.randint(0, world.herbivore_init_odds, (width, height), dtype=torch.uint8) == 0
                    ) * 255

        world_stats = world.update(step)

        display_manager.set_screen('energy_rgb', world.energy)
        display_manager.set_screen('scent_rgb', world.scent)
        display_manager.set_screen('plant_scent', world.plant.scent)
        display_manager.set_screen('herbivore_scent', world.herbivore.scent)
        display_manager.set_screen('predator_scent', world.predator.scent)
        display_manager.update()

        runtime_stats = {
            'current_screen': display_manager.screen_names[display_manager.current_screen],
            'fps': clock.get_fps(),
            'step': step
        }
        runtime_stats.update(world_stats)
        runtime_stats.update(get_mean_execution_times())

        print(json.dumps(runtime_stats, indent=4))

        clock.tick()
        step += 1


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        main(args)
