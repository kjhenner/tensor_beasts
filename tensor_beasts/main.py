import json
import torch
import argparse
import pygame

from tensor_beasts.display_manager import DisplayManager
from tensor_beasts.util import get_mean_execution_times
from tensor_beasts.world import World


def handle_events(pygame, display_manager, world):
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
                    torch.randint(0, world.herbivore_init_odds, (world.width, world.height), dtype=torch.uint8) == 0
                ) * 255
            elif event.key == pygame.K_p:
                world.predator.energy[:] = (
                    torch.randint(0, world.herbivore_init_odds, (world.width, world.height), dtype=torch.uint8) == 0
                ) * 255


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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the simulation in headless mode."
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    torch.set_default_device(torch.device(args.device))

    width, height = (args.size,) * 2

    world = World(size=args.size)

    clock = pygame.time.Clock()

    if args.headless:
        # Initialize pygame here as it won't be handled by the display manager
        pygame.init()
        display_manager = None
    else:
        screens = {
            'energy_rgb': torch.zeros((width, height, 3), dtype=torch.uint8),
            'offspring_counts': torch.zeros((width, height, 3), dtype=torch.uint8),
            'scent_rgb': torch.zeros((width, height, 3), dtype=torch.uint8),
            'plant_scent': torch.zeros((width, height, 3), dtype=torch.uint8),
            'herbivore_scent': torch.zeros((width, height, 3), dtype=torch.uint8),
            'predator_scent': torch.zeros((width, height, 3), dtype=torch.uint8),
        }
        display_manager = DisplayManager(width, height, screens)

    done, step = False, 0
    while not done:
        if display_manager is not None:
            handle_events(pygame, display_manager, world)

        world_stats = world.update(step)

        if not args.headless:
            display_manager.set_screen(
                'energy_rgb',
                torch.where(world.obstacle.mask.unsqueeze(-1).expand(-1, -1, 3), world.obstacle.mask.unsqueeze(-1).expand(-1, -1, 3) * 255, world.energy)
            )
            display_manager.set_screen('offspring_counts', world.herbivore.offspring_count)
            display_manager.set_screen('scent_rgb', world.scent)
            # display_manager.overlay_screen(
            #     'scent_rgb',
            #     world.energy
            # )
            display_manager.set_screen('plant_scent', world.plant.scent)
            display_manager.set_screen('herbivore_scent', world.herbivore.scent)
            display_manager.set_screen('predator_scent', world.predator.scent)
            display_manager.update()

        runtime_stats = {
            'current_screen': display_manager.screen_names[display_manager.current_screen] if not args.headless else "(headless)",
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
