import json
import torch
import argparse
import pygame

from tensor_beasts.display_manager import DisplayManager
from tensor_beasts.util import get_mean_execution_times, scale_tensor
from tensor_beasts.world import World


PAN_SPEED = 0.1


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
    parser.add_argument(
        "--store_buffer",
        action="store_true",
        help="Keep a state buffer."
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
        screens = [
            (
                'energy_rgb',
                lambda: torch.where(world.obstacle.mask.unsqueeze(-1).expand(-1, -1, 3), world.obstacle.mask.unsqueeze(-1).expand(-1, -1, 3) * 255, world.energy)
            ),
            (
                'scent_rgb',
                lambda: world.scent,
            ),
            (
                'herbivore_rgb',
                lambda: world.herbivore.energy.unsqueeze(-1).expand(-1, -1, 3),
            ),
        ]
        display_manager = DisplayManager(width, height)
        current_screen_idx = 0

    done, step = False, 0
    while not done:
        if display_manager is not None:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        display_manager.zoom_in()
                    elif event.key == pygame.K_MINUS:
                        display_manager.zoom_out()
                    elif event.key == pygame.K_LEFT:
                        display_manager.pan(PAN_SPEED, 0)
                    elif event.key == pygame.K_RIGHT:
                        display_manager.pan(-PAN_SPEED, 0)
                    elif event.key == pygame.K_UP:
                        display_manager.pan(0, -PAN_SPEED)
                    elif event.key == pygame.K_DOWN:
                        display_manager.pan(0, PAN_SPEED)
                    elif event.key == pygame.K_n:
                        current_screen_idx = (current_screen_idx + 1) % len(screens)
                    elif event.key == pygame.K_h:
                        world.initialize_herbivore()
                    elif event.key == pygame.K_p:
                        world.initialize_predator()

        _, world_stats = world.update()

        if not args.headless:
            display_manager.update(screens[current_screen_idx][1]())

        runtime_stats = {
            'current_screen': screens[current_screen_idx][0] if not args.headless else "(headless)",
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
