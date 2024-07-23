import json
import torch
import argparse
import pygame
import threading
import time

from omegaconf import DictConfig, OmegaConf
from pygame import mouse
import sys

from tensor_beasts.display_manager import DisplayManager
from tensor_beasts.entities.entity import Predator, Plant, Herbivore
from tensor_beasts.util import get_mean_execution_times
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
        default=None
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
    parser.add_argument(
        "--config_path",
        type=str,
        help="The path to the config file.",
        default="beast_config.yaml"
    )
    return parser.parse_args()


class WorldThread(threading.Thread):
    def __init__(self, world, update_screen_cb, clock, device, running=True, max_ups=None):
        super().__init__(daemon=True)
        self.world = world
        self.update_screen_cb = update_screen_cb
        self.clock = clock
        self.done = False
        self.running = running
        self.device = device
        self.ups = None
        self.max_ups = 1

    def stop(self):
        self.done = True

    def toggle_pause(self):
        self.running = not self.running

    def step(self):
        start = time.time()
        self.world.update()
        end = time.time()
        if self.max_ups:
            step_time = 1 / self.max_ups
            if end - start < step_time:
                time.sleep(step_time - (end - start))
        cur_ups = 1 / (end - start)
        self.ups = cur_ups if self.ups is None else (self.ups + cur_ups) / 2
        world_stats = {"ups": self.ups}
        self.update_screen_cb(self.world.step, world_stats)

    def run(self):
        torch.set_default_device(self.device)
        while not self.done:
            if self.running:
                self.step()


def main(config: DictConfig):
    if config.world.device == 'auto' or not config.world.device:
        if torch.cuda.is_available():
            config.world.device = "cuda"
        elif torch.backends.mps.is_available():
            config.world.device = "mps"
        else:
            config.world.device = "cpu"

    torch.set_default_device(torch.device(config.world.device))

    width, height = config.world.size

    world = World(config.world)

    clock = pygame.time.Clock()

    if args.headless:
        # Initialize pygame here as it won't be handled by the display manager
        pygame.init()
        display_manager = None
    else:
        screens = [
            # (
            #     'scent_rgb',
            #     lambda: world.td.get(('shared_features', 'scent')),
            # ),
            # (
            #     'energy_rgb',
            #     lambda: world.td.get(('shared_features', 'energy')),
            # ),
            # (
            #     'herbivore_rgb',
            #     lambda: world.herbivore.energy.unsqueeze(-1).expand(-1, -1, 3),
            # ),
            # (
            #     'fertility_map',
            #     lambda: world.get_feature("plant", "fertility_map").unsqueeze(-1).expand(-1, -1, 3) * 255,
            # ),
            (
                'elevation',
                lambda: world.get_feature("terrain", "elevation") * 255,
            ),
            (
                'water_level',
                lambda: world.get_feature("terrain", "water_level") / 20 * 255,
            )
        ]
        display_manager = DisplayManager(*config.world.size)
        current_screen_idx = 0

    def update_screen(step, world_stats):
        if not args.headless:
            display_manager.update_screen(screens[current_screen_idx][1]())
        runtime_stats = {
            'current_screen': screens[current_screen_idx][0] if not args.headless else "(headless)",
            'fps': clock.get_fps(),
            'step': step
        }
        runtime_stats.update(world_stats)
        runtime_stats.update(get_mean_execution_times())
        print(json.dumps(runtime_stats, indent=4))

    world_thread = WorldThread(
        world,
        update_screen,
        clock,
        device=config.world.device,
        running=config.world.running,
        max_ups=config.world.max_ups
    )
    world_thread.start()

    done, step = False, 0
    while not done:
        if display_manager is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    world_thread.stop()
                    world_thread.join(5)
                    pygame.quit()
                    sys.exit()
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
                        display_manager.update_screen(screens[current_screen_idx][1]())
                    elif event.key == pygame.K_h:
                        world.initialize_herbivore()
                    elif event.key == pygame.K_p:
                        world.initialize_predator()
                    elif event.key == pygame.K_SPACE:
                        world_thread.toggle_pause()
                    elif event.key == pygame.K_s:
                        if not world_thread.running:
                            world_thread.step()
                elif event.type == pygame.MOUSEBUTTONUP:
                    pos = mouse.get_pos()
                    x, y = display_manager.map_grid_position(*pos)
                    print(f"Inspecting: {(x, y)}")
                    world.inspect(x, y)
                elif event.type == pygame.MOUSEMOTION:
                    rel = mouse.get_rel()
                    if mouse.get_pressed()[0]:
                        display_manager.pan(-2 * rel[0] / height, 2 * rel[1] / height)
                elif event.type == pygame.MOUSEWHEEL:
                    display_manager.zoom_in(1 + event.y * 0.02)
                elif event.type == pygame.VIDEORESIZE:
                    width, height = event.w, event.h
                    display_manager.resize(event.w, event.h)

        if not args.headless:
            display_manager.update()

        clock.tick()
        step += 1


if __name__ == "__main__":
    args = parse_args()
    # Load the config
    config = OmegaConf.load(args.config_path)
    main(config)
