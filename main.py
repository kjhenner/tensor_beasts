import datetime
import json
import numpy
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from collections import defaultdict
import scipy.ndimage

from move import move


def safe_add(a, b):
    a += b
    a[a < b] = 255
    return a


def safe_sub(a, b):
    a -= b
    a[a > 255 - b] = 0
    return a


def eat(herbivore_energy, plant_energy, eat_max):
    eat_tensor = (herbivore_energy > 0) * np.min([plant_energy, np.ones(plant_energy.shape, dtype=np.uint8) + eat_max], axis=0)
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


class DisplayManager:

    def __init__(self, screen_width, screen_height, screens):
        self.width = screen_width * len(screens)
        self.height = screen_height
        self.clock = pygame.time.Clock()
        self.screens = screens
        self.display = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)

        self.display_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        pygame.init()
        glEnable(GL_TEXTURE_2D)

        # Initialize texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def update(self):
        # Update texture with the RGB image

        display = np.concatenate([self.screens[screen] for screen in self.screens], axis=1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, display)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()

        pygame.display.flip()
        pygame.time.wait(1)

    def set_screen(self, screen_name, screen):
        if len(screen.shape) < 3:
            screen = np.stack((screen, screen, screen), axis=-1)
        self.screens[screen_name] = screen


def conv2d(a, f):
    return scipy.ndimage.convolve(a, f, mode='constant')


def main():
    rng = np.random.default_rng()

    plant_energy_idx = 1
    seed_idx = 2
    herbivore_energy_idx = 3

    width = 512
    height = 512

    plant_init_odds = np.uint8(255)
    herbivore_init_odds = np.uint8(255)
    plant_growth_odds = np.uint8(255)
    plant_germination_odds = np.uint8(255)
    plant_crowding_odds = np.uint8(25)
    plant_seed_odds = np.uint8(255)
    herbivore_eat_max = np.uint8(32)

    clock = pygame.time.Clock()

    runtime_stats = {}

    screens = {
        'rgb': np.zeros((width, height, 3), dtype=np.uint8),
        # 'plant_crowding': np.zeros((width, height, 3), dtype=np.uint8),
        # 'plant_seed': np.zeros((width, height, 3), dtype=np.uint8)
    }

    display_manager = DisplayManager(width, height, screens)

    # Initialize a 1024x1024x8 tensor with zeros
    world_tensor = np.zeros((width, height, 8), dtype=np.uint8)

    # Set the plant channel to 1 at random locations
    world_tensor[:, :, plant_energy_idx] = (rng.integers(0, plant_init_odds, (width, height), dtype=np.uint8) == 0)
    world_tensor[:, :, herbivore_energy_idx] = (
        rng.integers(0, herbivore_init_odds, (width, height), dtype=np.uint8) == 0
    ) * 255

    plant_crowding_kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 2, 1, 1],
        [1, 2, 0, 2, 1],
        [1, 1, 2, 1, 1],
        [0, 1, 1, 1, 0],
    ], dtype=np.uint8)

    rand_size = 512
    rand_arrays = rng.integers(0, 255, (rand_size, width, height), dtype=np.uint8)
    time_data = defaultdict(list)

    done = False
    step = 0
    while not done:
        rand_array = rand_arrays[step % rand_size]

        plant_mask = numpy.array(world_tensor[:, :, plant_energy_idx], dtype=bool)
        plant_crowding = scipy.ndimage.convolve(plant_mask, plant_crowding_kernel, mode='constant')

        grow(world_tensor[:, :, plant_energy_idx], plant_growth_odds, plant_crowding, plant_crowding_odds, rand_array)

        world_tensor[:, :, seed_idx] |= plant_crowding > (rand_array % plant_seed_odds)

        germinate(world_tensor[:, :, seed_idx], world_tensor[:, :, plant_energy_idx], plant_germination_odds, rand_array)

        world_tensor[:, :, herbivore_energy_idx] = move(world_tensor[:, :, herbivore_energy_idx], divide_threshold=250)

        safe_sub(world_tensor[:, :, herbivore_energy_idx], 2)

        eat(world_tensor[:, :, herbivore_energy_idx], world_tensor[:, :, plant_energy_idx], herbivore_eat_max)

        plant_rgb = world_tensor[:, :, (0, plant_energy_idx, herbivore_energy_idx)]  # Extract RGB channels
        # seed_rgb = world_tensor[:, :, (seed_idx, 0, seed_idx)] * 255  # Extract RGB channels

        display_manager.set_screen('rgb', plant_rgb)
        display_manager.update()

        runtime_stats['fps'] = clock.get_fps()
        runtime_stats['seed_count'] = float(np.sum(world_tensor[:, :, seed_idx]))
        runtime_stats['plant_mass'] = float(np.sum(world_tensor[:, :, plant_energy_idx]))
        runtime_stats['herbivore_mass'] = float(np.sum(world_tensor[:, :, herbivore_energy_idx]))
        runtime_stats['step'] = step

        print(json.dumps(runtime_stats, indent=4))

        clock.tick()
        step += 1


if __name__ == "__main__":
    main()
