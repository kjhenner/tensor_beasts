from collections import deque
import time

import pygame
import torch
from pygame import DOUBLEBUF, OPENGL, RESIZABLE
from OpenGL.GL import (
    glBindTexture, glClear, glTexCoord2f, glVertex2f, glBegin, glEnd,
    glTexImage2D, glTexParameteri, glTexSubImage2D, glEnable, glGenTextures,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE, GL_LINEAR, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_QUADS,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, glLoadIdentity,
    GL_NEAREST, glViewport, GL_PROJECTION, GL_MODELVIEW, glMatrixMode
)
from OpenGL import GLU
import numpy as np


class DisplayManager:
    def __init__(self, world_width, world_height, world_thread=None):
        self.world_width = world_width
        self.world_height = world_height

        self.window_width = world_width
        self.window_height = world_height

        self.window_aspect = self.window_width / self.window_height
        self.world_aspect = self.world_width / self.world_height
        self.zoom_level = 1
        self.offset = [0, 0]
        self.clock = pygame.time.Clock()
        self.current_screen = 0
        self.display = pygame.display.set_mode((self.window_width, self.window_height), DOUBLEBUF | OPENGL | RESIZABLE)
        self.dirty = True
        self.pan_speed = 0.1
        self.world_thread = world_thread

        pygame.init()
        glEnable(GL_TEXTURE_2D)

        # Initialize texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        self.screen = np.zeros((self.world_height, self.world_width, 3), dtype=np.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.world_width, self.world_height, 0, GL_RGB, GL_UNSIGNED_BYTE, self.screen)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # Buffer for screens
        self.screen_buffer = deque()
        self.buffer_update_interval = 1  # Default 1 second
        self.last_update_time = time.time()

        if self.world_width < 256 and self.world_height < 256:
            self.resize(int(256 * self.world_aspect), 256)

    def update(self):
        current_time = time.time()
        if len(self.screen_buffer) > 0 and (current_time - self.last_update_time) >= self.buffer_update_interval:
            self.screen[:] = self.screen_buffer.popleft()
            self.last_update_time = current_time
            self.dirty = True

        if self.dirty:
            screen = np.flipud(self.screen)

            self.update_projection()
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.world_width, self.world_height, GL_RGB, GL_UNSIGNED_BYTE, screen)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(-self.world_aspect, -1)
            glTexCoord2f(1, 0); glVertex2f(self.world_aspect, -1)
            glTexCoord2f(1, 1); glVertex2f(self.world_aspect, 1)
            glTexCoord2f(0, 1); glVertex2f(-self.world_aspect, 1)
            glEnd()

            pygame.display.flip()
            self.dirty = False

        self.clock.tick(15)

    def update_from_buffer(self):
        while len(self.screen_buffer) > 0:
            self.update()

    def update_screen(self, screen: torch.Tensor):
        if screen.size() == (self.world_height, self.world_width):
            screen = screen.unsqueeze(-1).expand(-1, -1, 3)
        self.screen[:] = screen.cpu().numpy()
        self.dirty = True

    def add_screens_to_buffer(self, screens: torch.Tensor):
        current_time = time.time()

        for screen in list(screens):
            self.screen_buffer.append(screen.cpu().numpy())

        time_taken = time.time() - current_time
        self.buffer_update_interval = time_taken / max(len(screens), 1)

    def zoom_in(self, speed=1.1):
        self.zoom_level /= speed
        self.dirty = True

    def zoom_out(self):
        self.zoom_in(1 / 1.1)
        self.dirty = True

    def pan(self, dx, dy):
        self.offset[0] += dx / self.zoom_level
        self.offset[1] += dy / self.zoom_level
        self.dirty = True

    def update_projection(self):
        glViewport(0, 0, self.window_width, self.window_height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        world_aspect_ratio = self.world_aspect
        window_aspect_ratio = self.window_aspect

        if window_aspect_ratio > world_aspect_ratio:
            # Window is wider than world
            visible_height = 2
            visible_width = visible_height * window_aspect_ratio
        else:
            # Window is taller than world
            visible_width = 2 * world_aspect_ratio
            visible_height = visible_width / window_aspect_ratio

        # Apply zoom
        visible_width *= self.zoom_level
        visible_height *= self.zoom_level

        # Calculate boundaries with offset
        left = -visible_width / 2 + self.offset[0]
        right = visible_width / 2 + self.offset[0]
        bottom = -visible_height / 2 + self.offset[1]
        top = visible_height / 2 + self.offset[1]

        GLU.gluOrtho2D(left, right, bottom, top)

        glMatrixMode(GL_MODELVIEW)

    def resize(self, width, height):
        self.window_width = width
        self.window_height = height
        self.window_aspect = self.window_width / self.window_height
        self.dirty = True
        self.display = pygame.display.set_mode((self.window_width, self.window_height), DOUBLEBUF | OPENGL | RESIZABLE)
        self.update_projection()

    def map_grid_position(self, screen_x, screen_y):
        # Convert screen coordinates to normalized device coordinates
        ndc_x = (2 * screen_x / self.window_width) - 1
        ndc_y = 1 - (2 * screen_y / self.window_height)  # Flip y-coordinate

        # Calculate the visible area based on zoom and aspect ratios
        if self.window_aspect > self.world_aspect:
            visible_height = 2
            visible_width = visible_height * self.window_aspect
        else:
            visible_width = 2 * self.world_aspect
            visible_height = visible_width / self.window_aspect

        visible_width *= self.zoom_level
        visible_height *= self.zoom_level

        # Transform normalized device coordinates to world coordinates
        world_x = (ndc_x * visible_width / 2) + self.offset[0]
        world_y = (ndc_y * visible_height / 2) + self.offset[1]

        # Convert world coordinates to grid coordinates
        grid_x = int((world_x + self.world_aspect) / (2 * self.world_aspect) * self.world_width)
        grid_y = int((world_y + 1) / 2 * self.world_height)

        # Clamp values to ensure they're within the world bounds
        grid_x = max(0, min(grid_x, self.world_width - 1))
        grid_y = max(0, min(grid_y, self.world_height - 1))

        # Flip y-coordinate for screen array access
        grid_y = self.world_height - 1 - grid_y

        print(f"Screen value: {self.screen[grid_y][grid_x]}")
        return grid_x, grid_y