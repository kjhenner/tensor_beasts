import sys
from collections import deque
import time

import pygame
import torch
from pygame import DOUBLEBUF, OPENGL, RESIZABLE, mouse
from OpenGL.GL import (
    glBindTexture, glClear, glTexCoord2f, glVertex2f, glBegin, glEnd,
    glTexImage2D, glTexParameteri, glTexSubImage2D, glEnable, glGenTextures,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE, GL_LINEAR, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_QUADS,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, glLoadIdentity, glScale, glTranslate,
    GL_NEAREST, glViewport, GL_PROJECTION, GL_MODELVIEW, glMatrixMode
)
from OpenGL import GLU
import numpy as np


class DisplayManager:
    def __init__(self, world_width, world_height, world_thread=None):
        self.world_width = world_width
        self.world_height = world_height
        self.screen_width = world_width
        self.screen_height = world_height
        self.aspect_ratio = self.screen_width / self.screen_height
        self.zoom_level = 1
        self.offset = [0, 0]
        self.clock = pygame.time.Clock()
        self.current_screen = 0
        self.display = pygame.display.set_mode((self.screen_width, self.screen_height), DOUBLEBUF | OPENGL | RESIZABLE)
        self.dirty = True
        self.pan_speed = 0.1
        self.world_thread = world_thread

        self.display_array = np.zeros((self.world_height, self.world_width, 3), dtype=np.uint8)

        pygame.init()
        glEnable(GL_TEXTURE_2D)

        # Initialize texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.world_width, self.world_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        self.screen = np.zeros((self.world_height, self.world_width, 3), dtype=np.uint8)

        # Buffer for screens
        self.screen_buffer = deque()
        self.buffer_update_interval = 1  # Default 1 second
        self.last_update_time = time.time()

        if self.world_width < 256 and self.world_height < 256:
            self.resize(256, 256)

    def update(self):

        current_time = time.time()
        if len(self.screen_buffer) > 0 and (current_time - self.last_update_time) >= self.buffer_update_interval:
            self.screen[:] = self.screen_buffer.popleft()
            self.dirty = True
            self.last_update_time = current_time

        if self.dirty:
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.world_width, self.world_height, GL_RGB, GL_UNSIGNED_BYTE, self.screen)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glLoadIdentity()
            self.update_projection()

            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(-1, -1)
            glTexCoord2f(1, 0); glVertex2f(1, -1)
            glTexCoord2f(1, 1); glVertex2f(1, 1)
            glTexCoord2f(0, 1); glVertex2f(-1, 1)
            glEnd()

            pygame.display.flip()
            self.dirty = False
        self.clock.tick(15)

    def update_from_buffer(self):
        while len(self.screen_buffer) > 0:
            self.update()

    def update_screen(self, screen: torch.Tensor):
        self.dirty = True
        if screen.size() == (self.world_height, self.world_width):
            screen = screen.unsqueeze(-1).expand(-1, -1, 3)
        self.screen[:] = screen.cpu().numpy()

    def add_screens_to_buffer(self, screens: torch.Tensor):
        current_time = time.time()

        for screen in list(screens):
            self.screen_buffer.append(screen.cpu().numpy())

        time_taken = time.time() - current_time
        self.buffer_update_interval = time_taken / max(len(screens), 1)

    def zoom_in(self, speed=2.0):
        if self.zoom_level / speed >= 0.1:
            self.dirty = True
            self.zoom_level /= speed

    def zoom_out(self):
        self.zoom_in(1 / 1.1)

    def pan(self, dx, dy):
        self.dirty = True
        self.offset[0] += dx / self.zoom_level
        self.offset[1] += dy / self.zoom_level

    def update_projection(self):
        glViewport(0, 0, self.screen_width, self.screen_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        GLU.gluOrtho2D(
            -self.aspect_ratio * self.zoom_level + self.offset[0],
            self.aspect_ratio * self.zoom_level + self.offset[0],
            -1 * self.zoom_level + self.offset[1],
            1 * self.zoom_level + self.offset[1]
        )
        glMatrixMode(GL_MODELVIEW)

    def resize(self, width, height):
        self.screen_width = width
        self.screen_height = height
        self.aspect_ratio = width / height
        self.dirty = True
        self.display = pygame.display.set_mode((self.screen_width, self.screen_height), DOUBLEBUF | OPENGL | RESIZABLE)
        self.update_projection()

    def map_grid_position(self, screen_x, screen_y):
        # Convert screen coordinates to OpenGL coordinates
        gl_x = (screen_x / self.screen_width) * 2 - 1
        gl_y = 1 - (screen_y / self.screen_height) * 2

        # Apply zoom and pan
        world_x = (gl_x * self.aspect_ratio * self.zoom_level) + self.offset[0]
        world_y = (gl_y * self.zoom_level) + self.offset[1]

        # Convert to grid coordinates
        grid_x = int((world_x + self.aspect_ratio) / (2 * self.aspect_ratio) * self.world_width)
        grid_y = int((world_y + 1) / 2 * self.world_height)

        # Clamp values to ensure they're within the world bounds
        grid_x = max(0, min(grid_x, self.world_width - 1))
        grid_y = max(0, min(grid_y, self.world_height - 1))

        return grid_x, grid_y