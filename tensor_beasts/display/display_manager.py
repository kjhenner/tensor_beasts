from collections import deque
import time
from typing import Optional

import pygame
import torch
from pygame import DOUBLEBUF, OPENGL, RESIZABLE
from OpenGL.GL import (
    glBindTexture, glClear, glTexCoord2f, glVertex2f, glBegin, glEnd,
    glTexImage2D, glTexParameteri, glTexSubImage2D, glEnable, glGenTextures,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE, GL_LINEAR, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_QUADS,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, glLoadIdentity, glRasterPos2f, glDrawPixels, GL_RGBA,
    GL_NEAREST, glViewport, GL_PROJECTION, GL_MODELVIEW, glMatrixMode, glPushMatrix, glPopMatrix, glDeleteTextures
)
from OpenGL import GLU
import numpy as np


class DisplayManager:
    """DisplayManager class for rendering the world state using PyGame and OpenGL."""

    def __init__(
        self,
        display_width,
        display_height,
        world_thread=None
    ):
        self.display_width = display_width
        self.display_height = display_height

        self.window_width = display_width
        self.window_height = display_height

        self.window_aspect = self.window_width / self.window_height
        self.display_aspect = self.display_width / self.display_height

        self.display = pygame.display.set_mode((self.window_width, self.window_height), DOUBLEBUF | OPENGL | RESIZABLE)

        self.zoom_level = 1
        self.offset = [0, 0]
        self.pan_speed = 0.1
        self.world_thread = world_thread

        self.current_screen = 0

        pygame.font.init()
        self.font = pygame.font.Font(None, 16)

        self.dirty = True
        self.clock = pygame.time.Clock()
        pygame.init()
        self.texture = None
        glEnable(GL_TEXTURE_2D)

        self.screen = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        self.cell_value_data = np.zeros((self.display_height, self.display_width), dtype=np.uint8)

        self.create_texture(display_width, display_height)
        self.create_texture(display_width, display_height)

        if self.display_width < 256 and self.display_height < 256:
            self.resize(int(256 * self.display_aspect), 256)

    def create_texture(self, width, height):
        if self.texture is not None:
            glDeleteTextures([self.texture])
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        self.screen = np.zeros((height, width, 3), dtype=np.uint8)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, self.screen)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    def update(self):

        if self.dirty:
            # OpenGL uses bottom-left origin, so flip the screen
            screen = np.flipud(self.screen)
            if screen.shape[:2] != (self.display_height, self.display_width):
                self.display_height, self.display_width = screen.shape[:2]
                self.display_aspect = self.display_width / self.display_height
                self.create_texture(self.display_width, self.display_height)

            self.update_projection()
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                self.display_width,
                self.display_height,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                screen
            )

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glBegin(GL_QUADS)
            glTexCoord2f(0, 0)
            glVertex2f(-self.display_aspect, -1)
            glTexCoord2f(1, 0)
            glVertex2f(self.display_aspect, -1)
            glTexCoord2f(1, 1)
            glVertex2f(self.display_aspect, 1)
            glTexCoord2f(0, 1)
            glVertex2f(-self.display_aspect, 1)
            glEnd()

            # self.render_cell_values(self.cell_value_data)

            pygame.display.flip()
            self.dirty = False

        self.clock.tick(15)

    def render_cell_values(self, data: Optional[np.ndarray] = None):
        if data is None:
            return
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        GLU.gluOrtho2D(0, self.window_width, 0, self.window_height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        for y in range(self.display_height):
            for x in range(self.display_width):
                value = data[y, x]  # Assuming grayscale values
                if type(value) is not np.ndarray:
                    # Render a single centered value
                    screen_x, screen_y = self.screen_to_display(x, y, centered=True)
                    text_surface = self.font.render(str(value), False, (255, 255, 255))
                    text_width, text_height = text_surface.get_size()

                    text_data = pygame.image.tostring(text_surface, "RGBA", True)

                    glRasterPos2f(screen_x - text_width / 2, screen_y - text_height / 2)
                    glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
                elif len(value) == 4:
                    # render 2x2 grid
                    for i, val in enumerate(value):
                        screen_x, screen_y = self.screen_to_display(x + i % 2, y + i // 2, centered=False)
                        text_surface = self.font.render(str(val), False, (255, 255, 255))
                        text_width, text_height = text_surface.get_size()

                        text_data = pygame.image.tostring(text_surface, "RGBA", True)

                        glRasterPos2f(screen_x - text_width / 2, screen_y - text_height / 2)
                        glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
                elif len(value) == 8:
                    # render 3x3 grid
                    for i, val in enumerate(value):
                        if i < 4:
                            dx = x + (i % 3) * 0.333
                            dy = y + (i // 3) * 0.333
                        elif i == 4:
                            dx = x + 0.666
                            dy = y + 0.333
                        else:
                            dx = x + (i % 3) * 0.333
                            dy = y + 0.666

                        screen_x, screen_y = self.screen_to_display(dx, dy, centered=False)
                        # using numpy, render appropriate value using sciencific notation if needed
                        text = f"{val:.2e}" if val > 0.01 else f"{val:.2f}"
                        text_surface = self.font.render(text, False, (255, 255, 255))
                        text_width, text_height = text_surface.get_size()

                        text_data = pygame.image.tostring(text_surface, "RGBA", True)

                        glRasterPos2f(screen_x, screen_y - text_height)
                        glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def update_screen(
        self,
        screen_data: torch.Tensor,
        text_data: Optional[torch.Tensor] = None
    ):
        new_height, new_width = screen_data.shape[:2]

        if (new_height, new_width) != (self.display_height, self.display_width):
            self.screen = screen_data.cpu().numpy()
        else:
            self.screen[:] = screen_data.cpu().numpy()

        if text_data is not None:
            self.cell_value_data = text_data.cpu().numpy()
        else:
            self.cell_value_data = None

        self.dirty = True

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

        world_aspect_ratio = self.display_aspect
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

    def screen_to_display(self, screen_x, screen_y, centered=True):
        # Convert
        if centered:
            screen_x += 0.5
            screen_y += 0.5

        x = (screen_x / self.display_width) * (2 * self.display_aspect) - self.display_aspect
        y = (screen_y / self.display_height) * 2 - 1

        # Calculate the visible area based on zoom and aspect ratios
        if self.window_aspect > self.display_aspect:
            visible_height = 2
            visible_width = visible_height * self.window_aspect
        else:
            visible_width = 2 * self.display_aspect
            visible_height = visible_width / self.window_aspect

        visible_width *= self.zoom_level
        visible_height *= self.zoom_level

        # Apply zoom and offset
        x = (x - self.offset[0]) * (2 / visible_width)
        y = (y + self.offset[1]) * (2 / visible_height)

        # Convert world coordinates to screen coordinates
        display_x = int((x + 1) * self.window_width / 2)
        display_y = int((1 - y) * self.window_height / 2)  # Flip y-coordinate for screen

        return display_x, display_y

    def display_to_screen(self, display_x, display_y):

        # Convert screen coordinates to normalized device coordinates
        ndc_x = (2 * display_x / self.window_width) - 1
        ndc_y = 1 - (2 * display_y / self.window_height)  # Flip y-coordinate

        # Calculate the visible area based on zoom and aspect ratios
        if self.window_aspect > self.display_aspect:
            visible_height = 2
            visible_width = visible_height * self.window_aspect
        else:
            visible_width = 2 * self.display_aspect
            visible_height = visible_width / self.window_aspect

        visible_width *= self.zoom_level
        visible_height *= self.zoom_level

        # Transform normalized display coordinates to screen data coordinates
        screen_x = (ndc_x * visible_width / 2) + self.offset[0]
        screen_y = (ndc_y * visible_height / 2) + self.offset[1]

        screen_x = int((screen_x + self.display_aspect) / (2 * self.display_aspect) * self.display_width)
        screen_y = int((screen_y + 1) / 2 * self.display_height)

        # Clamp values to ensure they're within the world bounds
        screen_x = max(0, min(screen_x, self.display_width - 1))
        screen_y = max(0, min(screen_y, self.display_height - 1))

        # Flip y-coordinate for screen array access
        screen_y = self.display_height - 1 - screen_y

        return screen_x, screen_y
