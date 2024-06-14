import pygame
import torch
from pygame import DOUBLEBUF, OPENGL, RESIZABLE
from OpenGL.GL import (
    glBindTexture, glClear, glTexCoord2f, glVertex2f, glBegin, glEnd,
    glTexImage2D, glTexParameteri, glTexSubImage2D, glEnable, glGenTextures,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE, GL_LINEAR, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_QUADS,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, glLoadIdentity, glScale, glTranslate,
)
from OpenGL import GLU
import numpy as np


class DisplayManager:
    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        self.aspect_ratio = screen_width / screen_height
        self.zoom_level = 1
        self.offset = [0, 0]
        self.clock = pygame.time.Clock()
        self.current_screen = 0
        self.display = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        self.dirty = True

        self.display_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        pygame.init()
        glEnable(GL_TEXTURE_2D)

        # Initialize texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def update(self):
        if self.dirty:
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, self.screen)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glLoadIdentity()
            GLU.gluOrtho2D(
                -self.aspect_ratio, self.aspect_ratio, -1, 1
            )
            glScale(self.zoom_level, self.zoom_level, 1)
            glTranslate(self.offset[0], self.offset[1], 0)

            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(-1, -1)
            glTexCoord2f(1, 0); glVertex2f(1, -1)
            glTexCoord2f(1, 1); glVertex2f(1, 1)
            glTexCoord2f(0, 1); glVertex2f(-1, 1)
            glEnd()

            pygame.display.flip()
            self.dirty = False
        self.clock.tick(30)

    def update_screen(self, screen: torch.Tensor):
        self.dirty = True
        self.screen[:] = screen.cpu().numpy()

    def zoom_in(self, speed=2.0):
        if self.zoom_level * speed >= 1:
            self.dirty = True
            self.zoom_level *= speed

    def zoom_out(self):
        self.zoom_in(1 / 2)

    def pan(self, dx, dy):
        self.dirty = True
        self.offset[0] += dx / self.zoom_level
        self.offset[1] += dy / self.zoom_level

    def resize(self, width, height):
        self.aspect_ratio = width / height
        self.dirty = True
