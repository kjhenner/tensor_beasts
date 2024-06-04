from typing import Dict

import pygame
import torch
from pygame import DOUBLEBUF, OPENGL
import numpy as np
from OpenGL.GL import (
    glBindTexture, glClear, glTexCoord2f, glVertex2f, glBegin, glEnd,
    glTexImage2D, glTexParameteri, glTexSubImage2D, glEnable, glGenTextures,
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE, GL_LINEAR, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_QUADS,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, glLoadIdentity,
)


class DisplayManager:
    def __init__(self, screen_width, screen_height, screens: Dict[str, torch.Tensor]):
        self.width = screen_width
        self.height = screen_height
        self.zoom_level = 1
        self.clock = pygame.time.Clock()
        self.current_screen = 0
        self.screens = screens
        self.screen_names = list(screens.keys())
        self.display = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)

        self.display_array = torch.zeros((self.height, self.width, 3), dtype=torch.uint8)

        pygame.init()
        glEnable(GL_TEXTURE_2D)

        # Initialize texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def update(self):

        screen = self.screens[self.screen_names[self.current_screen]]
        if self.zoom_level > 1:
            zoom_divisor = int(self.width // self.zoom_level)
            screen = screen[:zoom_divisor, :zoom_divisor]
            display = screen.repeat_interleave(self.zoom_level, 0).repeat_interleave(self.zoom_level, 1)
        else:
            display = screen

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, display.cpu().numpy())

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()

        pygame.display.flip()
        self.clock.tick(60)

    def set_screen(self, screen_name, screen):
        if len(screen.shape) < 3:
            screen = torch.stack((screen, screen, screen), dim=-1)
        self.screens[screen_name] = screen

    def next_screen(self):
        self.current_screen = (self.current_screen + 1) % len(self.screen_names)

    def zoom_in(self):
        self.zoom_level *= 2

    def zoom_out(self):
        if self.zoom_level > 1:
            self.zoom_level //= 2
