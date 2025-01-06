import appkit
import pygame
import numpy as np
from boundary import Upstream
from settings import DURATION


class HydrographCurve(appkit.Item):
    def __init__(self, tag = None, rect: tuple[int, int, int, int] = None):
        super().__init__(tag)
        
        self.x_axis = np.linspace(0, DURATION, 100)
        self.y_axis = np.array([Upstream.inflow_hydrograph(t) for t in self.x_axis])
        
        self.x_scale = rect[2] / DURATION
        self.y_scale = rect[3] / max(self.y_axis)
        
        self.border = pygame.Rect(rect)
        self.surface = pygame.Surface((self.border.w, self.border.h))
        
        self.create_surface()
        
        
    def create_surface(self):
        self.surface.fill((255, 255, 255))
        
        pygame.draw.rect(self.surface, (0, 0, 0), (0, 0, self.border.w, self.border.h), width=1)
        
        points = [(x * self.x_scale,
                   self.border.h - y * self.y_scale)
                  for x, y in zip(self.x_axis, self.y_axis)]
        
        pygame.draw.aalines(self.surface, (0, 0, 0), False, points)
        
        
    def render(self, window):
        window.blit(self.surface, (self.border.x, self.border.y))
        