import appkit
import pygame
import numpy as np

class Background(appkit.Item):
    def __init__(self,
                 rect: tuple[int, int, int, int],
                 interval: tuple[float, float],
                 grid_lines: tuple[int, int],
                 padding: tuple[int, int, int, int],
                 tag = None):
        
        super().__init__(tag)
        
        self.rect = pygame.Rect(rect)
        self.x_interval = interval[0]
        self.y_interval = interval[1]
        self.padding = padding
        
        self.inner_border = pygame.Rect([self.padding[0],
                                   self.padding[2],
                                   rect[2] - self.padding[1] - self.padding[0],
                                   rect[3] - self.padding[3] - self.padding[2]])
        
        self.x_scale = self.inner_border.w / ((grid_lines[0] - 1) * self.x_interval)
        self.y_scale = self.inner_border.h / ((grid_lines[1] - 1) * self.y_interval)

        self.inner_border.w = (grid_lines[0] - 1) * int(self.x_scale * self.x_interval)
        self.inner_border.h = (grid_lines[1] - 1) * int(self.y_scale * self.y_interval)
        
        self.grid_font = pygame.font.Font(None, 18)
        
        self.surface = pygame.Surface((self.rect.w, self.rect.h))
        
        self.h_lines, self.v_lines = self.calc_coordinates()
        
        self.create_surface()
        
    
    def calc_coordinates(self):
        x1 = self.inner_border.x
        x2 = x1 + self.inner_border.w
        x_step = -int(self.x_interval  * self.x_scale)
                        
        y1 = self.padding[2]
        y2 = y1 + self.inner_border.h
        y_step = -int(self.y_interval * self.y_scale)
        
        h_lines = [y for y in range(y2, y1 - 1, y_step)]
        v_lines = [x for x in range(x2, x1 - 1, x_step)]
        
        return h_lines, v_lines
    
    
    def create_surface(self):
        self.surface.fill((255, 255, 255))
        self.create_grid()
        
        
    def render(self, window: pygame.Surface):
        window.blit(self.surface, (self.rect.x, self.rect.y))
        
        
    def create_grid(self):
        gray = (200, 200, 200)
        
        x1 = self.inner_border.x
        x2 = x1 + self.inner_border.w
        x_step = -int(self.x_interval  * self.x_scale)
                        
        y1 = self.padding[2]
        y2 = y1 + self.inner_border.h
        y_step = -int(self.y_interval * self.y_scale)
        
        # Horizontal lines
        for y in self.h_lines:
            pygame.draw.line(self.surface, gray, (x1, y), (x2, y))
            
            label = (y2 - y) / y_step * (-self.y_interval)
            label = str(label)
                
            text = self.grid_font.render(label, True, (0, 0, 0))
            self.surface.blit(text, (x1 - 30, y - 6))
            
        # Vertical lines
        for x in self.v_lines:
            pygame.draw.line(self.surface, gray, (x, y1), (x, y2))
            
            label = -(x2 - x) /  x_step * self.x_interval / 1000
            label = str(int(label))
                
            text = self.grid_font.render(label, True, (0, 0, 0))
            self.surface.blit(text, (x - 3, y2 + 10))
            
        pygame.draw.rect(self.surface, (0, 0, 0), self.inner_border, 1)