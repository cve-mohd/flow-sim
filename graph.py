import appkit
import pygame
import numpy as np
from background import Background

class Graph(appkit.Item):
    def __init__(self, y_data: list, V_data: list, delta_t: float, delta_x: float, t_upscaling: int, background: Background, tag = None):
        super().__init__(tag)
        
        self.y_data = np.array(y_data)
        self.V_data = np.array(V_data)
        
        self.t_upscaling = t_upscaling
        
        self.max_V_inv = 1. / np.max(self.V_data)
        
        self.delta_t = delta_t // self.t_upscaling
        
        self.time_step = 0
        self.seconds_per_frame = 30.
        self.frames_per_time_step = self.delta_t / self.seconds_per_frame
        self.frame_counter = 0
        
        self.rendered_y = self.y_data[self.time_step]
        self.rendered_x = [i * delta_x for i in range(len(self.rendered_y))]
        self.rendered_V = self.V_data[self.time_step]
        
        self.border = pygame.Rect(background.inner_border)
        self.x_interval = background.x_interval
        self.x_scale, self.y_scale = background.x_scale, background.y_scale
        self.y_correction = background.rect.y + 2
        
        self.surface = pygame.Surface((self.border.w, self.border.h), pygame.SRCALPHA)
        self.time_surface = None
        
        self.update_curve()           
                        
        self.upscaling_applied = True
        self.upscaling_counter = 1
        self.animation_running = True
            
            
    def update(self):
        self.frame_counter += 1
        
        if self.frame_counter >= self.frames_per_time_step:
            self.surface.fill((255, 255, 255, 0))
            
            self.frame_counter = 0
            self.time_step += 1
            converted_time_step = self.time_step // self.t_upscaling
                        
            if self.upscaling_applied:
                self.rendered_y = self.interpolate_arrays(
                    self.y_data[converted_time_step],
                    self.y_data[converted_time_step + 1],
                    factor=self.t_upscaling,
                    index=self.upscaling_counter - 1)
                
                self.rendered_V = self.interpolate_arrays(
                    self.V_data[converted_time_step],
                    self.V_data[converted_time_step + 1],
                    factor=self.t_upscaling,
                    index=self.upscaling_counter - 1)
                
                self.upscaling_counter += 1
                if self.upscaling_counter >= self.t_upscaling:
                    self.upscaling_applied = False
                    self.upscaling_counter = 1
                                
            else:
                self.rendered_y = self.y_data[converted_time_step]
                self.rendered_V = self.V_data[converted_time_step]
                self.upscaling_applied = True
                        
            self.update_curve()
            
            if self.time_step // self.t_upscaling >= len(self.y_data):
                self.animation_running = False
                

    
    def render(self, window: pygame.Surface):
        if not self.animation_running:
            return
        
        window.blit(self.surface, (self.border.x, self.border.y))
                
  
    def update_curve(self):
        xy_points = [(x, y) for x, y in zip(self.rendered_x, self.rendered_y)]
        coords = [(self.border.w - x * self.x_scale, self.y_correction + self.border.h - y * self.y_scale) for x, y in xy_points]
        
        pygame.draw.aalines(self.surface, (0, 0, 255), False, coords)
        
        
    def interpolate_arrays(self, array1: np.ndarray, array2: np.ndarray, factor: int, index: int) -> np.ndarray:
        """Return an array that is a linear interpolation between array1 and array2.
        """
        array_ = array1 + (array2 - array1) * ((index + 1) / float(factor))
        return array_