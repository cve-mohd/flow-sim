import appkit
import pygame
import numpy as np

class Visual(appkit.Item):
    def __init__(self, y_data: list, V_data: list, delta_t: float, y_smoothening, t_smoothening, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.y_data = np.array(y_data)
        self.V_data = np.array(V_data)
        
        self.y_smoothening = y_smoothening
        self.t_smoothening = t_smoothening
        
        self.max_V_inv = 1. / np.max(self.V_data)
        
        self.delta_t = delta_t // self.t_smoothening
        
        self.time_step = 0
        self.seconds_per_frame = 30.
        self.frames_per_time_step = self.delta_t / self.seconds_per_frame
        self.frame_counter = 0
        
        
        self.rendered_y = self.smoothen_array(self.y_data[self.time_step] , self.y_smoothening)
        self.rendered_V = self.smoothen_array(self.V_data[self.time_step] , self.y_smoothening)
        
        self.rendered_x = len(self.rendered_y)
                
        self.font = pygame.font.Font(None, 36)
        self.canvas = pygame.Rect(50, 50, 800, 600)
        
        self.y_scale = self.canvas.h // np.max(self.y_data)
        self.x_scale = self.canvas.w // self.rendered_x
        self.canvas.w = self.x_scale * self.rendered_x
                        
        self.smooth_t = True
        self.smoothing_counter = 1
        self.animation_running = True
        
        
    def interpolate_arrays(self, array1: np.ndarray, array2: np.ndarray, factor: int, index: int) -> np.ndarray:
        """Return an array that is a linear interpolation between array1 and array2.
        """
        array_ = array1 + (array2 - array1) * ((index + 1) / float(factor))
        return array_
            
            
    def smoothen_array(self, array_: np.ndarray, factor: int) -> np.ndarray:
        """Smoothens y_data and Q_data by interpolating between consecutive time steps.
        """
        smooth_array = []
        
        for i in range(len(array_) - 1):
            smooth_array.append(array_[i])
            
            for j in range(1, factor):
                smooth_array.append(array_[i] + (array_[i+1] - array_[i]) *  j / float(factor))
        
        smooth_array.append(array_[-1])
        
        smooth_array = np.array(smooth_array)
        
        return smooth_array
        
        
    def update(self):
        self.frame_counter += 1
        
        if self.frame_counter >= self.frames_per_time_step:
            self.frame_counter = 0
            self.time_step += 1
            converted_time_step = self.time_step // self.t_smoothening
                        
            if self.smooth_t:
                self.rendered_y = self.interpolate_arrays(
                    self.y_data[converted_time_step],
                    self.y_data[converted_time_step + 1],
                    factor=self.t_smoothening,
                    index=self.smoothing_counter - 1)
                
                self.rendered_V = self.interpolate_arrays(
                    self.V_data[converted_time_step],
                    self.V_data[converted_time_step + 1],
                    factor=self.t_smoothening,
                    index=self.smoothing_counter - 1)
                
                self.smoothing_counter += 1
                if self.smoothing_counter >= self.t_smoothening:
                    self.smooth_t = False
                    self.smoothing_counter = 1
                                
            else:
                self.rendered_y = self.y_data[converted_time_step]
                self.rendered_V = self.V_data[converted_time_step]
                self.smooth_t = True
            
            self.rendered_y = self.smoothen_array(self.rendered_y, self.y_smoothening)
            self.rendered_V = self.smoothen_array(self.rendered_V, self.y_smoothening)
            
            
            if self.time_step // self.t_smoothening >= len(self.y_data):
                self.animation_running = False

    
    def render(self, window: pygame.Surface):
        if not self.animation_running:
            return
                
        pygame.draw.rect(window, (0, 0, 0), self.canvas, 1)
        
        time = self.time_step * self.delta_t + self.frame_counter * self.seconds_per_frame
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        seconds = int(time % 60)
        text = self.font.render(f"{hours}:{minutes}:{seconds}", True, (0, 0, 0))
        window.blit(text, (10, 10))
                        
        for i in range(self.rendered_x):
            rg = 240 * (1 - self.rendered_V[i] * self.max_V_inv)
            C = (rg, rg, 255)
                        
            x = int(self.canvas.x + self.canvas.w - (i + 0.5) * self.x_scale)
            y1 = self.canvas.y + self.canvas.h
            y2 = y1 - self.rendered_y[i] * self.y_scale
            
            pygame.draw.line(window,
                             C,
                             (x, y1),
                             (x, y2),
                             self.x_scale)
            