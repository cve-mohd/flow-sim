import appkit
import pygame
import numpy as np

class Visual(appkit.Item):
    def __init__(self, y_data: list, Q_data: list, delta_t: int | float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.y_data = np.array(y_data)
        self.Q_data = np.array(Q_data)
        
        self.max_Q = np.max(self.Q_data)
        self.max_y = np.max(self.y_data)
        
        self.delta_t = delta_t
        self.time_step = 0
        self.seconds_per_frame = 30.
        self.frames_per_time_step = self.delta_t / self.seconds_per_frame
        self.frame_counter = 0
        
        self.canvas = pygame.Rect(50, 50, 800, 600)
        
        for i in range(2):
            self.smoothen_time_series()
            self.smoothen_space_series()
                
        self.animation_running = True
        
        
    def interpolate_arrays(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        """Return an array that is a linear interpolation between array1 and array2.
        """
        return array1 + 0.5 * (array2 - array1)
            
            
    def smoothen_time_series(self) -> None:
        """Smoothens y_data and Q_data by interpolating between consecutive time steps.
        """
        y_data_smooth = [self.y_data[0]]
        Q_data_smooth = [self.Q_data[0]]
        
        for i in range(len(self.y_data) - 1):
            y_data_smooth.append(self.interpolate_arrays(self.y_data[i], self.y_data[i + 1]))
            y_data_smooth.append(self.y_data[i + 1])
            
            Q_data_smooth.append(self.interpolate_arrays(self.Q_data[i], self.Q_data[i + 1]))
            Q_data_smooth.append(self.Q_data[i + 1])
            
        self.y_data = np.array(y_data_smooth)
        self.Q_data = np.array(Q_data_smooth)
        
        self.delta_t *= 0.5
        self.frames_per_time_step *= 0.5
    
    
    def smoothen_arrays(self, array_: np.ndarray) -> np.ndarray:
        """Smoothens a list by interpolating between consecutive elements.
        """
        new_array = [array_[0]]
        
        for i in range(len(array_) - 1):
            new_array.append(array_[i] + 0.5 * (array_[i + 1] - array_[i]))
            new_array.append(array_[i + 1])
            
        return np.array(new_array)
    
    
    def smoothen_space_series(self) -> None:
        """Smoothens y_data and Q_data by interpolating between consecutive space nodes.
        """
        y_data_smooth = []
        Q_data_smooth = []
        
        for i in range(len(self.y_data)):
            y_data_smooth.append(self.smoothen_arrays(self.y_data[i]))
            Q_data_smooth.append(self.smoothen_arrays(self.Q_data[i]))
            
        self.y_data = np.array(y_data_smooth)
        self.Q_data = np.array(Q_data_smooth)
        
        self.x_data = [i for i in range(len(self.y_data[0]))]
        
        
    def update(self):
        self.frame_counter += 1
        if self.frame_counter >= self.frames_per_time_step:
            self.frame_counter = 0
            self.time_step += 1
            
            if self.time_step >= len(self.y_data):
                self.animation_running = False

    
    def render(self, window: pygame.Surface):
        if not self.animation_running:
            return
                
        pygame.draw.rect(window, (0, 0, 0), self.canvas, 1)
        
        # Write the time on the top left corner
        font = pygame.font.Font(None, 36)
        text = font.render(f"Time: {self.time_step * self.delta_t} s", True, (0, 0, 0))
        window.blit(text, (10, 10))
        
        y_scale = self.canvas.h // self.max_y
        x_scale = self.canvas.w // len(self.y_data[0])
                
        for i in range(len(self.y_data[0])):
            # color should vary with Q_data[i] from white to blue
            fraction = self.Q_data[self.time_step][i] / self.max_Q
            C = (240 - int(240 * fraction),
                 240 - int(240 * fraction),
                 255)
                        
            x = self.canvas.x + self.canvas.w - i * x_scale - x_scale // 2
            y = self.canvas.y + self.canvas.h
            pygame.draw.line(window,
                             C,
                             (x, y),
                             (x, y - self.y_data[self.time_step][i] * y_scale),
                             x_scale)