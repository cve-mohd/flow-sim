import appkit
import pygame
import numpy as np

class Visual(appkit.Item):
    def __init__(self, y_data: list, V_data: list, delta_t: float, t_upscaling: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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
        self.rendered_V = self.V_data[self.time_step]
        
        self.n_nodes = len(self.rendered_y)
        
        self.time_font = pygame.font.Font(None, 24)
        self.grid_font = pygame.font.Font(None, 18)
        
        self.canvas = pygame.Rect(50, 50, 800, 600)
        
        self.y_scale = self.canvas.h // np.max(self.y_data)
        self.x_scale = self.canvas.w // (self.n_nodes - 1)
        self.canvas.w = self.x_scale * (self.n_nodes - 1) + 2
        
        self.update_points()
                        
        self.upscaling_applied = True
        self.upscaling_counter = 1
        self.animation_running = True
        
        
    def update_points(self):
        self.xy_points = [(self.canvas.x + self.canvas.w - i * self.x_scale - 1,
                               self.canvas.y + self.canvas.h - self.rendered_y[i] * self.y_scale) for i in range(self.n_nodes)]
        
        
    def interpolate_arrays(self, array1: np.ndarray, array2: np.ndarray, factor: int, index: int) -> np.ndarray:
        """Return an array that is a linear interpolation between array1 and array2.
        """
        array_ = array1 + (array2 - array1) * ((index + 1) / float(factor))
        return array_
                    
        
    def update(self):
        self.frame_counter += 1
        
        if self.frame_counter >= self.frames_per_time_step:
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
                        
            self.update_points()
            
            if self.time_step // self.t_upscaling >= len(self.y_data):
                self.animation_running = False

    
    def render(self, window: pygame.Surface):
        if not self.animation_running:
            return
        
        self.write_time(window)
                        
        self.draw_grid(window)
        pygame.draw.aalines(window, (0, 0, 255), False, self.xy_points)
                
                
    def write_time(self, window: pygame.Surface):
        time = self.time_step * self.delta_t # + self.frame_counter * self.seconds_per_frame
        
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        # seconds = int(time % 60)
        
        clock_string = f"{hours}"
        if hours < 10:
            clock_string = f"0{hours}"
            
        if minutes < 10:
            clock_string += f":0{minutes}"
        else:
            clock_string += f":{minutes}"
            
        clock_string += ":00"
        
        text = self.time_font.render(clock_string, True, (0, 0, 0))
        window.blit(text, (10, 10))
    
    
    def draw_grid(self, window: pygame.Surface):                  
        gray = (200, 200, 200)
        
        x1 = self.canvas.x
        x2 = x1 + self.canvas.w
        for i in range(6):
            y = self.canvas.y + self.canvas.h - i * 2.5 * self.y_scale
            pygame.draw.line(window, gray, (x1, y), (x2, y))
            
            label = str(i * 2.5)
            if i == 0:
                label = "0"
                
            text = self.grid_font.render(label, True, (0, 0, 0))
            window.blit(text, (x2 + 10, y - 6))
            
        y1 = self.canvas.y
        y2 = y1 + self.canvas.h
        for i in range(self.n_nodes):
            x = self.canvas.x + self.canvas.w - i * self.x_scale - 1
            pygame.draw.line(window, gray, (x, y1), (x, y2))
            
            label = str(i)
                
            text = self.grid_font.render(label, True, (0, 0, 0))
            window.blit(text, (x - 3, y2 + 10))
            
        pygame.draw.rect(window, (0, 0, 0), self.canvas, 1)
            