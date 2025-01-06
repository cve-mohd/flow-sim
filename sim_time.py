import pygame
import appkit
from graph import Graph

class Timer(appkit.Item):
    def __init__(self, tag = None, graph_: Graph = None):
        super().__init__(tag)
        
        self.graph = graph_
        self.time_step = -1
        self.time_font = pygame.font.Font(None, 28)
        self.surface = None
        
        
    def update(self):
        if self.time_step == self.graph.time_step:
            return
        
        self.time_step = self.graph.time_step
        time = self.time_step * self.graph.delta_t # + self.frame_counter * self.seconds_per_frame
        
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
        
        self.surface = self.time_font.render(clock_string, True, (0, 0, 0))
    
    
    def render(self, window):
        window.blit(self.surface, (10, 10))