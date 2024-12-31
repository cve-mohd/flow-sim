import pygame, pkg_resources

class Item:
    def __init__(self, tag: str = None):
        self.tag = tag

    def handleEvent(self, event: pygame.event.Event):
        pass

    def update(self):
        pass

    def render(self, window: pygame.Surface):
        pass

def coords_within_rect(coordinates: tuple[int, int],
                       rect: tuple[int, int, int, int]) -> bool:

    x, y = coordinates[0], coordinates[1]
    if x >= rect[0] and x < rect[0] + rect[2] and y > rect[1] and y < rect[1] + rect[3]:
        return True

class Button(Item):
    def __init__(self,
                 rect: tuple[int, int, int, int] = (0, 0, 100, 50),
                 border_radii: tuple[int, int, int, int] = (3, 3, 3, 3),
                 width: int = 0,
                 default_color: tuple[int, int, int] | tuple[int, int, int, int] = (255, 255, 255),
                 hover_color: tuple[int, int, int] | tuple[int, int, int, int] = (200, 200, 200),
                 focus_color: tuple[int, int, int] | tuple[int, int, int, int] = (100, 100, 100),
                 text: str = None,
                 text_size: int = 20,
                 text_color: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0),
                 font: pygame.font.Font = None,
                 click_action: str = 'press',
                 onClick = None,
                 onUnfocus = None,
                 active: bool = True):
        
        super().__init__(tag = 'button')
        
        self.rect = rect
        self.border_radii = border_radii
        self.width = width
        self.default_color = default_color
        self.hover_color = hover_color
        self.focus_color = focus_color
        
        self.text = text
        self.text_color = text_color
        self.font = font
        if self.font is None:
            self.font = pygame.font.Font(pkg_resources.resource_filename('myappkit', 'Courier Prime Code.ttf')
                                         , text_size)
        
        self.hovered = False
        self.focused = False
        self.click_action = click_action # 'press' or 'focus'
        self.onClick = onClick
        self.onUnfocus = onUnfocus
        self.active = active
        
        
    def render(self, window: pygame.Surface):
        if not self.active:
            return

        if self.focused:
            color = self.focus_color
        elif self.hovered:
            color = self.hover_color
        else:
            color = self.default_color
            
        pygame.draw.rect(window, color, self.rect, width=self.width,
                         border_top_left_radius=self.border_radii[0],
                         border_top_right_radius=self.border_radii[1],
                         border_bottom_right_radius=self.border_radii[2],
                         border_bottom_left_radius=self.border_radii[3])
        
        text_surface = self.font.render(self.text, False, self.text_color)
        text_rect = text_surface.get_rect()
        text_x = self.rect[0] + 0.5*self.rect[2] - 0.5*text_rect.w
        text_y = self.rect[1] + 0.5*self.rect[3] - 0.5*text_rect.h
        window.blit(text_surface, (text_x, text_y))

    def handleEvent(self, event: pygame.event.Event):
        if not self.active:
            return

        if event.type == pygame.MOUSEMOTION:
            x, y = pygame.mouse.get_pos()
            if coords_within_rect((x, y), self.rect):
                self.hovered = True
            else:
                self.hovered = False
                
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                if self.click_action == 'focus':
                    self.focused = not self.focused
                    if self.focused:
                        if self.onClick is not None:
                            self.onClick()
                    elif self.onUnfocus is not None:
                        self.onUnfocus()
                        
                elif self.click_action == 'press':
                    self.focused = True
                
        elif event.type == pygame.MOUSEBUTTONUP and self.click_action == 'press':
            if self.focused and self.hovered and self.onClick is not None:
                    self.onClick()

            self.focused = False

class TextLabel(Item):
    def __init__(self,
                 text: str = 'TextLabel',
                 color: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0),
                 size: int = 20,
                 coords: tuple[int, int] = (0, 0),
                 font: pygame.font.Font = None):

        super().__init__(tag = 'textlabel')

        self.text = text
        self.coords = coords
        self.color = pygame.Color(color)
        self.font = font
        if self.font is None:
            self.font = pygame.font.Font(pkg_resources.resource_filename('myappkit', 'Courier Prime Code.ttf')
                                         , size)

        self.surface = self.font.render(self.text, False, self.color)

    def render(self, window: pygame.Surface):
        window.blit(self.surface, self.coords)

class Activity:
    def __init__(self,
                 bgColor: tuple[int, int, int] | tuple[int, int, int, int] = (255, 255, 255),
                 frame_rate: int = 60):

        self.bgColor = pygame.Color(bgColor)
        self.color = pygame.Color((0, 0, 0))
        self.frame_rate = frame_rate
        self.items = []

    def setFrameRate(self, frame_rate: int):
        self.frame_rate = frame_rate

    def handleEvent(self, event: pygame.event.Event):
        for item in self.items:
            item.handleEvent(event)

    def update(self):
        for item in self.items:
            item.update()

    def render(self, window: pygame.Surface):
        window.fill(self.bgColor)

        for item in self.items:
            item.render(window)

        pygame.display.update()

    def addItem(self, item: Item):
        self.items.append(item)

class App:
    def __init__(self,
                 w: int = 600,
                 h: int = 400,
                 title: str = 'app'):
        
        if not pygame.get_init():
            pygame.init()
            
        self.window = pygame.display.set_mode((w, h))
        self.activities = {'home': Activity()} # default activity
        self.current_activity = 'home'
        self.FPS = pygame.time.Clock()
        self.setTitle(title)
        self.running = False

    def addActivity(self, name: str, activity: Activity):
        self.activities[name] = activity

    def resize(self, w: int, h: int):
        self.window = pygame.display.set_mode((w, h))

    def setCurrentActivity(self, activity_name: str):
        self.current_activity = activity_name

    def handleEvent(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            else:
                self.activities[self.current_activity].handleEvent(event)

    def update(self):
        self.activities[self.current_activity].update()

    def render(self):
        self.activities[self.current_activity].render(self.window)

    def run(self):
        self.running = True
        while self.running:
            self.handleEvent()
            self.update()
            self.render()

            self.FPS.tick(self.activities[self.current_activity].frame_rate)

        pygame.quit()
    
    def setTitle(self, title: str):
        pygame.display.set_caption(title)

    def setColor(self, color: tuple[int, int, int] | tuple[int, int, int, int]):
        self.color = pygame.Color(color)
