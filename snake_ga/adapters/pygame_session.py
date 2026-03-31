from __future__ import annotations

from pathlib import Path

import pygame

from snake_ga.domain.game_engine import SnakeGameEngine


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class PygameSession:
    """Pygame adapter: renders domain `SnakeGameEngine` state."""

    def __init__(self, engine: SnakeGameEngine, display: bool):
        self.engine = engine
        self.display = display
        self.game_width = engine.game_width
        self.game_height = engine.game_height
        self.gameDisplay: pygame.Surface | None = None
        self.bg: pygame.Surface | None = None
        self.snake_img: pygame.Surface | None = None
        self.food_img: pygame.Surface | None = None

    def init_pygame(self) -> None:
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("SnakeGen")
        root = _project_root()
        self.gameDisplay = pygame.display.set_mode((self.game_width, self.game_height + 60))
        self.bg = pygame.image.load(str(root / "img" / "background.png"))
        self.snake_img = pygame.image.load(str(root / "img" / "snakeBody.png"))
        self.food_img = pygame.image.load(str(root / "img" / "food2.png"))

    def pump_quit_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def wait(self, ms: int) -> None:
        pygame.time.wait(ms)

    def _display_ui(self, score: int, record: int) -> None:
        assert self.gameDisplay is not None and self.bg is not None
        myfont = pygame.font.SysFont("Segoe UI", 20)
        myfont_bold = pygame.font.SysFont("Segoe UI", 20, True)
        text_score = myfont.render("SCORE: ", True, (0, 0, 0))
        text_score_number = myfont.render(str(score), True, (0, 0, 0))
        text_highest = myfont.render("HIGHEST SCORE: ", True, (0, 0, 0))
        text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
        self.gameDisplay.blit(text_score, (45, 440))
        self.gameDisplay.blit(text_score_number, (120, 440))
        self.gameDisplay.blit(text_highest, (190, 440))
        self.gameDisplay.blit(text_highest_number, (350, 440))
        self.gameDisplay.blit(self.bg, (10, 10))

    def render(self, record: int) -> None:
        if not self.display:
            return
        assert self.gameDisplay is not None
        assert self.snake_img is not None and self.food_img is not None
        p = self.engine
        self.gameDisplay.fill((255, 255, 255))
        self._display_ui(p.score, record)
        if not p.crash:
            for i in range(p.snake_segments):
                x_temp, y_temp = p.position[len(p.position) - 1 - i]
                self.gameDisplay.blit(self.snake_img, (x_temp, y_temp))
            pygame.display.update()
        else:
            pygame.time.wait(300)
        self.gameDisplay.blit(self.food_img, (p.x_food, p.y_food))
        pygame.display.update()
