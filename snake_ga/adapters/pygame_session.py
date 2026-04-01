from __future__ import annotations

from pathlib import Path

import pygame

from snake_ga.domain.game_engine import SnakeGameEngine
from snake_ga.domain.multi_snake_engine import MultiSnakeGameEngine
from snake_ga.domain.tile_grid import TileKind


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _head_angle_degrees(dx: int, dy: int) -> float:
    """Assume base sprite faces right; pygame rotates counter-clockwise."""
    if dx > 0:
        return 0.0
    if dx < 0:
        return 180.0
    if dy < 0:
        return 90.0
    if dy > 0:
        return -90.0
    return 0.0


def _tint_surface(src: pygame.Surface, rgb: tuple[int, int, int]) -> pygame.Surface:
    """Colorize the snake sprite while preserving alpha."""
    out = src.copy()
    tint = pygame.Surface(src.get_size(), pygame.SRCALPHA)
    tint.fill((*rgb, 255))
    out.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return out


class PygameSession:
    """Pygame adapter: renders domain `SnakeGameEngine` state."""

    _margin = 10
    _hud_y = 440
    _hud_padding_x = 24

    def __init__(self, engine: SnakeGameEngine | MultiSnakeGameEngine, display: bool):
        self.engine = engine
        self.display = display
        self.game_width = engine.game_width
        self.game_height = engine.game_height
        self.gameDisplay: pygame.Surface | None = None
        self.bg: pygame.Surface | None = None
        self.snake_img: pygame.Surface | None = None
        self.food_img: pygame.Surface | None = None
        self._tail_img: pygame.Surface | None = None
        self._snake_green: pygame.Surface | None = None
        self._snake_blue: pygame.Surface | None = None
        self._snake_red: pygame.Surface | None = None
        self._tail_green: pygame.Surface | None = None
        self._tail_blue: pygame.Surface | None = None
        self._tail_red: pygame.Surface | None = None

    def init_pygame(self) -> None:
        pygame.init()
        pygame.font.init()
        root = _project_root()
        w, h = self.game_width, self.game_height + 60
        if self.display:
            pygame.display.set_caption("SnakeGen")
            self.gameDisplay = pygame.display.set_mode((w, h))
        else:
            # No visible window: render() is a no-op, but the loop still needs a surface for assets.
            self.gameDisplay = pygame.Surface((w, h))
        self.bg = pygame.image.load(str(root / "img" / "background.png"))
        snake_raw = pygame.image.load(str(root / "img" / "snakeBody.png"))
        food_raw = pygame.image.load(str(root / "img" / "food2.png"))
        # convert_alpha() needs a video mode; headless uses an offscreen Surface only.
        self.snake_img = snake_raw.convert_alpha() if self.display else snake_raw
        self.food_img = food_raw.convert_alpha() if self.display else food_raw
        sw, sh = self.snake_img.get_size()
        self._tail_img = pygame.transform.smoothscale(
            self.snake_img, (max(1, int(sw * 0.82)), max(1, int(sh * 0.82)))
        )
        # Mono snake is green; multi snakes are blue/red.
        self._snake_green = _tint_surface(self.snake_img, (72, 186, 104))
        self._snake_blue = _tint_surface(self.snake_img, (80, 130, 255))
        self._snake_red = _tint_surface(self.snake_img, (226, 88, 88))
        assert self._tail_img is not None
        self._tail_green = _tint_surface(self._tail_img, (72, 186, 104))
        self._tail_blue = _tint_surface(self._tail_img, (80, 130, 255))
        self._tail_red = _tint_surface(self._tail_img, (226, 88, 88))

    def pump_quit_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def wait(self, ms: int) -> None:
        pygame.time.wait(ms)

    def _draw_tiles(self) -> None:
        """Tint cells by tile kind (drawn on top of the frame image)."""
        assert self.gameDisplay is not None
        tg = self.engine.tile_grid
        margin = 20
        cell = 20
        for row in range(tg.rows):
            for col in range(tg.cols):
                k = tg.kind_at_cell(row, col)
                if k == TileKind.NORMAL:
                    continue
                rect = pygame.Rect(margin + col * cell, margin + row * cell, cell, cell)
                if k == TileKind.BONUS:
                    color = (200, 245, 200)
                elif k == TileKind.PENALTY:
                    color = (255, 210, 210)
                else:
                    color = (110, 110, 118)
                pygame.draw.rect(self.gameDisplay, color, rect)

    def _draw_grid(self) -> None:
        """Light grid on the playfield (helps read motion on a plain board)."""
        assert self.gameDisplay is not None
        m = self._margin
        inner_w = self.game_width - 2 * m
        inner_h = self.game_height - 2 * m
        color = (235, 238, 242)
        for x in range(0, inner_w + 1, 20):
            pygame.draw.line(
                self.gameDisplay,
                color,
                (m + x, m),
                (m + x, m + inner_h),
                1,
            )
        for y in range(0, inner_h + 1, 20):
            pygame.draw.line(
                self.gameDisplay,
                color,
                (m, m + y),
                (m + inner_w, m + y),
                1,
            )

    def _blit_segment(
        self,
        x: float,
        y: float,
        *,
        head: bool,
        tail: bool,
        dx: int,
        dy: int,
        body_img: pygame.Surface,
        tail_img: pygame.Surface,
    ) -> None:
        assert self.gameDisplay is not None
        cx, cy = x + 10, y + 10
        if head:
            angle = _head_angle_degrees(dx, dy)
            img = pygame.transform.rotate(body_img, angle)
            rect = img.get_rect(center=(cx, cy))
            self.gameDisplay.blit(img, rect)
        elif tail:
            rect = tail_img.get_rect(center=(cx, cy))
            self.gameDisplay.blit(tail_img, rect)
        else:
            rect = body_img.get_rect(center=(cx, cy))
            self.gameDisplay.blit(body_img, rect)

    def _display_ui(self, score: int, record: int) -> None:
        assert self.gameDisplay is not None and self.bg is not None
        myfont = pygame.font.SysFont("Segoe UI", 20)
        myfont_bold = pygame.font.SysFont("Segoe UI", 20, True)

        score_label = myfont.render("SCORE: ", True, (40, 44, 52))
        score_val = myfont.render(str(score), True, (40, 44, 52))
        hi_label = myfont.render("HIGHEST SCORE: ", True, (40, 44, 52))
        hi_val = myfont_bold.render(str(record), True, (26, 30, 36))

        pad = self._hud_padding_x
        y = self._hud_y

        self.gameDisplay.blit(score_label, (pad, y))
        self.gameDisplay.blit(score_val, (pad + score_label.get_width(), y))

        hi_block = hi_label.get_width() + hi_val.get_width() + 8
        hi_x = self.game_width - pad - hi_block
        self.gameDisplay.blit(hi_label, (hi_x, y))
        self.gameDisplay.blit(hi_val, (hi_x + hi_label.get_width() + 8, y))

        self.gameDisplay.blit(self.bg, (self._margin, self._margin))

    def _display_ui_multi(self, blue_score: int, red_score: int, record: int) -> None:
        assert self.gameDisplay is not None and self.bg is not None
        myfont = pygame.font.SysFont("Segoe UI", 20)
        myfont_bold = pygame.font.SysFont("Segoe UI", 20, True)

        blue_label = myfont.render("BLUE: ", True, (40, 44, 52))
        blue_val = myfont_bold.render(str(blue_score), True, (52, 92, 220))
        red_label = myfont.render("RED: ", True, (40, 44, 52))
        red_val = myfont_bold.render(str(red_score), True, (210, 64, 64))
        hi_label = myfont.render("HIGHEST COMBINED: ", True, (40, 44, 52))
        hi_val = myfont_bold.render(str(record), True, (26, 30, 36))

        pad = self._hud_padding_x
        y_top = self._hud_y
        y_bottom = self._hud_y + 22
        gap = 18

        # Line 1: both scores (never overlaps line 2)
        self.gameDisplay.blit(blue_label, (pad, y_top))
        x1 = pad + blue_label.get_width()
        self.gameDisplay.blit(blue_val, (x1, y_top))
        x2 = x1 + blue_val.get_width() + gap
        self.gameDisplay.blit(red_label, (x2, y_top))
        x3 = x2 + red_label.get_width()
        self.gameDisplay.blit(red_val, (x3, y_top))

        # Line 2: combined high score aligned with the score row's left margin.
        hi_x = pad
        self.gameDisplay.blit(hi_label, (hi_x, y_bottom))
        self.gameDisplay.blit(hi_val, (hi_x + hi_label.get_width() + 8, y_bottom))

        self.gameDisplay.blit(self.bg, (self._margin, self._margin))

    def _render_multi(self, record: int) -> None:
        assert self.gameDisplay is not None
        assert (
            self.snake_img is not None and self.food_img is not None and self._tail_img is not None
        )
        eng = self.engine
        assert isinstance(eng, MultiSnakeGameEngine)
        self.gameDisplay.fill((252, 253, 255))
        self._display_ui_multi(eng.snakes[0].score, eng.snakes[1].score, record)
        self._draw_tiles()
        self._draw_grid()

        # Draw both snakes even after crash so they do not disappear.
        if not (eng.snakes[0].crash and eng.snakes[1].crash):
            for idx, snake in enumerate(eng.snakes):
                body_img = self._snake_blue if idx == 0 else self._snake_red
                tail_img = self._tail_blue if idx == 0 else self._tail_red
                assert body_img is not None and tail_img is not None
                n = snake.snake_segments
                for i in range(n):
                    x_temp, y_temp = snake.position[len(snake.position) - 1 - i]
                    head = i == 0
                    tail = i == n - 1 and n > 1
                    self._blit_segment(
                        x_temp,
                        y_temp,
                        head=head,
                        tail=tail,
                        dx=snake.x_change,
                        dy=snake.y_change,
                        body_img=body_img,
                        tail_img=tail_img,
                    )
            pygame.display.update()
        else:
            pygame.time.wait(300)

        self.gameDisplay.blit(self.food_img, (eng.x_food, eng.y_food))
        pygame.display.update()

    def render(self, record: int) -> None:
        if not self.display:
            return
        assert self.gameDisplay is not None
        assert (
            self.snake_img is not None and self.food_img is not None and self._tail_img is not None
        )
        if isinstance(self.engine, MultiSnakeGameEngine):
            self._render_multi(record)
            return
        p = self.engine
        self.gameDisplay.fill((252, 253, 255))
        self._display_ui(p.score, record)
        self._draw_tiles()
        self._draw_grid()

        if not p.crash:
            assert self._snake_green is not None and self._tail_green is not None
            n = p.snake_segments
            for i in range(n):
                x_temp, y_temp = p.position[len(p.position) - 1 - i]
                head = i == 0
                tail = i == n - 1 and n > 1
                self._blit_segment(
                    x_temp,
                    y_temp,
                    head=head,
                    tail=tail,
                    dx=p.x_change,
                    dy=p.y_change,
                    body_img=self._snake_green,
                    tail_img=self._tail_green,
                )
            pygame.display.update()
        else:
            pygame.time.wait(300)

        self.gameDisplay.blit(self.food_img, (p.x_food, p.y_food))
        pygame.display.update()
