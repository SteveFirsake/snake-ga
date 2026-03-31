from __future__ import annotations

from random import randint

import numpy as np

from snake_ga.domain.models import GameSnapshot
from snake_ga.domain.tile_grid import TileGrid, tile_kind_index


def _straight_left_right_velocities(
    x_change: int, y_change: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Next step delta for straight, relative-left, relative-right (matches apply_move)."""
    straight = (x_change, y_change)
    if y_change == 0:
        left = (0, x_change)
        right = (0, -x_change)
    else:
        left = (-y_change, 0)
        right = (y_change, 0)
    return straight, left, right


class SnakeGameEngine:
    """Pure game rules (no pygame). Mirrors original Player / Game / Food behavior."""

    def __init__(
        self,
        game_width: int = 440,
        game_height: int = 440,
        tile_grid: TileGrid | None = None,
    ):
        self.game_width = game_width
        self.game_height = game_height
        # Inner grid: (game_width - 2*margin) / cell — default 20×20 cells @ 20px, margin 20
        self.tile_grid = tile_grid or TileGrid.all_normal(20, 20)
        if self.tile_grid.rows != 20 or self.tile_grid.cols != 20:
            raise ValueError(
                "Tile grid must be 20×20 cells for the default 440×440 board (20px cells, 20px margin)."
            )
        self.crash = False
        self.score = 0
        x = 0.45 * self.game_width
        y = 0.5 * self.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position: list[list[float]] = [[self.x, self.y]]
        self.snake_segments = 1
        self.eaten = False
        self.x_change = 20
        self.y_change = 0
        self.x_food = 240
        self.y_food = 200

    def reset(self) -> None:
        self.crash = False
        self.score = 0
        x = 0.45 * self.game_width
        y = 0.5 * self.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = [[self.x, self.y]]
        self.snake_segments = 1
        self.eaten = False
        self.x_change = 20
        self.y_change = 0
        self.x_food = 240
        self.y_food = 200

    def _lookahead_tile_indices(self) -> tuple[int, int, int, int]:
        px, py = self.x, self.y
        straight, left, right = _straight_left_right_velocities(self.x_change, self.y_change)
        u = tile_kind_index(self.tile_grid.kind_at_pixel(px, py))
        s = tile_kind_index(self.tile_grid.kind_at_pixel(px + straight[0], py + straight[1]))
        left_idx = tile_kind_index(self.tile_grid.kind_at_pixel(px + left[0], py + left[1]))
        r = tile_kind_index(self.tile_grid.kind_at_pixel(px + right[0], py + right[1]))
        return u, s, left_idx, r

    def snapshot(self) -> GameSnapshot:
        u, s, left_idx, r = self._lookahead_tile_indices()
        return GameSnapshot(
            game_width=self.game_width,
            game_height=self.game_height,
            player_x=self.x,
            player_y=self.y,
            position=[list(p) for p in self.position],
            x_change=self.x_change,
            y_change=self.y_change,
            food_x=self.x_food,
            food_y=self.y_food,
            eaten=self.eaten,
            crash=self.crash,
            score=self.score,
            tile_under_head=u,
            tile_straight=s,
            tile_left=left_idx,
            tile_right=r,
            obstacle_positions=(),
        )

    def _place_food(self) -> None:
        while True:
            x_rand = randint(20, self.game_width - 40)
            self.x_food = x_rand - x_rand % 20
            y_rand = randint(20, self.game_height - 40)
            self.y_food = y_rand - y_rand % 20
            if [self.x_food, self.y_food] in self.position:
                continue
            if self.tile_grid.is_blocked_pixel(self.x_food, self.y_food):
                continue
            return

    def _eat(self) -> None:
        if self.x == self.x_food and self.y == self.y_food:
            self._place_food()
            self.eaten = True
            self.score += 1

    def _update_position(self) -> None:
        x, y = self.x, self.y
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.snake_segments > 1:
                for i in range(0, self.snake_segments - 1):
                    self.position[i][0], self.position[i][1] = (
                        self.position[i + 1][0],
                        self.position[i + 1][1],
                    )
            self.position[-1][0] = x
            self.position[-1][1] = y

    def apply_move(self, move: np.ndarray) -> None:
        x, y = self.x, self.y
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.snake_segments += 1

        if np.array_equal(move, [1, 0, 0]):
            move_array = [self.x_change, self.y_change]
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:
            move_array = [self.y_change, 0]

        self.x_change, self.y_change = move_array
        next_x = x + self.x_change
        next_y = y + self.y_change
        if next_x < 20:
            next_x = self.game_width - 40
        elif next_x > self.game_width - 40:
            next_x = 20
        if next_y < 20:
            next_y = self.game_height - 40
        elif next_y > self.game_height - 40:
            next_y = 20
        blocked_by_inner_wall = self.tile_grid.is_blocked_pixel(next_x, next_y)
        if blocked_by_inner_wall:
            # Inner walls are barriers: movement is blocked, but snake does not die.
            self.x, self.y = x, y
        else:
            self.x, self.y = next_x, next_y

        moved = (self.x != x) or (self.y != y)
        self_body_collision = moved and ([self.x, self.y] in self.position)
        if (
            self_body_collision
        ):
            self.crash = True

        if not self.crash and moved:
            self.score += self.tile_grid.score_delta_on_enter(self.x, self.y)
            if self.score < 0:
                self.score = 0

        self._eat()
        self._update_position()
