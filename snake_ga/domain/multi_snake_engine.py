"""Two-snake game on one board (tile grid + shared food)."""

from __future__ import annotations

from dataclasses import dataclass
from random import randint
from typing import Literal

import numpy as np

from snake_ga.domain.game_engine import _straight_left_right_velocities
from snake_ga.domain.models import GameSnapshot
from snake_ga.domain.tile_grid import TileGrid, TileKind, tile_kind_index


def _next_velocity_from_move(move: np.ndarray, xc: int, yc: int) -> tuple[int, int]:
    move_array = [xc, yc]
    if np.array_equal(move, [1, 0, 0]):
        move_array = [xc, yc]
    elif np.array_equal(move, [0, 1, 0]) and yc == 0:
        move_array = [0, xc]
    elif np.array_equal(move, [0, 1, 0]) and xc == 0:
        move_array = [-yc, 0]
    elif np.array_equal(move, [0, 0, 1]) and yc == 0:
        move_array = [0, -xc]
    elif np.array_equal(move, [0, 0, 1]) and xc == 0:
        move_array = [yc, 0]
    return int(move_array[0]), int(move_array[1])


CollisionMode = Literal["head_to_head_both_die", "head_to_head_second_loses"]


@dataclass
class SnakeState:
    x: float
    y: float
    position: list[list[float]]
    x_change: int
    y_change: int
    snake_segments: int
    eaten: bool
    crash: bool
    score: int


class MultiSnakeGameEngine:
    """Two snakes on the same 20×20 tile map; one food; round-robin moves per tick."""

    def __init__(
        self,
        game_width: int = 440,
        game_height: int = 440,
        tile_grid: TileGrid | None = None,
        collision_mode: CollisionMode = "head_to_head_both_die",
    ):
        self.game_width = game_width
        self.game_height = game_height
        self.tile_grid = tile_grid or TileGrid.all_normal(20, 20)
        if self.tile_grid.rows != 20 or self.tile_grid.cols != 20:
            raise ValueError("Tile grid must be 20×20 cells for the default board.")
        self.collision_mode = collision_mode
        self.snakes: list[SnakeState] = []
        self.x_food = 240
        self.y_food = 200
        self._reset_snakes()
        self._place_food_initial()

    def _reset_snakes(self) -> None:
        self.snakes = [
            SnakeState(
                x=180.0,
                y=220.0,
                position=[[180.0, 220.0]],
                x_change=20,
                y_change=0,
                snake_segments=1,
                eaten=False,
                crash=False,
                score=0,
            ),
            SnakeState(
                x=380.0,
                y=220.0,
                position=[[380.0, 220.0]],
                x_change=-20,
                y_change=0,
                snake_segments=1,
                eaten=False,
                crash=False,
                score=0,
            ),
        ]

    def reset(self) -> None:
        self._reset_snakes()
        self._place_food_initial()

    def _place_food_initial(self) -> None:
        self.x_food = 240
        self.y_food = 200
        self._place_food()

    def _place_food(self) -> None:
        while True:
            x_rand = randint(20, self.game_width - 40)
            self.x_food = x_rand - x_rand % 20
            y_rand = randint(20, self.game_height - 40)
            self.y_food = y_rand - y_rand % 20
            occupied = [self.x_food, self.y_food]
            if any(occupied in s.position for s in self.snakes):
                continue
            # Keep regular food on neutral tiles only (no blocked/bonus/penalty overlap).
            if self.tile_grid.kind_at_pixel(self.x_food, self.y_food) != TileKind.NORMAL:
                continue
            return

    def _lookahead_tile_indices(self, s: SnakeState) -> tuple[int, int, int, int]:
        px, py = s.x, s.y
        straight, left, right = _straight_left_right_velocities(s.x_change, s.y_change)
        u = tile_kind_index(self.tile_grid.kind_at_pixel(px, py))
        s1 = tile_kind_index(self.tile_grid.kind_at_pixel(px + straight[0], py + straight[1]))
        left_idx = tile_kind_index(self.tile_grid.kind_at_pixel(px + left[0], py + left[1]))
        r = tile_kind_index(self.tile_grid.kind_at_pixel(px + right[0], py + right[1]))
        return u, s1, left_idx, r

    def snapshot(self, snake_id: int) -> GameSnapshot:
        s = self.snakes[snake_id]
        other = self.snakes[1 - snake_id]
        obstacle_positions = tuple((float(p[0]), float(p[1])) for p in other.position)
        u, s1, left_idx, r = self._lookahead_tile_indices(s)
        return GameSnapshot(
            game_width=self.game_width,
            game_height=self.game_height,
            player_x=s.x,
            player_y=s.y,
            position=[list(p) for p in s.position],
            x_change=s.x_change,
            y_change=s.y_change,
            food_x=self.x_food,
            food_y=self.y_food,
            eaten=s.eaten,
            crash=s.crash,
            score=s.score,
            tile_under_head=u,
            tile_straight=s1,
            tile_left=left_idx,
            tile_right=r,
            obstacle_positions=obstacle_positions,
        )

    def _grow_if_eaten(self, s: SnakeState) -> None:
        if s.eaten:
            s.position.append([s.x, s.y])
            s.eaten = False
            s.snake_segments += 1

    def _update_position(self, s: SnakeState) -> None:
        x, y = s.x, s.y
        if s.position[-1][0] != x or s.position[-1][1] != y:
            if s.snake_segments > 1:
                for i in range(0, s.snake_segments - 1):
                    s.position[i][0], s.position[i][1] = (
                        s.position[i + 1][0],
                        s.position[i + 1][1],
                    )
            s.position[-1][0] = x
            s.position[-1][1] = y

    def _wrap(self, x: float, y: float) -> tuple[float, float]:
        nx, ny = x, y
        if nx < 20:
            nx = self.game_width - 40
        elif nx > self.game_width - 40:
            nx = 20
        if ny < 20:
            ny = self.game_height - 40
        elif ny > self.game_height - 40:
            ny = 20
        return nx, ny

    def _apply_food_steal_on_contact(self, s0: SnakeState, s1: SnakeState) -> None:
        """On snake-vs-snake contact: steal 1 from higher score to lower score."""
        if s0.score == s1.score:
            return
        if s0.score > s1.score:
            s0.score = max(0, s0.score - 1)
            s1.score += 1
            return
        s1.score = max(0, s1.score - 1)
        s0.score += 1

    def apply_tick(self, moves: tuple[np.ndarray, np.ndarray]) -> None:
        """Apply one tick: both snakes choose moves; resolve collisions then update."""
        s0, s1 = self.snakes

        if not s0.crash:
            self._grow_if_eaten(s0)
        if not s1.crash:
            self._grow_if_eaten(s1)

        crash0 = bool(s0.crash)
        crash1 = bool(s1.crash)
        inter_hit0 = False
        inter_hit1 = False
        hx0 = hy0 = hx1 = hy1 = 0.0

        if not s0.crash:
            nx0 = _next_velocity_from_move(moves[0], s0.x_change, s0.y_change)
            s0.x_change, s0.y_change = nx0
            hx0, hy0 = s0.x + s0.x_change, s0.y + s0.y_change
        if not s1.crash:
            nx1 = _next_velocity_from_move(moves[1], s1.x_change, s1.y_change)
            s1.x_change, s1.y_change = nx1
            hx1, hy1 = s1.x + s1.x_change, s1.y + s1.y_change

        blocked0 = False
        blocked1 = False
        if not s0.crash and not s1.crash:
            if [hx0, hy0] == [hx1, hy1]:
                inter_hit0 = True
                inter_hit1 = True
            else:
                hx0, hy0 = self._wrap(hx0, hy0)
                hx1, hy1 = self._wrap(hx1, hy1)
                if not crash0 and self.tile_grid.is_blocked_pixel(hx0, hy0):
                    hx0, hy0 = s0.x, s0.y
                    blocked0 = True
                if not crash1 and self.tile_grid.is_blocked_pixel(hx1, hy1):
                    hx1, hy1 = s1.x, s1.y
                    blocked1 = True
                if not crash0 and not blocked0 and (
                    [hx0, hy0] in s0.position or [hx0, hy0] in s1.position
                ):
                    if [hx0, hy0] in s1.position:
                        inter_hit0 = True
                    else:
                        crash0 = True
                if not crash1 and not blocked1 and (
                    [hx1, hy1] in s1.position or [hx1, hy1] in s0.position
                ):
                    if [hx1, hy1] in s0.position:
                        inter_hit1 = True
                    else:
                        crash1 = True

        elif not s0.crash:
            hx0, hy0 = self._wrap(hx0, hy0)
            if self.tile_grid.is_blocked_pixel(hx0, hy0):
                hx0, hy0 = s0.x, s0.y
                blocked0 = True
            elif not blocked0 and ([hx0, hy0] in s0.position or [hx0, hy0] in s1.position):
                if [hx0, hy0] in s1.position:
                    inter_hit0 = True
                else:
                    crash0 = True
        elif not s1.crash:
            hx1, hy1 = self._wrap(hx1, hy1)
            if self.tile_grid.is_blocked_pixel(hx1, hy1):
                hx1, hy1 = s1.x, s1.y
                blocked1 = True
            elif not blocked1 and ([hx1, hy1] in s1.position or [hx1, hy1] in s0.position):
                if [hx1, hy1] in s0.position:
                    inter_hit1 = True
                else:
                    crash1 = True

        if inter_hit0 or inter_hit1:
            self._apply_food_steal_on_contact(s0, s1)

        s0.crash = crash0
        s1.crash = crash1

        if not s0.crash and not inter_hit0:
            s0.x, s0.y = hx0, hy0
        if not s1.crash and not inter_hit1:
            s1.x, s1.y = hx1, hy1

        for s in (s0, s1):
            if not s.crash:
                s.score += self.tile_grid.score_delta_on_enter(s.x, s.y)
                if s.score < 0:
                    s.score = 0

        self._eat_for_snake(s0)
        self._eat_for_snake(s1)

        if not s0.crash:
            self._update_position(s0)
        if not s1.crash:
            self._update_position(s1)

    def _eat_for_snake(self, s: SnakeState) -> None:
        if s.crash:
            return
        if s.x == self.x_food and s.y == self.y_food:
            self._place_food()
            s.eaten = True
            s.score += 1

    @property
    def any_alive(self) -> bool:
        return not self.snakes[0].crash or not self.snakes[1].crash

    @property
    def both_dead(self) -> bool:
        return self.snakes[0].crash and self.snakes[1].crash
