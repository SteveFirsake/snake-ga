from __future__ import annotations

from random import randint

import numpy as np

from snake_ga.domain.models import GameSnapshot


class SnakeGameEngine:
    """Pure game rules (no pygame). Mirrors original Player / Game / Food behavior."""

    def __init__(self, game_width: int = 440, game_height: int = 440):
        self.game_width = game_width
        self.game_height = game_height
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

    def snapshot(self) -> GameSnapshot:
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
        )

    def _place_food(self) -> None:
        while True:
            x_rand = randint(20, self.game_width - 40)
            self.x_food = x_rand - x_rand % 20
            y_rand = randint(20, self.game_height - 40)
            self.y_food = y_rand - y_rand % 20
            if [self.x_food, self.y_food] not in self.position:
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
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:
            move_array = [self.y_change, 0]

        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if (
            self.x < 20
            or self.x > self.game_width - 40
            or self.y < 20
            or self.y > self.game_height - 40
            or [self.x, self.y] in self.position
        ):
            self.crash = True

        self._eat()
        self._update_position()
