from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameSnapshot:
    """Immutable view of game state for policy and logging."""

    game_width: int
    game_height: int
    player_x: float
    player_y: float
    position: list[list[float]]
    x_change: int
    y_change: int
    food_x: int
    food_y: int
    eaten: bool
    crash: bool
    score: int
