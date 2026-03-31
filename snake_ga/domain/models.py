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
    # Map tile kinds as 0..3 indices (see tile_grid.tile_kind_index); default all normal
    tile_under_head: int = 0
    tile_straight: int = 0
    tile_left: int = 0
    tile_right: int = 0
