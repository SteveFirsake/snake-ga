from __future__ import annotations

from unittest.mock import patch

import numpy as np

from snake_ga.domain.game_engine import SnakeGameEngine
from snake_ga.domain.tile_grid import TileGrid


def test_initial_head_on_grid(engine: SnakeGameEngine) -> None:
    assert engine.x == 180.0
    assert engine.y == 220.0
    assert engine.position == [[180.0, 220.0]]
    assert engine.snake_segments == 1
    assert engine.crash is False
    assert engine.score == 0


def test_straight_move_advances_head(engine: SnakeGameEngine, straight: np.ndarray) -> None:
    engine.apply_move(straight)
    assert engine.x == 200.0
    assert engine.y == 220.0
    assert engine.crash is False


def test_reset_restores_start_state(engine: SnakeGameEngine, straight: np.ndarray) -> None:
    engine.apply_move(straight)
    engine.reset()
    assert engine.x == 180.0
    assert engine.y == 220.0
    assert engine.position == [[180.0, 220.0]]
    assert engine.snake_segments == 1


def test_right_turn_from_horizontal(engine: SnakeGameEngine) -> None:
    engine.apply_move(np.array([1.0, 0.0, 0.0]))
    assert engine.x_change == 20 and engine.y_change == 0
    engine.apply_move(np.array([0.0, 1.0, 0.0]))
    assert engine.x_change == 0 and engine.y_change == 20


def test_wall_crash_left() -> None:
    e = SnakeGameEngine(100, 100)
    e.x = 40.0
    e.y = 40.0
    e.position = [[40.0, 40.0]]
    e.x_change = -20
    e.y_change = 0
    e.apply_move(np.array([1.0, 0.0, 0.0]))
    assert e.crash is False
    e.apply_move(np.array([1.0, 0.0, 0.0]))
    assert e.crash is True


def test_eating_increments_score_and_sets_eaten(
    engine: SnakeGameEngine, straight: np.ndarray
) -> None:
    engine.x_food = 200
    engine.y_food = 220
    with patch("snake_ga.domain.game_engine.randint", return_value=100):
        engine.apply_move(straight)
    assert engine.score == 1
    assert engine.eaten is True


def test_snapshot_is_detached_copy(engine: SnakeGameEngine) -> None:
    snap = engine.snapshot()
    engine.x = 999.0
    assert snap.player_x == 180.0


def test_snapshot_lookahead_sees_bonus_tile_straight_ahead() -> None:
    rows = ["." * 20 for _ in range(20)]
    r = list(rows[10])
    r[9] = "+"
    rows[10] = "".join(r)
    grid = TileGrid(rows)
    eng = SnakeGameEngine(440, 440, tile_grid=grid)
    snap = eng.snapshot()
    assert snap.tile_straight == 1


def test_encode_state_after_snapshot(engine: SnakeGameEngine, straight: np.ndarray) -> None:
    from snake_ga.domain.state_encoding import STATE_VECTOR_SIZE, encode_state

    engine.apply_move(straight)
    vec = encode_state(engine.snapshot())
    assert vec.shape == (STATE_VECTOR_SIZE,)
    assert set(float(v) for v in vec).issubset({0.0, 1.0})
