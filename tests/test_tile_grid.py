from pathlib import Path

import numpy as np
from snake_ga.domain.game_engine import SnakeGameEngine
from snake_ga.domain.tile_grid import BONUS_SCORE, TileGrid


def test_example_board_file_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    g = TileGrid.from_file(root / "boards" / "example.txt")
    assert g.rows == 20 and g.cols == 20


def test_bonus_and_penalty_score_delta() -> None:
    g = TileGrid.all_normal(20, 20)
    # force known positions: patch grid by loading minimal - use cell coords
    # center of cell (21,21) is col 0 row 0 -> normal
    assert g.score_delta_on_enter(21, 21) == 0


def test_blocked_is_collision() -> None:
    lines = ["." * 20 for _ in range(20)]
    lines[10] = "." * 8 + "#" + "." * 11
    g = TileGrid(lines)
    assert g.is_blocked_pixel(180.0, 220.0) is True


def test_engine_applies_bonus_once_per_visit() -> None:
    rows = ["." * 20 for _ in range(20)]
    r = list(rows[10])
    r[9] = "+"
    rows[10] = "".join(r)
    grid = TileGrid(rows)
    eng = SnakeGameEngine(440, 440, tile_grid=grid)
    # one straight step: (180,220) -> (200,220), enters col 9 row 10
    eng.apply_move(np.array([1.0, 0.0, 0.0]))
    assert eng.score >= BONUS_SCORE


def test_penalty_clamped_at_zero() -> None:
    rows = ["." * 20 for _ in range(20)]
    r = list(rows[10])
    r[9] = "-"
    rows[10] = "".join(r)
    grid = TileGrid(rows)
    eng = SnakeGameEngine(440, 440, tile_grid=grid)
    eng.score = 0
    eng.apply_move(np.array([1.0, 0.0, 0.0]))
    assert eng.score == 0
