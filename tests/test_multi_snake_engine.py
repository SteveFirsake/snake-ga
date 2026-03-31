import numpy as np

from snake_ga.domain.multi_snake_engine import MultiSnakeGameEngine
from snake_ga.domain.tile_grid import TileGrid


def test_multi_snake_reset_and_snapshot_obstacles() -> None:
    g = TileGrid.all_normal(20, 20)
    eng = MultiSnakeGameEngine(440, 440, tile_grid=g)
    eng.reset()
    s0 = eng.snapshot(0)
    assert len(s0.obstacle_positions) == 1
    assert (380.0, 220.0) in s0.obstacle_positions


def test_multi_snake_tick_runs() -> None:
    eng = MultiSnakeGameEngine(440, 440)
    eng.reset()
    m0 = np.array([1.0, 0.0, 0.0])
    m1 = np.array([1.0, 0.0, 0.0])
    eng.apply_tick((m0, m1))
    assert eng.snakes[0].x != 180.0 or eng.snakes[1].x != 380.0
