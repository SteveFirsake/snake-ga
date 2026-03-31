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


def test_inter_snake_collision_steals_from_higher_score_and_no_death() -> None:
    eng = MultiSnakeGameEngine(440, 440, collision_mode="head_to_head_both_die")
    eng.reset()
    s0, s1 = eng.snakes
    s0.position = [[200.0, 220.0]]
    s0.x = 200.0
    s0.y = 220.0
    s0.x_change = 20
    s0.y_change = 0
    s0.score = 3

    s1.position = [[240.0, 220.0]]
    s1.x = 240.0
    s1.y = 220.0
    s1.x_change = -20
    s1.y_change = 0
    s1.score = 1

    # Both heads move to (220, 220): inter-snake collision.
    eng.apply_tick((np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])))
    assert s0.score == 2
    assert s1.score == 2
    assert s0.crash is False
    assert s1.crash is False


def test_multi_inner_wall_blocks_without_crash() -> None:
    rows = ["." * 20 for _ in range(20)]
    r = list(rows[10])
    r[10] = "#"
    rows[10] = "".join(r)
    eng = MultiSnakeGameEngine(440, 440, tile_grid=TileGrid(rows))
    eng.reset()
    s0 = eng.snakes[0]
    # Put snake0 at (200,220) moving right into blocked (220,220).
    s0.x = 200.0
    s0.y = 220.0
    s0.position = [[200.0, 220.0]]
    s0.x_change = 20
    s0.y_change = 0
    eng.apply_tick((np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])))
    assert s0.crash is False
    assert s0.x == 200.0 and s0.y == 220.0


def test_multi_outer_wall_wraps_without_crash() -> None:
    eng = MultiSnakeGameEngine(100, 100)
    eng.reset()
    s0 = eng.snakes[0]
    s0.x = 20.0
    s0.y = 40.0
    s0.position = [[20.0, 40.0]]
    s0.x_change = -20
    s0.y_change = 0
    eng.apply_tick((np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])))
    assert s0.crash is False
    assert s0.x == 60.0 and s0.y == 40.0
