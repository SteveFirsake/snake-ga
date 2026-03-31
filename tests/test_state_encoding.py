from __future__ import annotations

import numpy as np

from snake_ga.domain.models import GameSnapshot
from snake_ga.domain.state_encoding import compute_reward, encode_state


def _minimal_snapshot(**overrides) -> GameSnapshot:
    base = dict(
        game_width=440,
        game_height=440,
        player_x=200.0,
        player_y=220.0,
        position=[[180.0, 220.0], [200.0, 220.0]],
        x_change=20,
        y_change=0,
        food_x=300,
        food_y=220,
        eaten=False,
        crash=False,
        score=0,
    )
    base.update(overrides)
    return GameSnapshot(**base)


def test_compute_reward_crash() -> None:
    assert compute_reward(True, True) == -10
    assert compute_reward(True, False) == -10


def test_compute_reward_eat() -> None:
    assert compute_reward(False, True) == 10


def test_compute_reward_neutral() -> None:
    assert compute_reward(False, False) == 0


def test_encode_state_shape_and_binary() -> None:
    s = _minimal_snapshot()
    v = encode_state(s)
    assert v.dtype == np.float32
    assert v.shape == (11,)
    assert np.all((v == 0) | (v == 1))


def test_encode_state_food_direction_bits() -> None:
    s = _minimal_snapshot(player_x=200.0, food_x=100, food_y=220)
    v = encode_state(s)
    assert v[7] == 1.0
    assert v[8] == 0.0
