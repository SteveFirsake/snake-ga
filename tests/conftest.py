from __future__ import annotations

import numpy as np
import pytest

from snake_ga.domain.game_engine import SnakeGameEngine


@pytest.fixture
def engine() -> SnakeGameEngine:
    return SnakeGameEngine(440, 440)


@pytest.fixture
def straight() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)
