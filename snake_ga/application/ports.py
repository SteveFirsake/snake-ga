from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from snake_ga.domain.models import GameSnapshot


class GameEnginePort(Protocol):
    """Inbound port: game dynamics used by the training loop."""

    crash: bool
    score: int
    eaten: bool

    def reset(self) -> None: ...

    def snapshot(self) -> GameSnapshot: ...

    def apply_move(self, move: np.ndarray) -> None: ...


class PolicyLearnerPort(Protocol):
    """Inbound port: differentiable policy + replay memory."""

    epsilon: float
    memory: Any
    optimizer: Any

    def __call__(self, x: Any) -> Any: ...

    def train_short_memory(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None: ...

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None: ...

    def replay_new(self, memory: Any, batch_size: int) -> None: ...

    def forward(self, x: Any) -> Any: ...

    def parameters(self) -> Any: ...

    def state_dict(self) -> Any: ...

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any: ...

    def train(self, mode: bool = True) -> None: ...

    def to(self, device: Any) -> Any: ...


class ScorePlotterPort(Protocol):
    """Outbound port: optional training curves."""

    def plot(self, games: list[int], scores: list[float], train: bool) -> None: ...


class GameDisplayPort(Protocol):
    """Outbound port: pygame (or headless) session bound to a game engine."""

    engine: GameEnginePort

    def init_pygame(self) -> None: ...

    def pump_quit_events(self) -> None: ...

    def render(self, record: int) -> None: ...

    def wait(self, ms: int) -> None: ...
