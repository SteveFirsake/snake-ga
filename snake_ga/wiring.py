"""Composition root: bind ports to concrete adapters."""

from __future__ import annotations

from typing import Any

from snake_ga.adapters.plotting import SeabornScorePlotter
from snake_ga.adapters.pygame_session import PygameSession
from snake_ga.adapters.torch_dqn import TorchDQNAgent
from snake_ga.domain.game_engine import SnakeGameEngine


def build_session_agent_plotter(
    params: dict[str, Any],
) -> tuple[PygameSession, TorchDQNAgent, SeabornScorePlotter | None]:
    engine = SnakeGameEngine(440, 440)
    session = PygameSession(engine, bool(params["display"]))
    agent = TorchDQNAgent(params)
    plotter = SeabornScorePlotter() if params.get("plot_score") else None
    return session, agent, plotter
