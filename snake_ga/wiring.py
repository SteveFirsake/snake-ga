"""Composition root: bind ports to concrete adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from snake_ga.adapters.plotting import SeabornScorePlotter
from snake_ga.adapters.pygame_session import PygameSession
from snake_ga.application.ports import PolicyLearnerPort
from snake_ga.domain.game_engine import SnakeGameEngine
from snake_ga.domain.tile_grid import TileGrid
from snake_ga.policy_registry import build_policy

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _tile_grid_from_params(params: dict[str, Any]) -> TileGrid:
    p = params.get("board_path")
    if not p:
        return TileGrid.all_normal(20, 20)
    path = Path(p)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return TileGrid.from_file(path)


def build_session_agent_plotter(
    params: dict[str, Any],
) -> tuple[PygameSession, PolicyLearnerPort, SeabornScorePlotter | None]:
    engine = SnakeGameEngine(440, 440, tile_grid=_tile_grid_from_params(params))
    session = PygameSession(engine, bool(params["display"]))
    agent = build_policy(params)
    plotter = SeabornScorePlotter() if params.get("plot_score") else None
    return session, agent, plotter
