"""Composition root: bind ports to concrete adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from snake_ga.adapters.plotting import SeabornScorePlotter
from snake_ga.adapters.pygame_session import PygameSession
from snake_ga.application.ports import PolicyLearnerPort
from snake_ga.domain.game_engine import SnakeGameEngine
from snake_ga.domain.multi_snake_engine import MultiSnakeGameEngine
from snake_ga.domain.tile_grid import TileGrid
from snake_ga.policy_registry import LEARNED_POLICIES, build_policy

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


def build_multi_session_agents_plotter(
    params: dict[str, Any],
) -> tuple[PygameSession, tuple[PolicyLearnerPort, PolicyLearnerPort], SeabornScorePlotter | None]:
    engine = MultiSnakeGameEngine(
        440,
        440,
        tile_grid=_tile_grid_from_params(params),
        collision_mode=params.get("collision_mode", "head_to_head_both_die"),
    )
    session = PygameSession(engine, bool(params["display"]))
    n0 = params.get("policy_0", "heuristic")
    n1 = params.get("policy_1", "random")
    p0 = dict(params)
    p0["policy"] = n0
    p0["load_weights"] = n0 in LEARNED_POLICIES
    p1 = dict(params)
    p1["policy"] = n1
    p1["load_weights"] = n1 in LEARNED_POLICIES
    a0 = build_policy(p0)
    a1 = build_policy(p1)
    plotter = SeabornScorePlotter() if params.get("plot_score") else None
    return session, (a0, a1), plotter
