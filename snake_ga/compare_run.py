"""Run N policies with identical env settings and print scores (sequential, single-snake games)."""

from __future__ import annotations

from typing import Any

from snake_ga.application.run_loop import run_training_or_test
from snake_ga.policy_registry import LEARNED_POLICIES
from snake_ga.wiring import build_session_agent_plotter


def compare_policies(params: dict[str, Any], policies: list[str]) -> None:
    if len(policies) < 2:
        raise ValueError("compare_policies requires at least two policy names")
    base = dict(params)
    base["train"] = False
    base["test"] = True
    base["plot_score"] = False

    rows: list[tuple[str, float, float, float]] = []
    for name in policies:
        p = dict(base)
        p["policy"] = name
        p["load_weights"] = name in LEARNED_POLICIES
        print(f"\n=== Policy: {name}  (episodes={p['episodes']}) ===")
        session, agent, plotter = build_session_agent_plotter(p)
        total, mean, stdev = run_training_or_test(p, session, agent, plotter)
        rows.append((name, total, mean, stdev))

    print("\n--- Comparison (same board, speed, episode count) ---")
    w = 12
    print(f"  {'policy':{w}}  {'total':>10}  {'mean':>8}  {'stdev':>8}")
    for name, total, mean, stdev in rows:
        print(f"  {name:{w}}  {total:10.1f}  {mean:8.2f}  {stdev:8.2f}")
