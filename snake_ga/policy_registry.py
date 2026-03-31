"""Policy names and factory (single place for CLI + wiring + run loop checks)."""

from __future__ import annotations

from typing import Any

from snake_ga.adapters.baseline_policies import (
    HeuristicSurvivalPolicyAgent,
    LeftPolicyAgent,
    RandomPolicyAgent,
    RightPolicyAgent,
    StraightPolicyAgent,
)
from snake_ga.adapters.torch_dqn import TorchDQNAgent
from snake_ga.application.ports import PolicyLearnerPort

# All selectable policy ids (CLI --policy)
POLICY_CHOICES: tuple[str, ...] = (
    "dqn",
    "random",
    "straight",
    "left",
    "right",
    "heuristic",
)

# Policies implemented with a learned PyTorch checkpoint
LEARNED_POLICIES: frozenset[str] = frozenset({"dqn"})


def build_policy(params: dict[str, Any]) -> PolicyLearnerPort:
    name = params.get("policy", "dqn")
    if name == "dqn":
        return TorchDQNAgent(params)
    if name == "random":
        return RandomPolicyAgent()
    if name == "straight":
        return StraightPolicyAgent()
    if name == "left":
        return LeftPolicyAgent()
    if name == "right":
        return RightPolicyAgent()
    if name == "heuristic":
        return HeuristicSurvivalPolicyAgent()
    raise ValueError(f"Unknown policy {name!r}. Expected one of {POLICY_CHOICES}.")
