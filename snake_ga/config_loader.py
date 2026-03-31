"""Load run settings from JSON (optional env SNAKE_GA_CONFIG)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from snake_ga.domain import STATE_VECTOR_SIZE
from snake_ga.policy_registry import POLICY_CHOICES


def default_config_path_from_env() -> str | None:
    p = os.environ.get("SNAKE_GA_CONFIG")
    return p if p else None


def load_config_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")
    return data


def merge_config_into_params(params: dict[str, Any], data: dict[str, Any]) -> None:
    """Apply config keys onto params (same names as CLI / define_parameters where possible)."""
    if "board" in data:
        params["board_path"] = data["board"]
    if "board_path" in data:
        params["board_path"] = data["board_path"]
    if "episodes" in data:
        params["episodes"] = int(data["episodes"])
    if "display" in data:
        params["display"] = bool(data["display"])
    if "speed" in data:
        params["speed"] = int(data["speed"])
    if "train" in data:
        params["train"] = bool(data["train"])
    if "test" in data:
        params["test"] = bool(data["test"])
    if "plot_score" in data:
        params["plot_score"] = bool(data["plot_score"])
    if "policy" in data:
        p = str(data["policy"])
        if p not in POLICY_CHOICES:
            raise ValueError(f"Unknown policy in config: {p!r}")
        params["policy"] = p
    if "policies" in data:
        pols = data["policies"]
        if not isinstance(pols, list) or not pols:
            raise ValueError("config 'policies' must be a non-empty list of policy names")
        for p in pols:
            if str(p) not in POLICY_CHOICES:
                raise ValueError(f"Unknown policy in config policies: {p!r}")
        params["policies"] = [str(x) for x in pols]
        if len(params["policies"]) >= 2:
            params["policy_0"] = params["policies"][0]
            params["policy_1"] = params["policies"][1]
        if len(params["policies"]) >= 2 and "run_mode" not in data:
            params["run_mode"] = "compare"
    if "policy_0" in data:
        p = str(data["policy_0"])
        if p not in POLICY_CHOICES:
            raise ValueError(f"Unknown policy in config: {p!r}")
        params["policy_0"] = p
    if "policy_1" in data:
        p = str(data["policy_1"])
        if p not in POLICY_CHOICES:
            raise ValueError(f"Unknown policy in config: {p!r}")
        params["policy_1"] = p
    if "run_mode" in data:
        rm = str(data["run_mode"])
        if rm not in ("single", "compare", "multi_agent"):
            raise ValueError("run_mode must be 'single', 'compare', or 'multi_agent'")
        params["run_mode"] = rm
    if "collision_mode" in data:
        cm = str(data["collision_mode"])
        if cm not in ("head_to_head_both_die", "head_to_head_second_loses"):
            raise ValueError("collision_mode must be 'head_to_head_both_die' or 'head_to_head_second_loses'")
        params["collision_mode"] = cm
    if "state_dim" in data:
        params["state_dim"] = int(data["state_dim"])
    elif "state_dim" not in params:
        params["state_dim"] = STATE_VECTOR_SIZE
    if "weights_path" in data:
        params["weights_path"] = str(data["weights_path"])
    if "batch_size" in data:
        params["batch_size"] = int(data["batch_size"])
    if "memory_size" in data:
        params["memory_size"] = int(data["memory_size"])
    if "learning_rate" in data:
        params["learning_rate"] = float(data["learning_rate"])
