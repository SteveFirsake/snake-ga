from typing import Any

from snake_ga.config_loader import merge_config_into_params


def test_merge_config_policies_sets_compare_and_policy_slots() -> None:
    params: dict[str, Any] = {}
    merge_config_into_params(
        params,
        {
            "policies": ["heuristic", "random", "straight"],
            "episodes": 3,
            "display": False,
        },
    )
    assert params["run_mode"] == "compare"
    assert params["policies"] == ["heuristic", "random", "straight"]
    assert params["policy_0"] == "heuristic"
    assert params["policy_1"] == "random"


def test_merge_respects_explicit_run_mode() -> None:
    params: dict[str, Any] = {}
    merge_config_into_params(
        params,
        {
            "policies": ["heuristic", "random"],
            "run_mode": "multi_agent",
        },
    )
    assert params["run_mode"] == "multi_agent"
