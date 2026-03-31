import pytest
import torch

from snake_ga.adapters.baseline_policies import HeuristicSurvivalPolicyAgent
from snake_ga.domain import STATE_VECTOR_SIZE
from snake_ga.policy_registry import POLICY_CHOICES, build_policy


@pytest.mark.parametrize(
    "name",
    [p for p in POLICY_CHOICES if p != "dqn"],
)
def test_baseline_policy_forward_shape(name: str) -> None:
    params = {"policy": name}
    agent = build_policy(params)
    x = torch.zeros(1, STATE_VECTOR_SIZE)
    y = agent(x)
    assert y.shape == (1, 3)
    assert torch.allclose(y.sum(dim=-1), torch.ones(1), atol=1e-4)


def test_dqn_build_requires_params() -> None:
    params = {
        "policy": "dqn",
        "learning_rate": 1e-3,
        "first_layer_size": 10,
        "second_layer_size": 10,
        "third_layer_size": 10,
        "memory_size": 100,
        "weights_path": "weights/x.pt",
        "load_weights": False,
        "state_dim": STATE_VECTOR_SIZE,
    }
    agent = build_policy(params)
    y = agent(torch.zeros(1, STATE_VECTOR_SIZE))
    assert y.shape == (1, 3)


def test_heuristic_prefers_safe_straight() -> None:
    agent = HeuristicSurvivalPolicyAgent()
    # no danger -> straight
    x = torch.zeros(1, STATE_VECTOR_SIZE)
    y = agent(x)
    assert y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2]
