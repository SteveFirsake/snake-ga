"""
Non-learning policies (same PolicyLearnerPort as DQN for fair comparison).

Actions are [straight, right, left] in the game's relative frame (see domain engine).
"""

from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class _NonlearnablePolicy(nn.Module, ABC):
    """Shared no-op training hooks and no optimizer."""

    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1.0
        self.memory: collections.deque = collections.deque(maxlen=1)
        self.optimizer = None

    def train_short_memory(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass

    def replay_new(self, memory: Any, batch_size: int) -> None:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RandomPolicyAgent(_NonlearnablePolicy):
    """Random softmax logits (explore/exploit still driven by epsilon in the run loop)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        noise = torch.randn(b, 3, device=x.device, dtype=x.dtype)
        return torch.softmax(noise, dim=-1)


class StraightPolicyAgent(_NonlearnablePolicy):
    """Always relative straight [1, 0, 0]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        logits = torch.tensor([[8.0, 0.0, 0.0]], device=x.device, dtype=x.dtype).expand(b, -1)
        return torch.softmax(logits, dim=-1)


class LeftPolicyAgent(_NonlearnablePolicy):
    """Always relative left turn [0, 0, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        logits = torch.tensor([[0.0, 0.0, 8.0]], device=x.device, dtype=x.dtype).expand(b, -1)
        return torch.softmax(logits, dim=-1)


class RightPolicyAgent(_NonlearnablePolicy):
    """Always relative right turn [0, 1, 0]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        logits = torch.tensor([[0.0, 8.0, 0.0]], device=x.device, dtype=x.dtype).expand(b, -1)
        return torch.softmax(logits, dim=-1)


class HeuristicSurvivalPolicyAgent(_NonlearnablePolicy):
    """
    Rule-based: prefer straight if safe; else left if safe; else right if safe; else uniform.
    Uses danger bits from the state vector (indices 0–2: straight, right, left).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        out = torch.zeros(b, 3, device=x.device, dtype=x.dtype)
        for i in range(b):
            st = x[i]
            ds = float(st[0].item())
            d_right = float(st[1].item())
            d_left = float(st[2].item())
            if ds < 0.5:
                out[i, 0] = 8.0
            elif d_left < 0.5:
                out[i, 2] = 8.0
            elif d_right < 0.5:
                out[i, 1] = 8.0
            else:
                out[i, :] = 1.0
        return torch.softmax(out, dim=-1)
