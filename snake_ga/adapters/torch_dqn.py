import collections
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snake_ga.adapters.torch_device import DEVICE
from snake_ga.domain.state_encoding import STATE_VECTOR_SIZE


class TorchDQNAgent(torch.nn.Module):
    """PyTorch DQN policy; training targets match the original implementation."""

    def __init__(self, params: dict[str, Any]):
        super().__init__()
        self.reward = 0
        self.gamma = 0.9
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params["learning_rate"]
        self.epsilon = 1.0
        self.actual: list[Any] = []
        self.first_layer = params["first_layer_size"]
        self.second_layer = params["second_layer_size"]
        self.third_layer = params["third_layer_size"]
        self.memory: collections.deque[tuple[Any, ...]] = collections.deque(
            maxlen=params["memory_size"]
        )
        self.weights = params["weights_path"]
        self.load_weights = params["load_weights"]
        self.state_dim = int(params.get("state_dim", STATE_VECTOR_SIZE))
        self.optimizer = None
        self.network()

    def network(self) -> None:
        self.f1 = nn.Linear(self.state_dim, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 3)
        if self.load_weights:
            try:
                try:
                    state = torch.load(self.weights, map_location=DEVICE, weights_only=False)
                except TypeError:
                    state = torch.load(self.weights, map_location=DEVICE)
            except FileNotFoundError:
                print(
                    f"Warning: weights file not found: {self.weights!r}. Using random initialization."
                )
                return
            try:
                self.load_state_dict(state)
            except RuntimeError as e:
                print(
                    f"Warning: could not load checkpoint from {self.weights!r} ({e!s}).\n"
                    f"Using random initialization (network expects state_dim={self.state_dim}). "
                    "Retrain or replace with a matching weights file."
                )
                return
            print("weights loaded")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return F.softmax(self.f4(x), dim=-1)

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory: collections.deque, batch_size: int) -> None:
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = list(memory)
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(
                DEVICE
            )
            state_tensor = torch.tensor(
                np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True
            ).to(DEVICE)
            if not done:
                target = float(
                    reward + self.gamma * torch.max(self.forward(next_state_tensor)[0]).item()
                )
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            assert self.optimizer is not None
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()

    def train_short_memory(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(
            next_state.reshape((1, self.state_dim)), dtype=torch.float32
        ).to(DEVICE)
        state_tensor = torch.tensor(
            state.reshape((1, self.state_dim)), dtype=torch.float32, requires_grad=True
        ).to(DEVICE)
        if not done:
            target = float(
                reward + self.gamma * torch.max(self.forward(next_state_tensor[0])).item()
            )
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()
