import collections
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snake_ga.adapters.torch_device import DEVICE


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
        self.memory = collections.deque(maxlen=params["memory_size"])
        self.weights = params["weights_path"]
        self.load_weights = params["load_weights"]
        self.optimizer = None
        self.network()

    def network(self) -> None:
        self.f1 = nn.Linear(11, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 3)
        if self.load_weights:
            try:
                state = torch.load(self.weights, map_location=DEVICE, weights_only=False)
            except TypeError:
                state = torch.load(self.weights, map_location=DEVICE)
            self.load_state_dict(state)
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
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
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
        next_state_tensor = torch.tensor(next_state.reshape((1, 11)), dtype=torch.float32).to(
            DEVICE
        )
        state_tensor = torch.tensor(
            state.reshape((1, 11)), dtype=torch.float32, requires_grad=True
        ).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()
