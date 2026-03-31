from __future__ import annotations

import random
import statistics
from random import randint
from typing import Any, cast

import numpy as np
import torch
import torch.optim as optim

from snake_ga.application.ports import (
    GameDisplayPort,
    GameEnginePort,
    PolicyLearnerPort,
    ScorePlotterPort,
)
from snake_ga.domain.game_engine import SnakeGameEngine
from snake_ga.domain.state_encoding import STATE_VECTOR_SIZE, compute_reward, encode_state
from snake_ga.policy_registry import LEARNED_POLICIES

DEVICE = "cpu"


def _get_mean_stdev(array: list[float]) -> tuple[float, float]:
    return statistics.mean(array), statistics.stdev(array)


def _initialize_game(
    engine: GameEnginePort,
    agent: PolicyLearnerPort,
    batch_size: int,
) -> None:
    state_init1 = encode_state(engine.snapshot())
    action = np.array([1.0, 0.0, 0.0])
    engine.apply_move(action)
    state_init2 = encode_state(engine.snapshot())
    reward1 = compute_reward(engine.crash, engine.eaten)
    agent.remember(state_init1, action, reward1, state_init2, engine.crash)
    agent.replay_new(agent.memory, batch_size)


def run_training_or_test(
    params: dict[str, Any],
    session: GameDisplayPort,
    agent: PolicyLearnerPort,
    plotter: ScorePlotterPort | None,
) -> tuple[float, float, float]:
    """Main DQN loop; depends on ports only (hex: application core)."""
    session.init_pygame()

    agent = agent.to(DEVICE)
    if params.get("policy", "dqn") in LEARNED_POLICIES:
        agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params["learning_rate"])

    counter_games = 0
    score_plot: list[float] = []
    counter_plot: list[int] = []
    record = 0
    total_score = 0.0

    while counter_games < params["episodes"]:
        session.pump_quit_events()
        session.engine.reset()
        engine = cast(SnakeGameEngine, session.engine)

        _initialize_game(engine, agent, params["batch_size"])
        if params["display"]:
            session.render(record)

        steps = 0
        while (not engine.crash) and (steps < 100):
            if not params["train"]:
                agent.epsilon = 0.01
            else:
                agent.epsilon = 1 - (counter_games * params["epsilon_decay_linear"])

            state_old = encode_state(engine.snapshot())

            if random.uniform(0, 1) < agent.epsilon:
                final_move = np.eye(3)[randint(0, 2)]
            else:
                with torch.no_grad():
                    state_old_tensor = torch.tensor(
                        state_old.reshape((1, STATE_VECTOR_SIZE)), dtype=torch.float32
                    ).to(DEVICE)
                    prediction = agent(state_old_tensor)
                    final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

            engine.apply_move(final_move)
            state_new = encode_state(engine.snapshot())
            reward = compute_reward(engine.crash, engine.eaten)

            if reward > 0:
                steps = 0

            if params["train"]:
                agent.train_short_memory(state_old, final_move, reward, state_new, engine.crash)
                agent.remember(state_old, final_move, reward, state_new, engine.crash)

            record = max(engine.score, record)
            if params["display"]:
                session.render(record)
                session.wait(params["speed"])
            steps += 1

        if params["train"]:
            agent.replay_new(agent.memory, params["batch_size"])

        counter_games += 1
        total_score += engine.score
        print(f"Game {counter_games}      Score: {engine.score}")
        score_plot.append(float(engine.score))
        counter_plot.append(counter_games)

    mean, stdev = _get_mean_stdev(score_plot)
    if params["train"] and params.get("policy", "dqn") in LEARNED_POLICIES:
        torch.save(agent.state_dict(), params["weights_path"])
    if plotter is not None:
        plotter.plot(counter_plot, score_plot, bool(params["train"]))

    return total_score, mean, stdev


def run_test_only(
    params: dict[str, Any],
    session: GameDisplayPort,
    agent: PolicyLearnerPort,
    plotter: ScorePlotterPort | None,
) -> tuple[float, float, float]:
    """Compatibility wrapper used by Bayesian optimization (maximize total score)."""
    p = dict(params)
    p["load_weights"] = True
    p["train"] = False
    p["test"] = False
    return run_training_or_test(p, session, agent, plotter)
