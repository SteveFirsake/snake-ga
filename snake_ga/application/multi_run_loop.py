"""Two-snake tick loop with two policies (evaluation)."""

from __future__ import annotations

import random
import statistics
from random import randint
from typing import Any

import numpy as np
import torch

from snake_ga.application.ports import GameDisplayPort, PolicyLearnerPort, ScorePlotterPort
from snake_ga.domain.multi_snake_engine import MultiSnakeGameEngine
from snake_ga.domain.state_encoding import STATE_VECTOR_SIZE, encode_state
DEVICE = "cpu"


def _mean_stdev(arr: list[float]) -> tuple[float, float]:
    return statistics.mean(arr), statistics.stdev(arr)


def run_multi_agent_eval(
    params: dict[str, Any],
    session: GameDisplayPort,
    agents: tuple[PolicyLearnerPort, PolicyLearnerPort],
    plotter: ScorePlotterPort | None,
) -> tuple[float, float, float, float, float, float]:
    """Run episodes with two snakes; returns total0, mean0, stdev0, total1, mean1, stdev1."""
    engine = session.engine
    assert isinstance(engine, MultiSnakeGameEngine)

    session.init_pygame()
    a0, a1 = agents[0].to(DEVICE), agents[1].to(DEVICE)

    scores0: list[float] = []
    scores1: list[float] = []
    counter_plot: list[int] = []
    total0 = total1 = 0.0

    for game in range(params["episodes"]):
        session.pump_quit_events()
        engine.reset()
        record = max(max(scores0) if scores0 else 0, max(scores1) if scores1 else 0)

        steps = 0
        while not engine.both_dead and steps < 200:
            a0.epsilon = a1.epsilon = 0.01

            moves: list[np.ndarray] = []
            for sid, agent in enumerate((a0, a1)):
                if engine.snakes[sid].crash:
                    moves.append(np.array([1.0, 0.0, 0.0]))
                    continue
                st = encode_state(engine.snapshot(sid))
                if random.uniform(0, 1) < agent.epsilon:
                    mv = np.eye(3)[randint(0, 2)]
                else:
                    with torch.no_grad():
                        t = torch.tensor(st.reshape((1, STATE_VECTOR_SIZE)), dtype=torch.float32).to(
                            DEVICE
                        )
                        pred = agent(t)
                        mv = np.eye(3)[np.argmax(pred.detach().cpu().numpy()[0])]
                moves.append(mv)

            engine.apply_tick((moves[0], moves[1]))

            if params["display"]:
                session.render(int(record))
                session.wait(params["speed"])
            steps += 1

        s0, s1 = float(engine.snakes[0].score), float(engine.snakes[1].score)
        scores0.append(s0)
        scores1.append(s1)
        total0 += s0
        total1 += s1
        counter_plot.append(game + 1)
        print(f"Game {game + 1}   scores: snake0={s0}  snake1={s1}")

    m0, d0 = _mean_stdev(scores0)
    m1, d1 = _mean_stdev(scores1)
    if plotter is not None and scores0:
        combined = [a + b for a, b in zip(scores0, scores1)]
        plotter.plot(counter_plot, combined, False)
    return total0, m0, d0, total1, m1, d1
