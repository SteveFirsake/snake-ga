from __future__ import annotations

import argparse
import datetime
import distutils.util
from typing import Any

from snake_ga.application.run_loop import run_training_or_test
from snake_ga.wiring import build_session_agent_plotter


def define_parameters() -> dict[str, Any]:
    params: dict[str, Any] = {}
    params["epsilon_decay_linear"] = 1 / 100
    params["learning_rate"] = 0.00013629
    params["first_layer_size"] = 200
    params["second_layer_size"] = 20
    params["third_layer_size"] = 50
    params["episodes"] = 250
    params["memory_size"] = 2500
    params["batch_size"] = 1000
    params["weights_path"] = "weights/weights.h5"
    params["train"] = False
    params["test"] = True
    params["plot_score"] = True
    params["log_path"] = "logs/scores_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".txt"
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs="?", type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs="?", type=int, default=50)
    parser.add_argument("--bayesianopt", nargs="?", type=distutils.util.strtobool, default=False)
    args = parser.parse_args()

    params["display"] = bool(args.display)
    params["speed"] = args.speed

    if args.bayesianopt:
        from snake_ga.adapters.bayesian_optimizer import BayesianOptimizer

        print("Starting Bayesian optimization...")
        bayes_opt = BayesianOptimizer(params)
        optimized_params = bayes_opt.optimize_RL()
        if optimized_params is not None:
            params.update(optimized_params)
        print("Optimization complete. Using optimized parameters:", params)

    if params["train"]:
        print("Training with parameters:", params)
        params["load_weights"] = False
        session, agent, plotter = build_session_agent_plotter(params)
        run_training_or_test(params, session, agent, plotter)
    elif params["test"]:
        print("Testing with parameters:", params)
        params["train"] = False
        params["load_weights"] = True
        session, agent, plotter = build_session_agent_plotter(params)
        run_training_or_test(params, session, agent, plotter)
    else:
        print(
            "No training or testing mode specified. "
            "Please set train=True or test=True in parameters."
        )
