from __future__ import annotations

import argparse
import datetime
from typing import Any

from snake_ga.application.run_loop import run_training_or_test
from snake_ga.compare_run import compare_policies
from snake_ga.domain import STATE_VECTOR_SIZE
from snake_ga.policy_registry import LEARNED_POLICIES, POLICY_CHOICES
from snake_ga.wiring import build_session_agent_plotter


def _cli_strtobool(val: str) -> bool:
    """Parse truthy/falsey strings (replaces deprecated distutils.util.strtobool)."""
    v = val.lower()
    if v in ("y", "yes", "t", "true", "on", "1"):
        return True
    if v in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"invalid truth value: {val!r}")


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
    params["policy"] = "dqn"
    params["board_path"] = None
    params["state_dim"] = STATE_VECTOR_SIZE
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs="?", type=_cli_strtobool, default=True)
    parser.add_argument("--speed", nargs="?", type=int, default=50)
    parser.add_argument("--bayesianopt", nargs="?", type=_cli_strtobool, default=False)
    parser.add_argument(
        "--policy",
        choices=POLICY_CHOICES,
        default="dqn",
        help=(
            "Policy: dqn (learned); random/straight/left/right/heuristic are non-learning baselines."
        ),
    )
    parser.add_argument(
        "--board",
        default=None,
        metavar="PATH",
        help="Optional 20×20 tile map (. + - #). Example: boards/example.txt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of evaluation/training games (default 250; compare mode uses 50 if unset).",
    )
    parser.add_argument(
        "--compare",
        default=None,
        metavar="POLICY",
        choices=POLICY_CHOICES,
        help="Run a second policy after --policy and print a score table (same board & settings).",
    )
    args = parser.parse_args()

    params["display"] = bool(args.display)
    params["speed"] = args.speed
    params["policy"] = args.policy
    params["board_path"] = args.board
    if args.episodes is not None:
        params["episodes"] = args.episodes
    elif args.compare is not None:
        params["episodes"] = 50

    if args.bayesianopt and args.compare is not None:
        parser.error("--compare cannot be used with --bayesianopt")

    if args.compare is not None:
        compare_policies(params, args.policy, args.compare)
        return

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
        params["load_weights"] = params.get("policy", "dqn") in LEARNED_POLICIES
        session, agent, plotter = build_session_agent_plotter(params)
        run_training_or_test(params, session, agent, plotter)
    else:
        print(
            "No training or testing mode specified. "
            "Please set train=True or test=True in parameters."
        )
