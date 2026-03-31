from __future__ import annotations

import argparse
import datetime
from typing import Any

from snake_ga.application.multi_run_loop import run_multi_agent_eval
from snake_ga.application.run_loop import run_training_or_test
from snake_ga.compare_run import compare_policies
from snake_ga.config_loader import (
    default_config_path_from_env,
    load_config_file,
    merge_config_into_params,
)
from snake_ga.domain import STATE_VECTOR_SIZE
from snake_ga.policy_registry import LEARNED_POLICIES, POLICY_CHOICES
from snake_ga.wiring import build_multi_session_agents_plotter, build_session_agent_plotter


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
    params["run_mode"] = "single"
    params["policies"] = None
    params["policy_0"] = "heuristic"
    params["policy_1"] = "random"
    params["collision_mode"] = "head_to_head_both_die"
    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snake DQN / baselines. Use --config for JSON, or CLI flags (defaults)."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="JSON config (merged first). Env SNAKE_GA_CONFIG can also point to a file.",
    )
    parser.add_argument("--display", nargs="?", type=_cli_strtobool, default=None)
    parser.add_argument("--speed", type=int, default=None)
    parser.add_argument("--bayesianopt", nargs="?", type=_cli_strtobool, default=False)
    parser.add_argument(
        "--policy",
        choices=POLICY_CHOICES,
        default=None,
        help="Policy when run_mode is single (default from define_parameters or config).",
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
        help="Override number of games (default 250; compare CLI uses 50 if unset).",
    )
    parser.add_argument(
        "--compare",
        default=None,
        metavar="POLICY",
        choices=POLICY_CHOICES,
        help="Run a second policy after --policy (pair compare). Ignores config multi-policy list.",
    )
    args = parser.parse_args()

    params = define_parameters()
    cfg_path = args.config or default_config_path_from_env()
    if cfg_path:
        merge_config_into_params(params, load_config_file(cfg_path))

    if args.display is not None:
        params["display"] = bool(args.display)
    if args.speed is not None:
        params["speed"] = args.speed
    if args.policy is not None:
        params["policy"] = args.policy
    if args.board is not None:
        params["board_path"] = args.board
    if args.episodes is not None:
        params["episodes"] = args.episodes
    elif args.compare is not None:
        params["episodes"] = 50

    if args.bayesianopt and args.compare is not None:
        parser.error("--compare cannot be used with --bayesianopt")

    if args.bayesianopt and params.get("run_mode") == "compare":
        parser.error("config run_mode=compare cannot be used with --bayesianopt")

    if args.compare is not None:
        compare_policies(params, [params["policy"], args.compare])
        return

    if params.get("run_mode") == "multi_agent":
        if args.bayesianopt:
            parser.error("multi_agent mode cannot be used with --bayesianopt")
        pols = params.get("policies") or [params.get("policy_0"), params.get("policy_1")]
        if len(pols) < 2:
            parser.error("multi_agent requires at least two policies (config policies or policy_0/policy_1)")
        params["train"] = False
        params["test"] = True
        params["policy_0"] = pols[0]
        params["policy_1"] = pols[1]
        session, agents, plotter = build_multi_session_agents_plotter(params)
        run_multi_agent_eval(params, session, agents, plotter)
        return

    if params.get("run_mode") == "compare" and params.get("policies"):
        compare_policies(params, params["policies"])
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
