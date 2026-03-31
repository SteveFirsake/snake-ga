from __future__ import annotations

from typing import Any

from GPyOpt.methods import BayesianOptimization  # type: ignore[import-untyped]

from snake_ga.application.run_loop import run_training_or_test
from snake_ga.wiring import build_session_agent_plotter


class BayesianOptimizer:
    def __init__(self, params: dict[str, Any]):
        self.params = params.copy()
        self.best_score = float("-inf")
        self.best_params: dict[str, Any] | None = None

    def optimize_RL(self) -> dict[str, Any] | None:
        def optimize(inputs: list[list[float]]) -> float:
            print("Optimizing with inputs:", inputs)
            row = inputs[0]

            current_params = self.params.copy()
            current_params["learning_rate"] = row[0]
            lr_string = "{:.8f}".format(current_params["learning_rate"])[2:]
            current_params["first_layer_size"] = int(row[1])
            current_params["second_layer_size"] = int(row[2])
            current_params["third_layer_size"] = int(row[3])
            current_params["epsilon_decay_linear"] = int(row[4])

            current_params["name_scenario"] = "snake_lr{}_struct{}_{}_{}_eps{}".format(
                lr_string,
                current_params["first_layer_size"],
                current_params["second_layer_size"],
                current_params["third_layer_size"],
                current_params["epsilon_decay_linear"],
            )

            current_params["weights_path"] = (
                "weights/weights_" + current_params["name_scenario"] + ".h5"
            )
            current_params["load_weights"] = False
            current_params["train"] = True

            print("Testing parameters:", current_params)
            session, agent, plotter = build_session_agent_plotter(current_params)
            score, mean, stdev = run_training_or_test(current_params, session, agent, plotter)
            print(f"Results - Score: {score:.2f}, Mean: {mean:.2f}, Std dev: {stdev:.2f}")

            if score > self.best_score:
                self.best_score = score
                self.best_params = current_params.copy()
                print("New best parameters found!")

            with open(current_params["log_path"], "a", encoding="utf-8") as f:
                f.write(f"\nScenario: {current_params['name_scenario']}\n")
                f.write(f"Parameters: {current_params}\n")
                f.write(f"Score: {score}, Mean: {mean}, Std dev: {stdev}\n")

            return float(score)

        optim_params = [
            {"name": "learning_rate", "type": "continuous", "domain": (0.00005, 0.001)},
            {
                "name": "first_layer_size",
                "type": "discrete",
                "domain": (20, 50, 100, 200),
            },
            {
                "name": "second_layer_size",
                "type": "discrete",
                "domain": (20, 50, 100, 200),
            },
            {
                "name": "third_layer_size",
                "type": "discrete",
                "domain": (20, 50, 100, 200),
            },
            {
                "name": "epsilon_decay_linear",
                "type": "discrete",
                "domain": (
                    self.params["episodes"] * 0.2,
                    self.params["episodes"] * 0.4,
                    self.params["episodes"] * 0.6,
                    self.params["episodes"] * 0.8,
                    self.params["episodes"] * 1,
                ),
            },
        ]

        bayes_optimizer = BayesianOptimization(
            f=optimize,
            domain=optim_params,
            initial_design_numdata=6,
            acquisition_type="EI",
            exact_feval=True,
            maximize=True,
        )

        print("Starting Bayesian optimization...")
        bayes_optimizer.run_optimization(max_iter=20)

        print("\nOptimization Results:")
        print("Best learning rate:", bayes_optimizer.x_opt[0])
        print("Best first layer size:", bayes_optimizer.x_opt[1])
        print("Best second layer size:", bayes_optimizer.x_opt[2])
        print("Best third layer size:", bayes_optimizer.x_opt[3])
        print("Best epsilon decay:", bayes_optimizer.x_opt[4])
        print("Best score achieved:", self.best_score)

        return self.best_params
