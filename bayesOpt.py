# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:10:29 2020

@author: mauro
"""
from GPyOpt.methods import BayesianOptimization

from snakeClass import run

################################################
#   Set parameters for Bayesian Optimization   #
################################################


class BayesianOptimizer:
    def __init__(self, params):
        self.params = params.copy()  # Create a copy to avoid modifying original
        self.best_score = float("-inf")
        self.best_params = None

    def optimize_RL(self):
        def optimize(inputs):
            print("Optimizing with inputs:", inputs)
            inputs = inputs[0]

            # Variables to optimize
            current_params = self.params.copy()
            current_params["learning_rate"] = inputs[0]
            lr_string = "{:.8f}".format(current_params["learning_rate"])[2:]
            current_params["first_layer_size"] = int(inputs[1])
            current_params["second_layer_size"] = int(inputs[2])
            current_params["third_layer_size"] = int(inputs[3])
            current_params["epsilon_decay_linear"] = int(inputs[4])

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
            score, mean, stdev = run(current_params)
            print(
                f"Results - Score: {score:.2f}, Mean: {mean:.2f}, Std dev: {stdev:.2f}"
            )

            # Update best parameters if we found a better solution
            if score > self.best_score:
                self.best_score = score
                self.best_params = current_params.copy()
                print("New best parameters found!")

            # Log results
            with open(current_params["log_path"], "a") as f:
                f.write(f"\nScenario: {current_params['name_scenario']}\n")
                f.write(f"Parameters: {current_params}\n")
                f.write(f"Score: {score}, Mean: {mean}, Std dev: {stdev}\n")

            return score

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


##################
#      Main      #
##################
if __name__ == "__main__":
    # Define parameters
    params = {
        "episodes": 150,  # Number of episodes to run
        "gamma": 0.95,  # Discount factor
        "epsilon": 1,  # Starting exploration rate
        "epsilon_min": 0.01,  # Minimum exploration rate
    }

    # Define optimizer
    bayesOpt = BayesianOptimizer(params)
    bayesOpt.optimize_RL()
