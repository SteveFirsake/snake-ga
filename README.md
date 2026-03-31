# Deep Reinforcement Learning
## Project: Train AI to play Snake
*UPDATE:*

This project has been recently updated and improved:
- It is now possible to optimize the Deep RL approach using Bayesian Optimization.
- The code of Deep Reinforcement Learning was ported from Keras/TF to Pytorch. To see the old version of the code in Keras/TF, please refer to this repository: [snake-ga-tf](https://github.com/maurock/snake-ga-tf). 

## Introduction
The goal of this project is to develop an AI Bot able to learn how to play the popular game Snake from scratch. In order to do it, I implemented a Deep Reinforcement Learning algorithm. This approach consists in giving the system parameters related to its state, and a positive or negative reward based on its actions. No rules about the game are given, and initially the Bot has no information on what it needs to do. The goal for the system is to figure it out and elaborate a strategy to maximize the score - or the reward. \
We are going to see how a Deep Q-Learning algorithm learns how to play Snake, scoring up to 50 points and showing a solid strategy after only 5 minutes of training. \
Additionally, it is possible to run the Bayesian Optimization method to find the optimal parameters of the Deep neural network, as well as some parameters of the Deep RL approach.

## Install
Use [uv](https://docs.astral.sh/uv/) (Python 3.10+). Dependencies are declared in `pyproject.toml`; `torch` is resolved from the CPU wheel index to avoid pulling CUDA packages by default.

```bash
git clone git@github.com:maurock/snake-ga.git
cd snake-ga
uv sync
```

### Tests and lint
```bash
uv sync --group dev
uv run pytest
uv run ruff check snake_ga tests
uv run ruff format snake_ga tests
```

Optional: install [**just**](https://github.com/casey/just) for short task aliases (`just sync`, `just test`, `just check`, `just play`). It does not replace `uv`; it only names common commands. Run `just` in the repo root to list recipes.

**Testing strategy:** prioritize **fast unit tests on the domain** (pure rules, state vector, rewards) — they are deterministic and do not need pygame/torch. Add **narrow adapter tests** only where wiring is risky (e.g. optional smoke test with fakes). Reserve **full training/e2e** runs for manual or CI jobs with GPUs/time budgets, not the default test suite.

## Run
From the repository root:

```bash
uv run snake-ga
```

Legacy entry point (same CLI):

```bash
uv run python snakeClass.py
```

Arguments:

- `--display` — bool, default `True` (game window)
- `--speed` — int, default `50` (delay ms between frames when displaying)
- `--bayesianopt` — bool, default `False`
- `--policy` — `dqn` (default, loads weights in test mode) or baselines: `random`, `straight`, `left`, `right`, `heuristic` (no checkpoint; rules or fixed turns)
- `--board PATH` — optional **20×20** text map (one char per cell: `.` normal, `+` bonus, `-` penalty, `#` wall). Example: `uv run snake-ga --board boards/example.txt`
- `--compare POLICY` — run `--policy` first, then another policy, same board/episodes; prints a small table. Default compare length is **50** games unless you set `--episodes`. Example:  
  `uv run snake-ga --policy dqn --compare heuristic --board boards/example.txt --display false --speed 0 --episodes 20`
- `--episodes N` — override number of games (training or evaluation)

The default configuration runs a test with the DQN policy and pretrained weights.

To train the agent, set `params['train'] = True` in `snake_ga/cli.py` (`define_parameters()`), or extend the CLI.

Headless / faster runs: `--display False --speed 0` (pygame is still initialized, as in the original project).

## Project layout (hexagonal)
- `boards/` — optional ASCII tile maps (`20×20`: `. + - #`)
- `snake_ga/domain/` — pure game rules, tile grid, state encoding (no pygame/torch)

**State vector:** the DQN input is **27** floats: the original **11** binary features (danger, direction, food cues) plus **16** values = four **4-way one-hot** blocks for map tile kind (normal / bonus / penalty / wall) at the head cell and at the cell straight ahead, relative-left, and relative-right. Pretrained checkpoints built for the old **11**-dim input are **not** compatible; delete or replace `weights/weights.h5` and retrain after this change.
- `snake_ga/application/` — ports (interfaces) and the DQN run loop
- `snake_ga/adapters/` — pygame UI, PyTorch DQN, random baseline policy, plotting, Bayesian optimization
- `snake_ga/wiring.py` — composition root (binds adapters to the application)
- `snake_ga/policy_registry.py` — policy names and `build_policy()` (DQN + baselines)

## Optimize Deep RL with Bayesian Optimization

```bash
uv run snake-ga --bayesianopt True
```

Search space and optimizer settings live in `snake_ga/adapters/bayesian_optimizer.py` (`optimize_RL`).

## For Mac users
If the window does not refresh on macOS, pump the event queue after `pygame.display.update()` (for example call `pygame.event.get()` once) in `snake_ga/adapters/pygame_session.py` inside `render`.
