# Developer tasks. Requires: https://github.com/casey/just and https://docs.astral.sh/uv/
# List recipes: `just` or `just --list`

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Default: show available recipes
default:
    @just --list

# Install runtime + dev dependencies (pytest, ruff, mypy)
sync:
    uv sync --group dev

test:
    uv run pytest

lint:
    uv run ruff check snake_ga tests

typecheck:
    uv run mypy snake_ga

fmt:
    uv run ruff format snake_ga tests

# Lint + static types + tests — good pre-push / CI surrogate
check: lint typecheck test

# Run the game; args after `--` go to snake-ga, e.g. `just play -- --speed 100`
play *args:
    uv run snake-ga {{args}}

bayes:
    uv run snake-ga --bayesianopt true

# Compare two policies (headless, fast). For a visible window use `just compare-watch`.
compare *args:
    uv run snake-ga --policy dqn --compare heuristic --board boards/example.txt --display false --speed 0 --episodes 15 {{args}}

# JSON-driven run (see config/config.example.json, config/config.multi_agent.example.json)
config-run *args:
    uv run snake-ga --config config/config.multi_agent.example.json {{args}} --display true --speed 50 --episodes 50
