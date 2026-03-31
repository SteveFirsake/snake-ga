# Developer tasks. Requires: https://github.com/casey/just and https://docs.astral.sh/uv/
# List recipes: `just` or `just --list`

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Default: show available recipes
default:
    @just --list

# Install runtime + dev dependencies (pytest, ruff)
sync:
    uv sync --group dev

test:
    uv run pytest

lint:
    uv run ruff check snake_ga tests

fmt:
    uv run ruff format snake_ga tests

# Lint (no write) + tests — good pre-push / CI surrogate
check: lint test

# Run the game; args after `--` go to snake-ga, e.g. `just play -- --speed 100`
play *args:
    uv run snake-ga {{args}}

bayes:
    uv run snake-ga --bayesianopt true
