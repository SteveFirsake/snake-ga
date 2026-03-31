from __future__ import annotations

from operator import add

import numpy as np

from snake_ga.domain.models import GameSnapshot

# Original DQN features (danger, direction, food relative)
STATE_BASE_SIZE = 11
# One-hot tile kind (normal/bonus/penalty/blocked) × 4 positions: head, straight, left, right
TILE_ONE_HOT_DIM = 4
STATE_TILE_SIZE = 4 * TILE_ONE_HOT_DIM
STATE_VECTOR_SIZE = STATE_BASE_SIZE + STATE_TILE_SIZE


def _tile_one_hot4(idx: int) -> list[float]:
    return [1.0 if i == idx else 0.0 for i in range(TILE_ONE_HOT_DIM)]


def _hits_body_cell(
    cell: list[float],
    position: list[list[float]],
    obstacles: tuple[tuple[float, float], ...],
) -> bool:
    if cell in position:
        return True
    if not obstacles:
        return False
    return (cell[0], cell[1]) in obstacles


def encode_state(snapshot: GameSnapshot) -> np.ndarray:
    """27-dim vector: original 11-d DQN semantics + tile kinds (one-hot × 4)."""
    player = snapshot
    game_width = snapshot.game_width
    game_height = snapshot.game_height
    food_x = snapshot.food_x
    food_y = snapshot.food_y
    px = snapshot.player_x
    py = snapshot.player_y
    obs = snapshot.obstacle_positions

    state = [
        (
            player.x_change == 20
            and player.y_change == 0
            and (
                _hits_body_cell(list(map(add, player.position[-1], [20, 0])), player.position, obs)
                or player.position[-1][0] + 20 >= (game_width - 20)
            )
        )
        or (
            player.x_change == -20
            and player.y_change == 0
            and (
                _hits_body_cell(list(map(add, player.position[-1], [-20, 0])), player.position, obs)
                or player.position[-1][0] - 20 < 20
            )
        )
        or (
            player.x_change == 0
            and player.y_change == -20
            and (
                _hits_body_cell(list(map(add, player.position[-1], [0, -20])), player.position, obs)
                or player.position[-1][-1] - 20 < 20
            )
        )
        or (
            player.x_change == 0
            and player.y_change == 20
            and (
                _hits_body_cell(list(map(add, player.position[-1], [0, 20])), player.position, obs)
                or player.position[-1][-1] + 20 >= (game_height - 20)
            )
        ),
        (
            player.x_change == 0
            and player.y_change == -20
            and (
                _hits_body_cell(list(map(add, player.position[-1], [20, 0])), player.position, obs)
                or player.position[-1][0] + 20 > (game_width - 20)
            )
        )
        or (
            player.x_change == 0
            and player.y_change == 20
            and (
                _hits_body_cell(list(map(add, player.position[-1], [-20, 0])), player.position, obs)
                or player.position[-1][0] - 20 < 20
            )
        )
        or (
            player.x_change == -20
            and player.y_change == 0
            and (
                _hits_body_cell(list(map(add, player.position[-1], [0, -20])), player.position, obs)
                or player.position[-1][-1] - 20 < 20
            )
        )
        or (
            player.x_change == 20
            and player.y_change == 0
            and (
                _hits_body_cell(list(map(add, player.position[-1], [0, 20])), player.position, obs)
                or player.position[-1][-1] + 20 >= (game_height - 20)
            )
        ),
        (
            player.x_change == 0
            and player.y_change == 20
            and (
                _hits_body_cell(list(map(add, player.position[-1], [20, 0])), player.position, obs)
                or player.position[-1][0] + 20 > (game_width - 20)
            )
        )
        or (
            player.x_change == 0
            and player.y_change == -20
            and (
                _hits_body_cell(list(map(add, player.position[-1], [-20, 0])), player.position, obs)
                or player.position[-1][0] - 20 < 20
            )
        )
        or (
            player.x_change == 20
            and player.y_change == 0
            and (
                _hits_body_cell(list(map(add, player.position[-1], [0, -20])), player.position, obs)
                or player.position[-1][-1] - 20 < 20
            )
        )
        or (
            player.x_change == -20
            and player.y_change == 0
            and (
                _hits_body_cell(list(map(add, player.position[-1], [0, 20])), player.position, obs)
                or player.position[-1][-1] + 20 >= (game_height - 20)
            )
        ),
        player.x_change == -20,
        player.x_change == 20,
        player.y_change == -20,
        player.y_change == 20,
        food_x < px,
        food_x > px,
        food_y < py,
        food_y > py,
    ]

    base = np.asarray([1.0 if b else 0.0 for b in state], dtype=np.float32)
    tile = (
        _tile_one_hot4(snapshot.tile_under_head)
        + _tile_one_hot4(snapshot.tile_straight)
        + _tile_one_hot4(snapshot.tile_left)
        + _tile_one_hot4(snapshot.tile_right)
    )
    return np.concatenate([base, np.asarray(tile, dtype=np.float32)])


def compute_reward(crash: bool, eaten: bool) -> int:
    if crash:
        return -10
    if eaten:
        return 10
    return 0
