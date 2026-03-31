from __future__ import annotations

from operator import add

import numpy as np

from snake_ga.domain.models import GameSnapshot


def encode_state(snapshot: GameSnapshot) -> np.ndarray:
    """11-dim state vector (same semantics as the original DQN)."""
    player = snapshot
    game_width = snapshot.game_width
    game_height = snapshot.game_height
    food_x = snapshot.food_x
    food_y = snapshot.food_y
    px = snapshot.player_x
    py = snapshot.player_y

    state = [
        (
            player.x_change == 20
            and player.y_change == 0
            and (
                (list(map(add, player.position[-1], [20, 0])) in player.position)
                or player.position[-1][0] + 20 >= (game_width - 20)
            )
        )
        or (
            player.x_change == -20
            and player.y_change == 0
            and (
                (list(map(add, player.position[-1], [-20, 0])) in player.position)
                or player.position[-1][0] - 20 < 20
            )
        )
        or (
            player.x_change == 0
            and player.y_change == -20
            and (
                (list(map(add, player.position[-1], [0, -20])) in player.position)
                or player.position[-1][-1] - 20 < 20
            )
        )
        or (
            player.x_change == 0
            and player.y_change == 20
            and (
                (list(map(add, player.position[-1], [0, 20])) in player.position)
                or player.position[-1][-1] + 20 >= (game_height - 20)
            )
        ),
        (
            player.x_change == 0
            and player.y_change == -20
            and (
                (list(map(add, player.position[-1], [20, 0])) in player.position)
                or player.position[-1][0] + 20 > (game_width - 20)
            )
        )
        or (
            player.x_change == 0
            and player.y_change == 20
            and (
                (list(map(add, player.position[-1], [-20, 0])) in player.position)
                or player.position[-1][0] - 20 < 20
            )
        )
        or (
            player.x_change == -20
            and player.y_change == 0
            and (
                (list(map(add, player.position[-1], [0, -20])) in player.position)
                or player.position[-1][-1] - 20 < 20
            )
        )
        or (
            player.x_change == 20
            and player.y_change == 0
            and (
                (list(map(add, player.position[-1], [0, 20])) in player.position)
                or player.position[-1][-1] + 20 >= (game_height - 20)
            )
        ),
        (
            player.x_change == 0
            and player.y_change == 20
            and (
                (list(map(add, player.position[-1], [20, 0])) in player.position)
                or player.position[-1][0] + 20 > (game_width - 20)
            )
        )
        or (
            player.x_change == 0
            and player.y_change == -20
            and (
                (list(map(add, player.position[-1], [-20, 0])) in player.position)
                or player.position[-1][0] - 20 < 20
            )
        )
        or (
            player.x_change == 20
            and player.y_change == 0
            and (
                (list(map(add, player.position[-1], [0, -20])) in player.position)
                or player.position[-1][-1] - 20 < 20
            )
        )
        or (
            player.x_change == -20
            and player.y_change == 0
            and (
                (list(map(add, player.position[-1], [0, 20])) in player.position)
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

    for i in range(len(state)):
        state[i] = 1 if state[i] else 0

    return np.asarray(state, dtype=np.float32)


def compute_reward(crash: bool, eaten: bool) -> int:
    if crash:
        return -10
    if eaten:
        return 10
    return 0
