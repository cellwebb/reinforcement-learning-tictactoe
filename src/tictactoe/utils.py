from functools import lru_cache

WIN_CONDITIONS = [
    [0, 1, 2],  # Horizontal
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],  # Vertical
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],  # Diagonal
    [2, 4, 6],
]


@lru_cache(maxsize=19683)  # 3^9 possible board states
def get_available_moves(state: str) -> tuple[int, ...]:
    """Get available moves for a given board state."""
    return tuple(i for i, mark in enumerate(state) if mark == " ")


@lru_cache(maxsize=19683)
def is_winner(state: str, player: str) -> bool:
    """Check if the given player has won."""
    return any(all(state[i] == player for i in condition) for condition in WIN_CONDITIONS)


def is_draw(state: str) -> bool:
    """Check if the game is a draw."""
    return " " not in state


@lru_cache(maxsize=19683)
def opponent_wins_next_turn(state: str, player_mark: str) -> bool:
    """Check if the opponent can win in the next move."""
    opponent_mark = "O" if player_mark == "X" else "X"
    for opponent_action in get_available_moves(state):
        simulated_state = list(state)
        simulated_state[opponent_action] = opponent_mark
        simulated_state = "".join(simulated_state)
        if is_winner(simulated_state, opponent_mark):
            return True
    return False


@lru_cache(maxsize=19683)
def opponent_can_draw(state: str) -> bool:
    """Check if the opponent can force a draw."""
    return state.count(" ") == 1


@lru_cache(maxsize=19683)
def possible_next_states(state: str, mark: str) -> list[str]:
    """Get all possible next states for a given state."""
    return [
        "".join(
            state[:action] + mark + state[action + 1 :] if mark == " " else mark
            for action, mark in enumerate(state)
        )
        for action in get_available_moves(state)
    ]


# Precompute index mappings for transformations
ROTATE_90 = [6, 3, 0, 7, 4, 1, 8, 5, 2]
ROTATE_180 = [8, 7, 6, 5, 4, 3, 2, 1, 0]
ROTATE_270 = [2, 5, 8, 1, 4, 7, 0, 3, 6]
REFLECT_HORIZONTAL = [2, 1, 0, 5, 4, 3, 8, 7, 6]


def apply_transformation(board, mapping):
    """Apply a precomputed transformation mapping to a board state."""
    return "".join(board[mapping[i]] for i in range(9))


@lru_cache(maxsize=None)  # Cache all unique board states
def get_transformations_with_cache(board):
    """Generate all rotations and reflections of the board with caching."""
    transformations = {
        board,
        apply_transformation(board, ROTATE_90),
        apply_transformation(board, ROTATE_180),
        apply_transformation(board, ROTATE_270),
        apply_transformation(apply_transformation(board, REFLECT_HORIZONTAL), ROTATE_90),
        apply_transformation(apply_transformation(board, REFLECT_HORIZONTAL), ROTATE_180),
        apply_transformation(apply_transformation(board, REFLECT_HORIZONTAL), ROTATE_270),
        apply_transformation(board, REFLECT_HORIZONTAL),
    }
    return transformations
