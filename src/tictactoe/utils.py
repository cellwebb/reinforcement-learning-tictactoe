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
