from functools import lru_cache

from .constants import WIN_CONDITIONS, TRANSFORMATIONS, INVERSE_MAPPINGS


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


@lru_cache(maxsize=None)
def apply_transformation(state, mapping):
    """Apply a transformation mapping to a state."""
    if mapping is None:
        return state  # No transformation
    return "".join(state[mapping[i]] for i in range(9))


@lru_cache(maxsize=None)
def inverse_transform(move, transformation):
    """Apply the inverse transformation to a move index."""
    return INVERSE_MAPPINGS.get(transformation, list(range(9)))[move]


def find_matching_state_and_transform_back(state, q_table):
    """
    Rotate and reflect a state to find a match in the Q-table.
    If a match is found, return the move suggestions transformed back
    to the original orientation.
    """
    for transformation, mapping in TRANSFORMATIONS:
        transformed_state = apply_transformation(state, tuple(mapping) if mapping else None)
        if transformed_state in q_table:
            # Found a matching state in the Q-table
            suggested_moves = q_table[transformed_state]
            # Apply the inverse transformation to suggested moves
            transformed_moves = [
                inverse_transform(move, transformation) for move in suggested_moves
            ]
            return transformed_moves

    # No match found
    return None
