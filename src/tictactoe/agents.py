import random
import json
from .utils import get_available_moves, possible_next_states


class LearningAgent:
    """Q-Learning agent for playing Tic-Tac-Toe."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        policy_infile: str = None,
        policy_outfile: str = None,
    ):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

        if policy_infile:
            self.load_policy(policy_infile)
        else:
            self.q_table = {}

        self.policy_outfile = policy_outfile
        self.player_type = "agent"

    def get_q_value(self, state: str, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state: str, action: int, value: float) -> None:
        self.q_table[(state, action)] = value

    def choose_action(self, state: str, available_moves: tuple[int]) -> int:
        if not available_moves:
            raise ValueError("No available moves")

        if random.random() < self.epsilon:
            return random.choice(available_moves)
        else:
            q_values = {action: self.get_q_value(state, action) for action in available_moves}
            max_q = max(q_values.values())
            return random.choice([action for action, q in q_values.items() if q == max_q])

    def learn(
        self,
        state: str,
        action: int,
        reward: float,
        done: bool = False,  # whether the game is over
        next_state: str | None = None,
        player_mark: str = None,
    ) -> None:
        """Update Q-value based on reward and learned value."""
        current_q = self.get_q_value(state, action)

        if done or next_state is None:
            target_q = reward
        else:
            opponent_mark = "O" if player_mark == "X" else "X"
            potential_next_states = [
                "".join(state) for state in possible_next_states(next_state, opponent_mark)
            ]

            potential_future_rewards = []
            for potential_state in potential_next_states:
                potential_moves = get_available_moves("".join(potential_state))

                for move in potential_moves:
                    potential_future_rewards.append(self.get_q_value(potential_state, move))

            if potential_future_rewards:
                target_q = reward + self.gamma * max(potential_future_rewards)
            else:
                target_q = reward

        new_q = current_q + self.alpha * (target_q - current_q)
        self.update_q_value(state, action, new_q)

    def save_policy(self, filename: str = None) -> None:
        """Save the Q-table to a file."""
        if filename is None:
            filename = self.policy_outfile
        serialized_q_table = {
            self._serialize_key(state, action): value
            for (state, action), value in self.q_table.items()
        }
        with open(filename, "w") as f:
            json.dump(serialized_q_table, f)

    def load_policy(self, filename: str) -> None:
        with open(filename, "r") as f:
            serialized_q_table = json.load(f)
            self.q_table = {
                self._deserialize_key(key): value for key, value in serialized_q_table.items()
            }

    def _serialize_key(self, state: str, action: int) -> str:
        """Convert a state-action pair to a string key."""
        return f"{state}_{action}"

    def _deserialize_key(self, key: str) -> tuple[str, int]:
        """Convert a string key back to state-action pair."""
        state, action = key.rsplit("_", 1)
        return state, int(action)


class HumanPlayer:
    """Human player for playing Tic-Tac-Toe."""

    def __init__(self):
        self.player_type = "human"

    def choose_action(self, state: str, available_moves: list[int]) -> int:
        """Get move from human player via console input."""

        while True:
            try:
                print(f"Available moves (0-8): {available_moves}")
                move = int(input("Enter your move: "))
                if move in available_moves:
                    return move
                print("Invalid move, try again")
            except ValueError:
                print("Please enter an available move")
