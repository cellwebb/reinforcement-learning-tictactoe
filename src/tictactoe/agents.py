import random
import json
from pprint import pprint
from functools import lru_cache
from itertools import permutations
from .utils import get_available_moves


class LearningAgent:
    """Q-Learning agent for playing Tic-Tac-Toe."""

    def __init__(
        self,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        win_reward: float = 1.0,
        draw_reward: float = 0.5,
        loss_reward: float = -1.0,
        starting_q_value: float = 0.0,
        policy_infile: str = None,
        policy_outfile: str = None,
    ):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay

        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_reward = loss_reward

        self.starting_q_value = starting_q_value

        if policy_infile:
            self.load_policy(policy_infile)
        else:
            self.q_table = {}

        self.policy_outfile = policy_outfile
        self.player_type = "agent"

    def get_q_value(self, state: str, action: int) -> float:
        return self.q_table.get(state, {}).get(action, self.starting_q_value)

    def update_q_value(self, state: str, action: int, value: float) -> None:
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = value

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
        self, result: str, state_history: list[str], move_history: list[int], first_player: bool
    ) -> None:
        """Update Q-value based on reward and learned value."""
        turns = len(move_history)
        went_last = first_player if turns % 2 == 0 else not first_player
        i = turns - 1 if went_last else turns - 2
        player_mark = "X" if first_player else "O"

        if result == player_mark:
            reward = self.win_reward
        elif result == "draw":
            reward = self.draw_reward
        else:
            reward = self.loss_reward

        state = state_history[i]
        action = move_history[i]

        pprint(f"result: {result}, turns: {turns}, ")
        pprint(f"player_mark: {player_mark}, reward: {reward}")
        pprint(f"i: {i}, state: {state:}, action: {action}")
        pprint(state_history)

        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward - current_q)
        self.update_q_value(state, action, new_q)

        i -= 2
        while i >= 0:
            state = state_history[i]
            action = move_history[i]
            current_q = self.get_q_value(state, action)

            max_q = self.get_max_future_reward(state)

            new_q = current_q + self.alpha * (self.gamma * max_q - current_q)
            self.update_q_value(state, action, new_q)

            i -= 2

        # self.epsilon *= self.epsilon_decay

    def save_policy(self, filename: str = None) -> None:
        """Save the Q-table to a file."""
        if filename is None:
            filename = self.policy_outfile
        with open(filename, "w") as f:
            json.dump(self.q_table, f)
        print(f"Q-table saved to {filename}: {self.q_table}")  # Debug print

    def load_policy(self, filename: str) -> None:
        with open(filename, "r") as f:
            self.q_table = json.load(f)
        # action needs to be an int, but json keys are always strings
        self.q_table = {
            state: {int(action): value for action, value in actions.items()}
            for state, actions in self.q_table.items()
        }
        print(f"Q-table loaded from {filename}: {self.q_table}")  # Debug print

    @lru_cache(maxsize=None)
    def get_state_action_pairs(self, state: str):
        moves = get_available_moves(state)

        if len(moves) < 2:
            return []

        marks = ["X", "O"] if state.count("X") == state.count("O") else ["O", "X"]

        future_move_orders = permutations(moves, 2)
        new_states = []
        for move_order in future_move_orders:
            next_state = list(state)
            for move, mark in zip(move_order, marks):
                next_state[move] = mark
            new_states.append("".join(next_state))

        return [(state, action) for state in new_states for action in get_available_moves(state)]

    def get_max_future_reward(self, state: str) -> float:
        """Get the maximum future reward for the player's next move."""
        state_action_pairs = self.get_state_action_pairs(state)
        if not state_action_pairs:
            return 0

        return max(
            self.q_table.get(state, {}).get(action, -99) for state, action in state_action_pairs
        )


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

    def learn(
        self, result: str, state_history: list[str], move_history: list[int], first_player: bool
    ):
        pass
