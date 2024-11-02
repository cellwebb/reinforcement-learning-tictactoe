import random
import json
from typing import Optional


class TicTacToe:
    """Tic-Tac-Toe game environment."""

    def __init__(self, starting_player: str = "X"):
        self.board = [" " for _ in range(9)]
        self.current_player = starting_player

    def get_available_moves(self) -> list[int]:
        return [i for i, mark in enumerate(self.board) if mark == " "]

    def make_move(self, position: int):
        if self.board[position] != " ":
            raise ValueError("Invalid move")
        self.board[position] = self.current_player
        self.current_player = "O" if self.current_player == "X" else "X"

    def is_winner(self, player: str) -> bool:
        win_conditions = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        return any(all(self.board[i] == player for i in condition) for condition in win_conditions)

    def is_draw(self) -> bool:
        return " " not in self.board

    def get_state(self) -> tuple[str]:
        return tuple(self.board)

    def __str__(self):
        rows = ["|".join(self.board[i : i + 3]) for i in range(0, 9, 3)]
        return "\n-----\n".join(rows)


class LearningAgent:
    """Q-Learning agent for playing Tic-Tac-Toe."""

    def __init__(
        self, alpha: float = 0.1, gamma: float = 1.0, epsilon: float = 0.1, policy_file: str = None
    ):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

        if policy_file:
            self.load_policy(policy_file)
        else:
            self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def choose_action(self, state, available_moves):
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        else:
            q_values = {action: self.get_q_value(state, action) for action in available_moves}
            max_q = max(q_values.values())
            return random.choice([action for action, q in q_values.items() if q == max_q])

    def learn(
        self,
        state: tuple[str],
        action: int,
        reward: float,
        next_state: tuple[str],
        next_available_moves: Optional[list[int]],
    ):
        old_q = self.get_q_value(state, action)
        if next_available_moves:
            future_rewards = [
                self.get_q_value(next_state, next_action) for next_action in next_available_moves
            ]
            learned_value = reward + self.gamma * max(future_rewards)
        else:
            learned_value = reward
        self.update_q_value(state, action, old_q + self.alpha * (learned_value - old_q))

    def _serialize_key(self, state: tuple[str], action: int) -> str:
        """Convert a state-action pair to a string key."""
        return f"{state}_{action}"

    def _deserialize_key(self, key: str) -> tuple[tuple[str], int]:
        """Convert a string key back to state-action pair."""
        state_str, action = key.rsplit("_", 1)
        state = eval(state_str)  # Safe since we control the input format
        return state, int(action)

    def save_policy(self, filename: str):
        serialized_q_table = {
            self._serialize_key(state, action): value
            for (state, action), value in self.q_table.items()
        }
        with open(filename, "w") as f:
            json.dump(serialized_q_table, f)

    def load_policy(self, filename: str):
        with open(filename, "r") as f:
            serialized_q_table = json.load(f)
            self.q_table = {
                self._deserialize_key(key): value for key, value in serialized_q_table.items()
            }


def play_game(agent1, agent2):
    """Play a game of Tic-Tac-Toe between two agents."""
    env = TicTacToe()
    agents = {"X": agent1, "O": agent2}
    state = env.get_state()

    while True:
        player = env.current_player

        available_moves = env.get_available_moves()
        action = agents[player].choose_action(state, available_moves)
        env.make_move(action)

        next_state = env.get_state()

        if env.is_winner(player):
            agents[player].learn(state, action, 1, next_state, [])
            agents["O" if player == "X" else "X"].learn(state, action, -1, next_state, [])
            return player

        if env.is_draw():
            agents["X"].learn(state, action, 0.5, next_state, [])
            agents["O"].learn(state, action, 0.5, next_state, [])
            return "draw"

        agents[player].learn(state, action, 0, next_state, env.get_available_moves())
        state = next_state


def main():
    agent1 = LearningAgent()
    agent2 = LearningAgent()
    num_episodes = 100_000

    for _ in range(num_episodes):
        if random.random() < 0.5:
            play_game(agent1, agent2)
        else:
            play_game(agent2, agent1)

    agent1.save_policy("agent1.json")
    agent2.save_policy("agent2.json")


if __name__ == "__main__":
    main()
