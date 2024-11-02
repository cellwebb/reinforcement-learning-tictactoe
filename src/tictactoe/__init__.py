import random
import json
from typing import Optional

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


class TicTacToe:
    """Tic-Tac-Toe game environment."""

    def __init__(self, starting_player: str = "X"):
        self.board = [" " for _ in range(9)]
        self.current_player = starting_player

    def get_available_moves(self) -> list[int]:
        return [i for i, mark in enumerate(self.board) if mark == " "]

    def make_move(self, position: int) -> None:
        if self.board[position] != " ":
            raise ValueError("Invalid move")
        self.board[position] = self.current_player
        self.current_player = "O" if self.current_player == "X" else "X"

    def is_winner(self, player: str) -> bool:
        return any(all(self.board[i] == player for i in condition) for condition in WIN_CONDITIONS)

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

    def get_q_value(self, state: tuple[str], action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state: tuple[str], action: int, value: float) -> None:
        self.q_table[(state, action)] = value

    def choose_action(self, state: tuple[str], available_moves: list[int]) -> int:
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
    ) -> None:
        """Update Q-value based on reward and learned value."""
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

    def save_policy(self, filename: str) -> None:
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


class HumanPlayer:
    """Human player for playing Tic-Tac-Toe."""

    def choose_action(self, state: tuple[str], available_moves: list[int]) -> int:
        """Get move from human player via console input."""
        self.display_board(state)

        print(f"Available moves (0-8): {available_moves}")

        while True:
            try:
                move = int(input("Enter your move: "))
                if move in available_moves:
                    return move
                print("Invalid move, try again")
            except ValueError:
                print("Please enter a number between 0-8")

    def display_board(self, state: tuple[str]) -> None:
        """Display the board with numbered positions."""
        board = [" " if x == 0 else x for x in state]
        print("\n")
        print(f" {board[0]} | {board[1]} | {board[2]} ")
        print("-----------")
        print(f" {board[3]} | {board[4]} | {board[5]} ")
        print("-----------")
        print(f" {board[6]} | {board[7]} | {board[8]} ")
        print("\n")


def play_game(player1: LearningAgent | HumanPlayer, player2: LearningAgent | HumanPlayer) -> str:
    """Play a game of Tic-Tac-Toe between two agents or a human and an agent."""
    env = TicTacToe()

    players = {"X": player1, "O": player2}
    state = env.get_state()

    while True:
        player = env.current_player
        opponent = "O" if player == "X" else "X"

        available_moves = env.get_available_moves()
        action = players[player].choose_action(state, available_moves)
        env.make_move(action)

        next_state = env.get_state()

        if env.is_winner(player):
            # Only update AI agents
            if hasattr(players[player], "learn"):
                players[player].learn(state, action, 1, next_state, [])
            if hasattr(players[opponent], "learn"):
                players[opponent].learn(state, action, -1, next_state, [])
            return player

        if env.is_draw():
            # Only update AI agents
            for p in ["X", "O"]:
                if hasattr(players[p], "learn"):
                    players[p].learn(state, action, 0.5, next_state, [])
            return "draw"

        # Only update AI agents
        if hasattr(players[player], "learn"):
            players[player].learn(state, action, 0, next_state, env.get_available_moves())
        state = next_state


def play_against_ai(ai_agent, human_plays_first: bool = True) -> None:
    """Play a game against the AI agent."""
    human = HumanPlayer()

    if human_plays_first:
        agents = {"X": human, "O": ai_agent}
    else:
        agents = {"X": ai_agent, "O": human}

    print("Game starting! Positions are numbered 0-8, left to right, top to bottom")
    play_game(agents["X"], agents["O"])


def main():
    agent1 = LearningAgent()
    agent2 = LearningAgent()
    num_episodes = 100_000

    results = {"X": 0, "O": 0, "draw": 0}
    wins = {"Agent 1": 0, "Agent 2": 0, "draw": 0}

    for _ in range(num_episodes):
        if random.random() < 0.5:
            result = play_game(agent1, agent2)

            if result == "X":
                wins["Agent 1"] += 1
            elif result == "O":
                wins["Agent 2"] += 1
            else:
                wins["draw"] += 1
        else:
            result = play_game(agent2, agent1)

            if result == "X":
                wins["Agent 2"] += 1
            elif result == "O":
                wins["Agent 1"] += 1
            else:
                wins["draw"] += 1

        results[result] += 1

    print(f"Results: {results}")
    print(f"Wins: {wins}")

    agent1.save_policy("agent1.json")
    agent2.save_policy("agent2.json")


if __name__ == "__main__":
    main()