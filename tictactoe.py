import random


class TicTacToe:
    """Tic-Tac-Toe game environment."""

    def __init__(self, starting_player: str = "X"):
        self.board = [" " for _ in range(9)]
        self.current_player = starting_player

    def get_available_moves(self):
        return [i for i, mark in enumerate(self.board) if mark == " "]

    def make_move(self, position: int):
        if self.board[position] != " ":
            raise ValueError("Invalid move")
        self.board[position] = self.current_player
        self.current_player = "O" if self.current_player == "X" else "X"

    def is_winner(self, player: str):
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

    def is_draw(self):
        return " " not in self.board

    def get_state(self):
        return tuple(self.board)


class LearningAgent:
    """Q-Learning agent for playing Tic-Tac-Toe."""

    def __init__(self, alpha: float = 0.1, gamma: float = 1.0, epsilon: float = 0.1):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

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

    def learn(self, state, action, reward, next_state, next_available_moves):
        old_q = self.get_q_value(state, action)
        future_rewards = [
            self.get_q_value(next_state, next_action) for next_action in next_available_moves
        ]
        learned_value = reward + self.gamma * max(future_rewards)
        self.update_q_value(state, action, old_q + self.alpha * (learned_value - old_q))
