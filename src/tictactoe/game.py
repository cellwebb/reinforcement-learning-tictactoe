import random
from .utils import (
    get_available_moves,
    is_winner,
    is_draw,
    opponent_wins_next_turn,
    opponent_can_draw,
)
from .agents import LearningAgent, HumanPlayer


class TicTacToe:
    """Tic-Tac-Toe game environment."""

    def __init__(self, starting_player: str = "X"):
        self.board = [" " for _ in range(9)]
        self.current_player = starting_player
        self.state_history = [self.get_state()]
        self.move_history = []

    def make_move(self, position: int) -> None:
        if self.board[position] != " ":
            raise ValueError("Invalid move")
        self.board[position] = self.current_player
        self.current_player = "O" if self.current_player == "X" else "X"
        self.move_history.append(position)
        self.state_history.append(self.get_state())

    def get_state(self) -> str:
        return "".join(self.board)

    def __str__(self):
        return (
            "\n\n"
            f" {self.board[0]} | {self.board[1]} | {self.board[2]} \n"
            "-----------\n"
            f" {self.board[3]} | {self.board[4]} | {self.board[5]} \n"
            "-----------\n"
            f" {self.board[6]} | {self.board[7]} | {self.board[8]} \n\n"
        )


def play_game(player1: LearningAgent | HumanPlayer, player2: LearningAgent | HumanPlayer) -> str:
    """Play a game of Tic-Tac-Toe between two agents, two humans, or a human and an agent."""

    env = TicTacToe()

    players = {"X": player1, "O": player2}

    human_in_game = "human" in [player1.player_type, player2.player_type]
    if human_in_game:
        print(env)

    while True:
        player_mark = env.current_player

        state = env.get_state()
        available_moves = get_available_moves(state)

        if not available_moves:
            return "draw"

        for action in random.choices(available_moves):
            simulated_state = list(state)
            simulated_state[action] = player_mark
            simulated_state = "".join(simulated_state)
            if is_winner(simulated_state, player_mark):
                action = action
                break

        else:
            action = players[player_mark].choose_action(state, available_moves)
        env.make_move(action)

        if human_in_game:
            print(env)

        new_state = env.get_state()

        if players[player_mark].player_type == "agent":
            if is_winner(new_state, player_mark):
                reward = 100
                players[player_mark].learn(state, action, reward, done=True)
                return player_mark

            if is_draw(new_state):
                reward = 10
                players[player_mark].learn(state, action, reward, done=True)
                return "draw"

            if opponent_wins_next_turn(new_state, player_mark):
                reward = -1
                players[player_mark].learn(state, action, reward, done=True)
            elif opponent_can_draw(new_state):
                reward = 10
                players[player_mark].learn(state, action, reward, done=True)
            else:
                reward = 0
                players[player_mark].learn(state, action, reward, new_state, player_mark)


def play_against_ai(ai_agent, human_plays_first: bool = True) -> str:
    """Play a game against an AI agent."""
    human = HumanPlayer()

    stored_epsilon = ai_agent.epsilon
    ai_agent.epsilon = 0

    print("Game starting! Positions are numbered 0-8, left to right, top to bottom")

    if human_plays_first:
        result = play_game(human, ai_agent)
    else:
        result = play_game(ai_agent, human)

    if result == "draw":
        print("It's a draw!")
    elif (result == "X" and human_plays_first) or (result == "O" and not human_plays_first):
        print("You win!")
    else:
        print("AI wins!")

    ai_agent.epsilon = stored_epsilon

    return result
