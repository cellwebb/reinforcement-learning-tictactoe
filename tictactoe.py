class TicTacToe:

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

    def is_winner(self, player):
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
