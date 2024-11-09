import pytest
from tictactoe import TicTacToe, get_available_moves, is_winner, is_draw, play_game, LearningAgent


def test_initial_state(game):
    """Test that the game board is initialized correctly."""
    assert game.board == [" "] * 9
    assert game.current_player == "X"


def test_get_available_moves(game):
    """Test that the available moves are returned correctly."""
    assert list(get_available_moves(game.get_state())) == list(range(9))
    game.make_move(0)
    assert list(get_available_moves(game.get_state())) == list(range(1, 9))
    partial_board = "XO  X O  "
    assert get_available_moves(partial_board) == (2, 3, 5, 7, 8)


def test_make_move(game):
    """Test that moves are made correctly."""
    game.make_move(0)
    assert game.board[0] == "X"
    assert game.current_player == "O"


def test_make_invalid_move(game):
    """Test that invalid moves raise an error."""
    game.make_move(0)
    with pytest.raises(ValueError, match="Invalid move"):
        game.make_move(0)


@pytest.mark.parametrize(
    "state,player,expected",
    [
        ("XXX      ", "X", True),  # Horizontal win
        ("O  O  O  ", "O", True),  # Vertical win
        ("X   X   X", "X", True),  # Diagonal win
        ("OXOXOXXOX", "O", False),  # No win
    ],
)
def test_is_winner(state, player, expected):
    """Test various win conditions."""
    assert is_winner(state, player) == expected


@pytest.mark.parametrize(
    "state,expected",
    [
        ("XOXXOOOXX", True),  # Full board
        ("XOX OO OX", False),  # Not full
    ],
)
def test_is_draw(state, expected):
    """Test draw conditions."""
    assert is_draw(state) == expected


def test_get_state(game):
    """Test that the game state is returned correctly."""
    assert game.get_state() == "         "
    game.make_move(0)
    assert game.get_state() == "X        "


def test_full_game_play():
    """Test complete game between two agents."""
    agent1 = LearningAgent(epsilon=0)
    agent2 = LearningAgent(epsilon=0)
    result = play_game(agent1, agent2)
    assert result in ["X", "O", "draw"]


def test_state_history_tracking():
    """Test game state history tracking."""
    game = TicTacToe()
    assert len(game.state_history) == 1
    assert len(game.move_history) == 0

    game.make_move(0)
    assert len(game.state_history) == 2
    assert len(game.move_history) == 1


def test_multiple_game_outcomes():
    """Test different game outcomes."""
    outcomes = set()
    agent1 = LearningAgent(epsilon=0.5)
    agent2 = LearningAgent(epsilon=0.5)

    # Play multiple games to ensure we see all possible outcomes
    for _ in range(1000):
        outcome = play_game(agent1, agent2)
        outcomes.add(outcome)

    # Should see at least two different outcomes
    assert len(outcomes) >= 2


def test_str_representation():
    """Test string representation functions."""
    # Test board state formatting
    board = list("XO  X O  ")
    game = TicTacToe()
    game.board = board

    # Check board display format
    expected = "\n\n X | O |   \n-----------\n   | X |   \n-----------\n O |   |   \n\n"
    assert str(game) == expected


def test_cached_functions_consistency():
    """Test that cached functions maintain consistency."""
    board1 = "         "
    board2 = "         "

    # Test get_available_moves cache
    moves1 = get_available_moves(board1)
    moves2 = get_available_moves(board2)
    assert moves1 is moves2  # Should return same cached object

    # Test is_winner cache
    result1 = is_winner(board1, "X")
    result2 = is_winner(board2, "X")
    assert result1 is result2  # Should return same cached object
