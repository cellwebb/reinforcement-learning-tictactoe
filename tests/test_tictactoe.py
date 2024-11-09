import pytest
from unittest.mock import patch
from tictactoe import play_against_ai, train, LearningAgent

from tictactoe.utils import (
    opponent_wins_next_turn,
    opponent_can_draw,
    possible_next_states,
)


@patch("builtins.input", side_effect=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
def test_play_against_ai(mock_input, capsys):
    """Test game against AI.

    Provides a sequence of all possible moves to ensure the game can complete
    regardless of the AI's choices. The game will use moves in sequence until
    a game-ending condition is reached.
    """
    ai_agent = LearningAgent(epsilon=0)
    with capsys.disabled():
        result = play_against_ai(ai_agent, human_plays_first=True)
    assert result in ["X", "O", "draw"]


@patch("builtins.input", side_effect=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
def test_play_against_ai_ai_first(mock_input, capsys):
    """Test game against AI where AI plays first.

    Similar to test_play_against_ai but with AI making the first move.
    """
    ai_agent = LearningAgent(epsilon=0)
    with capsys.disabled():
        result = play_against_ai(ai_agent, human_plays_first=False)
    assert result in ["X", "O", "draw"]


def test_play_against_ai_returns_none():
    """Test play_against_ai function returns correctly."""
    ai_agent = LearningAgent(epsilon=0)
    with patch("builtins.input", side_effect=["0", "1", "2", "3", "4", "5", "6", "7", "8"]):
        result = play_against_ai(ai_agent, human_plays_first=True)
        assert result in ["X", "O", "draw"]


def test_train_with_different_episodes():
    """Test main function with different episode counts."""
    with patch("random.random", return_value=0.4):
        with patch("tictactoe.play_game", return_value="X"):
            train()


@pytest.mark.parametrize(
    "state,player,expected",
    [
        ("XX       ", "O", True),  # Opponent can win next turn
        ("XOXOXOXOX", "O", False),  # Opponent cannot win next turn
    ],
)
def test_opponent_wins_next_turn(state, player, expected):
    """Test if opponent can win in the next move."""
    assert opponent_wins_next_turn(state, player) == expected


@pytest.mark.parametrize(
    "state,expected",
    [
        ("XOXOXOXOX", False),  # Opponent cannot force a draw
        ("XOXOXO   ", True),  # Opponent can force a draw
    ],
)
def test_opponent_can_draw(state, expected):
    """Test if opponent can force a draw."""
    assert opponent_can_draw(state) == expected


@pytest.mark.parametrize(
    "state,mark,expected",
    [
        (
            "         ",
            "X",
            [
                "X        ",
                " X       ",
                "  X      ",
                "   X     ",
                "    X    ",
                "     X   ",
                "      X  ",
                "       X ",
                "        X",
            ],
        ),
        (
            "X        ",
            "O",
            [
                "XO       ",
                "X O      ",
                "X  O     ",
                "X   O    ",
                "X    O   ",
                "X     O  ",
                "X      O ",
                "X       O",
            ],
        ),
    ],
)
def test_possible_next_states(state, mark, expected):
    """Test possible next states for a given state."""
    assert possible_next_states(state, mark) == expected
