import pytest
from tictactoe import TicTacToe, LearningAgent


@pytest.fixture
def game():
    """Fixture to create a fresh game instance for each test."""
    return TicTacToe()


@pytest.fixture
def agent():
    """Fixture to create a fresh learning agent for each test."""
    return LearningAgent()
