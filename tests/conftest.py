import pytest
from tictactoe import TicTacToe, LearningAgent
import random  # New import


@pytest.fixture
def game():
    """Fixture to create a fresh game instance for each test."""
    return TicTacToe()


@pytest.fixture
def agent():
    """Fixture to create a fresh learning agent for each test."""
    return LearningAgent()


@pytest.fixture
def random_move_generator():
    """Fixture that returns a generator yielding random integers between 0 and 8 as strings."""
    return (str(random.randint(0, 8)) for _ in iter(int, 1))
