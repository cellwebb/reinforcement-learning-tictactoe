import pytest
from unittest.mock import patch
import os
from tictactoe import TicTacToe, LearningAgent, HumanPlayer, play_game, play_against_ai


@pytest.fixture
def game():
    """Fixture to create a fresh game instance for each test."""
    return TicTacToe()


@pytest.fixture
def agent():
    """Fixture to create a fresh learning agent for each test."""
    return LearningAgent()


def test_initial_state(game):
    """Test that the game board is initialized correctly."""
    assert game.board == [" "] * 9
    assert game.current_player == "X"


def test_get_available_moves(game):
    """Test that the available moves are returned correctly."""
    assert game.get_available_moves() == list(range(9))
    game.make_move(0)
    assert game.get_available_moves() == list(range(1, 9))


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
    "board,player,expected",
    [
        (["X", "X", "X", " ", " ", " ", " ", " ", " "], "X", True),  # Horizontal win
        (["O", " ", " ", "O", " ", " ", "O", " ", " "], "O", True),  # Vertical win
        (["X", " ", " ", " ", "X", " ", " ", " ", "X"], "X", True),  # Diagonal win
        (["O", "X", "O", "X", "O", "X", "X", "O", "X"], "O", False),  # No win
    ],
)
def test_is_winner(game, board, player, expected):
    """Test various win conditions."""
    game.board = board
    assert game.is_winner(player) == expected


@pytest.mark.parametrize(
    "board,expected",
    [
        (["X", "O", "X", "X", "O", "O", "O", "X", "X"], True),  # Full board
        (["X", "O", "X", " ", "O", "O", "O", "X", "X"], False),  # Not full
    ],
)
def test_is_draw(game, board, expected):
    """Test draw conditions."""
    game.board = board
    assert game.is_draw() == expected


def test_get_state(game):
    """Test that the game state is returned correctly."""
    assert game.get_state() == tuple([" "] * 9)
    game.make_move(0)
    assert game.get_state() == ("X", " ", " ", " ", " ", " ", " ", " ", " ")


# Learning Agent Tests
def test_agent_initialization(agent):
    """Test that the agent is initialized with correct parameters."""
    assert agent.alpha == 0.1
    assert agent.gamma == 1.0
    assert agent.epsilon == 0.1
    assert agent.q_table == {}


def test_agent_get_q_value(agent):
    """Test Q-value retrieval."""
    state = tuple([" "] * 9)
    action = 0
    assert agent.get_q_value(state, action) == 0.0
    agent.update_q_value(state, action, 1.0)
    assert agent.get_q_value(state, action) == 1.0


def test_agent_choose_action_exploitation():
    """Test that agent chooses best action when epsilon=0."""
    agent = LearningAgent(epsilon=0)
    state = tuple([" "] * 9)
    available_moves = [0, 1, 2]
    agent.update_q_value(state, 0, 1.0)
    agent.update_q_value(state, 1, 0.5)
    agent.update_q_value(state, 2, 0.2)
    assert agent.choose_action(state, available_moves) == 0


def test_agent_policy_save_load(tmp_path):
    """Test policy saving and loading."""
    agent = LearningAgent()
    state = tuple([" "] * 9)
    action = 0
    agent.update_q_value(state, action, 1.0)

    # Use tmp_path fixture for temporary file
    policy_file = tmp_path / "test_policy.json"
    agent.save_policy(str(policy_file))

    new_agent = LearningAgent(policy_file=str(policy_file))
    assert new_agent.get_q_value(state, action) == 1.0


# Human Player Tests
@patch("builtins.input", return_value="4")
def test_human_valid_move(mock_input):
    """Test valid human move."""
    player = HumanPlayer()
    state = tuple([" "] * 9)
    available_moves = [4, 5, 6]
    assert player.choose_action(state, available_moves) == 4


@patch("builtins.input", side_effect=["9", "abc", "5"])
def test_human_invalid_then_valid_move(mock_input):
    """Test invalid moves followed by valid move."""
    player = HumanPlayer()
    state = tuple([" "] * 9)
    available_moves = [4, 5, 6]
    assert player.choose_action(state, available_moves) == 5


# Game Play Tests
def test_full_game_play():
    """Test complete game between two agents."""
    agent1 = LearningAgent(epsilon=0)
    agent2 = LearningAgent(epsilon=0)
    result = play_game(agent1, agent2)
    assert result in ["X", "O", "draw"]


@patch("builtins.input", side_effect=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
def test_play_against_ai(mock_input):
    """Test game against AI.

    Provides a sequence of all possible moves to ensure the game can complete
    regardless of the AI's choices. The game will use moves in sequence until
    a game-ending condition is reached.
    """
    ai_agent = LearningAgent(epsilon=0)
    result = play_against_ai(ai_agent, human_plays_first=True)
    assert result in ["X", "O", "draw"]


@patch("builtins.input", side_effect=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
def test_play_against_ai_ai_first(mock_input):
    """Test game against AI where AI plays first.

    Similar to test_play_against_ai but with AI making the first move.
    """
    ai_agent = LearningAgent(epsilon=0)
    result = play_against_ai(ai_agent, human_plays_first=False)
    assert result in ["X", "O", "draw"]


def test_agent_learning():
    """Test that agent actually learns from experience."""
    agent = LearningAgent(alpha=1.0, gamma=1.0, epsilon=0)
    state = tuple([" "] * 9)
    action = 0
    next_state = ("X", " ", " ", " ", " ", " ", " ", " ", " ")

    # Test winning scenario
    agent.learn(state, action, 1.0, next_state, [])
    assert agent.get_q_value(state, action) == 1.0

    # Test losing scenario
    agent.learn(state, action, -1.0, next_state, [])
    assert agent.get_q_value(state, action) == -1.0


@pytest.mark.parametrize("human_plays_first", [True, False])
def test_play_against_ai_with_draw(human_plays_first):
    """Test game ending in a draw."""
    moves = [0, 4, 8, 2, 6, 3, 5, 1, 7]  # Force a draw
    with patch("builtins.input", side_effect=moves):
        ai_agent = LearningAgent(epsilon=0)
        result = play_against_ai(ai_agent, human_plays_first=human_plays_first)
        assert result == "draw"


def test_main_function():
    """Test the main training loop."""
    with patch("random.random", side_effect=[0.4, 0.6] * 50):  # Alternate between agents
        with patch("tictactoe.play_game") as mock_play:
            # Simulate alternating wins and draws
            mock_play.side_effect = ["X", "O", "draw"] * 33334

            # Run main with reduced episodes for testing
            from tictactoe import main

            main()


def test_game_str_representation(game):
    """Test the string representation of the game board."""
    game.make_move(0)  # X in position 0
    game.make_move(4)  # O in position 4
    expected = " X | |  \n-----\n  |O|  \n-----\n  | |  "
    assert str(game).replace("\n", "") == expected.replace("\n", "")


def test_display_board():
    """Test the human player's board display."""
    player = HumanPlayer()
    state = ("X", "O", " ", " ", "X", " ", "O", " ", " ")
    with patch("builtins.print") as mock_print:
        player.display_board(state)
        mock_print.assert_called()
