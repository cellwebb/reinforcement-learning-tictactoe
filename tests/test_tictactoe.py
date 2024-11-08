import pytest
from unittest.mock import patch
from tictactoe import (
    TicTacToe,
    LearningAgent,
    HumanPlayer,
    play_game,
    play_against_ai,
    get_available_moves,
    is_winner,
    is_draw,
    train,  # Now accessible directly from tictactoe
    cli,  # Now accessible directly from tictactoe
)
import itertools  # Add this import


def test_initial_state(game):
    """Test that the game board is initialized correctly."""
    assert game.board == [" "] * 9
    assert game.current_player == "X"


def test_get_available_moves(game):
    """Test that the available moves are returned correctly."""
    assert list(get_available_moves(game.get_state())) == list(range(9))
    game.make_move(0)
    assert list(get_available_moves(game.get_state())) == list(range(1, 9))


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


# Learning Agent Tests
def test_agent_initialization(agent):
    """Test that the agent is initialized with correct parameters."""
    assert 0 <= agent.alpha <= 1
    assert 0 <= agent.gamma <= 1
    assert 0 <= agent.epsilon <= 1
    assert agent.q_table == {}
    assert isinstance(agent.starting_q_value, (int, float))
    assert isinstance(agent.win_reward, (int, float))
    assert isinstance(agent.draw_reward, (int, float))
    assert isinstance(agent.loss_reward, (int, float))


def test_agent_get_q_value(agent):
    """Test Q-value retrieval."""
    state = "         "
    action = 0
    assert agent.get_q_value(state, action) == 0.0
    agent.update_q_value(state, action, 0.5)
    assert agent.get_q_value(state, action) == 0.5


def test_agent_choose_action_exploitation():
    """Test that agent chooses best action when epsilon=0."""
    agent = LearningAgent(epsilon=0)
    state = "         "
    available_moves = [0, 1, 2]
    agent.update_q_value(state, 0, 1.0)
    agent.update_q_value(state, 1, 0.5)
    agent.update_q_value(state, 2, 0.2)
    assert agent.choose_action(state, available_moves) == 0


def test_agent_policy_save_load(tmp_path):
    """Test policy saving and loading."""
    agent = LearningAgent()
    state = " " * 9
    action = 0
    agent.q_table[state] = {action: 1.0}

    # Use tmp_path fixture for temporary file
    policy_file = tmp_path / "test_policy.json"
    agent.save_policy(str(policy_file))

    # Load policy into a new agent
    new_agent = LearningAgent(policy_infile=str(policy_file))
    assert new_agent.get_q_value(state, action) == 1.0


# Human Player Tests
@patch("builtins.input", return_value="4")
def test_human_valid_move(mock_input):
    """Test valid human move."""
    player = HumanPlayer()
    state = "         "
    available_moves = [4, 5, 6]
    assert player.choose_action(state, available_moves) == 4


@patch("builtins.input", side_effect=["9", "abc", "5"])
def test_human_invalid_then_valid_move(mock_input):
    """Test invalid moves followed by valid move."""
    player = HumanPlayer()
    state = "         "
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


def test_train_function():
    """Test the main training loop."""
    with patch(
        "tictactoe.agents.random.random", side_effect=itertools.cycle([0.4, 0.6])
    ):  # Use correct patch path
        with patch("tictactoe.play_game") as mock_play:
            # Simulate alternating wins and draws
            mock_play.side_effect = ["X", "O", "draw"] * 33334

            # Run main with reduced episodes for testing
            train()  # Should now work correctly


def test_cached_get_available_moves():
    """Test the cached get_available_moves function."""
    # Test with empty board
    empty_board = "         "
    assert get_available_moves(empty_board) == tuple(range(9))

    # Test cache hit (same result object)
    result1 = get_available_moves(empty_board)
    result2 = get_available_moves(empty_board)
    assert result1 is result2

    # Test with partially filled board
    partial_board = "XO  X O  "
    assert get_available_moves(partial_board) == (2, 3, 5, 7, 8)

    # Test full board
    full_board = "XOXOXOXOX"
    assert get_available_moves(full_board) == tuple()

    # Test single move board
    single_move = "X        "
    assert len(get_available_moves(single_move)) == 8


def test_str_representation():
    """Test string representation functions."""
    # Test board state formatting
    board = list("XO  X O  ")
    game = TicTacToe()
    game.board = board

    # Check board display format
    expected = "\n\n X | O |   \n-----------\n   | X |   \n-----------\n O |   |   \n\n"
    assert str(game) == expected


def test_cached_is_winner():
    """Test the cached is_winner function."""
    empty_board = "         "
    assert not is_winner(empty_board, "X")

    winning_board = "XXX      "
    assert is_winner(winning_board, "X")

    # Test cache hit (same result object)
    result1 = is_winner(winning_board, "X")
    result2 = is_winner(winning_board, "X")
    assert result1 is result2


def test_cached_is_draw():
    """Test the cached is_draw function."""
    empty_board = "         "
    assert not is_draw(empty_board)

    draw_board = "XOXOXOXOX"
    assert is_draw(draw_board)

    # Test cache hit (same result object)
    result1 = is_draw(draw_board)
    result2 = is_draw(draw_board)
    assert result1 is result2


def test_learning_agent_str_loading():
    """Test loading agent from non-existent file gracefully fails."""
    with pytest.raises(FileNotFoundError):
        # Replace 'policy_file' with 'policy_infile'
        LearningAgent(policy_infile="nonexistent.json")


def test_human_player_type():
    """Test human player type attribute."""
    player = HumanPlayer()
    assert player.player_type == "human"


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


def test_cli_train(tmp_path, monkeypatch, capsys):
    """Test CLI training mode."""
    import sys
    import json

    # Define paths for outfile policy files within tmp_path
    agent1_outfile = tmp_path / "agent1.json"
    agent2_outfile = tmp_path / "agent2.json"

    # Create a temporary config.yaml with desired parameters
    config_content = f"""
num_episodes: 10

single_agent_training: false  # If true, only agent1 will be trained

# If policy_infile is not empty, the agent will load the Q-table from the file
# If policy_outfile is not empty, the agent will save the Q-table to the file
agents:
  agent1:
    policy_infile: ""
    policy_outfile: "{agent1_outfile}"
    alpha: 0.3
    gamma: 0.9
    epsilon: 0.05
  agent2:
    policy_infile: ""
    policy_outfile: "{agent2_outfile}"
    alpha: 0.2
    gamma: 0.8
    epsilon: 0.2
"""

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    # Mock argv with the 'train' command and '--config' argument
    test_args = ["tictactoe", "train", f"--config={config_file}"]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run CLI
    cli()  # Should now work correctly

    # Capture and assert output
    captured = capsys.readouterr()
    assert "Results:" in captured.out
    assert "Wins:" in captured.out

    # Assert that policy files are created
    assert agent1_outfile.exists()
    assert agent2_outfile.exists()

    # Optionally, check that the content is a valid Q-table (e.g., non-empty)
    with open(agent1_outfile, "r") as f:
        q_table1 = json.load(f)
        assert isinstance(q_table1, dict)
        assert len(q_table1) > 0  # Ensure Q-table is not empty

    with open(agent2_outfile, "r") as f:
        q_table2 = json.load(f)
        assert isinstance(q_table2, dict)
        assert len(q_table2) > 0  # Ensure Q-table is not empty


def test_cli_play_nonexistent_policy(capsys):
    """Test CLI play mode with nonexistent policy file."""
    import sys

    # Mock argv with the 'play' command and a nonexistent policy file
    with patch.object(sys, "argv", ["tictactoe", "play", "--policy=nonexistent.json"]):
        cli()  # Should now work correctly
        captured = capsys.readouterr()
        assert "Error: Policy file 'nonexistent.json' not found" in captured.out


def test_cli_play_valid(tmp_path, random_move_generator):
    """Test CLI play mode with valid policy using a random move generator."""
    import sys

    # Create a temporary policy file
    agent = LearningAgent()
    policy_file = tmp_path / "test_policy.json"
    agent.save_policy(str(policy_file))

    # Mock input for gameplay with the generator fixture and suppress print
    with patch("builtins.input", side_effect=random_move_generator), patch("builtins.print"):
        with patch.object(sys, "argv", ["tictactoe", "play", f"--policy={policy_file}"]):
            cli()  # Should now work correctly
