import pytest
from unittest.mock import patch
from tictactoe import LearningAgent, HumanPlayer


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


def test_learning_agent_str_loading():
    """Test loading agent from non-existent file gracefully fails."""
    with pytest.raises(FileNotFoundError):
        LearningAgent(policy_infile="nonexistent.json")


def test_human_player_type():
    """Test human player type attribute."""
    player = HumanPlayer()
    assert player.player_type == "human"
