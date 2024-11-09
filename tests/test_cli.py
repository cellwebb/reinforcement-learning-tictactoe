from itertools import cycle
from unittest.mock import patch
from tictactoe import cli, train, LearningAgent


def test_train_function():
    """Test the main training loop."""
    with patch("tictactoe.agents.random.random", side_effect=cycle([0.4, 0.6])):
        with patch("tictactoe.play_game") as mock_play:
            # Simulate alternating wins and draws
            mock_play.side_effect = ["X", "O", "draw"] * 33334

            # Run main with reduced episodes for testing
            train()  # Should now work correctly


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
