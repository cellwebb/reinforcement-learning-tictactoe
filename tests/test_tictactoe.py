import unittest
from unittest.mock import patch
from tictactoe import TicTacToe, LearningAgent, HumanPlayer, play_game, play_against_ai
import os


class TestTicTacToe(unittest.TestCase):

    def test_initial_state(self):
        """Test that the game board is initialized correctly."""
        game = TicTacToe()
        self.assertEqual(game.board, [" "] * 9)
        self.assertEqual(game.current_player, "X")

    def test_get_available_moves(self):
        """Test that the available moves are returned correctly."""
        game = TicTacToe()
        self.assertEqual(game.get_available_moves(), list(range(9)))
        game.make_move(0)
        self.assertEqual(game.get_available_moves(), list(range(1, 9)))

    def test_make_move(self):
        """Test that moves are made correctly and that invalid moves raise an error."""
        game = TicTacToe()
        game.make_move(0)
        self.assertEqual(game.board[0], "X")
        self.assertEqual(game.current_player, "O")
        with self.assertRaises(ValueError):
            game.make_move(0)

    def test_is_winner(self):
        """Test that the game correctly identifies a winner."""
        game = TicTacToe()
        game.board = ["X", "X", "X", " ", " ", " ", " ", " ", " "]
        self.assertTrue(game.is_winner("X"))
        self.assertFalse(game.is_winner("O"))

    def test_is_draw(self):
        """Test that the game correctly identifies a draw."""
        game = TicTacToe()
        game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertTrue(game.is_draw())
        game.board[0] = " "
        self.assertFalse(game.is_draw())

    def test_get_state(self):
        """Test that the game state is returned correctly."""
        game = TicTacToe()
        self.assertEqual(game.get_state(), tuple([" "] * 9))
        game.make_move(0)
        self.assertEqual(game.get_state(), ("X", " ", " ", " ", " ", " ", " ", " ", " "))


class TestLearningAgent(unittest.TestCase):

    def test_initialization(self):
        """Test that the agent is initialized with the correct parameters."""
        agent = LearningAgent()
        self.assertEqual(agent.alpha, 0.1)
        self.assertEqual(agent.gamma, 1.0)
        self.assertEqual(agent.epsilon, 0.1)
        self.assertEqual(agent.q_table, {})

    def test_get_q_value(self):
        """Test that the Q-value is returned correctly."""
        agent = LearningAgent()
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        action = 0
        self.assertEqual(agent.get_q_value(state, action), 0.0)
        agent.update_q_value(state, action, 1.0)
        self.assertEqual(agent.get_q_value(state, action), 1.0)

    def test_choose_action(self):
        """Test that the agent chooses the action with the highest Q-value."""
        agent = LearningAgent(epsilon=0)
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        available_moves = [0, 1, 2]
        agent.update_q_value(state, 0, 1.0)
        agent.update_q_value(state, 1, 0.5)
        agent.update_q_value(state, 2, 0.2)
        self.assertEqual(agent.choose_action(state, available_moves), 0)

    def test_learn(self):
        """Test that the agent learns from the environment correctly."""
        agent = LearningAgent()
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        action = 0
        reward = 1.0
        next_state = ("X", " ", " ", " ", " ", " ", " ", " ", " ")
        next_available_moves = [1, 2, 3, 4, 5, 6, 7, 8]
        agent.learn(state, action, reward, next_state, next_available_moves)
        self.assertNotEqual(agent.get_q_value(state, action), 0.0)

    def test_save_and_load_policy(self):
        """Test that the agent can save and load the policy."""
        agent = LearningAgent()
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        action = 0
        agent.update_q_value(state, action, 1.0)
        agent.save_policy("test_policy.json")
        new_agent = LearningAgent(policy_file="test_policy.json")
        self.assertEqual(new_agent.get_q_value(state, action), 1.0)
        os.remove("test_policy.json")

    def test_learn_with_empty_next_moves(self):
        """Test learning with empty next_available_moves."""
        agent = LearningAgent()
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        action = 0
        reward = 1.0
        next_state = ("X", " ", " ", " ", " ", " ", " ", " ", " ")
        # Test learning with empty next_available_moves
        agent.learn(state, action, reward, next_state, [])
        self.assertEqual(agent.get_q_value(state, action), agent.alpha * reward)

    def test_complex_serialization(self):
        """Test that complex state is preserved when saving and loading."""
        agent = LearningAgent()

        state = ("X", "O", " ", "X", " ", " ", "O", " ", " ")
        action = 4
        agent.update_q_value(state, action, 0.75)
        agent.save_policy("test_policy.json")

        new_agent = LearningAgent(policy_file="test_policy.json")
        self.assertEqual(new_agent.get_q_value(state, action), 0.75)

        os.remove("test_policy.json")


class TestHumanPlayer(unittest.TestCase):
    def setUp(self):
        self.player = HumanPlayer()

    @patch("builtins.input", return_value="4")
    def test_choose_action_valid(self, mock_input):
        """Test that the human player chooses a valid action."""
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        available_moves = [4, 5, 6]
        self.assertEqual(self.player.choose_action(state, available_moves), 4)

    @patch("builtins.input", side_effect=["9", "abc", "5"])
    def test_choose_action_invalid_then_valid(self, mock_input):
        """Test that the human player chooses a valid action after an invalid one."""
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        available_moves = [4, 5, 6]
        self.assertEqual(self.player.choose_action(state, available_moves), 5)


class TestGameplay(unittest.TestCase):
    def test_play_game(self):
        """Test that the game is played to completion."""
        agent1 = LearningAgent(epsilon=0)
        agent2 = LearningAgent(epsilon=0)
        result = play_game(agent1, agent2)
        self.assertIn(result, ["X", "O", "draw"])

    @patch("builtins.input", return_value="4")
    def test_play_against_ai(self, mock_input):
        """Test that the game is played to completion against the AI agent."""
        ai_agent = LearningAgent(epsilon=0)
        result = play_against_ai(ai_agent, human_plays_first=True)
        self.assertIn(result, ["X", "O", "draw"])


if __name__ == "__main__":
    unittest.main()
