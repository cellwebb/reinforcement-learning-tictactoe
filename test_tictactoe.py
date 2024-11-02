import unittest
from tictactoe import TicTacToe, LearningAgent, play_game


class TestTicTacToe(unittest.TestCase):

    def test_initial_state(self):
        game = TicTacToe()
        self.assertEqual(game.board, [" "] * 9)
        self.assertEqual(game.current_player, "X")

    def test_get_available_moves(self):
        game = TicTacToe()
        self.assertEqual(game.get_available_moves(), list(range(9)))
        game.make_move(0)
        self.assertEqual(game.get_available_moves(), list(range(1, 9)))

    def test_make_move(self):
        game = TicTacToe()
        game.make_move(0)
        self.assertEqual(game.board[0], "X")
        self.assertEqual(game.current_player, "O")
        with self.assertRaises(ValueError):
            game.make_move(0)

    def test_is_winner(self):
        game = TicTacToe()
        game.board = ["X", "X", "X", " ", " ", " ", " ", " ", " "]
        self.assertTrue(game.is_winner("X"))
        self.assertFalse(game.is_winner("O"))

    def test_is_draw(self):
        game = TicTacToe()
        game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertTrue(game.is_draw())
        game.board[0] = " "
        self.assertFalse(game.is_draw())

    def test_get_state(self):
        game = TicTacToe()
        self.assertEqual(game.get_state(), tuple([" "] * 9))
        game.make_move(0)
        self.assertEqual(game.get_state(), ("X", " ", " ", " ", " ", " ", " ", " ", " "))


class TestLearningAgent(unittest.TestCase):

    def test_initialization(self):
        agent = LearningAgent()
        self.assertEqual(agent.alpha, 0.1)
        self.assertEqual(agent.gamma, 1.0)
        self.assertEqual(agent.epsilon, 0.1)
        self.assertEqual(agent.q_table, {})

    def test_get_q_value(self):
        agent = LearningAgent()
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        action = 0
        self.assertEqual(agent.get_q_value(state, action), 0.0)
        agent.update_q_value(state, action, 1.0)
        self.assertEqual(agent.get_q_value(state, action), 1.0)

    def test_choose_action(self):
        agent = LearningAgent(epsilon=0)
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        available_moves = [0, 1, 2]
        agent.update_q_value(state, 0, 1.0)
        agent.update_q_value(state, 1, 0.5)
        agent.update_q_value(state, 2, 0.2)
        self.assertEqual(agent.choose_action(state, available_moves), 0)

    def test_learn(self):
        agent = LearningAgent()
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        action = 0
        reward = 1.0
        next_state = ("X", " ", " ", " ", " ", " ", " ", " ", " ")
        next_available_moves = [1, 2, 3, 4, 5, 6, 7, 8]
        agent.learn(state, action, reward, next_state, next_available_moves)
        self.assertNotEqual(agent.get_q_value(state, action), 0.0)

    def test_save_and_load_policy(self):
        agent = LearningAgent()
        state = (" ", " ", " ", " ", " ", " ", " ", " ", " ")
        action = 0
        agent.update_q_value(state, action, 1.0)
        agent.save_policy("test_policy.json")
        new_agent = LearningAgent(policy_file="test_policy.json")
        self.assertEqual(new_agent.get_q_value(state, action), 1.0)


class TestPlayGame(unittest.TestCase):

    def test_play_game(self):
        agent1 = LearningAgent()
        agent2 = LearningAgent()
        result = play_game(agent1, agent2)
        self.assertIn(result, ["X", "O", "draw"])


if __name__ == "__main__":
    unittest.main()
