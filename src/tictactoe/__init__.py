import random
import json
import yaml
from functools import lru_cache

WIN_CONDITIONS = [
    [0, 1, 2],  # Horizontal
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],  # Vertical
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],  # Diagonal
    [2, 4, 6],
]


@lru_cache(maxsize=19683)  # 3^9 possible board states
def get_available_moves(state: str) -> tuple[int, ...]:
    """Get available moves for a given board state."""
    return tuple(i for i, mark in enumerate(state) if mark == " ")


@lru_cache(maxsize=19683)
def is_winner(state: str, player: str) -> bool:
    """Check if the given player has won."""
    return any(all(state[i] == player for i in condition) for condition in WIN_CONDITIONS)


def is_draw(state: str) -> bool:
    """Check if the game is a draw."""
    return " " not in state


class TicTacToe:
    """Tic-Tac-Toe game environment."""

    def __init__(self, starting_player: str = "X"):
        self.board = [" " for _ in range(9)]
        self.current_player = starting_player
        self.state_history = [self.get_state()]
        self.move_history = []

    def make_move(self, position: int) -> None:
        if self.board[position] != " ":
            raise ValueError("Invalid move")
        self.board[position] = self.current_player
        self.current_player = "O" if self.current_player == "X" else "X"
        self.move_history.append(position)
        self.state_history.append(self.get_state())

    def get_state(self) -> tuple[str]:
        return "".join(self.board)

    def __str__(self):
        return (
            "\n\n"
            f" {self.board[0]} | {self.board[1]} | {self.board[2]} \n"
            "-----------\n"
            f" {self.board[3]} | {self.board[4]} | {self.board[5]} \n"
            "-----------\n"
            f" {self.board[6]} | {self.board[7]} | {self.board[8]} \n\n"
        )


class LearningAgent:
    """Q-Learning agent for playing Tic-Tac-Toe."""

    def __init__(
        self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1, policy_file: str = None
    ):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

        if policy_file:
            self.load_policy(policy_file)
        else:
            self.q_table = {}

        self.player_type = "agent"

    def get_q_value(self, state: str, action: int) -> float:
        return self.q_table.get((state, action), 1.0)

    def update_q_value(self, state: str, action: int, value: float) -> None:
        self.q_table[(state, action)] = value

    def choose_action(self, state: str, available_moves: tuple[int]) -> int:
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        else:
            q_values = {action: self.get_q_value(state, action) for action in available_moves}
            max_q = max(q_values.values())
            return random.choice([action for action, q in q_values.items() if q == max_q])

    def learn(
        self,
        state: str,
        action: int,
        reward: float,
        next_state: str | None = None,
        next_available_moves: tuple[int, ...] | None = None,
    ) -> None:
        """Update Q-value based on reward and learned value."""
        old_q = self.get_q_value(state, action)
        if next_available_moves:
            future_rewards = [
                self.get_q_value(next_state, next_action) for next_action in next_available_moves
            ]
            learned_value = reward + self.gamma * max(future_rewards)
        else:
            learned_value = reward
        self.update_q_value(state, action, old_q + self.alpha * (learned_value - old_q))

    def save_policy(self, filename: str) -> None:
        serialized_q_table = {
            self._serialize_key(state, action): value
            for (state, action), value in self.q_table.items()
        }
        with open(filename, "w") as f:
            json.dump(serialized_q_table, f)

    def load_policy(self, filename: str) -> None:
        with open(filename, "r") as f:
            serialized_q_table = json.load(f)
            self.q_table = {
                self._deserialize_key(key): value for key, value in serialized_q_table.items()
            }

    def _serialize_key(self, state: str, action: int) -> str:
        """Convert a state-action pair to a string key."""
        return f"{state}_{action}"

    def _deserialize_key(self, key: str) -> tuple[str, int]:
        """Convert a string key back to state-action pair."""
        state, action = key.rsplit("_", 1)
        return state, int(action)


class HumanPlayer:
    """Human player for playing Tic-Tac-Toe."""

    def __init__(self):
        self.player_type = "human"

    def choose_action(self, state: str, available_moves: list[int]) -> int:
        """Get move from human player via console input."""

        while True:
            try:
                print(f"Available moves (0-8): {available_moves}")
                move = int(input("Enter your move: "))
                if move in available_moves:
                    return move
                print("Invalid move, try again")
            except ValueError:
                print("Please enter an available move")


def play_game(player1: LearningAgent | HumanPlayer, player2: LearningAgent | HumanPlayer) -> str:
    """Play a game of Tic-Tac-Toe between two agents, two humans, or a human and an agent."""

    env = TicTacToe()

    players = {"X": player1, "O": player2}

    human_in_game = "human" in [player1.player_type, player2.player_type]
    if human_in_game:
        print(env)

    while True:
        player = env.current_player
        opponent = "O" if player == "X" else "X"

        state = env.get_state()
        available_moves = get_available_moves(state)
        action = players[player].choose_action(state, available_moves)
        env.make_move(action)

        if human_in_game:
            print(env)

        new_state = env.get_state()

        if is_winner(new_state, player):
            if players[player].player_type == "agent":
                players[player].update_q_value(state, action, 1.0)
            if players[opponent].player_type == "agent":
                players[opponent].update_q_value(env.state_history[-3], env.move_history[-2], -1.0)
            return player

        if is_draw(new_state):
            if players[player].player_type == "agent":
                players[player].update_q_value(state, action, 0.5)
            if players[opponent].player_type == "agent":
                players[opponent].update_q_value(env.state_history[-3], env.move_history[-2], 0.5)
            return "draw"

        if players[player].player_type == "agent":
            players[player].learn(state, action, 0, new_state, get_available_moves(new_state))


def play_against_ai(ai_agent, human_plays_first: bool = True) -> None:
    """Play a game against an AI agent."""
    human = HumanPlayer()

    stored_epsilon = ai_agent.epsilon
    ai_agent.epsilon = 0

    print("Game starting! Positions are numbered 0-8, left to right, top to bottom")

    if human_plays_first:
        result = play_game(human, ai_agent)
    else:
        result = play_game(ai_agent, human)

    if result == "draw":
        print("It's a draw!")
    elif (result == "X" and human_plays_first) or (result == "O" and not human_plays_first):
        print("You win!")
    else:
        print("AI wins!")

    ai_agent.epsilon = stored_epsilon

    return result


def train(
    agent1: LearningAgent = None,
    agent2: LearningAgent = None,
    num_episodes: int = 100,
    single_agent_training: bool = False,
    agent1_policy_file: str = None,
    agent2_policy_file: str = None,
) -> None:
    """Train two agents to play Tic-Tac-Toe against each other."""

    if not agent1:
        agent1 = LearningAgent()

    if single_agent_training:
        agent2 = agent1
    elif not agent2:
        agent2 = LearningAgent()

    results = {"X": 0, "O": 0, "draw": 0}
    wins = {"Agent 1": 0, "Agent 2": 0, "draw": 0}

    for _ in range(num_episodes):
        if random.random() < 0.5:
            result = play_game(agent1, agent2)

            if result == "X":
                wins["Agent 1"] += 1
            elif result == "O":
                wins["Agent 2"] += 1
            else:
                wins["draw"] += 1
        else:
            result = play_game(agent2, agent1)

            if result == "X":
                wins["Agent 2"] += 1
            elif result == "O":
                wins["Agent 1"] += 1
            else:
                wins["draw"] += 1

        results[result] += 1

    print(f"Results: {results}")
    print(f"Wins: {wins}")

    if agent1_policy_file:
        agent1.save_policy(agent1_policy_file)
    if agent2_policy_file:
        agent2.save_policy(agent2_policy_file)


def cli():
    """Command line interface for the Tic-Tac-Toe game."""
    import argparse

    parser = argparse.ArgumentParser(description="Tic-Tac-Toe with Q-Learning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for the 'train' command
    train_parser = subparsers.add_parser("train", help="Train the AI agent")
    train_parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of training episodes"
    )
    train_parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    train_parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    train_parser.add_argument(
        "--single-agent-training", action="store_true", help="Train a single agent"
    )
    train_parser.add_argument("--agent1", type=str, help="Policy file for agent 1")
    train_parser.add_argument("--agent2", type=str, help="Policy file for agent 2")
    train_parser.add_argument("--config", type=str, help="Path to configuration file")

    # Subparser for the 'play' command
    play_parser = subparsers.add_parser("play", help="Play against the AI agent")
    play_parser.add_argument("--policy", type=str, required=True, help="Policy file for loading")
    play_parser.add_argument("--ai-first", action="store_true", help="AI plays first")

    args = parser.parse_args()

    if args.command == "train":
        if args.config:

            with open(args.config, "r") as f:
                config = yaml.safe_load(f)

            args.num_episodes = config.get("num_episodes", args.num_episodes)
            args.alpha = config.get("alpha", args.alpha)
            args.gamma = config.get("gamma", args.gamma)
            args.epsilon = config.get("epsilon", args.epsilon)
            args.single_agent_training = config.get(
                "single_agent_training", args.single_agent_training
            )
            args.agent1 = config.get("agent1", args.agent1)
            args.agent2 = config.get("agent2", args.agent2)

        agent1 = LearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
        if args.single_agent_training:
            agent2 = agent1
        else:
            agent2 = LearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
        train(
            agent1=agent1,
            agent2=agent2,
            num_episodes=args.num_episodes,
            agent1_policy_file=args.agent1,
            agent2_policy_file=args.agent2,
        )
    elif args.command == "play":
        try:
            agent = LearningAgent(policy_file=args.policy)
            play_against_ai(agent, human_plays_first=not args.ai_first)
        except FileNotFoundError:
            print(f"Error: Policy file '{args.policy}' not found")


if __name__ == "__main__":
    cli()
