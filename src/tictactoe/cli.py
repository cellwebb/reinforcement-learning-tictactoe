import argparse
import yaml
from .agents import LearningAgent
from .game import play_against_ai
from .training import train as train_game  # Updated import


def cli():
    """Command line interface for the Tic-Tac-Toe game."""
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe with Q-Learning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for the 'train' command
    train_parser = subparsers.add_parser("train", help="Train the AI agent")
    train_parser.add_argument(
        "--config", type=str, help="Path to configuration file", default="config.yaml"
    )

    # Subparser for the 'play' command
    play_parser = subparsers.add_parser("play", help="Play against the AI agent")
    play_parser.add_argument("--policy", type=str, required=True, help="Policy file for loading")
    play_parser.add_argument("--ai-first", action="store_true", help="AI plays first")

    args = parser.parse_args()

    if args.command == "train":
        if args.config:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)

            num_episodes = config.get("num_episodes", 100)
            single_agent_training = config.get("single_agent_training", False)
            agents_config = config.get("agents", {})

            agent1_config = agents_config.get("agent1", {})
            agent2_config = agents_config.get("agent2", {})

        # Initialize Agent 1
        agent1 = LearningAgent(
            alpha=agent1_config.get("alpha", 0.1),
            gamma=agent1_config.get("gamma", 0.9),
            epsilon=agent1_config.get("epsilon", 0.1),
            policy_infile=agent1_config.get("policy_infile"),
            policy_outfile=agent1_config.get("policy_outfile"),
        )

        # Initialize Agent 2
        if single_agent_training:
            agent2 = agent1
        else:
            agent2 = LearningAgent(
                alpha=agent2_config.get("alpha", 0.1),
                gamma=agent2_config.get("gamma", 0.9),
                epsilon=agent2_config.get("epsilon", 0.1),
                policy_infile=agent2_config.get("policy_infile"),
                policy_outfile=agent2_config.get("policy_outfile"),
            )

        # Start Training
        train_game(
            agent1=agent1,
            agent2=agent2,
            num_episodes=num_episodes,
            agent1_policy_file=agent1_config.get("policy_outfile"),
            agent2_policy_file=agent2_config.get("policy_outfile"),
            switch_sides=True,
        )

    elif args.command == "play":
        try:
            agent = LearningAgent(
                policy_infile=args.policy,
                policy_outfile=None,  # Outfile not needed when playing
            )
            play_against_ai(agent, human_plays_first=not args.ai_first)
        except FileNotFoundError:
            print(f"Error: Policy file '{args.policy}' not found")


if __name__ == "__main__":
    cli()
