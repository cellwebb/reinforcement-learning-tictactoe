from .agents import LearningAgent
from .game import play_game
import random


def train(
    agent1: LearningAgent = None,
    agent2: LearningAgent = None,
    num_episodes: int = 100,
    single_agent_training: bool = False,
    agent1_policy_file: str = None,
    agent2_policy_file: str = None,
    switch_sides: bool = True,
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
        if switch_sides or random.random() < 0.5:
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
