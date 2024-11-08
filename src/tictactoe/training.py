from .agents import LearningAgent
from .game import play_game
import random
from tqdm import tqdm


def train(
    agent1: LearningAgent = None,
    agent2: LearningAgent = None,
    num_episodes: int = 1000,
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

    starting_alpha = (agent1.alpha, agent2.alpha)
    min_alpha = (agent1.min_alpha, agent2.min_alpha)
    starting_epsilon = (agent1.epsilon, agent2.epsilon)
    min_epsilon = (agent1.min_epsilon, agent2.min_epsilon)

    alpha_decay = [0, 0]
    epsilon_decay = [0, 0]
    for i in range(2):
        alpha_decay[i] = pow(min_alpha[i] / starting_alpha[i], 1.0 / num_episodes)
        epsilon_decay[i] = (starting_epsilon[i] - min_epsilon[i]) / num_episodes

    results = {"X": 0, "O": 0, "draw": 0}
    wins = {"Agent 1": 0, "Agent 2": 0, "draw": 0}

    for i in tqdm(range(num_episodes), desc="Training Progress"):
        if single_agent_training or (switch_sides and random.random() < 0.5):
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

        agent1.alpha = max(agent1.min_alpha, agent1.alpha * alpha_decay[0])
        agent2.alpha = max(agent2.min_alpha, agent2.alpha * alpha_decay[1])
        agent1.epsilon = max(agent1.min_epsilon, agent1.epsilon - epsilon_decay[0])
        agent2.epsilon = max(agent2.min_epsilon, agent2.epsilon - epsilon_decay[1])

    print(f"Results: {results}")
    if not single_agent_training:
        print(f"Wins: {wins}")

    if agent1_policy_file:
        agent1.save_policy(agent1_policy_file)
    if agent2_policy_file and not single_agent_training:
        agent2.save_policy(agent2_policy_file)
