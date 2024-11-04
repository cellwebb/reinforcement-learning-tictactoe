from tictactoe import LearningAgent, play_game, train, play_against_ai
from pprint import pprint

teddy = LearningAgent()
dummy = LearningAgent()

train(agent1=teddy, agent2=dummy, num_episodes=100000)
# play_game(teddy, teddy)

print("-" * 80)
pprint(teddy.q_table)
print("-" * 80)

# play_game(teddy, teddy)

# print("-" * 80)
# pprint(teddy.q_table)
# print("-" * 80)

# play_game(teddy, teddy)

# print("-" * 80)
# pprint(teddy.q_table)
# print("-" * 80)

# play_game(teddy, teddy)

# print("-" * 80)
# pprint(teddy.q_table)
# print("-" * 80)

# train(agent1=teddy, num_episodes=100000, single_agent_training=True)

# print("-" * 80)
# pprint(teddy.q_table)
# print("-" * 80)

# # print the q table where state is "         "
# state = " " * 9
# for action in range(9):
#     print(f"{action}: {teddy.q_table.get((state, action), None)}")

# play_game(teddy, teddy)

# print("-" * 80)
# pprint(teddy.q_table)
# print("-" * 80)

state = " " * 9
for action in range(9):
    print(f"{action}: {teddy.q_table.get(state, {}).get(action, None)}")

print("-" * 80)
print(f"teddy's epsilon: {teddy.epsilon}")
print("-" * 80)

# play_against_ai(dummy, human_plays_first=True)
play_against_ai(teddy, human_plays_first=True)
play_against_ai(teddy, human_plays_first=False)
