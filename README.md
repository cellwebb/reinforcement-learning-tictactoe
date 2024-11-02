# Reinforcement Learning Tic-Tac-Toe

A Python implementation of a Tic-Tac-Toe AI that learns optimal play through Q-learning, a model-free reinforcement learning algorithm.

## Features

- Q-learning agent that improves through self-play
- Human vs AI gameplay mode
- Save and load trained policies
- Configurable learning parameters (alpha, gamma, epsilon)

## Requirements

- Python 3.11+

## Installation

```bash
git clone https://github.com/yourusername/reinforcement-learning-tictactoe.git
cd reinforcement-learning-tictactoe
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Usage

### Training the AI

```bash
python3 tictactoe.py
```

This will train two agents through self-play for 1,000,000 episodes and save their policies to agent1.json and agent2.json.

### Playing Against the AI

```py
from tictactoe import LearningAgent, play_against_ai

# Load a trained agent
ai_agent = LearningAgent(policy_file="agent1.json")

# Play against the AI (True means human plays first)
play_against_ai(ai_agent, human_plays_first=True)
```

## Implementation Details

### Game Board Layout

The game board positions are numbered 0-8, arranged as follows:

```text
 0 | 1 | 2 
-----------
 3 | 4 | 5 
-----------
 6 | 7 | 8 
```

### Q-Learning Parameters

- Learning rate (alpha): 0.1
  - Controls how much new information overrides old information
- Discount factor (gamma): 1.0
  - Determines the importance of future rewards
- Exploration rate (epsilon): 0.1
  - Controls the exploration vs exploitation tradeoff

### Reward Structure

- Win: +1
- Loss: -1
- Draw: +0.5
- Ongoing: 0

### Results

TODO: Add chart of win results over n training games.

### Project Structure

- `tictactoe.py`: Main implementation of game environment and learning agents
- `test_tictactoe.py`: Unit tests
- `requirements.txt:` Project dependencies
- `agent1.json`/`agent2.json`: Saved AI policies (created after training)

### Classes

- ```TicTacToe```
  - Game environment class that manages the game state and rules.
- ```LearningAgent```
  - Q-learning agent that learns optimal play through experience.
- ```HumanPlayer```
  - Interface for human players to interact with the game.

### Running Tests

```bash
python3 -m pytest
```
