# flake8: noqa: F401

import random
import json
import yaml
from functools import lru_cache

from .utils import get_available_moves, is_winner, is_draw
from .game import TicTacToe, play_game, play_against_ai
from .agents import LearningAgent, HumanPlayer
from .training import train
from .cli import cli

if __name__ == "__main__":
    cli()
