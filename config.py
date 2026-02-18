# Functions to dictate how environment/agent reacts to movement/how things move around. Kind of the "miscellaneous" file.

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    """Configuration class to hold all experiment parameters."""

    # Environment settings
    grid_size: int
    slip_prob: float
    num_obstacles: int
    obstacle_move_prob: float
    goal_move_prob: float

    # Rewards
    movement_reward: int
    obstacle_penalty: int
    goal_reward: int

    # Agent settings
    agents: list[str]  # (e.g., ["RandomAgent", "GreedyAgent", "MCTSRandomAgent", "MCTSUctAgent"])
    mcts_iterations: int  # Number of MCTS iterations for MCTS agents
    mcts_rollout_depth: int
    mcts_ucb_c: float  # Exploration constant for MCTS UCT agent

    # Experiment settings
    num_trials: int
    visualize: bool
    output_dir: str


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file."""
    with Path(config_path).open() as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)

