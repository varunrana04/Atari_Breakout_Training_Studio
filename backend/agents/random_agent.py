"""
agents/random_agent.py
Random baseline agent — selects uniformly from {left, right, noop}.
Used as Phase 1 baseline to compare against trained DQN agents.
"""
import numpy as np
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions

    def select_action(self, state: np.ndarray) -> int:
        return np.random.randint(self.n_actions)
