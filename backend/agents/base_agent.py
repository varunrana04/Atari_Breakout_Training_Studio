"""
agents/base_agent.py
Abstract base class for all agents.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Given a state vector, return an action (0=left, 1=right, 2=noop)."""
        ...

    def on_episode_end(self, episode: int, total_reward: float):
        """Optional hook called at end of each episode."""
        pass
