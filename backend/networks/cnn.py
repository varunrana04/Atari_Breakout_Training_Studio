"""
networks/cnn.py
Q-Network and Dueling Q-Network architectures.
Input: state vector (obs_size,)
Output: Q-values for each action (N_ACTIONS,)

Architecture progression:
  DQN / Double DQN       → QNetwork (MLP)
  Dueling Double DQN     → DuelingQNetwork (MLP with Value + Advantage streams)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Standard Q-Network (MLP).
    Used for vanilla DQN and Double DQN.
    """
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network.
    Splits into Value stream V(s) and Advantage stream A(s, a).
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    Used for Dueling Double DQN.
    """
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        value = self.value_stream(features)              # (batch, 1)
        advantage = self.advantage_stream(features)      # (batch, n_actions)
        # Combine: subtract mean advantage for stability
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


def build_network(algorithm: str, obs_size: int, n_actions: int) -> nn.Module:
    """Factory function — returns the right network for the given algorithm."""
    if algorithm == 'dueling_double_dqn':
        return DuelingQNetwork(obs_size, n_actions)
    else:
        # dqn and double_dqn use same architecture; difference is in the training target
        return QNetwork(obs_size, n_actions)
