"""
agents/dqn_agent.py
Inference-only DQN agent.
Loads a saved checkpoint and selects greedy actions (epsilon=0).
Used in Human vs Agent and Agent vs Agent modes.
"""
import torch
import numpy as np
from agents.base_agent import BaseAgent
from networks.cnn import build_network
from training.checkpoint_manager import CheckpointManager


class DQNAgent(BaseAgent):
    """
    Loads a trained checkpoint and plays greedily (no exploration).
    """
    def __init__(self, checkpoint_path: str, obs_size: int, n_actions: int, algorithm: str = 'dueling_double_dqn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions

        self.net = build_network(algorithm, obs_size, n_actions).to(self.device)
        data = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(data['model_state_dict'])
        self.net.eval()

        self.episode_num = data.get('episode', 0)
        self.algorithm = algorithm
        self.meta = data.get('meta', {})
        print(f"[DQNAgent] Loaded {algorithm} checkpoint (episode {self.episode_num})")

    def select_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.net(s)
            return q_values.argmax(dim=1).item()

    @classmethod
    def from_episode(cls, episode: int, obs_size: int, n_actions: int):
        """Load agent from a checkpoint by episode number (picks last alphabetically if multiple match)."""
        mgr = CheckpointManager()
        import glob, os
        pattern = os.path.join(mgr.dir, f'checkpoint_ep{episode}_*.pt')
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No checkpoint for episode {episode}")
        path = sorted(matches)[-1]
        # Detect algorithm from filename
        algo = 'dueling_double_dqn'
        if 'double_dqn' in path and 'dueling' not in path:
            algo = 'double_dqn'
        elif 'dqn' in path and 'double' not in path and 'dueling' not in path:
            algo = 'dqn'
        return cls(path, obs_size, n_actions, algorithm=algo)

    @classmethod
    def from_name(cls, checkpoint_name: str, obs_size: int, n_actions: int):
        """Load agent from an exact checkpoint filename (e.g. 'checkpoint_ep3000_dqn.pt').
        Resolves to the checkpoints/ directory. Raises FileNotFoundError if not found."""
        import os
        mgr = CheckpointManager()
        # Accept both bare filename and full path
        if os.path.isabs(checkpoint_name) or os.sep in checkpoint_name or '/' in checkpoint_name:
            path = checkpoint_name
        else:
            path = os.path.join(mgr.dir, checkpoint_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Detect algorithm from filename
        name_lower = os.path.basename(path).lower()
        if 'dueling' in name_lower:
            algo = 'dueling_double_dqn'
        elif 'double' in name_lower:
            algo = 'double_dqn'
        else:
            algo = 'dqn'

        return cls(path, obs_size, n_actions, algorithm=algo)

