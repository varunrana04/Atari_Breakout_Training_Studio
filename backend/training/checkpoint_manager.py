"""
training/checkpoint_manager.py
Save and load model checkpoints to disk.
Saves to: checkpoints/checkpoint_ep{episode}_{algorithm}.pt
"""
import os
import glob
import torch


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR):
        self.dir = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def save(self, model, optimizer, episode: int, meta: dict = None) -> str:
        algo = (meta or {}).get('algorithm', 'dqn')
        path = os.path.join(self.dir, f'checkpoint_ep{episode}_{algo}.pt')
        torch.save({
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'meta': meta or {},
        }, path)
        print(f"[Checkpoint] Saved -> {path}")
        return path

    def load(self, model, optimizer, episode: int) -> dict | None:
        # Find any checkpoint matching episode number
        pattern = os.path.join(self.dir, f'checkpoint_ep{episode}_*.pt')
        matches = glob.glob(pattern)
        if not matches:
            print(f"[Checkpoint] No checkpoint found for episode {episode}")
            return None
        path = sorted(matches)[-1]
        data = torch.load(path, map_location='cpu')
        model.load_state_dict(data['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in data:
            optimizer.load_state_dict(data['optimizer_state_dict'])
        print(f"[Checkpoint] Loaded <- {path}")
        return data

    def list_checkpoints(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.dir, '*.pt')))
