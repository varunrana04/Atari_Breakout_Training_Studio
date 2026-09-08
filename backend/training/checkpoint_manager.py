"""
training/checkpoint_manager.py
Save and load model checkpoints to disk.
Saves to: checkpoints/checkpoint_ep{episode}_{algorithm}.pt
"""
import os
import glob
import json
import torch
import random
import numpy as np
import threading

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')


class CheckpointManager:
    _lock = threading.Lock()

    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR):
        self.dir = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def save(self, model, optimizer, epsilon: float, episode: int, meta: dict = None) -> str:
        meta = meta or {}
        meta['env_version'] = '1.0.0'
        meta['model_architecture'] = {
            'class': model.__class__.__name__,
            'total_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        algo = meta.get('algorithm', 'dqn')
        
        with self._lock:
            path = os.path.join(self.dir, f'checkpoint_ep{episode}_{algo}.pt')
            tmp_path = path + '.tmp'
            json_path = os.path.join(self.dir, f'checkpoint_ep{episode}_{algo}.json')
            tmp_json_path = json_path + '.tmp'
            
            torch.save({
                'episode': episode,
                'epsilon': epsilon,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'random_state': random.getstate(),
                'numpy_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'meta': meta or {},
            }, tmp_path)
            os.replace(tmp_path, path)
            
            # Save companion JSON for easy frontend inspection
            with open(tmp_json_path, 'w') as f:
                json.dump({
                    'episode': episode,
                    'epsilon': epsilon,
                    'algorithm': algo,
                    'meta': meta or {}
                }, f, indent=2)
            os.replace(tmp_json_path, json_path)
                
            print(f"[Checkpoint] Saved -> {path}")
            return path

    def load(self, model, optimizer, episode: int) -> dict | None:
        pattern = os.path.join(self.dir, f'checkpoint_ep{episode}_*.pt')
        matches = glob.glob(pattern)
        if not matches:
            print(f"[Checkpoint] No checkpoint found for episode {episode}")
            return None
            
        path = sorted(matches)[-1]
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        model.load_state_dict(data['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in data:
            optimizer.load_state_dict(data['optimizer_state_dict'])
            
        if 'random_state' in data:
            random.setstate(data['random_state'])
        if 'numpy_random_state' in data:
            np.random.set_state(data['numpy_random_state'])
        if 'torch_random_state' in data:
            torch.set_rng_state(data['torch_random_state'])
            
        print(f"[Checkpoint] Loaded <- {path}")
        return data

    def list_checkpoints(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.dir, '*.pt')))
