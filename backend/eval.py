import argparse
import numpy as np
import torch
import os
import sys

from env.breakout_env import BreakoutEnv
from networks.cnn import build_network

def evaluate_checkpoint(checkpoint_path: str, episodes: int = 20):
    """
    Evaluates a trained DQN checkpoint over a fixed set of seeds.
    Calculates: Mean reward, Std, Mean bricks destroyed, Board clear rate, Mean episode length.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        sys.exit(1)

    print(f"Loading checkpoint from: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract algorithm from path or assume dqn
    # The checkpoint name is usually like checkpoint_ep3000_dqn.pt
    algorithm = 'dqn'
    
    env = BreakoutEnv()
    obs_size = env.obs_size
    n_actions = env.N_ACTIONS
    
    model = build_network(algorithm, obs_size, n_actions)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback if saved directly
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    rewards = []
    bricks_destroyed_list = []
    episode_lengths = []
    board_clears = 0

    print(f"\n--- Running Evaluation over {episodes} fixed seeds ---")
    
    with torch.no_grad():
        for i in range(episodes):
            state = env.reset(seed=i)  # Fixed seeds 0 to episodes-1
            
            # Count total bricks at start to know when board is clear
            total_initial_bricks = sum(1 for b in env.bricks if not b['indestructible'])
            
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(s)
                # Epsilon = 0 (Pure exploitation)
                action = q_values.argmax(dim=1).item()
                
                state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
            bricks_remaining = info.get('bricks_remaining', 0)
            bricks_destroyed = total_initial_bricks - bricks_remaining
            
            if bricks_remaining == 0:
                board_clears += 1
                
            rewards.append(total_reward)
            bricks_destroyed_list.append(bricks_destroyed)
            episode_lengths.append(steps)
            
            print(f"Seed {i:>2} | Reward: {total_reward:>5.1f} | Bricks: {bricks_destroyed:>3} | Steps: {steps:>4} | Cleared: {'Yes' if bricks_remaining == 0 else 'No'}")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    mean_bricks = np.mean(bricks_destroyed_list)
    mean_len = np.mean(episode_lengths)
    
    print("\n" + "="*40)
    print(f"Evaluation — {episodes} episodes")
    print("="*40)
    print(f"Mean reward:       {mean_reward:.1f}")
    print(f"Std:               {std_reward:.1f}")
    print(f"Min/Max reward:    {min_reward:.1f} / {max_reward:.1f}")
    print(f"Mean bricks:       {mean_bricks:.1f}")
    print(f"Board clears:      {board_clears}/{episodes}")
    print(f"Mean episode len:  {mean_len:.0f}")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_ep3000_dqn.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--episodes", type=int, default=20, help="Number of fixed-seed episodes to run")
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint, args.episodes)
