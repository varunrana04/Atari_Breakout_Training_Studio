import argparse
import torch
import numpy as np
from env.breakout_env import BreakoutEnv
from networks.cnn import build_network
from training.checkpoint_manager import CheckpointManager

def evaluate(model_path, episodes=20, device='cpu'):
    env = BreakoutEnv()
    
    # Checkpoint Manager
    import os
    ckpt_mgr = CheckpointManager(checkpoint_dir=os.path.dirname(model_path))
    
    # Load model
    q_net = build_network('dueling_double_dqn', env.obs_size, env.N_ACTIONS).to(device)
    
    # We must mock an optimizer to load using the manager, or load weights directly
    data = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = data['model_state_dict']

    if 'shared.0.weight' in state_dict and state_dict['shared.0.weight'].shape[1] == 101:
        old_w = state_dict['shared.0.weight']
        new_w = torch.zeros(old_w.shape[0], 102, device=device)
        new_w[:, :5] = old_w[:, :5]
        new_w[:, 6:] = old_w[:, 5:]
        state_dict['shared.0.weight'] = new_w
        print("Patched checkpoint from 101 to 102 dimensions for evaluation.")
    elif 'net.0.weight' in state_dict and state_dict['net.0.weight'].shape[1] == 101:
        old_w = state_dict['net.0.weight']
        new_w = torch.zeros(old_w.shape[0], 102, device=device)
        new_w[:, :5] = old_w[:, :5]
        new_w[:, 6:] = old_w[:, 5:]
        state_dict['net.0.weight'] = new_w
        print("Patched checkpoint from 101 to 102 dimensions for evaluation.")

    q_net.load_state_dict(state_dict)
    
    # STRICT EVALUATION REQUIREMENTS
    q_net.eval()  # Disable dropout/batchnorm
    
    rewards = []
    
    print(f"Starting deterministic evaluation for {episodes} episodes...")
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        
        while not env.done:
            with torch.no_grad(): # No gradients
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_vals = q_net(state_t)
                action = int(q_vals.argmax(dim=1).item()) # epsilon=0
                
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            
        rewards.append(ep_reward)
        print(f"Episode {ep + 1}/{episodes} - Reward: {ep_reward:.1f}")
        
    avg_reward = np.mean(rewards)
    print("="*40)
    print(f"EVALUATION COMPLETE")
    print(f"Mean Reward: {avg_reward:.2f}")
    print(f"Max Reward:  {np.max(rewards):.2f}")
    print(f"Min Reward:  {np.min(rewards):.2f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    args = parser.parse_args()
    
    evaluate(args.model, episodes=args.episodes)
