"""
train.py
Standalone training script — runs DQN training from the command line
without requiring the web UI or WebSocket server.

Usage:
    python train.py --algorithm dueling_double_dqn --episodes 10000
    python train.py --algorithm dqn --episodes 5000 --lr 0.0001
    python train.py --compare  # trains all 4 agents and prints comparison table
"""
import argparse
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from env.breakout_env import BreakoutEnv
from networks.cnn import build_network
from training.replay_buffer import ReplayBuffer
from training.checkpoint_manager import CheckpointManager


def run_random_baseline(episodes: int = 200) -> dict:
    """Run random agent and return stats."""
    print("\n[Random Agent] Running baseline...")
    env = BreakoutEnv()
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_r = 0
        done = False
        while not done:
            action = np.random.randint(env.N_ACTIONS)
            _, r, done, _ = env.step(action)
            total_r += r
        rewards.append(total_r)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{episodes} | Avg reward: {np.mean(rewards[-50:]):.2f}")
    return {
        'algorithm': 'random',
        'avg_reward': float(np.mean(rewards)),
        'best_reward': float(np.max(rewards)),
        'episodes': episodes,
    }


def run_dqn_training(algorithm: str, episodes: int, hyperparams: dict, seed: int = 42) -> dict:
    """Run DQN training synchronously (no WebSocket). Returns stats dict."""
    import torch
    import torch.nn.functional as F

    print(f"\n[{algorithm.upper()}] Starting training for {episodes} episodes...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    env = BreakoutEnv()
    obs_size = env.obs_size
    n_actions = env.N_ACTIONS

    q_net      = build_network(algorithm, obs_size, n_actions).to(device)
    target_net = build_network(algorithm, obs_size, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer     = torch.optim.Adam(q_net.parameters(), lr=hyperparams.get('learning_rate', 0.00025))
    replay_buffer = ReplayBuffer(hyperparams.get('replay_buffer_size', 100_000))
    ckpt_mgr      = CheckpointManager()

    gamma               = hyperparams.get('gamma', 0.99)
    batch_size          = hyperparams.get('batch_size', 32)
    epsilon_start       = hyperparams.get('epsilon_start', 1.0)
    epsilon_final       = hyperparams.get('epsilon_final', 0.01)
    epsilon_decay_steps = hyperparams.get('epsilon_decay_steps', 1_000_000)
    target_update_steps = hyperparams.get('target_update_steps', 10_000)

    total_steps = 0
    all_rewards = []
    all_losses  = []
    start_time  = time.time()

    state = env.reset(seed=seed)
    episode_reward = 0

    import csv
    history_path = f"checkpoints/history_{algorithm}.csv"
    os.makedirs('checkpoints', exist_ok=True)
    csv_file = open(history_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'reward', 'avg_reward_100', 'avg_loss_100', 'avg_q_100', 'epsilon'])

    ep = 0
    while ep < episodes:
        # Epsilon
        eps = max(epsilon_final, epsilon_start - (total_steps / epsilon_decay_steps)
                  * (epsilon_start - epsilon_final))

        # Action selection
        if np.random.rand() < eps:
            action = np.random.randint(n_actions)
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = q_net(s).argmax(dim=1).item()

        next_state, reward, done, info = env.step(action)
        buffer_done = done or info.get('life_lost', False)
        replay_buffer.push(state, action, reward, next_state, buffer_done)
        state = next_state
        episode_reward += reward
        total_steps += 1

        # Train step
        if replay_buffer.is_ready:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            s  = torch.FloatTensor(states).to(device)
            a  = torch.LongTensor(actions).to(device)
            r  = torch.FloatTensor(rewards).to(device)
            s_ = torch.FloatTensor(next_states).to(device)
            d  = torch.FloatTensor(dones).to(device)

            current_q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if algorithm in ('double_dqn', 'dueling_double_dqn'):
                    best_a = q_net(s_).argmax(dim=1)
                    target_q_vals = target_net(s_).gather(1, best_a.unsqueeze(1)).squeeze(1)
                else:
                    target_q_vals = target_net(s_).max(dim=1).values
                target_q = r + gamma * target_q_vals * (1 - d)

            loss = F.smooth_l1_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()
            all_losses.append(loss.item())

        if total_steps % target_update_steps == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            ep += 1
            all_rewards.append(episode_reward)
            avg_r = np.mean(all_rewards[-100:])
            avg_l = np.mean(all_losses[-100:]) if all_losses else 0
            
            # Log to CSV
            csv_writer.writerow([ep, episode_reward, avg_r, avg_l, 0, eps])
            csv_file.flush()

            if ep % 100 == 0:
                elapsed = time.time() - start_time
                eps_hr  = ep / (elapsed / 3600) if elapsed > 0 else 0
                print(f"  Ep {ep:5d}/{episodes} | reward={episode_reward:6.1f} "
                      f"avg={avg_r:6.2f} eps={eps:.4f} loss={avg_l:.4f} "
                      f"speed={eps_hr:.0f}ep/hr")

            if ep % 500 == 0:
                meta = {
                    'environment_version': '1.0',
                    'state_dimension': obs_size,
                    'action_dimension': n_actions,
                    'algorithm': algorithm,
                    'seed': seed,
                    'hyperparameters': hyperparams,
                    'episode': ep,
                    'avg_reward': float(avg_r)
                }
                ckpt_mgr.save(q_net, optimizer, eps, ep, meta)

            state = env.reset()
            episode_reward = 0

    # Final save
    final_avg = float(np.mean(all_rewards[-100:]))
    meta = {
        'environment_version': '1.0',
        'state_dimension': obs_size,
        'action_dimension': n_actions,
        'algorithm': algorithm,
        'seed': seed,
        'hyperparameters': hyperparams,
        'episode': ep,
        'avg_reward': final_avg,
        'final': True,
    }
    ckpt_mgr.save(q_net, optimizer, eps, ep, meta)

    return {
        'algorithm': algorithm,
        'avg_reward': final_avg,
        'best_reward': float(np.max(all_rewards)),
        'episodes': episodes,
        'total_steps': total_steps,
    }


def print_comparison_table(results: list[dict]):
    print("\n" + "=" * 65)
    print(f"{'ALGORITHM':<25} {'AVG REWARD':>12} {'BEST EPISODE':>14} {'EPISODES':>10}")
    print("=" * 65)
    for r in results:
        print(f"  {r['algorithm']:<23} {r['avg_reward']:>12.2f} {r['best_reward']:>14.1f} {r['episodes']:>10,}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description='Breakout RL Training Script')
    parser.add_argument('--algorithm', default='dueling_double_dqn',
                        choices=['dqn', 'double_dqn', 'dueling_double_dqn'],
                        help='RL algorithm to train')
    parser.add_argument('--episodes', type=int, default=10_000, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=32, help='Replay batch size')
    parser.add_argument('--buffer-size', type=int, default=100_000, help='Replay buffer capacity')
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-final', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=int, default=1_000_000)
    parser.add_argument('--target-update', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    parser.add_argument('--compare', action='store_true',
                        help='Train random vs dqn algorithms and print comparison table')
    args = parser.parse_args()

    hp = {
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'batch_size': args.batch_size,
        'replay_buffer_size': args.buffer_size,
        'epsilon_start': args.epsilon_start,
        'epsilon_final': args.epsilon_final,
        'epsilon_decay_steps': args.epsilon_decay,
        'target_update_steps': args.target_update,
    }

    if args.compare:
        print("Training all agents for comparison...")
        results = []
        results.append(run_random_baseline(episodes=500))
        for algo in ['dqn']:
            results.append(run_dqn_training(algo, episodes=min(args.episodes, 3000), hyperparams=hp, seed=args.seed))
        print_comparison_table(results)
    else:
        result = run_dqn_training(args.algorithm, args.episodes, hp, seed=args.seed)
        print(f"\nTraining complete: avg_reward={result['avg_reward']:.2f}")


if __name__ == '__main__':
    main()
