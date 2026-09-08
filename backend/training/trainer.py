"""
training/trainer.py
Custom DQN training loop (from scratch — no stable-baselines).
Supports: DQN, Double DQN, Dueling Double DQN.
Streams training stats and game frames via an async callback.
"""
import asyncio
import csv
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from env.breakout_env import BreakoutEnv
from networks.cnn import build_network
from training.replay_buffer import ReplayBuffer
from training.checkpoint_manager import CheckpointManager


class Trainer:
    def __init__(self, hyperparams: dict, on_step_callback=None):
        self.hp = hyperparams
        self.on_step = on_step_callback  # async coroutine called every N steps

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Using device: {self.device}")

        algo = hyperparams.get('algorithm', 'dqn') # Default to dqn
        self.algo = algo
        self.env = BreakoutEnv()

        obs_size = self.env.obs_size
        n_actions = self.env.N_ACTIONS

        self.q_net     = build_network(algo, obs_size, n_actions).to(self.device)
        self.target_net = build_network(algo, obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=hyperparams.get('learning_rate', 0.00025)
        )

        self.replay_buffer = ReplayBuffer(hyperparams.get('replay_buffer_size', 100_000))
        self.checkpoint_mgr = CheckpointManager()

        # Hyperparameters
        self.gamma              = hyperparams.get('gamma', 0.99)
        self.batch_size         = hyperparams.get('batch_size', 32)
        self.epsilon_start      = hyperparams.get('epsilon_start', 1.0)
        self.epsilon_final      = hyperparams.get('epsilon_final', 0.01)
        self.epsilon_decay_steps = hyperparams.get('epsilon_decay_steps', 1_000_000)
        self.target_update_steps = hyperparams.get('target_update_steps', 10_000)
        self.total_episodes     = hyperparams.get('episodes', 10_000)

        # State
        self.total_steps = 0
        self.episode = 0
        self.paused = False
        self.stop_requested = False

        # Stats tracking
        self.recent_rewards = []
        self.recent_lengths = []
        self.recent_losses  = []
        self.recent_q_values = []
        self._start_time = time.time()

        # Thread pool for PyTorch training steps
        self.executor = ThreadPoolExecutor(max_workers=1)

        # CSV history log
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
        os.makedirs(log_dir, exist_ok=True)
        self._history_path = os.path.join(log_dir, f'training_history_{algo}.csv')
        self._csv_file = None
        self._csv_writer = None

    @property
    def epsilon(self) -> float:
        return max(
            self.epsilon_final,
            self.epsilon_start - (self.total_steps / self.epsilon_decay_steps)
            * (self.epsilon_start - self.epsilon_final)
        )

    def _select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.N_ACTIONS)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.q_net(s)
            return q.argmax(dim=1).item()

    def _train_step(self) -> tuple[float, float]:
        """Sample a batch and perform one gradient update. Returns (loss, mean_q)."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        s_ = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            if self.algo in ('double_dqn', 'dueling_double_dqn'):
                best_actions = self.q_net(s_).argmax(dim=1)
                target_q_vals = self.target_net(s_).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                target_q_vals = self.target_net(s_).max(dim=1).values

            target_q = r + self.gamma * target_q_vals * (1 - d)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        mean_q = current_q.mean().item()
        return loss.item(), mean_q

    async def _broadcast_loop(self):
        """Runs in background at 30 FPS, emitting the latest available state."""
        while not self.stop_requested:
            if self.on_step and getattr(self, '_latest_broadcast_data', None) is not None:
                try:
                    await self.on_step(*self._latest_broadcast_data)
                except Exception as e:
                    print(f"Broadcast error: {e}")
            await asyncio.sleep(1.0 / 30.0)

    async def run(self):
        """Main async training loop."""
        self.stop_requested = False
        self._start_time = time.time()
        
        # Start broadcaster
        self._latest_broadcast_data = None
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        # Open CSV history file
        self._csv_file = open(self._history_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['episode', 'reward', 'avg_reward_100', 'avg_loss_100', 'avg_q_100', 'epsilon'])

        state = self.env.reset(seed=42)

        episode_reward = 0
        episode_start_step = self.total_steps

        print(f"[Trainer] Starting {self.algo.upper()} training for {self.total_episodes} episodes")

        loop = asyncio.get_event_loop()

        while self.episode < self.total_episodes and not self.stop_requested:
            # Pause support
            while self.paused and not self.stop_requested:
                await asyncio.sleep(0.1)

            action = self._select_action(state)
            next_state, reward, done, info = self.env.step(action)
            buffer_done = done or info.get('life_lost', False)
            self.replay_buffer.push(state, action, reward, next_state, buffer_done)
            state = next_state
            episode_reward += reward
            self.total_steps += 1

            # Train (offloaded to threadpool)
            loss, mean_q = 0.0, 0.0
            if self.replay_buffer.is_ready:
                loss, mean_q = await loop.run_in_executor(self.executor, self._train_step)
                self.recent_losses.append(loss)
                self.recent_q_values.append(mean_q)

            # Sync target network
            if self.total_steps % self.target_update_steps == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Update latest broadcast state
            current_time = time.time()
            avg_reward = np.mean(self.recent_rewards[-100:]) if self.recent_rewards else 0.0
            avg_length = np.mean(self.recent_lengths[-100:]) if self.recent_lengths else 0.0
            avg_loss   = np.mean(self.recent_losses[-100:]) if self.recent_losses else 0.0
            avg_q      = np.mean(self.recent_q_values[-100:]) if self.recent_q_values else 0.0
            elapsed    = current_time - self._start_time
            eps_per_hr = self.episode / (elapsed / 3600) if elapsed > 0 else 0
            
            render = self.env.get_render_state()
            self._latest_broadcast_data = (
                {
                    'type': 'stats',
                    'episode': self.episode,
                    'reward': episode_reward,
                    'avg_reward': avg_reward,
                    'avg_length': avg_length,
                    'epsilon': self.epsilon,
                    'loss': avg_loss,
                    'q_value': avg_q,
                    'eps_per_hour': eps_per_hr,
                }, 
                {
                    'type': 'frame',
                    'frame': render,
                }
            )

            # Episode end
            if done:
                episode_len = self.total_steps - episode_start_step
                self.episode += 1
                self.recent_rewards.append(episode_reward)
                self.recent_lengths.append(episode_len)
                avg_reward = np.mean(self.recent_rewards[-100:])
                avg_loss   = np.mean(self.recent_losses[-100:]) if self.recent_losses else 0.0
                avg_q      = np.mean(self.recent_q_values[-100:]) if self.recent_q_values else 0.0
                
                if self.episode % 100 == 0:
                    print(f"[Ep {self.episode:5d}] reward={episode_reward:6.1f} "
                          f"avg={avg_reward:6.2f} eps={self.epsilon:.4f} loss={avg_loss:.4f}")

                # Write to CSV
                if self._csv_writer:
                    self._csv_writer.writerow([
                        self.episode, round(episode_reward, 2),
                        round(avg_reward, 2), round(avg_loss, 4),
                        round(avg_q, 4), round(self.epsilon, 4),
                    ])
                    self._csv_file.flush()

                # Auto-save every 500 episodes
                if self.episode % 500 == 0:
                    self.checkpoint_mgr.save(self.q_net, self.optimizer, self.epsilon, self.episode, {
                        'avg_reward': avg_reward, 'algorithm': self.algo,
                    })

                state = self.env.reset()
                episode_reward = 0
                episode_start_step = self.total_steps

                # Yield control to event loop every episode
                await asyncio.sleep(0)

        print(f"[Trainer] Training complete after {self.episode} episodes.")

        # Close CSV
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            
        if hasattr(self, '_broadcast_task'):
            self._broadcast_task.cancel()

        # Final checkpoint
        self.checkpoint_mgr.save(self.q_net, self.optimizer, self.epsilon, self.episode, {
            'avg_reward': float(np.mean(self.recent_rewards[-100:])) if self.recent_rewards else 0,
            'algorithm': self.algo, 'final': True,
        })
        self.executor.shutdown(wait=False)

    def pause(self): self.paused = True
    def resume(self): self.paused = False
    def stop(self): self.stop_requested = True

    def save_checkpoint(self):
        self.checkpoint_mgr.save(self.q_net, self.optimizer, self.epsilon, self.episode, {
            'algorithm': self.algo,
        })

    def load_checkpoint(self, episode: int):
        data = self.checkpoint_mgr.load(self.q_net, self.optimizer, episode)
        if data:
            self.episode = data.get('episode', episode)
            print(f"[Trainer] Loaded checkpoint from episode {self.episode}")
