"""
training/trainer.py
Custom DQN training loop (from scratch — no stable-baselines).
Supports: DQN, Double DQN, Dueling Double DQN.
Streams training stats and game frames via an async callback.
"""
import asyncio
import time
import os
import numpy as np
import torch
import torch.nn.functional as F

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

        algo = hyperparams.get('algorithm', 'dueling_double_dqn')
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
        self.recent_losses  = []
        self.recent_q_values = []
        self._start_time = time.time()

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
                # Double DQN: use online net to select action, target net to evaluate
                best_actions = self.q_net(s_).argmax(dim=1)
                target_q_vals = self.target_net(s_).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                # Vanilla DQN: use target net for both select and evaluate
                target_q_vals = self.target_net(s_).max(dim=1).values

            target_q = r + self.gamma * target_q_vals * (1 - d)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        mean_q = current_q.mean().item()
        return loss.item(), mean_q

    async def run(self):
        """Main async training loop."""
        self.stop_requested = False
        self._start_time = time.time()

        state = self.env.reset(seed=42)

        episode_reward = 0
        episode_start_step = 0

        print(f"[Trainer] Starting {self.algo.upper()} training for {self.total_episodes} episodes")

        while self.episode < self.total_episodes and not self.stop_requested:
            # Pause support
            while self.paused and not self.stop_requested:
                await asyncio.sleep(0.1)

            action = self._select_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            self.total_steps += 1

            # Train
            loss, mean_q = 0.0, 0.0
            if self.replay_buffer.is_ready:
                loss, mean_q = self._train_step()
                self.recent_losses.append(loss)
                self.recent_q_values.append(mean_q)

            # Sync target network
            if self.total_steps % self.target_update_steps == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Episode end
            if done:
                self.episode += 1
                self.recent_rewards.append(episode_reward)
                avg_reward = np.mean(self.recent_rewards[-100:])
                avg_loss   = np.mean(self.recent_losses[-100:]) if self.recent_losses else 0.0
                avg_q      = np.mean(self.recent_q_values[-100:]) if self.recent_q_values else 0.0
                elapsed    = time.time() - self._start_time
                eps_per_hr = self.episode / (elapsed / 3600) if elapsed > 0 else 0

                # Send stats + frame to frontend
                if self.on_step:
                    render = self.env.get_render_state()
                    await self.on_step({
                        'type': 'stats',
                        'episode': self.episode,
                        'reward': episode_reward,
                        'avg_reward': avg_reward,
                        'epsilon': self.epsilon,
                        'loss': avg_loss,
                        'q_value': avg_q,
                        'eps_per_hour': eps_per_hr,
                    }, {
                        'type': 'frame',
                        'frame': render,
                    })

                if self.episode % 100 == 0:
                    print(f"[Ep {self.episode:5d}] reward={episode_reward:6.1f} "
                          f"avg={avg_reward:6.2f} eps={self.epsilon:.4f} loss={avg_loss:.4f}")

                # Auto-save every 500 episodes
                if self.episode % 500 == 0:
                    self.checkpoint_mgr.save(self.q_net, self.optimizer, self.episode, {
                        'avg_reward': avg_reward, 'algorithm': self.algo,
                    })

                state = self.env.reset()
                episode_reward = 0

                # Yield control to event loop every episode
                await asyncio.sleep(0)

        print(f"[Trainer] Training complete after {self.episode} episodes.")

        # Final checkpoint
        self.checkpoint_mgr.save(self.q_net, self.optimizer, self.episode, {
            'avg_reward': float(np.mean(self.recent_rewards[-100:])) if self.recent_rewards else 0,
            'algorithm': self.algo, 'final': True,
        })

    def pause(self): self.paused = True
    def resume(self): self.paused = False
    def stop(self): self.stop_requested = True

    def save_checkpoint(self):
        self.checkpoint_mgr.save(self.q_net, self.optimizer, self.episode, {
            'algorithm': self.algo,
        })

    def load_checkpoint(self, episode: int):
        data = self.checkpoint_mgr.load(self.q_net, self.optimizer, episode)
        if data:
            self.episode = data.get('episode', episode)
            print(f"[Trainer] Loaded checkpoint from episode {self.episode}")
