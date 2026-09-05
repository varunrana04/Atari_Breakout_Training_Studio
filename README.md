# Breakout Training Studio

A browser-based Atari Breakout game with a full reinforcement learning training studio. Train a DQN agent, watch it learn in real time, compare algorithms, and pit agents against each other.

## Features

- **Breakout environment** — paddle, ball, bricks, lives, score; Python `BreakoutEnv` is the canonical environment used for both training and versus mode
- **3 RL algorithms written from scratch** — Vanilla DQN, Double DQN, Dueling Double DQN (no stable-baselines)
- **Live training studio** — watch the Python env run while training, see reward/loss/epsilon/Q-value charts update in real time
- **Configurable hyperparameters** — learning rate, gamma, epsilon schedule, batch size, replay buffer, target update frequency
- **Checkpoint save/load** — save training state and resume, compare different checkpoints
- **Versus mode** — Human vs Agent or Agent vs Agent: both sides run on the Python BreakoutEnv; the browser sends actions and renders the authoritative Python game state. DQN checkpoints run real PyTorch inference (ε=0), not a heuristic.
- **Single-command startup**

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Option 1 — Single command (recommended)

**Windows:**
```powershell
.\start.ps1
```

**Linux/Mac:**
```bash
./start.sh
```

### Option 2 — Manual

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

**Frontend** (in a new terminal):
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

---

## Training from CLI (no browser required)

```bash
cd backend

# Train a single algorithm
python train.py --algorithm dueling_double_dqn --episodes 10000

# Train with custom hyperparams
python train.py --algorithm dqn --episodes 5000 --lr 0.0001 --gamma 0.99

# Train all algorithms and compare
python train.py --compare
```

**CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--algorithm` | `dueling_double_dqn` | `dqn`, `double_dqn`, or `dueling_double_dqn` |
| `--episodes` | `10000` | Training episodes |
| `--lr` | `0.00025` | Learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--batch-size` | `32` | Replay batch size |
| `--buffer-size` | `100000` | Replay buffer capacity |
| `--epsilon-start` | `1.0` | Starting exploration |
| `--epsilon-final` | `0.01` | Minimum exploration |
| `--epsilon-decay` | `1000000` | Steps to decay epsilon |
| `--target-update` | `10000` | Steps between target net syncs |
| `--compare` | — | Train all agents + print table |

---

## Architecture

### RL Algorithm

We use **Dueling Double DQN** as the primary agent, with DQN and Double DQN as baselines:

```
DQN → Double DQN → Dueling Double DQN
```

**Why DQN and not PPO:** Breakout has a small, **3-action** discrete control space (left, right, noop). Value-based methods like DQN map cleanly onto "estimate value of each action, pick best". PPO's machinery (actor-critic, GAE, clipped surrogate) is unnecessary overhead and gives more ways to silently fail. DQN is the original DeepMind Breakout algorithm.

**Double DQN** (5-line change): Fixes Q-value overestimation by using the online network to *select* the action and the target network to *evaluate* it.

**Dueling DQN** (separate Value + Advantage streams): Helps the network learn which states are good even when action choice doesn't matter much.

> **Note on results:** In this experiment (3,000 episodes), vanilla DQN outperformed both variants. This is not unusual — the overestimation bias Double DQN corrects is not the primary bottleneck at this training budget. Both variants should be considered as architectural references, not as superior alternatives based on this run.

### State Representation

Rather than raw pixels (which require CNN + frame stacking + GPU), we use a compact state vector:

```
[paddle_center_x/W, ball_x/W, ball_y/H, ball_vx/10, ball_vy/10, brick_hp_grid...]
= 5 + 96 = 101 features
```

This trains ~10× faster on CPU and is fully interpretable — important for explaining what the agent learned.

### Environment API

```python
env = BreakoutEnv()
state = env.reset(seed=42)          # reproducible starting state
state, reward, done, info = env.step(action)  # action: 0=left, 1=right, 2=noop
```

**Python BreakoutEnv is the canonical environment.** It drives all RL training and all versus-mode games. The browser JavaScript game is for human play only — during versus mode, the browser sends keyboard actions to the backend, which runs `env.step()` and streams the authoritative game state back to the browser for rendering.

### WebSocket API

The backend streams training data to the frontend via WebSocket at `ws://localhost:8000/ws/train`.

**Client → Server:**
```json
{ "cmd": "start", "hyperparams": { ... } }
{ "cmd": "pause" }
{ "cmd": "resume" }
{ "cmd": "save_checkpoint" }
{ "cmd": "load_checkpoint", "episode": 500 }
```

**Server → Client (per episode):**
```json
{ "type": "stats", "episode": 100, "reward": 12.5, "avg_reward": 8.2, "epsilon": 0.85, "loss": 0.024, "q_value": 3.1, "eps_per_hour": 450 }
{ "type": "frame", "frame": { "paddle_x": 0.5, "ball_x": 0.4, "ball_y": 0.7, "bricks": [...] } }
```

---

## Project Structure

```
project2-breakout-studio/
├── backend/
│   ├── main.py                   # FastAPI server (training + versus WebSocket)
│   ├── train.py                  # Standalone CLI training script
│   ├── env/
│   │   └── breakout_env.py       # Python Breakout env (reset/step API)
│   ├── agents/
│   │   ├── base_agent.py         # Abstract agent
│   │   ├── random_agent.py       # Random baseline
│   │   └── dqn_agent.py          # Inference-only DQN agent (loads checkpoint)
│   ├── networks/
│   │   └── cnn.py                # QNetwork + DuelingQNetwork (PyTorch)
│   ├── training/
│   │   ├── replay_buffer.py      # Experience replay buffer
│   │   ├── trainer.py            # Full custom DQN training loop
│   │   └── checkpoint_manager.py # Save/load .pt checkpoints
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── game/
│   │   │   ├── breakout.js       # JS Breakout game engine
│   │   │   ├── levels.js         # 100 level definitions
│   │   │   └── powerups.js       # 6 power-up types
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx   # Start screen
│   │   │   ├── PlayerGame.jsx    # Human gameplay
│   │   │   ├── Studio.jsx        # Training studio
│   │   │   └── VersusMode.jsx    # Human vs Agent / Agent vs Agent
│   │   └── components/
│   │       ├── PauseMenu.jsx
│   │       ├── RewardChart.jsx   # 4-tab Chart.js charts
│   │       ├── StatsPanel.jsx    # Live training stats
│   │       ├── ControlPanel.jsx  # Hyperparams + controls
│   │       └── VersusSetupModal.jsx
│   └── package.json
├── checkpoints/                  # 20 saved .pt files (2 early-run artifacts + 18 regular)
├── start.ps1                     # Windows single-command startup
├── start.sh                      # Linux/Mac single-command startup
└── README.md
```

---

## Hyperparameters (Defaults)

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 0.00025 | Adam optimizer |
| Gamma | 0.99 | Discount factor |
| Replay buffer | 100,000 | Transitions stored |
| Batch size | 32 | Per gradient update |
| Epsilon start | 1.0 | Full exploration |
| Epsilon final | 0.01 | Minimum exploration |
| Epsilon decay | 1,000,000 steps | Linear annealing |
| Target update | 10,000 steps | Periodic sync |

> **On the training reward plateau:** Avg reward of ~203 for DQN (best result) corresponds to roughly 20 bricks cleared per episode — the agent never cleared a full level 1 (96 bricks × 10 = 960 points). The level-progression and power-up systems are extras added for human play; the trained agent operates entirely within level 1 and did not encounter the progressive difficulty during training.

---

## Reward Shaping

| Event | Reward |
|---|---|
| Brick destroyed | +10 |
| Life lost | −5 |
| Level cleared | +50 |
| All 100 levels complete | +20 |

---

## Training Results (3,000 Episodes Each)

```
ALGORITHM             AVG REWARD   BEST EPISODE   EPISODES
random                     27.60          125.0        500
dqn                       203.30          605.0      3,000
double_dqn                 73.30          485.0      3,000
dueling_double_dqn         82.60          455.0      3,000
```

All "avg reward" figures are the mean over the last 100 episodes of each run, at ε=0.01 (minimal exploration).

### What actually happened

**DQN (avg 203.30, best 605):**
- Episodes 0–500: Epsilon > 0.5, nearly random. Avg reward ~25.
- Episodes 500–700: Epsilon drops below 0.35. Qualitatively, gameplay showed improved paddle tracking — agent starts keeping the ball alive more consistently.
- Episodes 700–1,000: Avg reward rises to 150–200. Agent learns to clear bricks in the early part of the level.
- Episodes 1,000–3,000: Avg reward plateaus around 160–200. Best single episode: 605 points — roughly 60 bricks cleared, still short of clearing level 1 (960 points needed).

**Key finding:** The agent never cleared a full level in any recorded episode. Avg reward of 203 ≈ 20 bricks per episode. The level-progression system was never encountered during training.

**Double DQN (avg 73.30) and Dueling Double DQN (avg 82.60):**
- Both show similar exploration phases but plateau lower than vanilla DQN at 3,000 episodes.
- Both variants share the Double-DQN target computation. The underperformance versus vanilla at this episode count is not fully explained — it may reflect slower early learning from the Double DQN correction, or a subtle implementation issue. We did not run long enough to draw conclusions about which would perform better at a larger training budget.

**Random baseline (avg 27.60):** Confirms the environment produces sensible rewards for non-trivial play.

### What the agent learned
- Consistent paddle tracking (follows ball horizontally)
- Basic brick targeting (clears accessible bricks near the bottom of the grid)
- Avoiding immediate life loss

### What it never figured out
- Clearing a full level — the agent plateaus well below the points needed for level completion
- Deliberate shot angling to reach brick clusters at the top
- Power-ups — the JS human game has power-ups as a stretch feature; they are not represented in the state vector and have no effect on training


