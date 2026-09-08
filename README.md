# Atari Breakout Training Studio

A high-performance, fully asynchronous Reinforcement Learning environment and training studio for Atari Breakout. Built from scratch with a custom Python physics engine, a Dueling Double DQN agent, and a React real-time visualization dashboard.

## 🧠 RL Architecture & Agent Design

The agent is trained using a **Dueling Double Deep Q-Network (D3QN)**.
- **State Representation**: The game state is encoded as a 4-frame stacked tensor `(4, H, W)` allowing the CNN to infer ball velocity and trajectory.
- **Dueling Streams**: The CNN feature extractor splits into two separate fully-connected streams:
  1. **Value Stream**: Estimates the intrinsic value of the state $V(s)$.
  2. **Advantage Stream**: Estimates the relative advantage of each action $A(s, a)$.
  This allows the agent to learn that states where the ball is far away are "safe" regardless of the action taken.
- **Loss Function**: We utilize **Huber Loss** (`F.smooth_l1_loss`) to calculate the Temporal Difference (TD) error. This exponentially stabilizes gradient descent by preventing massive reward spikes (like breaking multiple bricks simultaneously) from exploding the neural weights.

## ⚙️ Physics Engine Details

Rather than wrapping `gym.Env` blindly, the backend implements a deterministic, 60Hz 2D physics engine:
- **AABB Collision Detection**: Axis-Aligned Bounding Boxes track the ball, paddle, and bricks. 
- **Deflection Geometry**: Paddle deflections are linearly interpolated based on where the ball strikes the paddle relative to its center, allowing the agent to "aim" the ball.
- **Terminal States**: Loss of life correctly emits `done=True` to the Replay Buffer to strictly enforce the Markov Property and teach the agent the penalty of death.

## 🚀 Setup & Installation

### Requirements
- Python 3.10+
- Node.js 18+
- PyTorch (CUDA recommended but CPU-compatible)

### Backend Initialization
```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Frontend Initialization
```bash
cd frontend
npm install
```

## 🎮 Running the Studio

Start the backend API (runs on `http://localhost:8000`):
```bash
cd backend
uvicorn main:app --reload
```

Start the React Frontend (runs on `http://localhost:5173`):
```bash
cd frontend
npm run dev
```

From the frontend Studio, you can:
- Observe the live epsilon decay, reward curve, and loss charting.
- Watch the agent learn to play in real-time.
- Pause the training loop instantly.
- Load `checkpoint_ep742_dueling_double_dqn.pt` and play against the trained agent in **Versus Mode**.

## 📂 Samples
Check the `samples/` directory for the `training_curve.png` plot showing the agent's convergence toward the optimal "tunneling" strategy, and `sample_state.json` detailing the WebSocket payload.
