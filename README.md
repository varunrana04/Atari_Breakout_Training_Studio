# Atari Breakout Training Studio

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)
![React](https://img.shields.io/badge/React-18-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![Architecture](https://img.shields.io/badge/Architecture-Asynchronous_RL-success.svg)

> **"Bridging rigorous asynchronous execution with dynamic reinforcement learning."**

## 🎯 The Mission
A high-performance, fully asynchronous Reinforcement Learning environment and training studio for Atari Breakout. Built from scratch with a custom Python physics engine, a Dueling Double DQN agent, and a React real-time visualization dashboard.

## 🏗 RL Neural Architecture

```mermaid
graph LR
    subgraph Environment
        A[Physics Engine]
    end
    
    subgraph Dueling Double DQN
        B[CNN Feature Extractor]
        C[Value Stream V]
        D[Advantage Stream A]
        B --> C
        B --> D
        C --> E(Aggregated Q-Values)
        D --> E
    end
    
    A -- "State (4xHxC)" --> B
    E -- Action --> A
    A -- Reward, Next State --> F[(Replay Buffer)]
    F -- Sample Mini-Batch --> B
```

## 🚀 Core Technology Stack
- **The Environment (`BreakoutEnv`)**: Engineered a deterministic, 60Hz physics engine from scratch using coordinate geometry. Implements an explicit `step(action)` and `reset()` interface to cleanly isolate state transitions.
- **The Agent (Dueling Double DQN)**: Developed a heavily customized DQN variant in `PyTorch`. 
  - **Dueling Streams**: The network flattens the Convolutional layers into a 512-node hidden layer, which splits into a **Value Stream** $V(s)$ and an **Advantage Stream** $A(s, a)$.
  - **Double Q-Learning**: Evaluates the greedy policy using the online network but estimates its value using an asynchronous Target Network, eliminating the positive maximization bias inherent in standard Q-Learning.
- **The Asynchronous Studio**: Built a FastAPI server that manages the training loop in a non-blocking background thread (`ThreadPoolExecutor`) while simultaneously pushing 60Hz state telemetry (paddle coordinates, ball vectors, brick arrays) to a React frontend via WebSockets.

## 📊 Quantitative Validation

| Metric | Measured Value | Target Standard | Note |
|--------|----------------|-----------------|------|
| **CNN Forward Latency** | `< 1 ms` | `< 5 ms` | Evaluated on CPU |
| **Max Training Convergence** | `12.1 Avg Reward` | `> 10.0` | Hit via "Tunneling Strategy" at Epoch 742 |
| **Replay Buffer Capacity** | `100,000` | N/A | Deque-based circular array |
| **Target Update Frequency** | `1000 Steps` | N/A | Hard-freezes target weights to prevent bias |

- **Convergence Details**: The Dueling Double DQN converged significantly faster than the baseline DQN. By Episode 500, the network established consistent ball-tracking. By Episode 742, it discovered the optimal "Tunneling Strategy" (destroying a single column to bounce the ball indefinitely against the ceiling), pushing max rewards toward $12.1+$.

## 💻 Setup & Installation

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

## ⚙️ Running the Studio

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
Check the `samples/` directory for the `training_curve.png` plot showing the agent's convergence, and `sample_state.json` detailing the exact WebSocket data schema.
