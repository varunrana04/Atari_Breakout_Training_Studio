# Project 2: Breakout Studio - Design Documents

## 1. High-Level Design (HLD)

### Overview
Breakout Studio is a dual-purpose reinforcement learning platform designed to train, evaluate, and visualize a Deep Q-Network (DQN) mastering a custom Breakout clone. The system runs physics and game logic completely in the backend, while streaming frame states over WebSockets to a React frontend for live visualisation and interactive human-vs-agent gameplay.

### Core Workflows
1. **Training Workflow**:
   - The user requests to start training via the frontend UI.
   - The backend spawns a non-blocking asyncio task to run the training loop (Double DQN or Standard DQN).
   - The `BreakoutEnv` physics engine ticks rapidly.
   - Model checkpoints are periodically saved to disk (`.pt` files).
   - Training statistics (loss, reward, epsilon) are streamed via WebSockets to the frontend for real-time charting and are persisted to CSV logs.

2. **Versus Mode Workflow**:
   - The user selects a specific training checkpoint from the UI to pit themselves (or another agent) against.
   - The backend loads the exact PyTorch state dict from the selected checkpoint file.
   - The backend maintains an authoritative `BreakoutEnv` instance.
   - The frontend connects via a dedicated `/ws/versus` WebSocket.
   - The backend streams 30fps frames to the frontend while simultaneously listening for human keyboard inputs (Left/Right) over the socket.
   - The frontend acts as a "dumb terminal", rendering the game state received from the backend onto HTML5 canvases.

## 2. Low-Level Design (LLD)

### Components

#### Backend (`FastAPI` & `PyTorch`)
- **`BreakoutEnv` (`env/breakout_env.py`)**: A vectorized, NumPy-based physics engine. Manages paddle bounds, ball-brick AABB collision detection, and score tracking. Defines the observation space (normalized positions/states) and action space (Left, Right, Stay).
- **`DQNAgent` (`agents/dqn_agent.py`)**: Implements the neural network (3-layer MLP) and action selection (epsilon-greedy). Includes `from_name()` class methods to safely construct agents from specific disk checkpoints.
- **`ReplayBuffer`**: Circular buffer for experience replay during DQN training to break correlation between consecutive samples.
- **`Trainer` (`training/trainer.py`)**: Manages the episodic loop. Implements Double DQN target network updates, Huber loss, gradient clipping, and robust CSV history logging (`training_history_dqn.csv`).
- **`main.py`**: Mounts the API and handles multiple concurrent WebSocket connections. Employs async generators to throttle and broadcast game ticks without blocking the main event loop.

#### Frontend (`React` + `Vite`)
- **`TrainingDashboard.jsx`**: Connects to the `/ws/train` socket. Uses `recharts` to plot incoming metrics. Displays live gameplay of the agent currently training.
- **`VersusMode.jsx`**: Connects to the `/ws/versus` socket. Renders split-screen canvases for Agent vs Human. Binds `keydown` events (Arrow/WASD) and transmits them upstream. Renders life bars and handles reconnection logic.

## 3. System Architecture

The architecture enforces a strict client-server model where the server is fully authoritative over state to ensure parity between training conditions and inference evaluation.

```
[ Frontend (React/Vite) ] <====================> [ Backend (FastAPI) ]
       |                            WebSockets           |
       |-- Plots Metrics                                 |-- [ Trainer ]
       |-- Renders Canvases                              |     -> Trains DQNAgent
       |-- Captures Keypresses                           |     -> Saves .pt Checkpoints
                                                         |     -> Writes .csv Logs
                                                         |
                                                         |-- [ BreakoutEnv ] (Authoritative)
                                                         |     -> Evaluates Actions
                                                         |     -> Computes Physics
```

## 4. System Design

### Data Flow & Storage
- **Checkpoints**: Saved strictly in the `checkpoints/` directory. Checkpoints are named structurally (e.g., `checkpoint_ep1000_dqn.pt`).
- **Logs**: Training trajectories are saved continuously to `training_history_<algo>.csv` to ensure data persists across server restarts, allowing users to analyze plateaus.
- **State Serialization**: The `BreakoutEnv` state is serialized into a lightweight JSON payload (containing normalized coordinates and an array of brick health values) which the frontend uses to paint primitives.

### Security and Robustness
- **Authoritative State**: By keeping all physics and game logic on the backend Python server, the system guarantees that the DQN agent is evaluated in the exact same mathematical environment it was trained in. Previous designs attempting to duplicate physics logic in JavaScript were abandoned due to unavoidable floating-point drift and synchronization issues.
- **Module Caching Avoidance**: The frontend utilizes explicit WebSockets and cleanly unmounts event listeners to prevent memory leaks or stale React closures.
- **Test Coverage**: Critical physics routines (collisions, paddle bounds, resets) are verified via `test_env.py` to prevent regression.

### Scalability Considerations
- **Training Throttle**: The async training loop utilizes `await asyncio.sleep(0)` to yield control back to the event loop, ensuring the WebSocket server remains highly responsive even during intensive backpropagation phases.
- **Inference Speed**: The `VersusMode` loop ticks at ~30Hz. Inference runs strictly on the CPU to avoid GPU tensor-transfer latency overhead for small MLPs.
