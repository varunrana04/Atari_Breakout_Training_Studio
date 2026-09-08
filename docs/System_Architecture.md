# System Architecture

## Overview
The Atari Breakout Studio consists of a high-performance Python backend serving a physics engine and RL training loop, alongside a modern React interface for real-time visualization.

## Components
1. **Frontend Studio (React + Vite + Chart.js)**
   - Renders the Breakout game canvas dynamically using incoming WebSocket JSON payloads.
   - Displays real-time training telemetry on multiple charts (Reward, Ep Length, Loss, Epsilon, Q-Value).
   - Exposes a control panel to configure hyperparameters and start/pause training.
   - "Versus Mode" allows humans to play against the trained `.pt` weights via mouse inputs.

2. **Backend Server (FastAPI)**
   - Manages asynchronous WebSockets to push `60Hz` state updates to the UI.
   - Spawns the reinforcement learning background task without blocking the event loop.

3. **Environment Engine (`BreakoutEnv`)**
   - A custom-built 2D physics engine implemented purely in Python using coordinate geometry.
   - Exposes a standard Gym-like interface (`reset()`, `step(action)`).
   - Handles paddle bounding boxes, ball velocity deflection, brick grid destruction, and scoring/lives.

4. **Reinforcement Learning Controller (`trainer.py`)**
   - Contains a generalized training loop with Replay Buffer sampling.
   - Manages target network updates.
   - Automatically saves model checkpoints (`.pt`) and writes out historical telemetry (`.csv`).

## Data Flow (Live Training)
1. The `Trainer` invokes `env.step(action)` to progress the physics engine.
2. The `Trainer` updates the Neural Network weights based on the loss computed from a sample of the Replay Buffer.
3. Every step, the `Trainer` packages the updated physics state (paddle x, ball x/y, brick array) into a JSON dictionary and queues it.
4. The FastAPI `WebSocket` endpoint consumes the queue and transmits it to the browser.
5. `Studio.jsx` receives the state via `onMessage` and triggers a `requestAnimationFrame` render to draw the state on the canvas.
