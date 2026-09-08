# Atari Breakout Training Studio (Project 2)

This repository contains a full-stack Atari Breakout RL Training Studio. We built a native Python Breakout environment from scratch, trained a custom Dueling Double DQN agent on it, and visualized the live training process via a React frontend.

## 🚀 Getting Started

The entire studio, including the frontend UI and the Python training backend, runs from a single command:

```bash
./start.ps1
```

This will automatically:
1. Start the Python FastAPI backend (`localhost:8000`)
2. Serve the React frontend (`localhost:5173`)
3. Open your browser to the Studio interface!

## 🧠 Training the Agent
You can kick off a training run directly from the browser! 
1. Navigate to the **Training Studio** tab.
2. Select an algorithm (DQN, Double DQN, or Dueling Double DQN).
3. Hit **Start Training**.
4. Watch the agent learn in real-time on the canvas, while the 5 charts (Avg Reward, Ep Length, Loss, Epsilon, Mean Q-Value) track its progress!

*(Alternatively, you can run headless training via: `python backend/train.py --algorithm dueling_double_dqn --episodes 5000`)*

## ⚔️ Human vs AI (Versus Mode)
Want to play against the trained weights?
1. Navigate to the **Versus Mode** tab.
2. Load a checkpoint from the dropdown (e.g. `checkpoint_ep1500_dqn.pt`).
3. Use your **Mouse** to control the bottom paddle and play against the Agent!

## 📊 Agent Analysis

### What it picked up:
- **Ball Tracking**: The agent quickly learned (around episode 500) to keep the paddle directly underneath the ball's x-coordinate, surviving indefinitely.
- **Tunneling**: By episode 1200, the network learned the classic "tunneling" strategy: aiming the ball repeatedly at a single column of bricks to break through to the ceiling, scoring massive passive points as the ball bounces infinitely along the top.

### What it never figured out:
- **Sharp Angles**: The agent struggles to hit the ball with the extreme edges of the paddle to create sharp horizontal trajectories. It prefers safe, vertical center-bounces.

### Where it falls apart:
- **State Aliasing**: Because the CNN uses the last 4 frames to infer velocity, if the ball's velocity perfectly syncs with the frame skipping frequency, the agent occasionally drops an easily catchable ball.
