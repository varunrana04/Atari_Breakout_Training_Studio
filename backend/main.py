"""
main.py
FastAPI backend for the Breakout Training Studio.

Endpoints:
  WS  /ws/train         — training control + live stats/frames
  WS  /ws/versus        — run two agents (or human+agent) simultaneously
  GET /api/checkpoints  — list saved checkpoint files
"""
import asyncio
import json
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from training.trainer import Trainer
from training.checkpoint_manager import CheckpointManager

app = FastAPI(title="Breakout Training Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global training state ──────────────────────────────────────────────────
_trainer: Trainer | None = None
_train_task: asyncio.Task | None = None


# ── Training WebSocket ─────────────────────────────────────────────────────
@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    global _trainer, _train_task
    await ws.accept()
    print("[WS/train] Client connected")

    async def send_update(stats: dict, frame: dict):
        try:
            await ws.send_text(json.dumps(stats))
            await ws.send_text(json.dumps(frame))
        except Exception:
            pass

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get('cmd')

            if cmd == 'start':
                if _train_task and not _train_task.done():
                    if _trainer:
                        _trainer.stop()
                    await asyncio.sleep(0.2)

                hyperparams = msg.get('hyperparams', {})
                _trainer = Trainer(hyperparams, on_step_callback=send_update)
                _train_task = asyncio.create_task(_trainer.run())
                await ws.send_text(json.dumps({'type': 'ack', 'msg': 'Training started'}))

            elif cmd == 'pause':
                if _trainer:
                    _trainer.pause()
                await ws.send_text(json.dumps({'type': 'ack', 'msg': 'Paused'}))

            elif cmd == 'resume':
                if _trainer:
                    _trainer.resume()
                await ws.send_text(json.dumps({'type': 'ack', 'msg': 'Resumed'}))

            elif cmd == 'stop':
                if _trainer:
                    _trainer.stop()
                await ws.send_text(json.dumps({'type': 'ack', 'msg': 'Stopped'}))

            elif cmd == 'save_checkpoint':
                if _trainer:
                    _trainer.save_checkpoint()
                    await ws.send_text(json.dumps({
                        'type': 'ack',
                        'msg': f'Checkpoint saved (ep {_trainer.episode})',
                    }))

            elif cmd == 'load_checkpoint':
                episode = msg.get('episode', 0)
                if _trainer:
                    _trainer.load_checkpoint(episode)
                    await ws.send_text(json.dumps({
                        'type': 'ack',
                        'msg': f'Loaded checkpoint ep {episode}',
                    }))

    except WebSocketDisconnect:
        print("[WS/train] Client disconnected")
        if _trainer:
            _trainer.stop()


# ── Versus WebSocket ───────────────────────────────────────────────────────
@app.websocket("/ws/versus")
async def ws_versus(ws: WebSocket):
    """
    Run two Breakout environments in parallel.
    mode='human_vs_agent'  → left env runs freely (human controls via messages),
                              right env runs with a loaded DQN agent.
    mode='agent_vs_agent'  → both envs run with loaded agents.

    Client sends:
      { cmd: 'start', mode: 'human_vs_agent'|'agent_vs_agent',
        agent1_episode: int, agent2_episode: int }
      { cmd: 'action', action: 0|1|2 }  (for human left side)
      { cmd: 'stop' }

    Server sends every tick:
      { type: 'frame',
        left:  { paddle_x, paddle_w, ball_x, ball_y, bricks, score, lives, done },
        right: { ... } }
    """
    await ws.accept()
    print("[WS/versus] Client connected")

    from env.breakout_env import BreakoutEnv
    from agents.random_agent import RandomAgent

    running = False
    stop_flag = False
    human_action = 2  # default noop

    env_left = BreakoutEnv()
    env_right = BreakoutEnv()
    agent_left = None
    agent_right = None

    async def game_loop():
        nonlocal running, stop_flag, human_action
        state_left = env_left.reset(seed=1)
        state_right = env_right.reset(seed=1)
        score_left = 0
        score_right = 0

        while running and not stop_flag:
            # Select actions
            action_l = human_action if agent_left is None else agent_left.select_action(state_left)
            action_r = agent_right.select_action(state_right) if agent_right else 2

            # Step envs
            state_left, r_l, done_l, info_l = env_left.step(action_l)
            state_right, r_r, done_r, info_r = env_right.step(action_r)
            score_left += r_l
            score_right += r_r

            if done_l:
                state_left = env_left.reset(seed=1)
                score_left = 0
            if done_r:
                state_right = env_right.reset(seed=1)
                score_right = 0

            frame = {
                'type': 'frame',
                'left': {**env_left.get_render_state(), 'score': info_l['score'], 'lives': info_l['lives']},
                'right': {**env_right.get_render_state(), 'score': info_r['score'], 'lives': info_r['lives']},
            }
            try:
                await ws.send_text(json.dumps(frame))
            except Exception:
                break

            await asyncio.sleep(0.016)  # ~60fps

    loop_task: asyncio.Task | None = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get('cmd')

            if cmd == 'start':
                mode = msg.get('mode', 'human_vs_agent')
                
                # Build agents
                cp1 = msg.get('agent1_checkpoint')   # full filename e.g. "checkpoint_ep3000_dqn.pt"
                cp2 = msg.get('agent2_checkpoint')
                obs_size = env_left.obs_size
                n_actions = env_left.N_ACTIONS

                def _make_agent(checkpoint_name):
                    """Load DQN from exact checkpoint filename, or fall back to Random."""
                    if not checkpoint_name:
                        return RandomAgent(), 'Random agent'
                    try:
                        from agents.dqn_agent import DQNAgent
                        agent = DQNAgent.from_name(checkpoint_name, obs_size, n_actions)
                        label = checkpoint_name.replace('checkpoint_', '').replace('.pt', '')
                        return agent, label
                    except (FileNotFoundError, Exception) as e:
                        print(f"[WS/versus] Could not load checkpoint {checkpoint_name}: {e}")
                        return RandomAgent(), 'Random agent'

                if mode == 'agent_vs_agent':
                    agent_left,  label1 = _make_agent(cp1)
                    agent_right, label2 = _make_agent(cp2)
                else:  # human_vs_agent
                    agent_left  = None
                    label1      = 'Human'
                    agent_right, label2 = _make_agent(ep1)

                running = True
                stop_flag = False
                if loop_task:
                    loop_task.cancel()
                loop_task = asyncio.create_task(game_loop())
                await ws.send_text(json.dumps({
                    'type': 'ack',
                    'msg': f'{mode} started',
                    'agent_label': label2,
                }))


            elif cmd == 'action':
                human_action = int(msg.get('action', 2))

            elif cmd == 'stop':
                running = False
                stop_flag = True
                if loop_task:
                    loop_task.cancel()
                await ws.send_text(json.dumps({'type': 'ack', 'msg': 'Stopped'}))

    except WebSocketDisconnect:
        print("[WS/versus] Client disconnected")
        running = False
        if loop_task:
            loop_task.cancel()


# ── REST endpoints ─────────────────────────────────────────────────────────
@app.get("/api/checkpoints")
def list_checkpoints():
    mgr = CheckpointManager()
    files = mgr.list_checkpoints()
    result = []
    for f in files:
        name = os.path.basename(f)
        # Parse ep and algorithm from filename: checkpoint_ep500_dueling_double_dqn.pt
        parts = name.replace('.pt', '').split('_ep')
        episode = int(parts[1].split('_')[0]) if len(parts) > 1 else 0
        algo = '_'.join(parts[1].split('_')[1:]) if len(parts) > 1 else 'unknown'
        result.append({'path': f, 'episode': episode, 'algorithm': algo, 'name': name})
    return {'checkpoints': result}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
