/**
 * VersusMode.jsx
 * Human vs Agent  |  Agent vs Agent
 *
 * Architecture: Python BreakoutEnv is the canonical environment.
 * The browser does NOT run its own physics. It connects to /ws/versus,
 * the backend runs BreakoutEnv.step() using the loaded DQN checkpoints,
 * and sends authoritative game state back every tick for the browser to render.
 *
 * For Human vs Agent: human key presses are sent as { cmd: 'action', action: 0|1|2 }.
 * For Agent vs Agent: both sides use loaded .pt checkpoints via DQNAgent.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { CANVAS_W, CANVAS_H } from '../game/breakout.js';

const WS_URL = 'ws://localhost:8000/ws/versus';

// ── Canvas renderer ────────────────────────────────────────────────────────
// Draws the Python-authoritative game state onto a canvas.
function renderState(canvas, state) {
  if (!canvas || !state) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  // Background
  ctx.fillStyle = '#060d1a';
  ctx.fillRect(0, 0, W, H);

  // Bricks
  const brickData = state.bricks || [];
  const COLS = 12, ROWS = 8;
  const bW = W / COLS;
  const bH = (H * 0.45) / ROWS;
  const BRICK_COLORS = ['', '#3b8ff5', '#10d97c', '#f97316'];

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const hp = brickData[r * COLS + c] ?? 0;
      if (hp <= 0) continue;
      const x = c * bW + 1;
      const y = 40 + r * bH + 1;
      ctx.fillStyle = BRICK_COLORS[Math.min(hp, 3)] || '#3b8ff5';
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x, y, bW - 2, bH - 2);
      ctx.globalAlpha = 1;
    }
  }

  // Ball
  const bx = (state.ball_x ?? 0.5) * W;
  const by = (state.ball_y ?? 0.5) * H;
  ctx.beginPath();
  ctx.arc(bx, by, 7, 0, Math.PI * 2);
  ctx.fillStyle = '#f0f6ff';
  ctx.fill();

  // Paddle
  const px = (state.paddle_x ?? 0.5) * W;
  const pw = (state.paddle_w ?? 0.15) * W;
  ctx.fillStyle = '#3b8ff5';
  ctx.beginPath();
  ctx.roundRect(px - pw / 2, H - 18, pw, 10, 4);
  ctx.fill();
}

// ── Versus Mode Component ──────────────────────────────────────────────────
export default function VersusMode({ mode, agent1Checkpoint, agent2Checkpoint, onExit }) {
  const leftCanvasRef  = useRef(null);
  const rightCanvasRef = useRef(null);
  const wsRef          = useRef(null);
  const keysRef        = useRef({});

  const [status, setStatus]       = useState('connecting'); // connecting | running | error
  const [errorMsg, setErrorMsg]   = useState('');
  const [leftScore,  setLeftScore]  = useState(0);
  const [rightScore, setRightScore] = useState(0);
  const [leftLives,  setLeftLives]  = useState(3);
  const [rightLives, setRightLives] = useState(3);
  const [agentLabel, setAgentLabel] = useState('DQN Agent');
  const [countdown, setCountdown]  = useState(3);

  const isHumanVsAgent = mode === 'human_vs_agent';

  // ── Connect to backend WebSocket ─────────────────────────────────────────
  useEffect(() => {
    let started = false;
    const t1 = setTimeout(() => setCountdown(2), 1000);
    const t2 = setTimeout(() => setCountdown(1), 2000);
    const t3 = setTimeout(() => {
      setCountdown(0);

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('running');
        ws.send(JSON.stringify({
          cmd: 'start',
          mode,
          agent1_checkpoint: agent1Checkpoint ?? null,
          agent2_checkpoint: agent2Checkpoint ?? null,
        }));
        started = true;
      };

      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);

        if (data.type === 'error') {
          setStatus('error');
          setErrorMsg(data.msg || 'Unknown error');
          return;
        }

        if (data.type === 'ack') {
          // e.g. "human_vs_agent started (DQN ep3000)"
          if (data.agent_label) setAgentLabel(data.agent_label);
          return;
        }

        if (data.type === 'frame') {
          // Render Python-authoritative state onto both canvases
          renderState(leftCanvasRef.current,  data.left);
          renderState(rightCanvasRef.current, data.right);

          // Update score/lives HUD
          if (data.left.score  !== undefined) setLeftScore(data.left.score);
          if (data.right.score !== undefined) setRightScore(data.right.score);
          if (data.left.lives  !== undefined) setLeftLives(data.left.lives);
          if (data.right.lives !== undefined) setRightLives(data.right.lives);
        }
      };

      ws.onerror = () => {
        setStatus('error');
        setErrorMsg('WebSocket connection failed. Make sure the backend is running on port 8000.');
      };

      ws.onclose = () => {
        if (started) setStatus('connecting');
      };
    }, 3000);

    return () => {
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3);
      if (wsRef.current) {
        wsRef.current.send(JSON.stringify({ cmd: 'stop' }));
        wsRef.current.close();
      }
    };
  }, [mode, agent1Episode, agent2Episode, isHumanVsAgent]);

  // ── Keyboard input: send human actions to backend ─────────────────────────
  useEffect(() => {
    if (!isHumanVsAgent) return;

    const sendAction = (action) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ cmd: 'action', action }));
      }
    };

    // Send action on keydown, send noop on keyup
    const onDown = (e) => {
      if (e.code === 'ArrowLeft' || e.code === 'KeyA') sendAction(0);
      if (e.code === 'ArrowRight' || e.code === 'KeyD') sendAction(1);
    };
    const onUp = (e) => {
      if (['ArrowLeft','KeyA','ArrowRight','KeyD'].includes(e.code)) sendAction(2);
    };

    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup',   onUp);
    return () => {
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup',   onUp);
    };
  }, [isHumanVsAgent]);

  const handleRestart = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        cmd: 'start',
        mode,
        agent1_checkpoint: agent1Checkpoint ?? null,
        agent2_checkpoint: agent2Checkpoint ?? null,
      }));
    }
    setLeftScore(0); setRightScore(0);
    setLeftLives(3); setRightLives(3);
  };

  const leftLabel  = isHumanVsAgent ? 'YOU' : `Agent 1 — ${agent1Checkpoint ?? 'random'}`;
  const rightLabel = isHumanVsAgent
    ? `${agentLabel} — ${agent1Checkpoint ?? 'random'}`
    : `Agent 2 — ${agent2Checkpoint ?? 'random'}`;

  return (
    <div style={styles.root}>
      {/* Top bar */}
      <div style={styles.topBar}>
        <span style={styles.title}>{isHumanVsAgent ? 'Human vs Agent' : 'Agent vs Agent'}</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {status === 'running' && <span className="badge badge-green">Live</span>}
          {status === 'connecting' && <span className="badge badge-blue">Connecting…</span>}
          {status === 'error' && <span className="badge badge-orange">Error</span>}
          <button className="btn btn-sm" onClick={handleRestart} id="btn-restart-versus">Restart</button>
          <button className="btn btn-sm" onClick={onExit} id="btn-exit-versus">Exit</button>
        </div>
      </div>

      {isHumanVsAgent && (
        <p style={styles.hint}>← → or A D to move · The right side is the trained DQN checkpoint</p>
      )}

      {status === 'error' && (
        <div style={styles.errorBox}>{errorMsg}</div>
      )}

      {/* Countdown overlay */}
      {countdown > 0 && (
        <div style={styles.countdownWrap}>
          <span style={styles.countdownNum}>{countdown}</span>
        </div>
      )}

      {/* Arenas */}
      <div style={styles.arenas}>
        {/* Left */}
        <div style={styles.arenaBlock}>
          <div style={styles.arenaHeader}>
            <span style={styles.arenaLabel}>{leftLabel}</span>
            <span style={styles.arenaScore}>{leftScore}</span>
          </div>
          <LivesBar lives={leftLives} />
          <canvas
            ref={leftCanvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            style={styles.canvas}
            id="canvas-versus-left"
          />
        </div>

        <div style={styles.vsDivider}><span style={styles.vsText}>VS</span></div>

        {/* Right */}
        <div style={styles.arenaBlock}>
          <div style={styles.arenaHeader}>
            <span style={styles.arenaLabel}>{rightLabel}</span>
            <span style={styles.arenaScore}>{rightScore}</span>
          </div>
          <LivesBar lives={rightLives} />
          <canvas
            ref={rightCanvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            style={styles.canvas}
            id="canvas-versus-right"
          />
        </div>
      </div>

      <p style={styles.footer}>
        Both sides run on the Python BreakoutEnv — same physics the agent was trained on.
        {isHumanVsAgent ? ' Right side runs your trained DQN checkpoint at ε=0.' : ' Both agents run greedy DQN inference (ε=0).'}
      </p>
    </div>
  );
}

function LivesBar({ lives }) {
  return (
    <div style={{ display: 'flex', gap: 5, marginBottom: 4 }}>
      {[1, 2, 3].map(i => (
        <div key={i} style={{
          width: 10, height: 10, borderRadius: '50%',
          background: i <= lives ? 'var(--accent-red)' : 'var(--bg-700)',
          transition: 'background 0.2s',
        }} />
      ))}
    </div>
  );
}

const styles = {
  root: {
    minHeight: '100vh', display: 'flex', flexDirection: 'column',
    alignItems: 'center', padding: '20px 16px', gap: 12,
    background: 'var(--bg-950)',
  },
  topBar: {
    width: '100%', maxWidth: 1020,
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  },
  title: { fontSize: 20, fontWeight: 700, letterSpacing: '0.02em' },
  hint: {
    color: 'var(--text-muted)', fontSize: 12,
    background: 'var(--bg-800)', borderRadius: 6, padding: '5px 14px', margin: 0,
  },
  errorBox: {
    background: 'rgba(249,115,22,0.08)', border: '1px solid rgba(249,115,22,0.3)',
    borderRadius: 8, padding: '10px 16px', maxWidth: 600,
    color: '#fdba74', fontSize: 13,
  },
  arenas: { display: 'flex', gap: 0, alignItems: 'flex-start' },
  arenaBlock: { display: 'flex', flexDirection: 'column', alignItems: 'center' },
  arenaHeader: {
    width: '100%', display: 'flex', justifyContent: 'space-between',
    alignItems: 'baseline', padding: '0 4px', marginBottom: 4,
  },
  arenaLabel: {
    fontSize: 11, fontWeight: 800, letterSpacing: '0.1em',
    textTransform: 'uppercase', color: 'var(--text-muted)',
  },
  arenaScore: {
    fontFamily: 'var(--font-mono)', fontSize: 24, fontWeight: 700,
    color: 'var(--text-primary)',
  },
  canvas: { display: 'block', border: '1px solid var(--border)', borderRadius: 6 },
  vsDivider: {
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    padding: '0 20px', marginTop: 80,
  },
  vsText: { fontSize: 26, fontWeight: 900, color: 'var(--text-muted)', letterSpacing: '0.12em' },
  countdownWrap: {
    position: 'fixed', inset: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'rgba(15,23,42,0.75)', zIndex: 50, pointerEvents: 'none',
  },
  countdownNum: {
    fontSize: 120, fontWeight: 900, color: 'var(--accent-blue)',
    textShadow: '0 0 60px rgba(59,130,246,0.6)',
  },
  footer: { color: 'var(--text-muted)', fontSize: 11, textAlign: 'center', margin: 0, maxWidth: 700 },
};
