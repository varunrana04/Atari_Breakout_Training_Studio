import { useRef, useState, useEffect, useCallback } from 'react';
import RewardChart from '../components/RewardChart.jsx';
import StatsPanel from '../components/StatsPanel.jsx';
import ControlPanel from '../components/ControlPanel.jsx';
import { CANVAS_W, CANVAS_H } from '../game/breakout.js';

const WS_URL = 'ws://localhost:8000/ws/train';

const DEFAULT_HYPERPARAMS = {
  algorithm: 'dueling_double_dqn',
  episodes: 10000,
  learning_rate: 0.00025,
  gamma: 0.99,
  replay_buffer_size: 100000,
  batch_size: 32,
  epsilon_start: 1.0,
  epsilon_final: 0.01,
  epsilon_decay_steps: 1000000,
  target_update_steps: 10000,
};

export default function Studio({ settings, onExit, onVersus }) {
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const animRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [training, setTraining] = useState(false);
  const [hyperparams, setHyperparams] = useState(DEFAULT_HYPERPARAMS);

  // Live stats
  const [stats, setStats] = useState({
    episode: 0, reward: 0, avgReward: 0,
    epsilon: 1.0, loss: 0, qValue: 0, speed: 0,
  });

  // Chart data
  const [rewardHistory, setRewardHistory] = useState([]);
  const [lossHistory, setLossHistory] = useState([]);
  const [epsilonHistory, setEpsilonHistory] = useState([]);
  const [qHistory, setQHistory] = useState([]);

  // Live game frame
  const [gameFrame, setGameFrame] = useState(null);

  // ── WebSocket connection ──────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setTraining(false); };
    ws.onerror = () => { setConnected(false); };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'stats') {
          setStats({
            episode: data.episode,
            reward: data.reward,
            avgReward: data.avg_reward,
            epsilon: data.epsilon,
            loss: data.loss,
            qValue: data.q_value,
            speed: data.eps_per_hour,
          });
          setRewardHistory(h => [...h.slice(-500), { x: data.episode, y: data.avg_reward }]);
          setLossHistory(h => [...h.slice(-500), { x: data.episode, y: data.loss }]);
          setEpsilonHistory(h => [...h.slice(-500), { x: data.episode, y: data.epsilon }]);
          setQHistory(h => [...h.slice(-500), { x: data.episode, y: data.q_value }]);
        }

        if (data.type === 'frame') {
          setGameFrame(data.frame); // { bricks, ballX, ballY, paddleX, paddleW }
        }
      } catch (_) {}
    };
  }, []);

  // Draw game frame from backend state onto canvas
  useEffect(() => {
    if (!gameFrame || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    drawFrame(ctx, gameFrame, CANVAS_W, CANVAS_H);
  }, [gameFrame]);

  const sendCommand = (cmd, payload = {}) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ cmd, ...payload }));
  };

  const handleStart = () => {
    if (!connected) connect();
    setTimeout(() => {
      sendCommand('start', { hyperparams, powerUpsEnabled: settings.powerUpsEnabled });
      setTraining(true);
      setRewardHistory([]); setLossHistory([]); setEpsilonHistory([]); setQHistory([]);
    }, 300);
  };

  const handlePause = () => {
    sendCommand(training ? 'pause' : 'resume');
    setTraining(t => !t);
  };

  const handleSave = () => sendCommand('save_checkpoint');
  const handleLoad = () => {
    const ep = prompt('Enter episode number to load checkpoint:');
    if (ep) sendCommand('load_checkpoint', { episode: parseInt(ep) });
  };

  // Cleanup on unmount
  useEffect(() => () => { wsRef.current?.close(); }, []);

  return (
    <div style={styles.root}>
      {/* Top bar */}
      <div style={styles.topBar}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={styles.topTitle}>Training Studio</span>
          <span style={styles.connDot} className={connected ? 'badge badge-green' : 'badge badge-orange'}>
            {connected ? 'Connected' : 'Offline'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-sm" onClick={onVersus} id="btn-studio-versus">
            Versus Mode
          </button>
          <button className="btn btn-sm btn-danger" onClick={onExit} id="btn-exit-studio">
            Exit
          </button>
        </div>
      </div>

      <div style={styles.mainLayout}>
        {/* Left column: game canvas + stats */}
        <div style={styles.leftCol}>
          <div className="panel" style={{ padding: 12 }}>
            <canvas
              ref={canvasRef}
              width={CANVAS_W}
              height={CANVAS_H}
              style={styles.canvas}
              id="studio-canvas"
            />
            {!connected && (
              <div style={styles.offlineOverlay}>
                <p style={styles.offlineText}>Start backend to see live gameplay</p>
                <code style={styles.offlineCode}>python backend/main.py</code>
              </div>
            )}
          </div>
          <StatsPanel stats={stats} />
        </div>

        {/* Right column: charts + controls */}
        <div style={styles.rightCol}>
          <div className="panel">
            <RewardChart
              rewardData={rewardHistory}
              lossData={lossHistory}
              epsilonData={epsilonHistory}
              qData={qHistory}
            />
          </div>
          <ControlPanel
            hyperparams={hyperparams}
            onHyperparamsChange={setHyperparams}
            training={training}
            connected={connected}
            onConnect={connect}
            onStart={handleStart}
            onPause={handlePause}
            onSave={handleSave}
            onLoad={handleLoad}
          />
        </div>
      </div>
    </div>
  );
}

// ── Lightweight frame renderer ──────────────────────────────────────────────
function drawFrame(ctx, frame, W, H) {
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, W, H);

  // Bricks
  if (frame.bricks) {
    for (const b of frame.bricks) {
      if (!b.alive) continue;
      const colors = { 1: '#e2e8f0', 2: '#60a5fa', 3: '#f97316', 9: '#6b7280' };
      ctx.fillStyle = colors[b.hp] ?? '#94a3b8';
      ctx.fillRect(b.x, b.y, b.w, b.h);
    }
  }

  // Ball
  if (frame.ballX != null) {
    ctx.beginPath();
    ctx.arc(frame.ballX * W, frame.ballY * H, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#f8fafc';
    ctx.fill();
  }

  // Paddle
  if (frame.paddleX != null) {
    const pw = (frame.paddleW ?? 80);
    const px = frame.paddleX * W - pw / 2;
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(px, H - 40, pw, 12);
  }
}

const styles = {
  root: { minHeight: '100vh', display: 'flex', flexDirection: 'column', padding: 16, gap: 16 },
  topBar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '10px 4px',
  },
  topTitle: { fontSize: 18, fontWeight: 700, marginRight: 12 },
  connDot: { verticalAlign: 'middle' },
  mainLayout: { display: 'flex', gap: 16, flex: 1, alignItems: 'flex-start' },
  leftCol: { display: 'flex', flexDirection: 'column', gap: 12, flexShrink: 0 },
  rightCol: { display: 'flex', flexDirection: 'column', gap: 12, flex: 1, minWidth: 0 },
  canvas: { display: 'block', borderRadius: 4 },
  offlineOverlay: {
    marginTop: 8,
    padding: '10px 14px',
    background: 'var(--bg-900)',
    borderRadius: 'var(--radius-sm)',
    textAlign: 'center',
  },
  offlineText: { color: 'var(--text-muted)', fontSize: 13, marginBottom: 6 },
  offlineCode: {
    fontFamily: 'var(--font-mono)',
    fontSize: 12,
    color: 'var(--accent-green)',
    background: 'rgba(34,197,94,0.08)',
    padding: '4px 10px',
    borderRadius: 4,
  },
};
