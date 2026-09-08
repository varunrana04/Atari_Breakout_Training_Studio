import { useEffect, useRef, useState } from 'react';
import { CANVAS_W, CANVAS_H } from '../game/breakout.js';

const WS_URL = 'ws://localhost:8001/ws/versus';

function renderState(canvas, state) {
  if (!canvas || !state) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  
  ctx.fillStyle = '#060d1a';
  ctx.fillRect(0, 0, W, H);

  // Bricks
  const brickData = state.bricks || [];
  const COLS = 12, ROWS = 8;
  const bW = 34, bH = 18;
  const PADDING = 4, OFFSET_X = 10, OFFSET_Y = 60;

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const hp = brickData[r * COLS + c] ?? 0;
      if (hp <= 0) continue;
      const colors = { 1: '#e2e8f0', 2: '#60a5fa', 3: '#f97316', 9: '#6b7280' };
      ctx.fillStyle = colors[hp] ?? '#94a3b8';
      ctx.fillRect(OFFSET_X + c * (bW + PADDING), OFFSET_Y + r * (bH + PADDING), bW, bH);
    }
  }

  // Ball
  if (state.ball_x != null) {
    ctx.beginPath();
    ctx.arc(state.ball_x * W, state.ball_y * H, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#f0f6ff';
    ctx.fill();
  }

  // Paddle
  if (state.paddle_x != null) {
    const pw = (state.paddle_w * W || 80);
    const px = state.paddle_x * W - pw / 2;
    ctx.fillStyle = '#3b8ff5';
    ctx.beginPath();
    ctx.roundRect(px, H - 40, pw, 12, 4);
    ctx.fill();
  }
}

function LivesBar({ lives }) {
  return (
    <div style={{ display: 'flex', gap: 5, marginBottom: 4 }}>
      {[1, 2, 3].map(i => (
        <div key={i} style={{
          width: 10, height: 10, borderRadius: '50%',
          background: i <= lives ? '#ef4444' : 'var(--bg-700)',
        }} />
      ))}
    </div>
  );
}

export default function VersusMode({ mode, agent1Checkpoint, agent2Checkpoint, onExit }) {
  const leftRef  = useRef(null);
  const rightRef = useRef(null);
  const wsRef    = useRef(null);

  const [status,     setStatus]     = useState('connecting');
  const [errorMsg,   setErrorMsg]   = useState('');
  const [leftScore,  setLeftScore]  = useState(0);
  const [rightScore, setRightScore] = useState(0);
  const [leftLives,  setLeftLives]  = useState(3);
  const [rightLives, setRightLives] = useState(3);
  const [countdown,  setCountdown]  = useState(3);
  const [agentLabel, setAgentLabel] = useState('DQN Agent');

  const isHuman = mode === 'human_vs_agent';

  useEffect(() => {
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
      };

      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type === 'error') {
          setStatus('error');
          setErrorMsg(data.msg || 'Unknown error');
        } else if (data.type === 'ack') {
          if (data.agent_label) setAgentLabel(data.agent_label);
        } else if (data.type === 'frame') {
          renderState(leftRef.current,  data.left);
          renderState(rightRef.current, data.right);
          if (data.left.score  != null) setLeftScore(data.left.score);
          if (data.right.score != null) setRightScore(data.right.score);
          if (data.left.lives  != null) setLeftLives(data.left.lives);
          if (data.right.lives != null) setRightLives(data.right.lives);
        }
      };

      ws.onerror = () => {
        setStatus('error');
        setErrorMsg('WebSocket failed. Is the backend running on port 8000?');
      };

      ws.onclose = () => setStatus('connecting');
    }, 3000);

    return () => {
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3);
      if (wsRef.current) {
        try { wsRef.current.send(JSON.stringify({ cmd: 'stop' })); } catch (_) {}
        wsRef.current.close();
      }
    };
  }, [mode, agent1Checkpoint, agent2Checkpoint]);

  useEffect(() => {
    if (!isHuman) return;
    const send = (a) => wsRef.current?.readyState === 1 &&
      wsRef.current.send(JSON.stringify({ cmd: 'action', action: a }));
    
    // Keyboard support
    const onDown = (e) => {
      if (e.code === 'ArrowLeft'  || e.code === 'KeyA') send(0);
      if (e.code === 'ArrowRight' || e.code === 'KeyD') send(1);
    };
    const onUp   = (e) => {
      if (['ArrowLeft','KeyA','ArrowRight','KeyD'].includes(e.code)) send(2);
    };
    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup',   onUp);

    // Mouse support
    let mouseTimeout;
    const onMouseMove = (e) => {
      if (e.movementX < 0) send(0);
      else if (e.movementX > 0) send(1);
      
      clearTimeout(mouseTimeout);
      mouseTimeout = setTimeout(() => send(2), 50); // stop if mouse stops
    };
    window.addEventListener('mousemove', onMouseMove);

    return () => {
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup',   onUp);
      window.removeEventListener('mousemove', onMouseMove);
      clearTimeout(mouseTimeout);
    };
  }, [isHuman]);

  const restart = () => {
    if (wsRef.current?.readyState === 1) {
      wsRef.current.send(JSON.stringify({
        cmd: 'start', mode,
        agent1_checkpoint: agent1Checkpoint ?? null,
        agent2_checkpoint: agent2Checkpoint ?? null,
      }));
    }
    setLeftScore(0); setRightScore(0); setLeftLives(3); setRightLives(3);
  };

  const leftLabel  = isHuman ? 'YOU' : `Agent 1`;
  const rightLabel = isHuman ? agentLabel : `Agent 2`;

  return (
    <div style={{ minHeight:'100vh', display:'flex', flexDirection:'column',
      alignItems:'center', padding:'20px 16px', gap:12, background:'var(--bg-950)' }}>

      {/* Top bar */}
      <div style={{ width:'100%', maxWidth:1020, display:'flex',
        alignItems:'center', justifyContent:'space-between' }}>
        <span style={{ fontSize:20, fontWeight:700 }}>
          {isHuman ? 'Human vs Agent' : 'Agent vs Agent'}
        </span>
        <div style={{ display:'flex', gap:8, alignItems:'center' }}>
          {status === 'running'    && <span className="badge badge-green">Live</span>}
          {status === 'connecting' && <span className="badge badge-blue">Connecting…</span>}
          {status === 'error'      && <span className="badge badge-orange">Error</span>}
          <button className="btn btn-sm" onClick={restart} id="btn-restart-versus">Restart</button>
          <button className="btn btn-sm" onClick={onExit}  id="btn-exit-versus">Exit</button>
        </div>
      </div>

      {isHuman && (
        <p style={{ color:'var(--text-muted)', fontSize:12, margin:0,
          background:'var(--bg-800)', borderRadius:6, padding:'5px 14px' }}>
          ← → or A D to move · Right side is the trained DQN agent
        </p>
      )}

      {status === 'error' && (
        <div style={{ background:'rgba(249,115,22,0.08)', border:'1px solid rgba(249,115,22,0.3)',
          borderRadius:8, padding:'10px 16px', maxWidth:600, color:'#fdba74', fontSize:13 }}>
          {errorMsg}
        </div>
      )}

      {/* Countdown overlay */}
      {countdown > 0 && (
        <div style={{ position:'fixed', inset:0, display:'flex', alignItems:'center',
          justifyContent:'center', background:'rgba(15,23,42,0.75)', zIndex:50 }}>
          <span style={{ fontSize:120, fontWeight:900, color:'var(--accent-blue)',
            textShadow:'0 0 60px rgba(59,130,246,0.6)' }}>{countdown}</span>
        </div>
      )}

      {/* Arenas */}
      <div style={{ display:'flex', gap:0, alignItems:'flex-start' }}>
        {mode !== 'watch_agent' && (
          <>
            <div style={{ display:'flex', flexDirection:'column', alignItems:'center' }}>
              <div style={{ width:'100%', display:'flex', justifyContent:'space-between',
                alignItems:'baseline', padding:'0 4px', marginBottom:4 }}>
                <span style={{ fontSize:11, fontWeight:800, letterSpacing:'0.1em',
                  textTransform:'uppercase', color:'var(--text-muted)' }}>{leftLabel}</span>
                <span style={{ fontFamily:'var(--font-mono)', fontSize:24, fontWeight:700 }}>{leftScore}</span>
              </div>
              <LivesBar lives={leftLives} />
              <canvas ref={leftRef} width={CANVAS_W} height={CANVAS_H}
                id="canvas-versus-left"
                style={{ display:'block', border:'1px solid var(--border)', borderRadius:6 }} />
            </div>

            <div style={{ display:'flex', flexDirection:'column', alignItems:'center',
              justifyContent:'center', padding:'0 20px', marginTop:80 }}>
              <span style={{ fontSize:26, fontWeight:900, color:'var(--text-muted)' }}>VS</span>
            </div>
          </>
        )}

        <div style={{ display:'flex', flexDirection:'column', alignItems:'center' }}>
          <div style={{ width:'100%', display:'flex', justifyContent:'space-between',
            alignItems:'baseline', padding:'0 4px', marginBottom:4 }}>
            <span style={{ fontSize:11, fontWeight:800, letterSpacing:'0.1em',
              textTransform:'uppercase', color:'var(--text-muted)' }}>{mode === 'watch_agent' ? agentLabel : rightLabel}</span>
            <span style={{ fontFamily:'var(--font-mono)', fontSize:24, fontWeight:700 }}>{rightScore}</span>
          </div>
          <LivesBar lives={rightLives} />
          <canvas ref={rightRef} width={CANVAS_W} height={CANVAS_H}
            id="canvas-versus-right"
            style={{ display:'block', border:'1px solid var(--border)', borderRadius:6 }} />
        </div>
      </div>

      <p style={{ color:'var(--text-muted)', fontSize:11, textAlign:'center', margin:0, maxWidth:700 }}>
        {mode === 'watch_agent' ? 'The agent is playing on the Python BreakoutEnv using real DQN checkpoint inference (ε=0).' :
         isHuman ? 'Both sides run on the Python BreakoutEnv. Right side uses real DQN checkpoint inference (ε=0).'
                 : 'Both sides run on the Python BreakoutEnv. Both agents use real DQN checkpoint inference (ε=0).'}
      </p>
    </div>
  );
}
