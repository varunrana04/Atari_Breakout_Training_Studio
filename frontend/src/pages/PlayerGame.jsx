import { useEffect, useRef, useState } from 'react';
import { BreakoutGame, CANVAS_W, CANVAS_H } from '../game/breakout.js';
import PauseMenu from '../components/PauseMenu.jsx';

export default function PlayerGame({ settings, onExit }) {
  const canvasRef = useRef(null);
  const gameRef = useRef(null);
  const [paused, setPaused] = useState(false);
  const [score, setScore] = useState(0);
  const [lives, setLives] = useState(3);
  const [level, setLevel] = useState(1);
  const [gameOver, setGameOver] = useState(false);
  const [won, setWon] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const game = new BreakoutGame(canvas, 'player', {
      powerUpsEnabled: settings.powerUpsEnabled,
      onScoreChange: setScore,
      onLivesChange: setLives,
      onLevelChange: setLevel,
      onGameOver: (finalScore, didWin) => {
        setGameOver(true);
        if (didWin) setWon(true);
      },
      // This fires from both ESC key AND the Pause button — single source of truth
      onPauseToggle: (isPaused) => setPaused(isPaused),
    });
    gameRef.current = game;
    game.start();

    // Click canvas to launch ball
    const handleClick = () => { if (!game.paused && game.waitingForLaunch) game.launch(); };
    canvas.addEventListener('click', handleClick);

    return () => {
      game.destroy();
      canvas.removeEventListener('click', handleClick);
    };
  }, []);

  const handlePause = () => {
    const game = gameRef.current;
    if (!game || game.gameOver) return;
    // togglePause() handles both game.paused flip AND onPauseToggle callback
    game.togglePause();
  };

  const handleResume = () => {
    const game = gameRef.current;
    if (!game) return;
    game.paused = false;
    game.onPauseToggle?.(false);
    setPaused(false);
  };

  const handleRestart = () => {
    const game = gameRef.current;
    if (!game) return;
    game.reset();
    setScore(0); setLives(3); setLevel(1); setGameOver(false); setWon(false); setPaused(false);
    game.paused = false;
  };

  return (
    <div style={styles.root}>
      {/* HUD */}
      <div style={styles.hud}>
        <div style={styles.hudItem}>
          <span className="stat-label">Score</span>
          <span className="stat-value">{score}</span>
        </div>
        <div style={styles.hudItem}>
          <span className="stat-label">Level</span>
          <span className="stat-value">{level} / 100</span>
        </div>
        <div style={styles.hudItem}>
          <span className="stat-label">Lives</span>
          <span className="stat-value" style={{ color: lives <= 1 ? 'var(--accent-red)' : undefined }}>
            {'● '.repeat(lives).trim()}
          </span>
        </div>
        <button
          className="btn btn-sm"
          onClick={handlePause}
          id="btn-pause"
          style={{ marginLeft: 'auto' }}
        >
          {paused ? 'Paused' : 'Pause'}
        </button>
      </div>

      {/* Game canvas */}
      <div style={styles.canvasWrapper}>
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          style={styles.canvas}
          id="game-canvas"
        />

        {/* Pause overlay */}
        {paused && !gameOver && (
          <PauseMenu onResume={handleResume} onExit={onExit} />
        )}

        {/* Game over overlay */}
        {gameOver && (
          <div style={styles.overlay}>
            <div className="panel" style={styles.overlayPanel}>
              <h2 style={styles.overlayTitle}>{won ? 'You Win!' : 'Game Over'}</h2>
              <p style={styles.overlayScore}>Final Score: <strong>{score}</strong></p>
              <div style={styles.overlayActions}>
                <button className="btn btn-green" onClick={handleRestart} id="btn-restart">
                  Play Again
                </button>
                <button className="btn" onClick={onExit} id="btn-exit-gameover">
                  Exit
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  root: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '20px',
    gap: 16,
  },
  hud: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: 32,
    width: '100%',
    maxWidth: 480,
    padding: '12px 16px',
    background: 'var(--bg-800)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
  },
  hudItem: { display: 'flex', flexDirection: 'column', gap: 2 },
  canvasWrapper: { position: 'relative' },
  canvas: {
    display: 'block',
    borderRadius: 8,
    border: '1px solid var(--border)',
  },
  overlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(15,23,42,0.85)',
    borderRadius: 8,
  },
  overlayPanel: { textAlign: 'center', minWidth: 260 },
  overlayTitle: { fontSize: 28, fontWeight: 800, marginBottom: 8 },
  overlayScore: { color: 'var(--text-secondary)', marginBottom: 24 },
  overlayActions: { display: 'flex', gap: 12, justifyContent: 'center' },
};
