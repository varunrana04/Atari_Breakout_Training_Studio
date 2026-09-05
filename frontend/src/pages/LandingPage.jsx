import { useState } from 'react';

export default function LandingPage({ onPlayPlayer, onOpenStudio, onVersus, settings, onSettingsChange }) {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div style={styles.root}>
      {/* Top-left brand */}
      <div style={styles.header}>
        <div style={styles.brandDot} />
        <span style={styles.brandLabel}>SAPIEN ROBOTICS</span>
      </div>

      {/* Background glow */}
      <div style={styles.bgGlow} />

      {/* Title block */}
      <div style={styles.titleBlock}>
        <div style={styles.eyebrow}>Reinforcement Learning · 100 Levels · DQN</div>
        <h1 style={styles.title}>
          <span className="hero-gradient">Breakout</span>
        </h1>
        <p style={styles.subtitle}>TRAINING STUDIO</p>
        <p style={styles.description}>
          Train a DQN agent from scratch, watch it learn in real time,<br />
          or play the game yourself across 100 progressive levels.
        </p>
      </div>

      {/* Main action cards */}
      <div style={styles.cards}>
        <ActionCard
          id="btn-open-studio"
          onClick={onOpenStudio}
          primary
          icon="⬡"
          title="Training Studio"
          desc="Train DQN / Double DQN / Dueling agent with live charts and Gemini AI advisor"
        />
        <ActionCard
          id="btn-play-player"
          onClick={onPlayPlayer}
          icon="▷"
          title="Play"
          desc="Play Breakout yourself — 100 levels, power-ups, live score tracking"
        />
        <ActionCard
          id="btn-versus"
          onClick={onVersus}
          icon="⇌"
          title="Versus Mode"
          desc="Human vs AI or AI vs AI — split-screen side-by-side"
        />
      </div>

      {/* Settings */}
      <button
        className="btn btn-sm"
        onClick={() => setShowSettings(s => !s)}
        id="btn-settings"
        style={{ opacity: 0.7 }}
      >
        {showSettings ? '— Close Settings' : '⚙ Settings'}
      </button>

      {showSettings && (
        <div className="panel" style={styles.settingsPanel}>
          <p style={styles.settingsTitle}>Settings</p>
          <hr className="divider" />
          <div style={styles.settingsRow}>
            <div>
              <div style={styles.settingName}>Power-ups</div>
              <div style={styles.settingDesc}>Enable falling power-ups during gameplay</div>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.powerUpsEnabled}
                onChange={e => onSettingsChange({ ...settings, powerUpsEnabled: e.target.checked })}
                id="toggle-powerups"
              />
              <span className="toggle-slider" />
            </label>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={styles.footer}>
        <span style={styles.footerText}>100 levels · 6 power-ups · Gemini AI Advisor</span>
      </div>
    </div>
  );
}

function ActionCard({ id, onClick, primary, icon, title, desc }) {
  return (
    <button
      id={id}
      onClick={onClick}
      className="card-hover"
      style={{
        ...styles.card,
        ...(primary ? styles.cardPrimary : {}),
      }}
    >
      <span style={styles.cardIcon}>{icon}</span>
      <div style={styles.cardText}>
        <div style={styles.cardTitle}>{title}</div>
        <div style={styles.cardDesc}>{desc}</div>
      </div>
      <span style={styles.cardArrow}>→</span>
    </button>
  );
}

const styles = {
  root: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '60px 20px 40px',
    gap: 28,
    position: 'relative',
    overflow: 'hidden',
  },
  bgGlow: {
    position: 'absolute',
    top: '20%', left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 600, height: 400,
    background: 'radial-gradient(ellipse at center, rgba(59,143,245,0.12) 0%, transparent 70%)',
    pointerEvents: 'none',
    zIndex: 0,
  },
  header: {
    position: 'absolute',
    top: 24, left: 32,
    display: 'flex', alignItems: 'center', gap: 8,
    zIndex: 2,
  },
  brandDot: {
    width: 8, height: 8,
    borderRadius: '50%',
    background: 'var(--accent-blue)',
    boxShadow: '0 0 8px var(--accent-blue)',
  },
  brandLabel: {
    fontSize: 10,
    fontWeight: 800,
    letterSpacing: '0.18em',
    textTransform: 'uppercase',
    color: 'var(--text-muted)',
  },
  titleBlock: {
    textAlign: 'center',
    maxWidth: 560,
    position: 'relative', zIndex: 1,
  },
  eyebrow: {
    fontSize: 10, fontWeight: 700, letterSpacing: '0.14em',
    textTransform: 'uppercase', color: 'var(--accent-blue)',
    marginBottom: 12,
    opacity: 0.8,
  },
  title: {
    fontSize: 88,
    fontWeight: 900,
    letterSpacing: '-0.04em',
    lineHeight: 1,
  },
  subtitle: {
    fontSize: 13,
    fontWeight: 700,
    color: 'var(--text-muted)',
    marginTop: 10,
    letterSpacing: '0.28em',
    textTransform: 'uppercase',
  },
  description: {
    marginTop: 18,
    fontSize: 14,
    color: 'var(--text-secondary)',
    lineHeight: 1.8,
  },
  cards: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
    width: '100%',
    maxWidth: 420,
    position: 'relative', zIndex: 1,
  },
  card: {
    display: 'flex', alignItems: 'center', gap: 16,
    padding: '16px 20px',
    background: 'var(--bg-800)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    cursor: 'pointer',
    text: 'left',
    width: '100%',
    transition: 'transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s',
  },
  cardPrimary: {
    background: 'linear-gradient(135deg, rgba(43,109,232,0.25) 0%, rgba(30,79,181,0.15) 100%)',
    border: '1px solid rgba(59,143,245,0.4)',
    boxShadow: '0 0 24px rgba(59,143,245,0.12)',
  },
  cardIcon: {
    fontSize: 22, color: 'var(--accent-blue)', flexShrink: 0,
    width: 36, textAlign: 'center',
  },
  cardText: { flex: 1, textAlign: 'left' },
  cardTitle: { fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 3 },
  cardDesc: { fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 },
  cardArrow: { color: 'var(--text-muted)', fontSize: 16, flexShrink: 0 },
  settingsPanel: { width: '100%', maxWidth: 400, position: 'relative', zIndex: 1 },
  settingsTitle: { fontSize: 12, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-secondary)' },
  settingsRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 },
  settingName: { fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' },
  settingDesc: { fontSize: 11, color: 'var(--text-muted)', marginTop: 2 },
  footer: { position: 'absolute', bottom: 20, zIndex: 1 },
  footerText: { fontSize: 11, color: 'var(--text-muted)', letterSpacing: '0.04em' },
};
