export default function PauseMenu({ onResume, onExit }) {
  return (
    <div style={styles.overlay}>
      <div className="panel" style={styles.panel}>
        <p style={styles.title}>Paused</p>
        <hr className="divider" />
        <div style={styles.actions}>
          <button className="btn btn-primary" onClick={onResume} id="btn-resume">
            Resume
          </button>
          <button className="btn btn-danger" onClick={onExit} id="btn-exit-pause">
            Exit to Menu
          </button>
        </div>
      </div>
    </div>
  );
}

const styles = {
  overlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(15,23,42,0.88)',
    borderRadius: 8,
    zIndex: 10,
  },
  panel: {
    minWidth: 220,
    textAlign: 'center',
  },
  title: {
    fontSize: 18,
    fontWeight: 700,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'var(--text-secondary)',
  },
  actions: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
};
