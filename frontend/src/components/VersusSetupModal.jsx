import { useState } from 'react';

export default function VersusSetupModal({ checkpoints, onStart, onClose }) {
  const [mode, setMode] = useState('human_vs_agent');
  const [ep1, setEp1] = useState('');
  const [ep2, setEp2] = useState('');

  const handleStart = () => {
    onStart({
      mode,
      agent1Episode: ep1 ? parseInt(ep1) : null,
      agent2Episode: ep2 ? parseInt(ep2) : null,
    });
  };

  return (
    <div style={styles.backdrop}>
      <div className="panel" style={styles.modal}>
        <p style={styles.title}>Versus Mode Setup</p>
        <hr className="divider" />

        {/* Mode selection */}
        <div style={{ marginBottom: 16 }}>
          <label>Mode</label>
          <div style={styles.modeRow}>
            {[
              { id: 'human_vs_agent', label: 'Human vs Agent' },
              { id: 'agent_vs_agent', label: 'Agent vs Agent' },
            ].map(m => (
              <button
                key={m.id}
                className="btn btn-sm"
                style={{
                  borderColor: mode === m.id ? 'var(--accent-blue)' : 'var(--border)',
                  color: mode === m.id ? 'var(--accent-blue)' : 'var(--text-muted)',
                  background: 'transparent',
                }}
                onClick={() => setMode(m.id)}
                id={`mode-${m.id}`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Checkpoint pickers */}
        <div style={styles.fields}>
          <div>
            <label htmlFor="ep1-input">
              {mode === 'human_vs_agent' ? 'Agent Checkpoint (episode)' : 'Agent 1 Checkpoint (episode)'}
            </label>
            <input
              id="ep1-input"
              className="input"
              placeholder="e.g. 500  (leave blank for random)"
              value={ep1}
              onChange={e => setEp1(e.target.value)}
            />
          </div>
          {mode === 'agent_vs_agent' && (
            <div>
              <label htmlFor="ep2-input">Agent 2 Checkpoint (episode)</label>
              <input
                id="ep2-input"
                className="input"
                placeholder="e.g. 1000  (leave blank for random)"
                value={ep2}
                onChange={e => setEp2(e.target.value)}
              />
            </div>
          )}
        </div>

        {/* Available checkpoints list */}
        {checkpoints.length > 0 ? (
          <div style={styles.cpList}>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
              Saved Checkpoints
            </p>
            {checkpoints.map(cp => (
              <div key={cp.name} style={styles.cpItem}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>
                  ep{cp.episode} — {cp.algorithm}
                </span>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="btn btn-sm" style={{ padding: '2px 8px', fontSize: 11 }}
                    onClick={() => setEp1(String(cp.episode))} id={`use-ep1-${cp.episode}`}>
                    Use as {mode === 'human_vs_agent' ? 'Agent' : 'Agent 1'}
                  </button>
                  {mode === 'agent_vs_agent' && (
                    <button className="btn btn-sm" style={{ padding: '2px 8px', fontSize: 11 }}
                      onClick={() => setEp2(String(cp.episode))} id={`use-ep2-${cp.episode}`}>
                      Agent 2
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={styles.emptyState}>
            <p style={{ color: 'var(--text-muted)', fontSize: 12, margin: 0 }}>
              No saved checkpoints yet. Leave the episode field blank to use a <strong>Random Agent</strong> as the opponent.
            </p>
            <p style={{ color: 'var(--text-muted)', fontSize: 11, margin: '6px 0 0' }}>
              Train an agent in the Training Studio first to save checkpoints, then load them here.
            </p>
          </div>
        )}

        <hr className="divider" />
        <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
          <button className="btn btn-sm" onClick={onClose} id="btn-versus-cancel">Cancel</button>
          <button className="btn btn-sm btn-primary" onClick={handleStart} id="btn-versus-start">
            Start
          </button>
        </div>
      </div>
    </div>
  );
}

const styles = {
  backdrop: {
    position: 'fixed', inset: 0,
    background: 'rgba(15,23,42,0.85)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    zIndex: 100,
  },
  modal: { width: 420, maxHeight: '90vh', overflowY: 'auto' },
  title: {
    fontSize: 15, fontWeight: 700, letterSpacing: '0.06em',
    textTransform: 'uppercase', color: 'var(--text-secondary)',
  },
  modeRow: { display: 'flex', gap: 8, marginTop: 6 },
  fields: { display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 16 },
  cpList: {
    background: 'var(--bg-900)', borderRadius: 6, padding: 10,
    marginBottom: 12, display: 'flex', flexDirection: 'column', gap: 6,
  },
  cpItem: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '4px 0', borderBottom: '1px solid var(--border)',
  },
  emptyState: {
    background: 'var(--bg-900)', borderRadius: 6, padding: '12px 14px',
    marginBottom: 12, borderLeft: '3px solid var(--accent-blue)',
  },
};
