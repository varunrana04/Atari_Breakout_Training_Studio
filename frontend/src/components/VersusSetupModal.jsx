/**
 * VersusSetupModal.jsx
 * Lets the user pick mode + specific checkpoint (by full name, not just episode)
 * so the backend can load the exact .pt file without ambiguity between algorithms.
 */
import { useState } from 'react';

export default function VersusSetupModal({ checkpoints, onStart, onClose }) {
  const [mode, setMode] = useState('human_vs_agent');
  const [cp1, setCp1] = useState('');  // full checkpoint name e.g. "checkpoint_ep3000_dqn.pt"
  const [cp2, setCp2] = useState('');

  const handleStart = () => {
    onStart({
      mode,
      agent1Checkpoint: cp1 || null,
      agent2Checkpoint: cp2 || null,
    });
  };

  // Build a friendly label: "ep3000 — DQN" from "checkpoint_ep3000_dqn.pt"
  const cpLabel = (cp) => {
    const name = cp.name;
    const ep   = cp.episode;
    const algo = (cp.algorithm || 'unknown').replace(/_/g, ' ').toUpperCase();
    return `ep${ep} — ${algo}`;
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

        {/* Checkpoint selectors */}
        <div style={styles.fields}>
          <div>
            <label htmlFor="cp1-select">
              {mode === 'human_vs_agent' ? 'Agent Checkpoint' : 'Agent 1 Checkpoint'}
            </label>
            {checkpoints.length > 0 ? (
              <select
                id="cp1-select"
                className="input"
                value={cp1}
                onChange={e => setCp1(e.target.value)}
                style={{ marginTop: 4 }}
              >
                <option value="">— Random Agent (no checkpoint) —</option>
                {checkpoints.map(cp => (
                  <option key={cp.name} value={cp.name}>{cpLabel(cp)}</option>
                ))}
              </select>
            ) : (
              <p style={styles.noCheckpoints}>No checkpoints saved yet. Train first!</p>
            )}
          </div>

          {mode === 'agent_vs_agent' && (
            <div>
              <label htmlFor="cp2-select">Agent 2 Checkpoint</label>
              {checkpoints.length > 0 ? (
                <select
                  id="cp2-select"
                  className="input"
                  value={cp2}
                  onChange={e => setCp2(e.target.value)}
                  style={{ marginTop: 4 }}
                >
                  <option value="">— Random Agent (no checkpoint) —</option>
                  {checkpoints.map(cp => (
                    <option key={cp.name} value={cp.name}>{cpLabel(cp)}</option>
                  ))}
                </select>
              ) : (
                <p style={styles.noCheckpoints}>No checkpoints saved yet.</p>
              )}
            </div>
          )}
        </div>

        {/* Summary of what will load */}
        {cp1 && (
          <div style={styles.summary}>
            <span style={{ color: 'var(--accent-blue)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              {mode === 'human_vs_agent' ? 'Agent' : 'Agent 1'}: {cp1}
            </span>
            {mode === 'agent_vs_agent' && cp2 && (
              <span style={{ color: 'var(--accent-blue)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                Agent 2: {cp2}
              </span>
            )}
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
  modal: { width: 440, maxHeight: '90vh', overflowY: 'auto' },
  title: {
    fontSize: 15, fontWeight: 700, letterSpacing: '0.06em',
    textTransform: 'uppercase', color: 'var(--text-secondary)',
  },
  modeRow: { display: 'flex', gap: 8, marginTop: 6 },
  fields: { display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 16 },
  noCheckpoints: {
    color: 'var(--text-muted)', fontSize: 12, margin: '6px 0 0',
    fontStyle: 'italic',
  },
  summary: {
    background: 'var(--bg-900)', borderRadius: 6, padding: '8px 12px',
    marginBottom: 12, display: 'flex', flexDirection: 'column', gap: 4,
    borderLeft: '3px solid var(--accent-blue)',
  },
};
