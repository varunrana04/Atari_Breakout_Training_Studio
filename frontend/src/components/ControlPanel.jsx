const ALGORITHMS = [
  { id: 'dqn',              label: 'DQN',              badge: 'Baseline' },
  { id: 'double_dqn',       label: 'Double DQN',       badge: 'Recommended' },
  { id: 'dueling_double_dqn', label: 'Dueling Double DQN', badge: 'Best' },
];

export default function ControlPanel({
  hyperparams, onHyperparamsChange,
  training, connected,
  onConnect, onStart, onPause, onSave, onLoad,
}) {
  const set = (key, val) => onHyperparamsChange({ ...hyperparams, [key]: val });

  return (
    <div className="panel" style={styles.root}>
      <p style={styles.sectionTitle}>Algorithm</p>
      <div style={styles.algoRow}>
        {ALGORITHMS.map(a => (
          <button
            key={a.id}
            className="btn btn-sm"
            style={{
              ...styles.algoBtn,
              borderColor: hyperparams.algorithm === a.id ? 'var(--accent-blue)' : 'var(--border)',
              color: hyperparams.algorithm === a.id ? 'var(--accent-blue)' : 'var(--text-secondary)',
            }}
            onClick={() => set('algorithm', a.id)}
            id={`algo-${a.id}`}
            disabled={training}
          >
            {a.label}
            <span
              className={`badge ${a.id === 'dueling_double_dqn' ? 'badge-green' : a.id === 'double_dqn' ? 'badge-blue' : ''}`}
              style={{ marginLeft: 4, fontSize: 9 }}
            >
              {a.badge}
            </span>
          </button>
        ))}
      </div>

      <hr className="divider" />
      <p style={styles.sectionTitle}>Hyperparameters</p>

      <div style={styles.grid}>
        <Field label="Episodes" id="hp-episodes"
          value={hyperparams.episodes}
          onChange={v => set('episodes', v)}
          disabled={training} />
        <Field label="Learning Rate" id="hp-lr"
          value={hyperparams.learning_rate}
          onChange={v => set('learning_rate', v)}
          disabled={training} />
        <Field label="Gamma" id="hp-gamma"
          value={hyperparams.gamma}
          onChange={v => set('gamma', v)}
          disabled={training} />
        <Field label="Batch Size" id="hp-batch"
          value={hyperparams.batch_size}
          onChange={v => set('batch_size', v)}
          disabled={training} />
        <Field label="Replay Buffer" id="hp-buffer"
          value={hyperparams.replay_buffer_size}
          onChange={v => set('replay_buffer_size', v)}
          disabled={training} />
        <Field label="Epsilon Start" id="hp-eps-start"
          value={hyperparams.epsilon_start}
          onChange={v => set('epsilon_start', v)}
          disabled={training} />
        <Field label="Epsilon Final" id="hp-eps-final"
          value={hyperparams.epsilon_final}
          onChange={v => set('epsilon_final', v)}
          disabled={training} />
        <Field label="Target Update" id="hp-target"
          value={hyperparams.target_update_steps}
          onChange={v => set('target_update_steps', v)}
          disabled={training} />
      </div>

      <hr className="divider" />

      {/* Action buttons */}
      <div style={styles.actionRow}>
        {!connected ? (
          <button className="btn btn-primary" onClick={onConnect} id="btn-connect" style={{ flex: 1 }}>
            Connect to Backend
          </button>
        ) : !training ? (
          <button className="btn btn-green" onClick={onStart} id="btn-start-train" style={{ flex: 1 }}>
            Start Training
          </button>
        ) : (
          <button className="btn btn-primary" onClick={onPause} id="btn-pause-train" style={{ flex: 1 }}>
            Pause
          </button>
        )}
      </div>

      <div style={styles.actionRow}>
        <button className="btn btn-sm" onClick={onSave} disabled={!connected} id="btn-save" style={{ flex: 1 }}>
          Save Checkpoint
        </button>
        <button className="btn btn-sm" onClick={onLoad} disabled={!connected} id="btn-load" style={{ flex: 1 }}>
          Load Checkpoint
        </button>
      </div>

      <div style={{ marginTop: 4 }}>
        <p style={{ color: 'var(--text-muted)', fontSize: 11, lineHeight: 1.6 }}>
          Tip: Training may plateau for 500+ episodes before the agent starts learning. This is expected — the replay buffer needs to fill first.
        </p>
      </div>
    </div>
  );
}

function Field({ label, id, value, onChange, disabled }) {
  return (
    <div>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        className="input"
        value={value}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
      />
    </div>
  );
}

const styles = {
  root: { display: 'flex', flexDirection: 'column', gap: 12 },
  sectionTitle: {
    fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
    textTransform: 'uppercase', color: 'var(--text-muted)',
  },
  algoRow: { display: 'flex', gap: 8, flexWrap: 'wrap' },
  algoBtn: { background: 'transparent', flexDirection: 'column', alignItems: 'flex-start', height: 'auto', padding: '8px 12px' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px 16px' },
  actionRow: { display: 'flex', gap: 8 },
};
