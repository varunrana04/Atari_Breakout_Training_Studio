export default function StatsPanel({ stats }) {
  const items = [
    { label: 'Episode',     value: stats.episode.toLocaleString(), color: 'var(--text-primary)' },
    { label: 'Reward',      value: stats.reward.toFixed(1),        color: 'var(--accent-green)' },
    { label: 'Avg Reward',  value: stats.avgReward.toFixed(2),     color: 'var(--accent-green)' },
    { label: 'Epsilon',     value: stats.epsilon.toFixed(4),       color: 'var(--accent-blue)' },
    { label: 'Loss',        value: stats.loss.toFixed(4),          color: 'var(--accent-orange)' },
    { label: 'Mean Q',      value: stats.qValue.toFixed(3),        color: '#a855f7' },
    { label: 'Speed',       value: `${Math.round(stats.speed)} ep/hr`, color: 'var(--text-secondary)' },
  ];

  return (
    <div className="panel" style={styles.root}>
      {items.map(item => (
        <div key={item.label} style={styles.item}>
          <span className="stat-label">{item.label}</span>
          <span style={{ ...styles.val, color: item.color }}>{item.value}</span>
        </div>
      ))}
    </div>
  );
}

const styles = {
  root: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '12px 16px',
    padding: 14,
  },
  item: { display: 'flex', flexDirection: 'column', gap: 3 },
  val: { fontFamily: 'var(--font-mono)', fontSize: 15, fontWeight: 700 },
};
