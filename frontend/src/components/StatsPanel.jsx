export default function StatsPanel({ stats }) {
  const formatVal = (val, decimals) => {
    if (typeof val !== 'number' || isNaN(val)) return '0';
    if (Math.abs(val) > 9999 || (Math.abs(val) < 0.001 && val !== 0)) {
      return val.toExponential(2);
    }
    return val.toFixed(decimals);
  };

  const items = [
    { label: 'Episode',     value: stats.episode.toLocaleString(), color: 'var(--text-primary)' },
    { label: 'Reward',      value: formatVal(stats.reward, 1),     color: 'var(--accent-green)' },
    { label: 'Avg Reward',  value: formatVal(stats.avgReward, 2),  color: 'var(--accent-green)' },
    { label: 'Epsilon',     value: formatVal(stats.epsilon, 4),    color: 'var(--accent-blue)' },
    { label: 'Loss',        value: formatVal(stats.loss, 4),       color: 'var(--accent-orange)' },
    { label: 'Mean Q',      value: formatVal(stats.qValue, 3),     color: '#a855f7' },
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
