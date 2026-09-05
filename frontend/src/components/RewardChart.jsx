import {
  Chart as ChartJS,
  LineElement, PointElement, LinearScale,
  CategoryScale, Filler, Tooltip, Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { useState } from 'react';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Filler, Tooltip, Legend);

const CHARTS = [
  { key: 'reward',  label: 'Avg Reward',   color: '#22c55e', dataKey: 'rewardData' },
  { key: 'loss',    label: 'Loss',          color: '#f97316', dataKey: 'lossData' },
  { key: 'epsilon', label: 'Epsilon',       color: '#3b82f6', dataKey: 'epsilonData' },
  { key: 'qvalue',  label: 'Mean Q-Value',  color: '#a855f7', dataKey: 'qData' },
];

function makeDataset(data, color, label) {
  return {
    label,
    data: data.map(d => d.y),
    borderColor: color,
    backgroundColor: color + '18',
    borderWidth: 1.5,
    pointRadius: 0,
    fill: true,
    tension: 0.3,
  };
}

const CHART_OPTIONS = (title) => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#1e293b',
      titleColor: '#94a3b8',
      bodyColor: '#f8fafc',
      borderColor: 'rgba(255,255,255,0.08)',
      borderWidth: 1,
    },
  },
  scales: {
    x: {
      ticks: { color: '#64748b', maxTicksLimit: 6, font: { size: 10 } },
      grid: { color: 'rgba(255,255,255,0.04)' },
    },
    y: {
      ticks: { color: '#64748b', maxTicksLimit: 5, font: { size: 10 } },
      grid: { color: 'rgba(255,255,255,0.04)' },
    },
  },
});

export default function RewardChart({ rewardData, lossData, epsilonData, qData }) {
  const [activeChart, setActiveChart] = useState('reward');

  const dataMap = { rewardData, lossData, epsilonData, qData };
  const active = CHARTS.find(c => c.key === activeChart);
  const currentData = dataMap[active.dataKey] ?? [];

  const chartData = {
    labels: currentData.map(d => d.x),
    datasets: [makeDataset(currentData, active.color, active.label)],
  };

  return (
    <div>
      {/* Tab selector */}
      <div style={styles.tabs}>
        {CHARTS.map(c => (
          <button
            key={c.key}
            className={`btn btn-sm${activeChart === c.key ? '' : ''}`}
            style={{
              ...styles.tab,
              borderColor: activeChart === c.key ? c.color : 'var(--border)',
              color: activeChart === c.key ? c.color : 'var(--text-muted)',
            }}
            onClick={() => setActiveChart(c.key)}
            id={`tab-chart-${c.key}`}
          >
            {c.label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div style={styles.chartWrap}>
        {currentData.length === 0 ? (
          <div style={styles.empty}>No data yet — start training to see charts</div>
        ) : (
          <Line data={chartData} options={CHART_OPTIONS(active.label)} />
        )}
      </div>
    </div>
  );
}

const styles = {
  tabs: { display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' },
  tab: { background: 'transparent' },
  chartWrap: { height: 200, position: 'relative' },
  empty: {
    height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: 'var(--text-muted)', fontSize: 13,
  },
};
