/**
 * levels.js
 * Defines all 100 levels of Breakout with deterministic, clearly progressive difficulty.
 *
 * Difficulty axes that scale per level:
 *   - Ball speed (most impactful — player FEELS this immediately)
 *   - Brick rows (more bricks = more work per level)
 *   - Brick HP (2-hit and 3-hit bricks introduced gradually)
 *   - Indestructible bricks (from level 60+)
 *   - Paddle width shrinks (from level 40+, makes it harder to track)
 *
 * Seeded RNG per level so layouts are consistent on replay.
 */

// ── Seeded pseudo-random (LCG) ───────────────────────────────────────────────
function seededRng(seed) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

// ── Speed formula ────────────────────────────────────────────────────────────
// Level 1  → ×1.0  (base 3.5 px/frame)
// Level 25 → ×1.5
// Level 50 → ×2.0
// Level 75 → ×2.5
// Level 100 → ×3.0  (hard cap applied in engine)
export function getLevelSpeedMultiplier(level) {
  return 1.0 + (level - 1) * (2.0 / 99); // linear from 1.0 to 3.0
}

// ── Paddle width multiplier (shrinks from level 40) ──────────────────────────
export function getLevelPaddleMultiplier(level) {
  if (level <= 40) return 1.0;
  // Shrinks from 1.0 at lv40 to 0.55 at lv100
  return Math.max(0.55, 1.0 - (level - 40) * (0.45 / 60));
}

// ── Brick HP distribution by level ──────────────────────────────────────────
function getHpDistribution(level) {
  if (level <= 10)  return [[1, 1.00]];
  if (level <= 20)  return [[1, 0.75], [2, 0.25]];
  if (level <= 30)  return [[1, 0.50], [2, 0.50]];
  if (level <= 40)  return [[1, 0.25], [2, 0.60], [3, 0.15]];
  if (level <= 55)  return [[1, 0.10], [2, 0.45], [3, 0.45]];
  if (level <= 70)  return [[2, 0.30], [3, 0.60], [9, 0.10]];
  if (level <= 85)  return [[2, 0.10], [3, 0.65], [9, 0.25]];
  return             [[3, 0.55], [9, 0.45]];
}

// ── Row / column count by level ──────────────────────────────────────────────
function getGridSize(level) {
  if (level <= 10)  return { rows: 3, cols: 8  };
  if (level <= 20)  return { rows: 4, cols: 9  };
  if (level <= 35)  return { rows: 5, cols: 10 };
  if (level <= 55)  return { rows: 6, cols: 11 };
  if (level <= 75)  return { rows: 7, cols: 11 };
  return                    { rows: 8, cols: 12 };
}

// ── Layout generator ─────────────────────────────────────────────────────────
function generateLayout(rows, cols, hpDistribution, rng) {
  const layout = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const rand = rng();
      let hp = 1;
      let cumulative = 0;
      for (const [val, prob] of hpDistribution) {
        cumulative += prob;
        if (rand < cumulative) { hp = val; break; }
      }
      row.push(hp);
    }
    layout.push(row);
  }
  return layout;
}

// ── Main export ───────────────────────────────────────────────────────────────
export function getLevel(levelNumber) {
  const level = Math.max(1, Math.min(100, levelNumber));
  const rng = seededRng(level * 9973 + 12345); // unique seed per level

  const { rows, cols } = getGridSize(level);
  const hpDist = getHpDistribution(level);
  const layout = generateLayout(rows, cols, hpDist, rng);

  return {
    level,
    rows,
    cols,
    layout,
    speedMultiplier:  getLevelSpeedMultiplier(level),
    paddleMultiplier: getLevelPaddleMultiplier(level),
  };
}

export const TOTAL_LEVELS = 100;
