/**
 * powerups.js
 * Defines power-up types, their effects, and the PowerUp entity.
 */

export const POWERUP_TYPES = {
  WIDE_PADDLE: {
    id: 'WIDE_PADDLE',
    label: 'Wide Paddle',
    color: '#22c55e',
    duration: 10000,  // ms
    apply: (game) => { game.paddle.widthMultiplier = 1.5; },
    remove: (game) => { game.paddle.widthMultiplier = 1.0; },
  },
  SLOW_BALL: {
    id: 'SLOW_BALL',
    label: 'Slow Ball',
    color: '#3b82f6',
    duration: 8000,
    apply: (game) => { game.balls.forEach(b => { b.vx *= 0.7; b.vy *= 0.7; }); },
    remove: (game) => { game.balls.forEach(b => { b.vx /= 0.7; b.vy /= 0.7; }); },
  },
  EXTRA_LIFE: {
    id: 'EXTRA_LIFE',
    label: 'Extra Life',
    color: '#ef4444',
    duration: 0,  // instant
    apply: (game) => { game.lives = Math.min(game.lives + 1, 5); },
    remove: () => {},
  },
  MULTI_BALL: {
    id: 'MULTI_BALL',
    label: 'Multi-Ball',
    color: '#f59e0b',
    duration: 0,  // lasts until all extra balls are lost
    apply: (game) => {
      const ref = game.balls[0];
      if (!ref) return;
      for (let i = 0; i < 2; i++) {
        game.balls.push({
          x: ref.x, y: ref.y,
          vx: ref.vx * (i === 0 ? 1.2 : -0.9),
          vy: ref.vy * (i === 0 ? -1.1 : -1.0),
          radius: ref.radius,
          isExtra: true,
        });
      }
    },
    remove: () => {},
  },
  FIRE_BALL: {
    id: 'FIRE_BALL',
    label: 'Fire Ball',
    color: '#f97316',
    duration: 8000,
    apply: (game) => { game.fireBallActive = true; },
    remove: (game) => { game.fireBallActive = false; },
  },
  SHIELD: {
    id: 'SHIELD',
    label: 'Shield',
    color: '#a855f7',
    duration: 8000,
    apply: (game) => { game.shieldActive = true; },
    remove: (game) => { game.shieldActive = false; },
  },
};

const POWERUP_KEYS = Object.keys(POWERUP_TYPES);

export class PowerUp {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.width = 32;
    this.height = 16;
    this.speed = 2.5;
    this.type = POWERUP_TYPES[POWERUP_KEYS[Math.floor(Math.random() * POWERUP_KEYS.length)]];
    this.collected = false;
  }

  update() {
    this.y += this.speed;
  }

  draw(ctx) {
    ctx.fillStyle = this.type.color;
    ctx.fillRect(this.x - this.width / 2, this.y - this.height / 2, this.width, this.height);
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(this.x - this.width / 2, this.y - this.height / 2, this.width, this.height);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(this.type.label.toUpperCase(), this.x, this.y);
  }

  isOffScreen(canvasHeight) {
    return this.y - this.height / 2 > canvasHeight;
  }

  collidesWithPaddle(paddle) {
    return (
      this.x + this.width / 2 > paddle.x &&
      this.x - this.width / 2 < paddle.x + paddle.currentWidth() &&
      this.y + this.height / 2 > paddle.y &&
      this.y - this.height / 2 < paddle.y + paddle.height
    );
  }
}
