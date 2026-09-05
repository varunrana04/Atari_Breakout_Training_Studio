/**
 * breakout.js
 * Core Breakout game engine.
 * Runs on a <canvas> element. Supports Player mode, Agent mode (driven externally),
 * 100 levels, power-ups, pause/resume, and exposes a step() API for RL training.
 */

import { getLevel } from './levels.js';
import { PowerUp } from './powerups.js';

// ─── Constants ───────────────────────────────────────────────────────────────
const CANVAS_W = 480;
const CANVAS_H = 600;
const BRICK_PADDING = 4;
const BRICK_OFFSET_TOP = 60;
const BRICK_OFFSET_LEFT = 10;
const PADDLE_HEIGHT = 12;
const PADDLE_BASE_WIDTH = 80;
const BALL_RADIUS = 7;
const BASE_BALL_SPEED = 3.5;  // base speed — easier start
const MAX_BALL_SPEED = 12.0;  // hard cap so it stays humanly playable
const POWERUP_SPAWN_CHANCE = 0.18; // 18% chance per brick destroyed

// Brick HP colors
const HP_COLORS = {
  1: '#e2e8f0',  // light grey
  2: '#60a5fa',  // blue
  3: '#f97316',  // orange
  9: '#6b7280',  // indestructible (dark grey)
};

// ─── Paddle ──────────────────────────────────────────────────────────────────
class Paddle {
  constructor() {
    this.baseWidth = PADDLE_BASE_WIDTH;
    this.widthMultiplier = 1.0;
    this.height = PADDLE_HEIGHT;
    this.x = CANVAS_W / 2 - this.baseWidth / 2;
    this.y = CANVAS_H - 40;
    this.speed = 7;
  }

  currentWidth() { return this.baseWidth * this.widthMultiplier; }

  moveLeft() { this.x = Math.max(0, this.x - this.speed); }
  moveRight() { this.x = Math.min(CANVAS_W - this.currentWidth(), this.x + this.speed); }
  moveTo(targetX) {
    this.x = Math.max(0, Math.min(CANVAS_W - this.currentWidth(), targetX - this.currentWidth() / 2));
  }

  draw(ctx) {
    const w = this.currentWidth();
    ctx.fillStyle = '#f8fafc';
    ctx.beginPath();
    ctx.roundRect(this.x, this.y, w, this.height, 4);
    ctx.fill();
  }

  center() { return this.x + this.currentWidth() / 2; }
  reset(paddleMult = 1.0) {
    this.widthMultiplier = paddleMult;
    this.x = CANVAS_W / 2 - this.currentWidth() / 2;
  }
}

// ─── Ball ────────────────────────────────────────────────────────────────────
function makeBall(speedMult = 1) {
  const angle = (Math.random() * 60 + 60) * (Math.PI / 180); // 60–120 deg upward
  const speed = Math.min(BASE_BALL_SPEED * speedMult, MAX_BALL_SPEED);
  return {
    x: CANVAS_W / 2,
    y: CANVAS_H - 70,
    vx: speed * Math.cos(angle) * (Math.random() < 0.5 ? 1 : -1),
    vy: -speed * Math.sin(angle),
    radius: BALL_RADIUS,
    isExtra: false,
  };
}

// ─── Brick ───────────────────────────────────────────────────────────────────
function makeBricks(layout, brickW, brickH) {
  const bricks = [];
  for (let r = 0; r < layout.length; r++) {
    for (let c = 0; c < layout[r].length; c++) {
      const hp = layout[r][c];
      if (hp === 0) continue;
      bricks.push({
        x: BRICK_OFFSET_LEFT + c * (brickW + BRICK_PADDING),
        y: BRICK_OFFSET_TOP + r * (brickH + BRICK_PADDING),
        w: brickW,
        h: brickH,
        hp,
        maxHp: hp,
        indestructible: hp === 9,
        alive: true,
      });
    }
  }
  return bricks;
}

// ─── BreakoutGame ─────────────────────────────────────────────────────────────
export class BreakoutGame {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {'player' | 'agent' | 'watch'} mode
   * @param {object} options
   */
  constructor(canvas, mode = 'player', options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.mode = mode;
    this.powerUpsEnabled = options.powerUpsEnabled ?? true;
    this.skipInput = options.skipInput ?? false;  // true = no keyboard/mouse listeners
    this.onScoreChange = options.onScoreChange ?? null;
    this.onLivesChange = options.onLivesChange ?? null;
    this.onLevelChange = options.onLevelChange ?? null;
    this.onGameOver = options.onGameOver ?? null;
    this.onLevelComplete = options.onLevelComplete ?? null;
    // Called whenever pause state changes — lets React update its overlay
    this.onPauseToggle = options.onPauseToggle ?? null;

    this.level = 1;
    this.score = 0;
    this.lives = 3;
    this.paused = false;
    this.gameOver = false;
    this.levelComplete = false;
    this.waitingForLaunch = true; // ball sitting on paddle before first launch

    this.activePowerUps = []; // { type, expiresAt }
    this.fallingPowerUps = [];
    this.fireBallActive = false;
    this.shieldActive = false;

    this.paddle = new Paddle();
    this._loadLevel(this.level);

    this._rafId = null;
    this._lastTime = 0;
    this._keys = {};
    this._bindInput();
  }

  // ─── Level Loading ──────────────────────────────────────────────────────────
  _loadLevel(levelNum) {
    const levelDef = getLevel(levelNum);
    this._speedMult = levelDef.speedMultiplier;
    this._paddleMult = levelDef.paddleMultiplier ?? 1.0;

    const cols = levelDef.cols;
    const brickW = Math.floor((CANVAS_W - BRICK_OFFSET_LEFT * 2 - (cols - 1) * BRICK_PADDING) / cols);
    const brickH = 18;

    this.bricks = makeBricks(levelDef.layout, brickW, brickH);
    this.balls = [makeBall(this._speedMult)];
    this.waitingForLaunch = true;
    this.fallingPowerUps = [];
    this.activePowerUps = [];
    this.fireBallActive = false;
    this.shieldActive = false;
    // Apply paddle width multiplier from level
    this.paddle.reset(this._paddleMult);
    this.levelComplete = false;
    this.onLevelChange?.(levelNum);
  }

  // ─── Input Binding ──────────────────────────────────────────────────────────
  _bindInput() {
    if (this.skipInput) return;  // agent-side game: no input binding
    this._onKeyDown = (e) => {
      this._keys[e.code] = true;
      if (e.code === 'Escape') {
        this.togglePause();
        // Notify React so the pause overlay renders
        this.onPauseToggle?.(this.paused);
      }
      if (e.code === 'Space' && this.waitingForLaunch) this.launch();
    };
    this._onKeyUp = (e) => { this._keys[e.code] = false; };
    this._onMouseMove = (e) => {
      if (this.mode !== 'player') return;
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = CANVAS_W / rect.width;
      this.paddle.moveTo((e.clientX - rect.left) * scaleX);
    };
    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    this.canvas.addEventListener('mousemove', this._onMouseMove);
  }

  destroy() {
    if (!this.skipInput) {
      window.removeEventListener('keydown', this._onKeyDown);
      window.removeEventListener('keyup', this._onKeyUp);
      this.canvas.removeEventListener('mousemove', this._onMouseMove);
    }
    this.stop();
  }

  // ─── Game Loop ──────────────────────────────────────────────────────────────
  start() {
    this._lastTime = performance.now();
    const loop = (timestamp) => {
      const dt = Math.min(timestamp - this._lastTime, 32); // cap at ~30fps delta
      this._lastTime = timestamp;
      if (!this.paused && !this.gameOver) this._update(dt);
      this._draw();
      this._rafId = requestAnimationFrame(loop);
    };
    this._rafId = requestAnimationFrame(loop);
  }

  stop() {
    if (this._rafId) cancelAnimationFrame(this._rafId);
    this._rafId = null;
  }

  togglePause() {
    if (this.gameOver) return;
    this.paused = !this.paused;
    // Always notify React (called from both ESC key and React button)
    this.onPauseToggle?.(this.paused);
  }

  launch() {
    this.waitingForLaunch = false;
  }

  // ─── Agent API ──────────────────────────────────────────────────────────────
  /**
   * Called by external agent to take an action.
   * action: 0 = left, 1 = right, 2 = stay
   * Returns: { state, reward, done }
   */
  agentStep(action) {
    if (action === 0) this.paddle.moveLeft();
    else if (action === 1) this.paddle.moveRight();
    if (this.waitingForLaunch) this.launch();

    const prevScore = this.score;
    const prevLives = this.lives;
    this._update(16); // ~1 frame at 60fps

    const reward = (this.score - prevScore) + (this.lives < prevLives ? -10 : 0);
    const done = this.gameOver;
    return { state: this.getState(), reward, done };
  }

  /**
   * Returns a compact game state object for the agent.
   */
  getState() {
    const mainBall = this.balls[0] ?? { x: 0, y: 0, vx: 0, vy: 0 };
    const brickGrid = this.bricks.map(b => b.alive ? b.hp : 0);
    return {
      paddleX: this.paddle.center() / CANVAS_W,
      ballX: mainBall.x / CANVAS_W,
      ballY: mainBall.y / CANVAS_H,
      ballVX: mainBall.vx / 10,
      ballVY: mainBall.vy / 10,
      lives: this.lives,
      score: this.score,
      brickGrid,
    };
  }

  reset(seed = null) {
    this.level = 1;
    this.score = 0;
    this.lives = 3;
    this.gameOver = false;
    this.paddle = new Paddle();
    this._loadLevel(this.level);
    return this.getState();
  }

  // ─── Update Logic ────────────────────────────────────────────────────────────
  _update(dt) {
    if (this.waitingForLaunch) {
      // Ball sits on paddle
      const ball = this.balls[0];
      if (ball) {
        ball.x = this.paddle.center();
        ball.y = this.paddle.y - ball.radius - 1;
      }
      // Player input during waiting
      if (this.mode === 'player') {
        if (this._keys['ArrowLeft'] || this._keys['KeyA']) this.paddle.moveLeft();
        if (this._keys['ArrowRight'] || this._keys['KeyD']) this.paddle.moveRight();
      }
      return;
    }

    // Player keyboard paddle
    if (this.mode === 'player') {
      if (this._keys['ArrowLeft'] || this._keys['KeyA']) this.paddle.moveLeft();
      if (this._keys['ArrowRight'] || this._keys['KeyD']) this.paddle.moveRight();
    }

    // Update balls
    const now = Date.now();
    const ballsToRemove = [];
    for (const ball of this.balls) {
      ball.x += ball.vx;
      ball.y += ball.vy;

      // Wall collisions
      if (ball.x - ball.radius <= 0) { ball.x = ball.radius; ball.vx = Math.abs(ball.vx); }
      if (ball.x + ball.radius >= CANVAS_W) { ball.x = CANVAS_W - ball.radius; ball.vx = -Math.abs(ball.vx); }
      if (ball.y - ball.radius <= 0) { ball.y = ball.radius; ball.vy = Math.abs(ball.vy); }

      // Paddle collision
      const pw = this.paddle.currentWidth();
      if (
        ball.y + ball.radius >= this.paddle.y &&
        ball.y + ball.radius <= this.paddle.y + this.paddle.height + 5 &&
        ball.x >= this.paddle.x - 5 &&
        ball.x <= this.paddle.x + pw + 5 &&
        ball.vy > 0
      ) {
        // Angle based on hit position (left = more left, right = more right)
        const hitPos = (ball.x - this.paddle.center()) / (pw / 2); // -1 to 1
        const angle = hitPos * 60 * (Math.PI / 180); // max 60 deg
        const speed = Math.sqrt(ball.vx ** 2 + ball.vy ** 2);
        ball.vx = speed * Math.sin(angle);
        ball.vy = -Math.abs(speed * Math.cos(angle));
        ball.y = this.paddle.y - ball.radius - 1;
      }

      // Shield collision
      if (this.shieldActive && ball.y + ball.radius >= CANVAS_H - 8 && ball.vy > 0) {
        ball.vy = -Math.abs(ball.vy);
      }

      // Ball lost
      if (ball.y - ball.radius > CANVAS_H) {
        if (ball.isExtra) {
          ballsToRemove.push(ball);
        } else {
          this.lives -= 1;
          this.onLivesChange?.(this.lives);
          if (this.lives <= 0) {
            this.gameOver = true;
            this.onGameOver?.(this.score);
          } else {
            this.waitingForLaunch = true;
            ball.x = this.paddle.center();
            ball.y = this.paddle.y - ball.radius - 1;
            const angle = (Math.random() * 60 + 60) * (Math.PI / 180);
            const speed = BASE_BALL_SPEED * this._speedMult;
            ball.vx = speed * Math.cos(angle) * (Math.random() < 0.5 ? 1 : -1);
            ball.vy = -speed * Math.sin(angle);
            // Remove extra balls on life loss
            this.balls = [ball];
          }
        }
      }

      // Brick collisions
      for (const brick of this.bricks) {
        if (!brick.alive) continue;
        if (this._ballHitsBrick(ball, brick)) {
          if (!brick.indestructible) {
            const damage = this.fireBallActive ? 99 : 1;
            brick.hp -= damage;
            if (brick.hp <= 0) {
              brick.alive = false;
              this.score += 10;
              this.onScoreChange?.(this.score);
              // Spawn power-up
              if (this.powerUpsEnabled && Math.random() < POWERUP_SPAWN_CHANCE) {
                this.fallingPowerUps.push(
                  new PowerUp(brick.x + brick.w / 2, brick.y + brick.h / 2)
                );
              }
            }
          }
          // Reflect ball
          if (!this.fireBallActive || brick.indestructible) {
            const overlapLeft = ball.x + ball.radius - brick.x;
            const overlapRight = brick.x + brick.w - (ball.x - ball.radius);
            const overlapTop = ball.y + ball.radius - brick.y;
            const overlapBottom = brick.y + brick.h - (ball.y - ball.radius);
            const minH = Math.min(overlapLeft, overlapRight);
            const minV = Math.min(overlapTop, overlapBottom);
            if (minH < minV) ball.vx = -ball.vx;
            else ball.vy = -ball.vy;
          }
        }
      }
    }

    // Remove lost extra balls
    this.balls = this.balls.filter(b => !ballsToRemove.includes(b));

    // Update falling power-ups
    for (const pu of this.fallingPowerUps) {
      pu.update();
      if (pu.collidesWithPaddle(this.paddle) && !pu.collected) {
        pu.collected = true;
        pu.type.apply(this);
        if (pu.type.duration > 0) {
          this.activePowerUps.push({ type: pu.type, expiresAt: now + pu.type.duration });
        }
      }
    }
    this.fallingPowerUps = this.fallingPowerUps.filter(
      pu => !pu.isOffScreen(CANVAS_H) && !pu.collected
    );

    // Expire active power-ups
    for (const ap of this.activePowerUps) {
      if (now > ap.expiresAt) ap.type.remove(this);
    }
    this.activePowerUps = this.activePowerUps.filter(ap => now <= ap.expiresAt);

    // Check level complete
    const anyAliveDestructible = this.bricks.some(b => b.alive && !b.indestructible);
    if (!anyAliveDestructible && !this.levelComplete) {
      this.levelComplete = true;
      this.score += 50; // level clear bonus
      this.onScoreChange?.(this.score);
      if (this.level < 100) {
        setTimeout(() => {
          this.level += 1;
          this._loadLevel(this.level);
          this.onLevelChange?.(this.level);
        }, 1200);
      } else {
        this.gameOver = true;
        this.onGameOver?.(this.score, true); // true = won
      }
    }
  }

  _ballHitsBrick(ball, brick) {
    return (
      ball.x + ball.radius > brick.x &&
      ball.x - ball.radius < brick.x + brick.w &&
      ball.y + ball.radius > brick.y &&
      ball.y - ball.radius < brick.y + brick.h
    );
  }

  // ─── Draw ────────────────────────────────────────────────────────────────────
  _draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    // Background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Shield bar
    if (this.shieldActive) {
      ctx.fillStyle = 'rgba(168,85,247,0.5)';
      ctx.fillRect(0, CANVAS_H - 8, CANVAS_W, 8);
    }

    // Bricks
    for (const brick of this.bricks) {
      if (!brick.alive) continue;
      const color = HP_COLORS[brick.indestructible ? 9 : brick.hp] ?? '#94a3b8';
      ctx.fillStyle = color;
      ctx.fillRect(brick.x, brick.y, brick.w, brick.h);
      // Crack overlay for multi-hit bricks
      if (!brick.indestructible && brick.hp < brick.maxHp) {
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.fillRect(brick.x, brick.y, brick.w * (1 - brick.hp / brick.maxHp), brick.h);
      }
    }

    // Balls
    for (const ball of this.balls) {
      ctx.beginPath();
      ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
      ctx.fillStyle = this.fireBallActive ? '#f97316' : '#f8fafc';
      ctx.fill();
      if (this.fireBallActive) {
        ctx.shadowColor = '#f97316';
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // Paddle
    this.paddle.draw(ctx);

    // Power-ups
    for (const pu of this.fallingPowerUps) pu.draw(ctx);

    // Waiting overlay
    if (this.waitingForLaunch && !this.gameOver) {
      ctx.fillStyle = 'rgba(248,250,252,0.5)';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('SPACE / CLICK to launch', CANVAS_W / 2, this.paddle.y - 20);
    }

    // Level complete flash
    if (this.levelComplete && this.level < 100) {
      ctx.fillStyle = 'rgba(15,23,42,0.7)';
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = '#22c55e';
      ctx.font = 'bold 28px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`LEVEL ${this.level} COMPLETE`, CANVAS_W / 2, CANVAS_H / 2);
    }

    // Game over overlay
    if (this.gameOver) {
      ctx.fillStyle = 'rgba(15,23,42,0.85)';
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 32px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(this.level >= 100 ? 'YOU WIN!' : 'GAME OVER', CANVAS_W / 2, CANVAS_H / 2 - 16);
      ctx.font = '18px Inter, sans-serif';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(`Score: ${this.score}`, CANVAS_W / 2, CANVAS_H / 2 + 20);
    }
  }
}

export { CANVAS_W, CANVAS_H };
