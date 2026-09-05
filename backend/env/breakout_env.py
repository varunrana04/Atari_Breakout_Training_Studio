"""
breakout_env.py
Python-side Breakout environment for RL training.
Exposes the standard API: reset(seed) → state, step(action) → (state, reward, done, info)
No rendering — pure game logic, fast.
"""
import numpy as np
import random

# ── Constants ──────────────────────────────────────────────────────────────────
CANVAS_W = 480
CANVAS_H = 600
PADDLE_Y = CANVAS_H - 40
PADDLE_BASE_W = 80
PADDLE_H = 12
BALL_R = 7
BASE_SPEED = 4.5
BRICK_OFFSET_TOP = 60
BRICK_OFFSET_LEFT = 10
BRICK_PADDING = 4
MAX_SPEED_MULT = 2.5


def _make_level(level: int):
    """Generate a brick layout for a given level number (1–100)."""
    speed_mult = min(1.0 + (level - 1) * 0.015, MAX_SPEED_MULT)
    rng = random.Random(level)  # deterministic per level

    if level <= 10:
        rows, cols = 4, 10
        hp_dist = [(1, 1.0)]
    elif level <= 25:
        rows, cols = 5, 10
        hp_dist = [(1, 0.8), (2, 0.2)]
    elif level <= 50:
        rows, cols = 6, 11
        hp_dist = [(1, 0.5), (2, 0.4), (3, 0.1)]
    elif level <= 75:
        rows, cols = 7, 11
        hp_dist = [(1, 0.2), (2, 0.4), (3, 0.35), (9, 0.05)]
    else:
        rows, cols = 8, 12
        hp_dist = [(1, 0.1), (2, 0.2), (3, 0.55), (9, 0.15)]

    brick_w = (CANVAS_W - BRICK_OFFSET_LEFT * 2 - (cols - 1) * BRICK_PADDING) // cols
    brick_h = 18

    bricks = []
    for r in range(rows):
        for c in range(cols):
            val = rng.random()
            hp = 1
            cumulative = 0.0
            for (v, prob) in hp_dist:
                cumulative += prob
                if val < cumulative:
                    hp = v
                    break
            bricks.append({
                'x': BRICK_OFFSET_LEFT + c * (brick_w + BRICK_PADDING),
                'y': BRICK_OFFSET_TOP + r * (brick_h + BRICK_PADDING),
                'w': brick_w, 'h': brick_h,
                'hp': hp, 'max_hp': hp,
                'indestructible': hp == 9,
                'alive': True,
            })

    return bricks, speed_mult, brick_w, brick_h


class BreakoutEnv:
    """
    Breakout environment compatible with a gym-like API.
    State vector: [paddle_x_norm, ball_x_norm, ball_y_norm, ball_vx_norm, ball_vy_norm]
    + simplified brick presence grid (flattened, fixed max size).
    Actions: 0=left, 1=right, 2=noop
    """
    ACTION_LEFT  = 0
    ACTION_RIGHT = 1
    ACTION_NOOP  = 2
    N_ACTIONS    = 3

    # Fixed grid size for state (max bricks we'll represent)
    GRID_SIZE = 8 * 12  # max rows × max cols

    def __init__(self):
        self.level = 1
        self.seed_val = None
        self._init_state()

    def _init_state(self):
        self.paddle_x = CANVAS_W / 2 - PADDLE_BASE_W / 2
        self.paddle_w = PADDLE_BASE_W

        angle = np.random.uniform(np.radians(60), np.radians(120))
        speed = BASE_SPEED
        self.ball_x = CANVAS_W / 2
        self.ball_y = CANVAS_H - 70
        self.ball_vx = speed * np.cos(angle) * (1 if np.random.rand() < 0.5 else -1)
        self.ball_vy = -speed * np.sin(angle)

        self.bricks, self.speed_mult, self._bw, self._bh = _make_level(self.level)
        self.lives = 3
        self.score = 0
        self.done = False
        self.waiting_launch = True
        self.steps = 0

    def reset(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed_val = seed
        self.level = 1
        self.done = False
        self.steps = 0
        self._init_state()
        return self._get_obs()

    def step(self, action: int):
        assert not self.done, "Call reset() before step() after episode ends."
        self.steps += 1

        # Move paddle
        paddle_speed = 7
        if action == self.ACTION_LEFT:
            self.paddle_x = max(0, self.paddle_x - paddle_speed)
        elif action == self.ACTION_RIGHT:
            self.paddle_x = min(CANVAS_W - self.paddle_w, self.paddle_x + paddle_speed)

        if self.waiting_launch:
            self.waiting_launch = False

        prev_score = self.score
        prev_lives = self.lives

        self._physics_step()

        reward = (self.score - prev_score) * 1.0
        if self.lives < prev_lives:
            reward -= 5.0
        if self.done and self.lives > 0:
            reward += 20.0  # completed all levels

        info = {
            'score': self.score,
            'lives': self.lives,
            'level': self.level,
            'bricks_remaining': sum(1 for b in self.bricks if b['alive'] and not b['indestructible']),
        }
        return self._get_obs(), reward, self.done, info

    def _physics_step(self):
        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Wall collisions
        if self.ball_x - BALL_R <= 0:
            self.ball_x = BALL_R
            self.ball_vx = abs(self.ball_vx)
        if self.ball_x + BALL_R >= CANVAS_W:
            self.ball_x = CANVAS_W - BALL_R
            self.ball_vx = -abs(self.ball_vx)
        if self.ball_y - BALL_R <= 0:
            self.ball_y = BALL_R
            self.ball_vy = abs(self.ball_vy)

        # Paddle collision
        paddle_center = self.paddle_x + self.paddle_w / 2
        if (self.ball_y + BALL_R >= PADDLE_Y and
                self.ball_y + BALL_R <= PADDLE_Y + PADDLE_H + 6 and
                self.ball_x >= self.paddle_x - 4 and
                self.ball_x <= self.paddle_x + self.paddle_w + 4 and
                self.ball_vy > 0):
            hit_pos = (self.ball_x - paddle_center) / (self.paddle_w / 2)
            hit_pos = np.clip(hit_pos, -1, 1)
            angle = hit_pos * np.radians(60)
            speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
            self.ball_vx = speed * np.sin(angle)
            self.ball_vy = -abs(speed * np.cos(angle))
            self.ball_y = PADDLE_Y - BALL_R - 1

        # Ball lost
        if self.ball_y - BALL_R > CANVAS_H:
            self.lives -= 1
            if self.lives <= 0:
                self.done = True
                return
            # Reset ball
            self.ball_x = CANVAS_W / 2
            self.ball_y = CANVAS_H - 70
            angle = np.random.uniform(np.radians(60), np.radians(120))
            speed = BASE_SPEED * self.speed_mult
            self.ball_vx = speed * np.cos(angle) * (1 if np.random.rand() < 0.5 else -1)
            self.ball_vy = -speed * np.sin(angle)
            self.waiting_launch = False

        # Brick collisions
        for brick in self.bricks:
            if not brick['alive']:
                continue
            bx, by, bw, bh = brick['x'], brick['y'], brick['w'], brick['h']
            if (self.ball_x + BALL_R > bx and
                    self.ball_x - BALL_R < bx + bw and
                    self.ball_y + BALL_R > by and
                    self.ball_y - BALL_R < by + bh):
                if not brick['indestructible']:
                    brick['hp'] -= 1
                    if brick['hp'] <= 0:
                        brick['alive'] = False
                        self.score += 10
                # Reflect ball
                overlap_l = self.ball_x + BALL_R - bx
                overlap_r = bx + bw - (self.ball_x - BALL_R)
                overlap_t = self.ball_y + BALL_R - by
                overlap_b = by + bh - (self.ball_y - BALL_R)
                min_h = min(overlap_l, overlap_r)
                min_v = min(overlap_t, overlap_b)
                if min_h < min_v:
                    self.ball_vx = -self.ball_vx
                else:
                    self.ball_vy = -self.ball_vy
                break  # one brick per step

        # Level complete?
        if not any(b['alive'] and not b['indestructible'] for b in self.bricks):
            if self.level < 100:
                self.level += 1
                self.score += 50
                self.bricks, self.speed_mult, self._bw, self._bh = _make_level(self.level)
                speed = BASE_SPEED * self.speed_mult
                self.ball_x = CANVAS_W / 2
                self.ball_y = CANVAS_H - 70
                angle = np.random.uniform(np.radians(60), np.radians(120))
                self.ball_vx = speed * np.cos(angle) * (1 if np.random.rand() < 0.5 else -1)
                self.ball_vy = -speed * np.sin(angle)
            else:
                self.done = True

    def _get_obs(self) -> np.ndarray:
        """
        Returns a compact state vector:
        [paddle_x_norm, ball_x_norm, ball_y_norm, ball_vx_norm, ball_vy_norm,
         ...brick_presence (GRID_SIZE,)]
        Total: 5 + GRID_SIZE = 5 + 96 = 101 features
        """
        state = np.zeros(5 + self.GRID_SIZE, dtype=np.float32)
        state[0] = (self.paddle_x + self.paddle_w / 2) / CANVAS_W
        state[1] = self.ball_x / CANVAS_W
        state[2] = self.ball_y / CANVAS_H
        state[3] = self.ball_vx / 10.0
        state[4] = self.ball_vy / 10.0
        for i, brick in enumerate(self.bricks[:self.GRID_SIZE]):
            state[5 + i] = float(brick['alive']) * (brick['hp'] / 3.0)
        return state

    @property
    def obs_size(self) -> int:
        return 5 + self.GRID_SIZE

    def get_render_state(self) -> dict:
        """Returns a dict for the frontend WebSocket renderer.
        bricks: flat list of HP values (0=dead), length = GRID_SIZE,
                matches the 8-row × 12-col grid the frontend draws.
        """
        brick_hp = [0] * self.GRID_SIZE
        for i, b in enumerate(self.bricks[:self.GRID_SIZE]):
            brick_hp[i] = b['hp'] if b['alive'] else 0
        return {
            'paddle_x': (self.paddle_x + self.paddle_w / 2) / CANVAS_W,
            'paddle_w': self.paddle_w / CANVAS_W,
            'ball_x':   self.ball_x / CANVAS_W,
            'ball_y':   self.ball_y / CANVAS_H,
            'bricks':   brick_hp,
            'level':    self.level,
        }
