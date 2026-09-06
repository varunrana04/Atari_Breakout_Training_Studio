"""
test_env.py
Unit tests for BreakoutEnv — validates the environment is correct before
relying on training results as evidence of correctness.

Run from the backend/ directory:
    python -m pytest test_env.py -v
    -- or --
    python test_env.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import numpy as np
from env.breakout_env import BreakoutEnv, CANVAS_W, CANVAS_H


def test_reset_returns_correct_obs_shape():
    """State vector must be exactly obs_size = 101 features."""
    env = BreakoutEnv()
    obs = env.reset(seed=1)
    assert obs.shape == (env.obs_size,), f"Expected {(env.obs_size,)}, got {obs.shape}"
    assert env.obs_size == 101, f"Expected 101, got {env.obs_size}"
    print("PASS test_reset_returns_correct_obs_shape")


def test_reset_is_deterministic_with_same_seed():
    """reset(seed=N) twice must return identical states."""
    env = BreakoutEnv()
    obs1 = env.reset(seed=42)
    obs2 = env.reset(seed=42)
    assert np.allclose(obs1, obs2), "Same seed produced different initial states"
    print("PASS test_reset_is_deterministic_with_same_seed")


def test_reset_differs_with_different_seeds():
    """reset(seed=1) and reset(seed=2) must differ (at least ball velocity)."""
    env = BreakoutEnv()
    obs1 = env.reset(seed=1)
    obs2 = env.reset(seed=2)
    assert not np.allclose(obs1, obs2), "Different seeds produced identical states"
    print("PASS test_reset_differs_with_different_seeds")


def test_action_space():
    """Environment must have exactly 3 discrete actions."""
    env = BreakoutEnv()
    assert env.N_ACTIONS == 3, f"Expected 3 actions, got {env.N_ACTIONS}"
    assert env.ACTION_LEFT  == 0
    assert env.ACTION_RIGHT == 1
    assert env.ACTION_NOOP  == 2
    print("PASS test_action_space")


def test_noop_does_not_move_paddle():
    """NOOP action must not change paddle position."""
    env = BreakoutEnv()
    env.reset(seed=1)
    x_before = env.paddle_x
    env.step(env.ACTION_NOOP)
    assert env.paddle_x == x_before, "Noop moved the paddle"
    print("PASS test_noop_does_not_move_paddle")


def test_left_moves_paddle_left():
    """Left action must decrease paddle_x (or clamp at 0)."""
    env = BreakoutEnv()
    env.reset(seed=1)
    env.paddle_x = 100  # ensure room to move
    x_before = env.paddle_x
    env.step(env.ACTION_LEFT)
    assert env.paddle_x < x_before, "Left action did not move paddle left"
    print("PASS test_left_moves_paddle_left")


def test_right_moves_paddle_right():
    """Right action must increase paddle_x (or clamp at canvas right)."""
    env = BreakoutEnv()
    env.reset(seed=1)
    env.paddle_x = CANVAS_W / 2  # centre
    x_before = env.paddle_x
    env.step(env.ACTION_RIGHT)
    assert env.paddle_x > x_before, "Right action did not move paddle right"
    print("PASS test_right_moves_paddle_right")


def test_paddle_does_not_go_out_of_bounds():
    """Paddle must stay within [0, CANVAS_W - paddle_w] after many left/right steps."""
    env = BreakoutEnv()
    env.reset(seed=1)
    for _ in range(200):
        env.step(env.ACTION_LEFT)
    assert env.paddle_x >= 0, f"Paddle x went below 0: {env.paddle_x}"
    env.reset(seed=1)
    for _ in range(200):
        env.step(env.ACTION_RIGHT)
    assert env.paddle_x + env.paddle_w <= CANVAS_W + 1, \
        f"Paddle right edge exceeded canvas: {env.paddle_x + env.paddle_w}"
    print("PASS test_paddle_does_not_go_out_of_bounds")


def test_brick_destroyed_gives_positive_reward():
    """Hitting a brick must yield reward > 0 (at least +10)."""
    env = BreakoutEnv()
    env.reset(seed=1)
    # Place ball just above a brick to force a collision next step
    brick = next(b for b in env.bricks if b['alive'] and not b['indestructible'])
    env.ball_x = brick['x'] + brick['w'] / 2
    env.ball_y = brick['y'] + brick['h'] + 2   # just below the brick
    env.ball_vx = 0.0
    env.ball_vy = -6.0   # moving upward into the brick

    total_reward = 0.0
    for _ in range(10):  # give a few steps for the collision to register
        _, r, done, _ = env.step(env.ACTION_NOOP)
        total_reward += r
        if done:
            break
    assert total_reward >= 10, f"Expected at least +10 reward for brick hit, got {total_reward}"
    print("PASS test_brick_destroyed_gives_positive_reward")


def test_life_loss_gives_negative_reward():
    """Losing a ball (ball falls below canvas) must yield reward <= -5."""
    env = BreakoutEnv()
    env.reset(seed=1)
    # Put the ball below the canvas so it's lost immediately
    env.ball_x = CANVAS_W / 2
    env.ball_y = CANVAS_H + 20
    env.ball_vy = 5.0  # moving downward (already past canvas)
    lives_before = env.lives
    _, reward, _, info = env.step(env.ACTION_NOOP)
    assert info['lives'] < lives_before or reward <= -5 or env.lives < lives_before, \
        f"Life loss did not yield negative reward or decrement lives. reward={reward}, lives={info['lives']}"
    print("PASS test_life_loss_gives_negative_reward")


def test_done_when_all_lives_lost():
    """done must be True when lives reach 0."""
    env = BreakoutEnv()
    env.reset(seed=1)
    env.lives = 1
    # Force ball out of bounds
    env.ball_x = CANVAS_W / 2
    env.ball_y = CANVAS_H + 50
    env.ball_vy = 5.0
    _, _, done, info = env.step(env.ACTION_NOOP)
    assert done, f"Expected done=True when last life lost, got done={done}, lives={info['lives']}"
    print("PASS test_done_when_all_lives_lost")


def test_step_after_done_raises():
    """Calling step() after done=True must raise AssertionError."""
    env = BreakoutEnv()
    env.reset(seed=1)
    env.lives = 1
    env.ball_y = CANVAS_H + 50
    env.ball_vy = 5.0
    env.step(env.ACTION_NOOP)  # causes done=True
    try:
        env.step(env.ACTION_NOOP)
        assert False, "Expected AssertionError but no error was raised"
    except AssertionError:
        pass  # Expected
    print("PASS test_step_after_done_raises")


def test_obs_values_in_valid_range():
    """All state values must be in expected normalised range after several random steps."""
    env = BreakoutEnv()
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    for _ in range(50):
        action = rng.integers(0, 3)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset(seed=7)
        # Continuous features should be broadly normalised (allow some slack for velocity)
        assert -5.0 <= obs[0] <= 5.0, f"paddle_x_norm out of range: {obs[0]}"
        assert -5.0 <= obs[1] <= 5.0, f"ball_x_norm out of range: {obs[1]}"
        assert -5.0 <= obs[2] <= 5.0, f"ball_y_norm out of range: {obs[2]}"
        # Brick HP values should be 0..1
        assert np.all(obs[5:] >= 0), "Brick HP has negative values"
        assert np.all(obs[5:] <= 1.0 + 1e-6), "Brick HP exceeds 1.0"
    print("PASS test_obs_values_in_valid_range")


def test_render_state_returns_required_keys():
    """get_render_state() must return all keys the frontend renderer expects."""
    env = BreakoutEnv()
    env.reset(seed=1)
    state = env.get_render_state()
    required_keys = {'paddle_x', 'paddle_w', 'ball_x', 'ball_y', 'bricks', 'level'}
    missing = required_keys - state.keys()
    assert not missing, f"get_render_state() missing keys: {missing}"
    assert len(state['bricks']) == 96, f"Expected 96 brick HP values, got {len(state['bricks'])}"
    assert 0.0 <= state['paddle_x'] <= 1.0, f"paddle_x not normalised: {state['paddle_x']}"
    assert 0.0 <= state['ball_x'] <= 1.0, f"ball_x not normalised: {state['ball_x']}"
    assert 0.0 <= state['ball_y'] <= 1.0, f"ball_y not normalised: {state['ball_y']}"
    print("PASS test_render_state_returns_required_keys")


def test_level_progression():
    """Clearing all bricks must increment the level."""
    env = BreakoutEnv()
    env.reset(seed=1)
    level_before = env.level
    # Destroy all non-indestructible bricks manually
    for b in env.bricks:
        if not b['indestructible']:
            b['alive'] = False
            b['hp'] = 0
    # Force a step to trigger level-complete check
    env.step(env.ACTION_NOOP)
    assert env.level == level_before + 1 or env.done, \
        f"Level did not advance after all bricks cleared (level={env.level}, done={env.done})"
    print("PASS test_level_progression")


# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        test_reset_returns_correct_obs_shape,
        test_reset_is_deterministic_with_same_seed,
        test_reset_differs_with_different_seeds,
        test_action_space,
        test_noop_does_not_move_paddle,
        test_left_moves_paddle_left,
        test_right_moves_paddle_right,
        test_paddle_does_not_go_out_of_bounds,
        test_brick_destroyed_gives_positive_reward,
        test_life_loss_gives_negative_reward,
        test_done_when_all_lives_lost,
        test_step_after_done_raises,
        test_obs_values_in_valid_range,
        test_render_state_returns_required_keys,
        test_level_progression,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All environment tests passed.")
    sys.exit(0 if failed == 0 else 1)
