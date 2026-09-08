import pytest
import numpy as np
from env.breakout_env import BreakoutEnv, CANVAS_W, CANVAS_H, PADDLE_Y, PADDLE_BASE_W, PADDLE_H, BALL_R

def test_determinism():
    """Verify that environment seeding produces exactly identical trajectories."""
    env1 = BreakoutEnv()
    env2 = BreakoutEnv()
    
    state1 = env1.reset(seed=42)
    state2 = env2.reset(seed=42)
    
    np.testing.assert_array_equal(state1, state2, "Initial states with same seed are not identical")
    
    for _ in range(50):
        action = np.random.randint(0, 3)
        s1, r1, d1, _ = env1.step(action)
        s2, r2, d2, _ = env2.step(action)
        
        np.testing.assert_array_equal(s1, s2, "States diverged after same actions")
        assert r1 == r2, "Rewards diverged"
        assert d1 == d2, "Done flags diverged"

def test_collision_tunneling():
    """Verify high-speed ball does not tunnel through the paddle."""
    env = BreakoutEnv()
    env.reset()
    
    env.paddle_x = CANVAS_W / 2 - PADDLE_BASE_W / 2
    env.paddle_w = PADDLE_BASE_W
    
    # Place ball exactly above the paddle, moving down extremely fast
    env.ball_x = env.paddle_x + env.paddle_w / 2
    env.ball_y = PADDLE_Y - BALL_R - 1
    env.ball_vx = 0.0
    env.ball_vy = 30.0 # High speed, would tunnel without substepping
    
    env.step(2)
    
    # Ball should have bounced UP (vy < 0)
    assert env.ball_vy < 0, "Ball tunneled through the paddle at high speed!"
    assert env.ball_y < PADDLE_Y, "Ball is stuck inside or below the paddle!"

def test_brick_tunneling():
    """Verify high-speed ball does not tunnel through bricks."""
    env = BreakoutEnv()
    env.reset()
    
    # Place a brick
    env.bricks = [{
        'x': 100, 'y': 100, 'w': 40, 'h': 20,
        'hp': 1, 'max_hp': 1, 'indestructible': False, 'alive': True
    }]
    
    # Place ball exactly above the brick, moving down extremely fast
    env.ball_x = 120
    env.ball_y = 100 - BALL_R - 1
    env.ball_vx = 0.0
    env.ball_vy = 30.0
    
    env.step(2)
    
    # Ball should have bounced UP (vy < 0) and brick should be destroyed
    assert env.ball_vy < 0, "Ball tunneled through the brick!"
    assert env.bricks[0]['alive'] == False, "Brick was not destroyed!"
    assert env.ball_y < 100, "Ball is stuck inside the brick!"

def test_clamping():
    """Verify paddle cannot move out of bounds."""
    env = BreakoutEnv()
    env.reset()
    
    env.paddle_w = PADDLE_BASE_W
    
    # Move right to the edge
    env.paddle_x = CANVAS_W - env.paddle_w - 1
    for _ in range(10):
        env.step(1)
        
    assert env.paddle_x + env.paddle_w <= CANVAS_W, "Paddle moved out of bounds to the right!"
    
    # Move left to the edge
    env.paddle_x = 1
    for _ in range(10):
        env.step(0)
        
    assert env.paddle_x >= 0, "Paddle moved out of bounds to the left!"

def test_state_shape():
    """Verify that the state vector is strictly 102 dimensions."""
    env = BreakoutEnv()
    obs = env.reset()
    
    assert obs.shape == (102,), f"Expected observation shape (102,), got {obs.shape}"
    assert env.obs_size == 102, f"Expected env.obs_size to be 102, got {env.obs_size}"

def test_rewards():
    """Verify the formal reward function (+10/brick, -5/life, +50/clear)."""
    env = BreakoutEnv()
    env.reset()
    
    # 1. Test Brick Hit Reward (+10)
    # Give the first brick 1 HP and ensure it gets hit
    for b in env.bricks:
        b['alive'] = False
        
    # Leave one other alive to prevent level clear bonus
    env.bricks[1] = {'x': 0, 'y': 0, 'w': 10, 'h': 10, 'hp': 1, 'max_hp': 1, 'indestructible': False, 'alive': True}
    
    env.bricks[0] = {'x': 100, 'y': 100, 'w': 40, 'h': 20, 'hp': 1, 'max_hp': 1, 'indestructible': False, 'alive': True}
    env.ball_x = 120
    env.ball_y = 100 - BALL_R - 1
    env.ball_vx = 0.0
    env.ball_vy = 30.0
    
    # We will need to set score to 0 to be sure
    env.score = 0
    _, reward_hit, _, _ = env.step(2) # don't move
    # 1 hp damaged = +10 score
    assert reward_hit == 10.0, f"Expected +10 reward for hitting brick, got {reward_hit}"
    
    # 2. Test Life Lost Reward (-5)
    env.reset()
    env.score = 0
    env.lives = 3
    env.ball_x = CANVAS_W / 2
    env.ball_y = CANVAS_H + BALL_R + 10 # below canvas
    env.ball_vy = 10.0
    
    _, reward_death, _, _ = env.step(2)
    assert reward_death == -5.0, f"Expected -5 reward for losing a life, got {reward_death}"
    
    # 3. Test Level Clear Reward (+50)
    env.reset()
    env.score = 0
    env.lives = 3
    # Kill all bricks except one
    for b in env.bricks:
        b['alive'] = False
        
    env.bricks[0] = {'x': 100, 'y': 100, 'w': 40, 'h': 20, 'hp': 1, 'max_hp': 1, 'indestructible': False, 'alive': True}
    env.ball_x = 120
    env.ball_y = 100 - BALL_R - 1
    env.ball_vx = 0.0
    env.ball_vy = 30.0
    
    _, reward_clear, done, _ = env.step(2)
    # Destroyed brick (+10) + level clear (+50) = 60
    assert reward_clear == 60.0, f"Expected +60 reward for clearing level, got {reward_clear}"
    assert done == True, "Expected environment to be done when level clears"

import os
import torch
from networks.cnn import build_network
from training.checkpoint_manager import CheckpointManager

def test_checkpoint_restart(tmp_path):
    """Verify that checkpoint saving and loading correctly persists weights and metadata."""
    env = BreakoutEnv()
    
    # 1. Create a model and save a checkpoint
    model1 = build_network('dqn', env.obs_size, 3)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    
    ckpt_mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
    meta1 = {'algorithm': 'dqn', 'custom_val': 42}
    
    ckpt_mgr.save(model1, optimizer1, epsilon=0.5, episode=10, meta=meta1)
    
    # 2. Simulate process restart by creating a completely new model
    model2 = build_network('dqn', env.obs_size, 3)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    # 3. Load the checkpoint
    loaded_data = ckpt_mgr.load(model2, optimizer2, episode=10)
    
    # 4. Assert weights are now identical
    assert torch.equal(model1.net[0].weight, model2.net[0].weight), "Weights were not restored correctly!"
    
    # 5. Assert metadata was enriched and preserved
    loaded_meta = loaded_data.get('meta', {})
    assert loaded_meta.get('custom_val') == 42
    assert loaded_meta.get('env_version') == '1.0.0'
    assert 'model_architecture' in loaded_meta
    assert loaded_meta['model_architecture']['class'] == 'QNetwork'
