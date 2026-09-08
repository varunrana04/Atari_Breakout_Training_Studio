import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from env.breakout_env import BreakoutEnv

def test_websocket_broadcast_schema():
    """
    Validates that the rendered state combined with step info 
    produces the exact schema required by the live visualization protocol.
    """
    env = BreakoutEnv()
    env.reset(seed=1)
    
    # Take a step to get step info
    obs, reward, done, info = env.step(env.ACTION_NOOP)
    
    # Simulate the dictionary merge that happens in main.py's WebSocket loop
    render_state = env.get_render_state()
    frame_state = {**render_state, 'score': info['score'], 'lives': info['lives']}
    
    # Assert exact schema keys
    required_keys = {'paddle_x', 'paddle_w', 'ball_x', 'ball_y', 'bricks', 'score', 'lives'}
    assert required_keys.issubset(frame_state.keys()), f"Missing keys in broadcast frame: {required_keys - frame_state.keys()}"
    
    # Assert valid ranges to protect visualization protocol
    assert 0.0 <= frame_state['paddle_x'] <= 1.0, f"paddle_x out of bounds: {frame_state['paddle_x']}"
    assert 0.0 <= frame_state['paddle_w'] <= 1.0, f"paddle_w out of bounds: {frame_state['paddle_w']}"
    assert 0.0 <= frame_state['ball_x'] <= 1.0, f"ball_x out of bounds: {frame_state['ball_x']}"
    assert 0.0 <= frame_state['ball_y'] <= 1.5, f"ball_y out of bounds (can fall below canvas): {frame_state['ball_y']}"
    
    assert isinstance(frame_state['score'], int), "score must be an integer"
    assert frame_state['score'] >= 0, "score must be non-negative"
    
    assert isinstance(frame_state['lives'], int), "lives must be an integer"
    assert 0 <= frame_state['lives'] <= 3, "lives must be between 0 and 3"
    
    assert isinstance(frame_state['bricks'], list), "bricks must be a list"
    assert len(frame_state['bricks']) == 96, f"expected 96 bricks in grid, got {len(frame_state['bricks'])}"
    for hp in frame_state['bricks']:
        assert isinstance(hp, (int, float)), f"brick hp must be numeric, got {type(hp)}"
        assert hp >= 0, f"brick hp cannot be negative, got {hp}"
        
    print("PASS test_websocket_broadcast_schema")

if __name__ == '__main__':
    test_websocket_broadcast_schema()
