# Low Level Design

## Breakout Physics Engine (`breakout_env.py`)
- **Coordinates**: The canvas is arbitrarily scaled to $480 \times 640$. All elements are calculated in absolute pixels but normalized to `[0, 1]` floats before transmission to the Neural Network or Frontend.
- **Collision Detection**: 
  - Paddle: A simple AABB (Axis-Aligned Bounding Box) collision. If the ball enters the paddle's rectangle and `velocity_y > 0`, it deflects. The deflection angle is linearly interpolated based on the intersection point relative to the paddle center, allowing the player/agent to "aim".
  - Bricks: Represented as a flat array of integers (HP). The ball's AABB is checked against each active brick. If a collision occurs, the ball is physically pushed out of the brick's geometry before its velocity is inverted to prevent "phasing".

## Neural Network (`network.py`)
- **Input**: `(Batch, Channels, Height, Width)` where Channels = 4 (frame stacking).
- **Architecture**:
  - `Conv2d(4, 32, kernel=8, stride=4)`
  - `Conv2d(32, 64, kernel=4, stride=2)`
  - `Conv2d(64, 64, kernel=3, stride=1)`
  - `Linear(flattened, 512)`
- **Dueling Streams**:
  - `Value Stream`: `Linear(512, 1)` -> $V(s)$
  - `Advantage Stream`: `Linear(512, num_actions)` -> $A(s, a)$
- **Aggregation**: $Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|A|}\sum A(s,a)\right)$

## Training Loop (`trainer.py`)
- Uses **Huber Loss** (`F.smooth_l1_loss`) to prevent exploding gradients when TD errors are large.
- The Replay Buffer has a capacity of `100,000` transitions.
- A **Target Network** is synchronized with the primary network every `1000` steps.
