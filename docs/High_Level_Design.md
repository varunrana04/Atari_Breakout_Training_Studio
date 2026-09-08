# High Level Design

## The Reinforcement Learning Triad
1. **The Environment**: Simulated state containing geometry logic. The agent interacts with it by submitting an action `A` and receiving a new state `S'`, a reward `R`, and a boolean flag `done`.
2. **The Memory (Replay Buffer)**: A circular array storing tuples of `(S, A, R, S', done)`. This allows the agent to sample uncorrelated batches of past experiences, stabilizing training (Experience Replay).
3. **The Agent (Neural Network)**:
   - Evaluates the current State `S` using a Convolutional Neural Network (CNN) to process the pixel/grid matrix.
   - Outputs Q-values (expected future cumulative reward) for each possible action (Left, Right, No-Op).
   - Trained by minimizing the temporal difference (TD) error between its current Q-value prediction and the target Q-value (computed using the Bellman Equation).

## Subsystems
- **Dueling Architecture**: Splits the final hidden layer of the neural network into two separate streams: one estimating the scalar *Value* of being in the current state, and another estimating the *Advantage* of taking each action. This helps the agent learn which states are inherently valuable regardless of the action taken (e.g., when the ball is safely far away from the paddle).
- **Double Q-Learning**: Utilizes a secondary "Target" neural network with frozen weights to calculate the target Q-values during loss computation. This prevents the primary network from aggressively overestimating Q-values.
