# Phase 6 Notes — Reinforcement Learning Agent

## What this phase produced

### Neural Architectures (`models/rl/`)
1. **`q_network.py`**: A standard PyTorch MLP that predicts a Q-value (expected total future reward) for a given state.
2. **`actor_critic.py`**: A dual-headed network for PPO. It shares a representation base, outputting both an Action Logit (Actor) and a State Value (Critic).

### Training Environment (`reinforcement_learning/`)
1. **`env.py`**: Implements `SymbolicExecutionEnv`, conforming to the standard Gymnasium interface. It acts as an offline simulator holding a queue of active symbolic states. 
    - **State**: The matrix of feature vectors in the current queue.
    - **Action**: Index of the state to pop and explore.
    - **Reward**: Change in coverage.
2. **`dqn.py`**: The `DQNAgent` using a Replay Buffer and epsilon-greedy exploration.
3. **`ppo.py`**: The `PPOAgent` using clipped objective ratios to ensure monotonic policy improvement.
4. **`trainer.py`**: Centralizes the `gym.Env` episode rollout loop.

## Architecture Context

Why map Symbolic Execution to RL?
Standard ML Ranking (Phase 4) is greedy: it predicts which state will yield the most coverage *right now*. RL plans for the future. By using the Q-value or Advantage, the agent learns that sometimes it is optimal to explore a "boring" path now (taking a short-term hit in reward) because it leads to a massive coverage payoff deeper down the execution tree.

### Dynamic Action Space Challenge
Unlike standard RL games (where actions are fixed like UP, DOWN, LEFT, RIGHT), here the action space is **the index of the state in the queue**. As states fork and terminate, the queue changes in size and content. 
We solve this by padding the queue to `max_queue_size` in the observation matrix. The networks evaluate all slots, but we apply a `valid_mask` (setting invalid slot Q-values or Actor Logits to $-\infty$) before applying `argmax()` or `softmax()`, ensuring the agent never selects an empty slot.

## Verification (Definition of Done)
1. Environment respects the Gym interface (`reset` and `step`).
2. Replay Buffer properly captures and uniformly samples `(state, action, reward, next_state, done)` tuples.
3. Neural Networks run forward passes on batched queue states without crashing due to dimensional mismatches.
4. PPO correctly clips objective ratios and updates the policy gradient.
