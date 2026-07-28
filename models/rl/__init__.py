"""
models.rl — Phase 6.

Responsibility: neural network architectures (Q-network for DQN,
actor-critic for PPO) used by `reinforcement_learning/`. Kept separate from
`reinforcement_learning/` itself so the training loop / environment code
doesn't need to know the network internals (SOLID: single responsibility).
"""
