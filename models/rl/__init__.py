"""
models.rl — Phase 6.

Responsibility: neural network architectures (Q-network for DQN,
actor-critic for PPO) used by `reinforcement_learning/`. Kept separate from
`reinforcement_learning/` itself so the training loop / environment code
doesn't need to know the network internals (SOLID: single responsibility).
"""

from models.rl.q_network import QNetwork
from models.rl.actor_critic import ActorCriticNetwork

__all__ = [
    "QNetwork",
    "ActorCriticNetwork",
]
