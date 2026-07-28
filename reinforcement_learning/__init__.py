"""
reinforcement_learning — Phase 6.

Responsibility: a Gym-style environment wrapping KLEE state selection, plus
DQN and PPO training loops.

    State  = ExecutionStateFeatures (+ optional LLM BranchPrediction, + ML
             PriorityScore) for every currently-active KLEE execution state.
    Action = index of the execution state to explore next.
    Reward = marginal increase in branch coverage, minus penalties for
             timeout / solver overhead / state explosion (see
             configs/config.yaml: reinforcement_learning.reward).
"""

from reinforcement_learning.env import SymbolicExecutionEnv
from reinforcement_learning.dqn import DQNAgent
from reinforcement_learning.ppo import PPOAgent
from reinforcement_learning.trainer import RLTrainer

__all__ = [
    "SymbolicExecutionEnv",
    "DQNAgent",
    "PPOAgent",
    "RLTrainer",
]

