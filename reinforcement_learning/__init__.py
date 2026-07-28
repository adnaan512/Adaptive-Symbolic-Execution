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

Planned public interface:

    class SymbolicExecutionEnv(gym.Env): ...
    class DQNAgent: ...
    class PPOAgent: ...
    def train(agent_cls, env, config) -> TrainedAgent: ...

RQ3 (does RL improve over static ML ranking?) is answered by comparing this
module's trained-policy coverage against the Phase-4 static ranker's
coverage, both against the Phase-2 baseline heuristics, in `evaluation/`.
"""
