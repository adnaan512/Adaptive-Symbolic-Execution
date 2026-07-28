"""
Gymnasium-compatible Environment for Symbolic Execution.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from backend.core.schemas import ExecutionStateFeatures

logger = logging.getLogger(__name__)


class SymbolicExecutionEnv(gym.Env if gym else object):
    """
    Simulates the KLEE state selection process as an RL environment.
    
    In a real run (Phase 7), this environment acts as a wrapper around the live KLEE process.
    For Phase 6, we implement the Gym interface so DQN/PPO can be trained offline
    on pre-collected trace datasets (or a mock queue).
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_queue_size: int = 100, max_steps: int = 1000):
        super().__init__()
        if gym is None:
            raise ImportError("gymnasium is required for SymbolicExecutionEnv")
            
        self.max_queue_size = max_queue_size
        self.max_steps = max_steps
        
        # State: a fixed-size queue of execution states (padded with zeros if < max)
        # Each state has 14 features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.max_queue_size, 14), 
            dtype=np.float32
        )
        
        # Action: index of the state to select from the queue
        self.action_space = spaces.Discrete(self.max_queue_size)
        
        self.current_step = 0
        self.state_queue: list[ExecutionStateFeatures] = []
        self.last_coverage = 0.0

    def _get_obs(self) -> np.ndarray:
        """Construct the observation matrix."""
        obs = np.zeros((self.max_queue_size, 14), dtype=np.float32)
        for i, state in enumerate(self.state_queue[:self.max_queue_size]):
            obs[i] = state.to_vector()
        return obs

    def reset(
        self, 
        *, 
        seed: int | None = None, 
        options: dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to start a new symbolic execution trace."""
        super().reset(seed=seed)
        self.current_step = 0
        self.last_coverage = 0.0
        
        # Initialize with a dummy initial state
        initial_state = ExecutionStateFeatures(
            state_id="init-0", execution_depth=0, num_path_constraints=0,
            constraint_complexity=0.0, num_symbolic_variables=0, solver_time_ms=0.0,
            instruction_count=0, branch_count=0, loop_depth=0,
            distance_to_uncovered_branch=10.0, current_branch_coverage=0.0,
            memory_object_count=0, num_forks=0, state_age=0.0, path_length=0
        )
        self.state_queue = [initial_state]
        
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step: pick the state at `action`, simulate KLEE execution.
        """
        self.current_step += 1
        
        # In a real environment, stepping invalid/empty index would be caught by mask.
        # Here we just apply a heavy penalty if the action is invalid.
        if action >= len(self.state_queue):
            reward = -10.0
            terminated = False
            truncated = self.current_step >= self.max_steps
            return self._get_obs(), reward, terminated, truncated, {"invalid_action": True}

        # Simulate state popping and forking
        selected_state = self.state_queue.pop(action)
        
        # MOCK SIMULATION: Randomly increase coverage and spawn new states
        new_coverage = min(1.0, self.last_coverage + np.random.uniform(0.0, 0.05))
        reward = new_coverage - self.last_coverage
        self.last_coverage = new_coverage
        
        # Spawn 0 to 2 new states based on the selected state
        num_new = np.random.randint(0, 3)
        for i in range(num_new):
            new_state = selected_state.model_copy(update={
                "state_id": f"state-{self.current_step}-{i}",
                "execution_depth": selected_state.execution_depth + 1,
                "current_branch_coverage": new_coverage
            })
            if len(self.state_queue) < self.max_queue_size:
                self.state_queue.append(new_state)

        terminated = len(self.state_queue) == 0 or new_coverage >= 0.99
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), float(reward), terminated, truncated, {"coverage": new_coverage}

    def render(self) -> None:
        pass
