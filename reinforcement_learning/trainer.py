"""
Training Loop for Reinforcement Learning Agents (Phase 6).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reinforcement_learning.env import SymbolicExecutionEnv

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Orchestrates the training of an RL agent over the SymbolicExecutionEnv.
    """
    
    def __init__(self, agent: Any, env: SymbolicExecutionEnv, episodes: int = 100):
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.history: list[float] = []

    def train(self) -> list[float]:
        """
        Run the training loop for the specified number of episodes.
        Returns the history of total rewards per episode.
        """
        for episode in range(self.episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0
            
            while not done:
                # Valid mask: True where the queue slot is actually filled
                # We know from env.py that an empty slot is all zeros (including state_id length which we simulate here via non-zero check)
                # A simpler mask: anything with execution_depth > 0 or distance_to_uncovered_branch > 0
                # We'll use instruction_count as a proxy, or just track queue size.
                # In env, the first N rows are filled.
                num_active = len(self.env.state_queue)
                valid_mask = np.zeros(self.env.max_queue_size, dtype=bool)
                if num_active > 0:
                    valid_mask[:num_active] = True
                    
                action = self.agent.select_action(obs, valid_mask)
                
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Check Agent Type via hasattr to avoid strict isinstance imports
                if hasattr(self.agent, "update_target_network"):
                    # DQN API
                    self.agent.step(obs, action, reward, next_obs, done)
                else:
                    # PPO API
                    self.agent.step(reward, done)
                    
                obs = next_obs
                total_reward += reward

            # End of episode tasks
            if hasattr(self.agent, "update_target_network"):
                # DQN Target Network Update
                if episode % 10 == 0:
                    self.agent.update_target_network()
            else:
                # PPO Learn
                self.agent.learn()
                
            self.history.append(total_reward)
            logger.info(f"Episode {episode + 1}/{self.episodes} - Total Reward: {total_reward:.4f}")
            
        return self.history
