"""
Tests for Phase 6: Reinforcement Learning components.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.rl.actor_critic import ActorCriticNetwork
from models.rl.q_network import QNetwork
from reinforcement_learning.dqn import DQNAgent
from reinforcement_learning.env import SymbolicExecutionEnv
from reinforcement_learning.ppo import PPOAgent
from reinforcement_learning.trainer import RLTrainer


class TestRLEnvironment:
    def test_env_reset(self):
        env = SymbolicExecutionEnv(max_queue_size=10, max_steps=50)
        obs, info = env.reset()
        
        assert obs.shape == (10, 14)
        assert env.current_step == 0
        assert len(env.state_queue) == 1
        assert env.state_queue[0].state_id == "init-0"

    def test_env_step_valid_action(self):
        env = SymbolicExecutionEnv(max_queue_size=10, max_steps=50)
        env.reset()
        
        # Action 0 is valid since state_queue has 1 element
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert obs.shape == (10, 14)
        assert isinstance(reward, float)
        assert reward >= 0.0
        assert "coverage" in info

    def test_env_step_invalid_action(self):
        env = SymbolicExecutionEnv(max_queue_size=10, max_steps=50)
        env.reset()
        
        # Action 5 is invalid
        obs, reward, terminated, truncated, info = env.step(5)
        
        assert reward == -10.0
        assert info.get("invalid_action") is True


class TestRLNetworks:
    def test_q_network_forward(self):
        net = QNetwork(input_dim=14, hidden_dim=32)
        state_tensor = torch.randn(8, 14)  # batch size 8
        q_values = net(state_tensor)
        assert q_values.shape == (8, 1)

    def test_actor_critic_forward(self):
        net = ActorCriticNetwork(input_dim=14, hidden_dim=32)
        state_tensor = torch.randn(8, 14)
        action_scores, state_values = net(state_tensor)
        assert action_scores.shape == (8, 1)
        assert state_values.shape == (8, 1)


class TestRLAgents:
    @pytest.fixture
    def mock_env(self):
        return SymbolicExecutionEnv(max_queue_size=5, max_steps=10)

    def test_dqn_agent_train_loop(self, mock_env):
        agent = DQNAgent(
            input_dim=14, hidden_dim=32, buffer_size=10, batch_size=2, epsilon_decay=10
        )
        trainer = RLTrainer(agent, mock_env, episodes=2)
        history = trainer.train()
        
        assert len(history) == 2
        assert len(agent.memory) > 0

    def test_ppo_agent_train_loop(self, mock_env):
        agent = PPOAgent(
            input_dim=14, hidden_dim=32, ppo_epochs=1
        )
        trainer = RLTrainer(agent, mock_env, episodes=2)
        history = trainer.train()
        
        assert len(history) == 2
        # After training episode, ppo learn clears buffer
        assert len(agent.states) == 0
