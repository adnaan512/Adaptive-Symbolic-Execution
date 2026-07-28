"""
Deep Q-Network (DQN) Agent for Reinforcement Learning (Phase 6).
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None

from models.rl.q_network import QNetwork

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    DQN Agent with Experience Replay.
    Selects states to explore using an epsilon-greedy strategy over Q-values.
    """
    
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 1000,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent.")
            
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.steps_done = 0
        
        self.q_net = QNetwork(input_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=buffer_size)

    def select_action(self, state_queue_obs: np.ndarray, valid_mask: np.ndarray) -> int:
        """
        Select an action (index of the state in the queue) using epsilon-greedy.
        
        Args:
            state_queue_obs: Shape (max_queue_size, 14)
            valid_mask: Boolean array of shape (max_queue_size,) indicating non-empty slots.
        """
        self.steps_done += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 / self.epsilon_decay))
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return 0  # Fallback if queue is empty
            
        if random.random() < self.epsilon:
            return int(np.random.choice(valid_indices))
            
        self.q_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_queue_obs).to(self.device)
            q_values = self.q_net(state_tensor).squeeze(-1) # shape (max_queue_size,)
            
            # Mask out invalid states by setting their Q-values to -infinity
            q_values[~torch.BoolTensor(valid_mask).to(self.device)] = float('-inf')
            
            best_action = torch.argmax(q_values).item()
            return int(best_action)

    def step(
        self, 
        obs: np.ndarray, 
        action: int, 
        reward: float, 
        next_obs: np.ndarray, 
        done: bool
    ) -> None:
        """Store experience and train."""
        self.memory.append((obs, action, reward, next_obs, done))
        
        if len(self.memory) >= self.batch_size:
            self._learn()

    def _learn(self) -> None:
        batch = random.sample(self.memory, self.batch_size)
        
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
        
        obs_tensor = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        action_tensor = torch.LongTensor(action_batch).to(self.device).unsqueeze(-1)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(-1)
        next_obs_tensor = torch.FloatTensor(np.array(next_obs_batch)).to(self.device)
        done_tensor = torch.FloatTensor(done_batch).to(self.device).unsqueeze(-1)
        
        self.q_net.train()
        
        # We selected `action_tensor` which is the index in the max_queue_size dimension.
        # We need to extract the Q-value for the specific state that was chosen.
        # obs_tensor: (batch_size, max_queue_size, 14)
        # q_net(obs_tensor): (batch_size, max_queue_size, 1)
        curr_q = self.q_net(obs_tensor)
        curr_q = curr_q.gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)  # (batch_size, 1)
        
        with torch.no_grad():
            next_q = self.target_net(next_obs_tensor).max(1)[0] # max over max_queue_size
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * next_q
            
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())
