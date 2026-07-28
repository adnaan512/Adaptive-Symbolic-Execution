"""
Proximal Policy Optimization (PPO) Agent for Reinforcement Learning (Phase 6).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except ImportError:
    torch = None
    nn = None
    optim = None
    Categorical = None

from models.rl.actor_critic import ActorCriticNetwork

logger = logging.getLogger(__name__)


class PPOAgent:
    """
    PPO Agent with an Actor-Critic architecture.
    Uses clipped objective to safely update the policy without destructive large steps.
    """
    
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        ppo_epochs: int = 4,
        device: str = "cpu"
    ):
        if torch is None:
            raise ImportError("PyTorch is required for PPOAgent.")
            
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        
        self.ac_net = ActorCriticNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        
        # Rollout buffer
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.valid_masks: list[np.ndarray] = []

    def select_action(self, state_queue_obs: np.ndarray, valid_mask: np.ndarray) -> int:
        """
        Sample an action from the policy distribution, masked by valid states.
        """
        self.ac_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_queue_obs).to(self.device)
            # action_scores: (max_queue_size, 1)
            action_scores, _ = self.ac_net(state_tensor)
            action_scores = action_scores.squeeze(-1)
            
            # Mask invalid slots
            mask_tensor = torch.BoolTensor(valid_mask).to(self.device)
            action_scores[~mask_tensor] = float('-inf')
            
            # If all are masked (empty queue fallback)
            if (~mask_tensor).all():
                return 0
                
            probs = torch.softmax(action_scores, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
            self.states.append(state_queue_obs)
            self.actions.append(action.item())
            self.log_probs.append(dist.log_prob(action).item())
            self.valid_masks.append(valid_mask)
            
            return int(action.item())

    def step(self, reward: float, done: bool) -> None:
        """Record reward and done flag. If done, we don't automatically train in this basic setup;
        training is typically called externally at the end of an episode or rollout."""
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self) -> None:
        """
        Compute advantages and update Actor-Critic using PPO clipped loss.
        """
        if len(self.states) == 0:
            return
            
        self.ac_net.train()
        
        # Calculate Returns (Discounted rewards)
        returns = []
        discounted_sum = 0.0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_sum = 0.0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
            
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        # Normalize returns
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        old_states = torch.FloatTensor(np.array(self.states)).to(self.device)
        old_actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        valid_masks_tensor = torch.BoolTensor(np.array(self.valid_masks)).to(self.device)
        
        for _ in range(self.ppo_epochs):
            # Evaluate current policy
            # old_states is (batch_size, max_queue_size, 14)
            action_scores, state_values = self.ac_net(old_states)
            action_scores = action_scores.squeeze(-1) # (batch_size, max_queue_size)
            
            # Apply mask
            action_scores[~valid_masks_tensor] = float('-inf')
            
            probs = torch.softmax(action_scores, dim=-1)
            dist = Categorical(probs)
            
            curr_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            
            # We need a single state value per batch item. 
            # In our setup, we can approximate V(global_state) as the max or mean of valid V(s).
            # Here we use the mean of valid states for simplicity.
            masked_values = state_values.squeeze(-1).masked_fill(~valid_masks_tensor, 0.0)
            valid_counts = valid_masks_tensor.sum(dim=-1).clamp(min=1).float()
            global_values = masked_values.sum(dim=-1) / valid_counts
            
            advantages = returns_tensor - global_values.detach()
            
            # PPO Ratio
            ratios = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(global_values, returns_tensor)
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.valid_masks.clear()
