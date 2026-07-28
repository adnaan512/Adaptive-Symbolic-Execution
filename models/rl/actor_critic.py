"""
Actor-Critic architecture for Proximal Policy Optimization (PPO) in Phase 6.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class ActorCriticNetwork(nn.Module if nn else object):
    """
    Combined Actor-Critic network.
    
    In Symbolic Execution, the action space is dynamic (choosing from N states currently in the queue).
    This network takes a state feature vector and outputs a hidden representation.
    
    - The Critic outputs V(s) to estimate the value of the current global queue state.
    - The Actor outputs a score for the state. When Softmax is applied over all scores in the queue, 
      it yields the probability distribution of selecting each state.
    """
    
    def __init__(self, input_dim: int = 14, hidden_dim: int = 128):
        super().__init__()
        if nn is None:
            raise ImportError("PyTorch is required to use ActorCriticNetwork.")
            
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_head = nn.Linear(hidden_dim, 1)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_features: Tensor of shape (batch_size, input_dim)
            
        Returns:
            action_scores: Tensor of shape (batch_size, 1) -> logits for the softmax policy
            state_values: Tensor of shape (batch_size, 1) -> V(s) baseline for advantage
        """
        hidden = self.shared_net(state_features)
        action_scores = self.actor_head(hidden)
        state_values = self.critic_head(hidden)
        return action_scores, state_values
