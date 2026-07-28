"""
Deep Q-Network (DQN) architecture for Reinforcement Learning (Phase 6).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class QNetwork(nn.Module if nn else object):
    """
    Neural Network predicting the Q-value (expected future reward) of selecting a given state.
    
    The input is the 14-dimensional ExecutionStateFeatures vector.
    The output is a single scalar Q-value for that state.
    
    In DQN for symbolic execution, the Action space is dynamic (the queue of states).
    We evaluate QNetwork(state) for all states in the queue, and select the state with the highest Q.
    """
    
    def __init__(self, input_dim: int = 14, hidden_dim: int = 128):
        super().__init__()
        if nn is None:
            raise ImportError("PyTorch is required to use QNetwork.")
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_features: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Q-values: Tensor of shape (batch_size, 1)
        """
        return self.net(state_features)
