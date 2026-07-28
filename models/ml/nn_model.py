"""
Neural Network model for Ranking Symbolic Execution States.
Implemented using PyTorch.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

from models.ml.base import RankingModel


class MLP(nn.Module if nn else object):
    """Simple Multi-Layer Perceptron for tabular state features."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralNetRanker:
    """PyTorch-based MLP regressor for predicting state priority."""
    
    def __init__(
        self, 
        input_dim: int = 14, 
        hidden_dim: int = 64, 
        learning_rate: float = 1e-3, 
        batch_size: int = 32, 
        epochs: int = 50,
        device: str = "cpu"
    ):
        if torch is None:
            raise ImportError("PyTorch is required for NeuralNetRanker")
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)
        
        self.model = MLP(input_dim, hidden_dim).to(self.device)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """
        Train the MLP on the dataset using MSE Loss.
        """
        if len(X) == 0:
            warnings.warn("NeuralNetRanker.fit called with empty dataset.")
            self.is_fitted = True
            return

        # Prepare dataloader
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Determine actual batch size (avoid issues with dataset smaller than batch_size)
        actual_batch_size = min(self.batch_size, len(dataset))
        dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        epochs_to_run = kwargs.get("epochs", self.epochs)
        
        for _ in range(epochs_to_run):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
            
        if len(X) == 0:
            return np.array([])
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy().flatten()

    def save(self, path: str | Path) -> None:
        """Save the PyTorch model state_dict."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'is_fitted': self.is_fitted
        }, str(path))

    def load(self, path: str | Path) -> None:
        """Load the PyTorch model state_dict."""
        checkpoint = torch.load(str(path), map_location=self.device)
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        
        # Re-initialize architecture just in case dims changed
        self.model = MLP(self.input_dim, self.hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = checkpoint['is_fitted']
