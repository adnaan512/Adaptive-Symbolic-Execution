"""
Base Protocol for all Machine Learning Ranking Models.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class RankingModel(Protocol):
    """
    Protocol defining the interface for all Phase 4 ML ranking models.
    
    Adhering to SOLID principles (Dependency Inversion), the rest of the system
    (e.g. RL loops or KLEE state-selection hooks) relies only on this Protocol,
    allowing us to hot-swap XGBoost, Random Forest, or Neural Networks without
    changing orchestration code.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """
        Train the model on the state features (X) to predict the target score (y).
        
        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features). The extracted state features.
        y : np.ndarray
            Shape (n_samples,). The target priority score (e.g. coverage gain).
        **kwargs : Any
            Model-specific training arguments (e.g., epochs, learning_rate).
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the priority score for a batch of states.
        
        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features). The state features.
            
        Returns
        -------
        np.ndarray
            Shape (n_samples,). The predicted scores. Higher is better.
        """
        ...

    def save(self, path: str | Path) -> None:
        """
        Serialize the trained model to disk.
        """
        ...

    def load(self, path: str | Path) -> None:
        """
        Load a trained model from disk.
        """
        ...


def save_pickle_model(model: Any, path: str | Path) -> None:
    """Helper to save a scikit-learn-like model using pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_pickle_model(path: str | Path) -> Any:
    """Helper to load a scikit-learn-like model using pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)
