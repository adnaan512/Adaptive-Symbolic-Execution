"""
Unified training orchestrator for ML Ranking Models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.core.schemas import ExecutionStateFeatures
from models.ml.base import RankingModel

logger = logging.getLogger(__name__)


def prepare_training_data(
    states: list[ExecutionStateFeatures], 
    labels: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert structured states and labels into numpy arrays for training.
    
    Parameters
    ----------
    states : list[ExecutionStateFeatures]
        List of extracted execution states.
    labels : list[float]
        The target coverage gain or priority scores.
        
    Returns
    -------
    X, y : tuple[np.ndarray, np.ndarray]
        The feature matrix and label vector.
    """
    if len(states) != len(labels):
        raise ValueError("Number of states must match number of labels.")
        
    if not states:
        return np.array([]), np.array([])
        
    X = np.array([state.to_vector() for state in states])
    y = np.array(labels)
    return X, y


class ModelTrainer:
    """
    Orchestrates training and evaluating RankingModels on given datasets.
    """
    
    def __init__(self, model: RankingModel):
        self.model = model

    def train(
        self, 
        states: list[ExecutionStateFeatures], 
        labels: list[float], 
        **kwargs: Any
    ) -> None:
        """
        Train the underlying model on the provided states and labels.
        """
        logger.info(f"Preparing training data for {len(states)} samples.")
        X, y = prepare_training_data(states, labels)
        
        logger.info(f"Fitting model {self.model.__class__.__name__}...")
        self.model.fit(X, y, **kwargs)
        logger.info("Model fitting complete.")

    def evaluate_mse(
        self, 
        states: list[ExecutionStateFeatures], 
        labels: list[float]
    ) -> float:
        """
        Evaluate the model using Mean Squared Error.
        """
        X, y_true = prepare_training_data(states, labels)
        if len(X) == 0:
            return 0.0
            
        y_pred = self.model.predict(X)
        mse = float(np.mean((y_true - y_pred) ** 2))
        return mse
