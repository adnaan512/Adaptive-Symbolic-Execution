"""
Tree-based ML models for Ranking Symbolic Execution States.
Includes Random Forest, XGBoost, and LightGBM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# We wrap the imports so the module can still be imported if
# a specific library is missing (useful for testing environments).
try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from models.ml.base import save_pickle_model, load_pickle_model


class RandomForestRanker:
    """Random Forest regressor for predicting state priority."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        if RandomForestRegressor is None:
            raise ImportError("scikit-learn is required for RandomForestRanker")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        save_pickle_model(self, path)

    def load(self, path: str | Path) -> None:
        loaded = load_pickle_model(path)
        self.model = loaded.model
        self.is_fitted = loaded.is_fitted


class XGBoostRanker:
    """XGBoost regressor for predicting state priority."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42):
        if xgb is None:
            raise ImportError("xgboost is required for XGBoostRanker")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            objective="reg:squarederror"
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        save_pickle_model(self, path)

    def load(self, path: str | Path) -> None:
        loaded = load_pickle_model(path)
        self.model = loaded.model
        self.is_fitted = loaded.is_fitted


class LightGBMRanker:
    """LightGBM regressor for predicting state priority."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42):
        if lgb is None:
            raise ImportError("lightgbm is required for LightGBMRanker")
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        save_pickle_model(self, path)

    def load(self, path: str | Path) -> None:
        loaded = load_pickle_model(path)
        self.model = loaded.model
        self.is_fitted = loaded.is_fitted
