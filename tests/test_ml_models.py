"""
Tests for Phase 4: Machine Learning Ranking Models.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.schemas import ExecutionStateFeatures
from models.ml import (
    LightGBMRanker,
    NeuralNetRanker,
    RandomForestRanker,
    RankingModel,
    XGBoostRanker,
)
from models.ml.trainer import ModelTrainer, prepare_training_data


@pytest.fixture
def dummy_data():
    """Generate a tiny synthetic dataset (X, y) for model testing."""
    X = np.random.rand(10, 14)
    y = np.random.rand(10)
    return X, y


@pytest.fixture
def dummy_states():
    """Generate dummy ExecutionStateFeatures."""
    states = []
    for i in range(10):
        states.append(ExecutionStateFeatures(
            state_id=f"state-{i}",
            execution_depth=i,
            num_path_constraints=i * 2,
            constraint_complexity=i * 0.5,
            num_symbolic_variables=1,
            solver_time_ms=10.0,
            instruction_count=100,
            branch_count=10,
            loop_depth=0,
            distance_to_uncovered_branch=5.0,
            current_branch_coverage=0.5,
            memory_object_count=2,
            num_forks=1,
            state_age=1.0,
            path_length=i * 10
        ))
    labels = [float(i) / 10.0 for i in range(10)]
    return states, labels


class TestModelTrainer:
    def test_prepare_training_data(self, dummy_states):
        states, labels = dummy_states
        X, y = prepare_training_data(states, labels)
        assert X.shape == (10, 14)
        assert y.shape == (10,)
        
    def test_prepare_training_data_mismatch(self, dummy_states):
        states, labels = dummy_states
        with pytest.raises(ValueError):
            prepare_training_data(states, labels[:-1])


class TestMLModels:
    @pytest.mark.parametrize("model_class", [
        RandomForestRanker,
        XGBoostRanker,
        LightGBMRanker,
        NeuralNetRanker,
    ])
    def test_model_fit_predict(self, model_class, dummy_data):
        """Test that all models conform to the RankingModel protocol and can fit/predict."""
        X, y = dummy_data
        
        try:
            model = model_class()
        except ImportError:
            pytest.skip(f"Skipping {model_class.__name__} due to missing dependency.")

        # Ensure it conforms to protocol visually (duck typing check is implicitly done by usage)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "save")
        assert hasattr(model, "load")

        # Fit
        model.fit(X, y, epochs=1)  # Epochs arg is just ignored by tree models, used by NN
        
        # Predict
        preds = model.predict(X)
        assert preds.shape == (10,)
        
    @pytest.mark.parametrize("model_class", [
        RandomForestRanker,
        XGBoostRanker,
        LightGBMRanker,
        NeuralNetRanker,
    ])
    def test_model_save_load(self, model_class, dummy_data, tmp_path):
        """Test model serialization round-trip."""
        X, y = dummy_data
        
        try:
            model = model_class()
        except ImportError:
            pytest.skip(f"Skipping {model_class.__name__} due to missing dependency.")
            
        model.fit(X, y, epochs=1)
        preds_original = model.predict(X)
        
        save_path = tmp_path / "model.pkl"
        model.save(save_path)
        assert save_path.exists()
        
        new_model = model_class()
        new_model.load(save_path)
        
        preds_loaded = new_model.predict(X)
        np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-5, atol=1e-5)

    def test_neural_net_empty_data(self):
        try:
            model = NeuralNetRanker()
        except ImportError:
            pytest.skip("PyTorch not installed.")
            
        # Fitting on empty shouldn't crash
        model.fit(np.array([]), np.array([]))
        
        # Predicting on empty should return empty array
        preds = model.predict(np.array([]))
        assert len(preds) == 0

    def test_predict_unfitted_raises(self, dummy_data):
        try:
            model = RandomForestRanker()
        except ImportError:
            pytest.skip("scikit-learn not installed.")
            
        with pytest.raises(RuntimeError, match="Model is not fitted"):
            model.predict(dummy_data[0])
