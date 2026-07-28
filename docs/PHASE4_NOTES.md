# Phase 4 Notes — Machine Learning Ranking Models

## What this phase produced

### New modules

| File | Purpose |
|------|---------|
| `models/ml/base.py` | Contains the `RankingModel` Protocol, enforcing SOLID dependency inversion across all our ML algorithms. |
| `models/ml/tree_models.py` | Contains implementations for `RandomForestRanker`, `XGBoostRanker`, and `LightGBMRanker` built on top of Scikit-Learn. |
| `models/ml/nn_model.py` | Contains `NeuralNetRanker`, a PyTorch-based Multi-Layer Perceptron (MLP) for continuous state-score prediction. |
| `models/ml/trainer.py` | Contains `ModelTrainer`, unifying data formatting (`ExecutionStateFeatures` → NumPy/Torch Tensors) and training evaluation. |
| `docs/design/label_construction.md` | Documents the design choices regarding ML targets (why regression over classification, and how to discount time horizons). |

### New tests

| File | Tests |
|------|-------|
| `tests/test_ml_models.py` | Unit tests covering protocol conformance, fitting, predicting, error handling for unfitted predictions, PyTorch edge cases, and disk serialization round-tripping for all 4 model types. |

---

## Architecture Context

Phase 4 transforms the Symbolic Execution state selection from a static heuristic into an ML-driven predictive problem. 

### Why Regression?
KLEE naturally uses heuristics like "Distance to Uncovered Instructions." We are learning an abstract meta-heuristic: "How much coverage gain do we expect if we follow this state?"
By predicting a continuous score (regression), KLEE can dynamically sort its internal state queue. 

### Duck Typing & Protocol
By exposing only the `RankingModel` interface:
```python
def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None: ...
def predict(self, X: np.ndarray) -> np.ndarray: ...
def save(self, path: str | Path) -> None: ...
def load(self, path: str | Path) -> None: ...
```
We decoupled Phase 4 entirely from Phase 7 (Evaluation) and Phase 6 (Reinforcement Learning). We can hot-swap `XGBoostRanker` with `NeuralNetRanker` in 1 line of configuration.

### Disk Serialization
Tree models serialize using the standard `pickle` approach, while PyTorch strictly uses `state_dict` loading. The `save()` and `load()` implementations abstract this entirely.

---

## Verification (Definition of Done)

1. The test suite successfully trains all 4 models on a synthetic 14-dimensional feature vector.
2. The PyTorch MLP gracefully handles empty datasets (which can occur in certain edge-case symbolic execution boundaries).
3. The dataset pipeline accurately maps arrays to Pydantic definitions and raises `ValueError` if lengths mismatch.
4. Models successfully serialize to and load from temporary directories.
