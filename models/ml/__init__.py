"""
models.ml — Phase 4.

Responsibility: train/serve ranking models that map an
`ExecutionStateFeatures` vector (+ optional LLM confidence) to a
`PriorityScore`.

Planned public interface:

    class RankingModel(Protocol):
        def fit(self, X, y) -> None: ...
        def predict(self, X) -> np.ndarray: ...

    RandomForestRanker, XGBoostRanker, LightGBMRanker, NeuralNetRanker,
    (optional) GNNRanker — all implement `RankingModel` so `evaluation/` and
    `reinforcement_learning/` can swap models without code changes (SOLID:
    dependency inversion).

Training data for Phase 4 is bootstrapped by running KLEE runs to
completion under each baseline heuristic and labeling states by whether
choosing them led to a subsequent coverage gain within a short horizon —
see `docs/design/label_construction.md` (added in Phase 4).
"""

from models.ml.base import RankingModel
from models.ml.tree_models import RandomForestRanker, XGBoostRanker, LightGBMRanker

__all__ = [
    "RankingModel",
    "RandomForestRanker",
    "XGBoostRanker",
    "LightGBMRanker",
]
