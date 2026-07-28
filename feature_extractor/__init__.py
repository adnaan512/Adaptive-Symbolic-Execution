"""
feature_extractor — Phase 3.

Responsibility: turn a raw KLEE execution-state log entry into a validated
:class:`backend.core.schemas.ExecutionStateFeatures` instance.

Public interface:

    extract_features(raw_state: dict) -> ExecutionStateFeatures
    extract_batch(raw_states: list[dict]) -> list[ExecutionStateFeatures]
    save_dataset_csv(states: list[ExecutionStateFeatures], path: str) -> None
    save_dataset_jsonl(states: list[ExecutionStateFeatures], path: str) -> None

Depends only on `backend.core.schemas` — does not import `klee/`, `models/`,
`llm/`, or `reinforcement_learning/` directly, so it can be unit-tested with
synthetic dict fixtures without a KLEE installation.
"""

from feature_extractor.extractor import extract_features, extract_batch
from feature_extractor.exceptions import FeatureExtractionError, ValidationError

__all__ = [
    "extract_features",
    "extract_batch",
    "FeatureExtractionError",
    "ValidationError",
]
