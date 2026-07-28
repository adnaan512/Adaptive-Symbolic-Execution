"""
feature_extractor — Phase 3.

Responsibility: turn a raw KLEE execution-state log entry into a validated
:class:`backend.core.schemas.ExecutionStateFeatures` instance.

Planned public interface:

    extract_features(raw_state: dict) -> ExecutionStateFeatures
    extract_batch(raw_states: list[dict]) -> list[ExecutionStateFeatures]
    save_dataset(states: list[ExecutionStateFeatures], path: str) -> None

Depends only on `backend.core.schemas` — does not import `klee/`, `models/`,
`llm/`, or `reinforcement_learning/` directly, so it can be unit-tested with
synthetic dict fixtures without a KLEE installation.
"""
