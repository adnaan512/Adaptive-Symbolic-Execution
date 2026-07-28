"""Phase 1 smoke tests for the shared Pydantic schemas."""

import pytest
from pydantic import ValidationError

from backend.core.config import load_config
from backend.core.schemas import BranchPrediction, ExecutionStateFeatures, SearchHeuristic


def _sample_features(**overrides) -> ExecutionStateFeatures:
    base = dict(
        state_id="s-0001",
        execution_depth=3,
        num_path_constraints=5,
        constraint_complexity=1.2,
        num_symbolic_variables=2,
        solver_time_ms=12.5,
        instruction_count=340,
        branch_count=4,
        loop_depth=1,
        distance_to_uncovered_branch=2.0,
        current_branch_coverage=0.42,
        memory_object_count=6,
        num_forks=1,
        state_age=0.8,
        path_length=10,
    )
    base.update(overrides)
    return ExecutionStateFeatures(**base)


def test_execution_state_features_round_trip():
    feats = _sample_features()
    vec = feats.to_vector()
    assert len(vec) == 14
    assert all(isinstance(v, float) for v in vec)


def test_feature_names_match_config_order():
    cfg = load_config()
    feats = _sample_features()
    vec = feats.to_vector()
    assert len(vec) == len(cfg.features.names)


def test_coverage_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        _sample_features(current_branch_coverage=1.5)


def test_branch_prediction_confidence_bounds():
    pred = BranchPrediction(branch=7, confidence=0.92, reason="nested conditional")
    assert 0.0 <= pred.confidence <= 1.0

    with pytest.raises(ValidationError):
        BranchPrediction(branch=7, confidence=1.5, reason="bad")


def test_search_heuristic_enum_has_ai_guided_arm():
    assert SearchHeuristic.AI_GUIDED == "ai-guided"
