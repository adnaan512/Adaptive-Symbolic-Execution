"""
Tests for Phase 3: feature_extractor module.
Verifies parsing robustly handles missing/bad types, and round-trips
CSV/JSONL dataset generation for ML/RL use.
"""

import pytest
import uuid
import json
from pathlib import Path

from backend.core.schemas import ExecutionStateFeatures
from feature_extractor import (
    extract_features,
    extract_batch,
    save_dataset_csv,
    load_dataset_csv,
    save_dataset_jsonl,
    load_dataset_jsonl,
    ValidationError,
    FeatureExtractionError,
)


@pytest.fixture
def valid_raw_state():
    return {
        "state_id": "test-state-1",
        "execution_depth": 5,
        "num_path_constraints": 10,
        "constraint_complexity": 2.5,
        "num_symbolic_variables": 3,
        "solver_time_ms": 150.5,
        "instruction_count": 1000,
        "branch_count": 50,
        "loop_depth": 2,
        "distance_to_uncovered_branch": 15.0,
        "current_branch_coverage": 0.85,
        "memory_object_count": 25,
        "num_forks": 4,
        "state_age": 10.2,
        "path_length": 55,
    }


class TestExtractor:
    def test_extract_features_valid(self, valid_raw_state):
        features = extract_features(valid_raw_state)
        assert isinstance(features, ExecutionStateFeatures)
        assert features.state_id == "test-state-1"
        assert features.execution_depth == 5
        assert features.current_branch_coverage == 0.85

    def test_extract_features_missing_id_generates_uuid(self, valid_raw_state):
        del valid_raw_state["state_id"]
        features = extract_features(valid_raw_state)
        assert features.state_id is not None
        # Should be a valid UUID
        uuid.UUID(features.state_id)

    def test_extract_features_defaults(self):
        # Empty dict should populate with zero/defaults
        features = extract_features({})
        assert features.execution_depth == 0
        assert features.constraint_complexity == 0.0
        assert features.distance_to_uncovered_branch == -1.0

    def test_extract_features_type_casting(self, valid_raw_state):
        valid_raw_state["execution_depth"] = "42"  # String integer
        valid_raw_state["current_branch_coverage"] = "0.99"  # String float
        features = extract_features(valid_raw_state)
        assert features.execution_depth == 42
        assert features.current_branch_coverage == 0.99

    def test_extract_features_validation_error(self, valid_raw_state):
        valid_raw_state["current_branch_coverage"] = 1.5  # Coverage > 1.0 is invalid
        with pytest.raises(ValidationError):
            extract_features(valid_raw_state)
            
    def test_extract_batch(self, valid_raw_state):
        states = [valid_raw_state, valid_raw_state.copy()]
        states[1]["state_id"] = "test-state-2"
        
        features_list = extract_batch(states)
        assert len(features_list) == 2
        assert features_list[0].state_id == "test-state-1"
        assert features_list[1].state_id == "test-state-2"


class TestDatasetPersistence:
    def test_csv_roundtrip(self, valid_raw_state, tmp_path):
        features = extract_features(valid_raw_state)
        csv_path = tmp_path / "dataset.csv"
        
        save_dataset_csv([features, features], csv_path)
        assert csv_path.exists()
        
        loaded = load_dataset_csv(csv_path)
        assert len(loaded) == 2
        assert loaded[0].state_id == features.state_id
        assert loaded[0].current_branch_coverage == features.current_branch_coverage
        
    def test_jsonl_roundtrip(self, valid_raw_state, tmp_path):
        features = extract_features(valid_raw_state)
        jsonl_path = tmp_path / "dataset.jsonl"
        
        save_dataset_jsonl([features, features], jsonl_path)
        assert jsonl_path.exists()
        
        loaded = load_dataset_jsonl(jsonl_path)
        assert len(loaded) == 2
        assert loaded[1].state_id == features.state_id
        assert loaded[1].constraint_complexity == features.constraint_complexity

    def test_load_csv_missing_file(self):
        with pytest.raises(FeatureExtractionError):
            load_dataset_csv(Path("does_not_exist.csv"))

    def test_load_jsonl_missing_file(self):
        with pytest.raises(FeatureExtractionError):
            load_dataset_jsonl(Path("does_not_exist.jsonl"))

    def test_save_empty_list_does_not_create_file(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        save_dataset_csv([], csv_path)
        assert not csv_path.exists()
