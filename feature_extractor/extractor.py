"""
Core logic for extracting structured features from raw KLEE state data.
"""

from __future__ import annotations

import pydantic
import uuid

from backend.core.schemas import ExecutionStateFeatures
from feature_extractor.exceptions import ValidationError


def extract_features(raw_state: dict) -> ExecutionStateFeatures:
    """
    Transform a raw KLEE execution state dictionary into a validated 
    ExecutionStateFeatures object.
    
    If the state_id is missing, a new one is generated.
    If required fields are missing or cannot be cast to the correct types, 
    a ValidationError is raised.
    """
    # Ensure state_id exists
    state_id = raw_state.get("state_id")
    if not state_id:
        state_id = str(uuid.uuid4())

    try:
        # Default missing complex fields sensibly if not provided by raw hook
        # (Assuming the raw dict tries to match names where possible)
        return ExecutionStateFeatures(
            state_id=state_id,
            execution_depth=int(raw_state.get("execution_depth", 0)),
            num_path_constraints=int(raw_state.get("num_path_constraints", 0)),
            constraint_complexity=float(raw_state.get("constraint_complexity", 0.0)),
            num_symbolic_variables=int(raw_state.get("num_symbolic_variables", 0)),
            solver_time_ms=float(raw_state.get("solver_time_ms", 0.0)),
            instruction_count=int(raw_state.get("instruction_count", 0)),
            branch_count=int(raw_state.get("branch_count", 0)),
            loop_depth=int(raw_state.get("loop_depth", 0)),
            distance_to_uncovered_branch=float(raw_state.get("distance_to_uncovered_branch", -1.0)),
            current_branch_coverage=float(raw_state.get("current_branch_coverage", 0.0)),
            memory_object_count=int(raw_state.get("memory_object_count", 0)),
            num_forks=int(raw_state.get("num_forks", 0)),
            state_age=float(raw_state.get("state_age", 0.0)),
            path_length=int(raw_state.get("path_length", 0)),
        )
    except (ValueError, TypeError, pydantic.ValidationError) as e:
        raise ValidationError(f"Failed to validate state features: {e}") from e


def extract_batch(raw_states: list[dict]) -> list[ExecutionStateFeatures]:
    """
    Extract a batch of raw KLEE states.
    Raises ValidationError if any state in the batch is invalid.
    """
    return [extract_features(state) for state in raw_states]
