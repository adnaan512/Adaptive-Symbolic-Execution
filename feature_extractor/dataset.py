"""
Dataset persistence logic for ML and RL consumption.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from backend.core.schemas import ExecutionStateFeatures
from feature_extractor.exceptions import FeatureExtractionError


def save_dataset_csv(states: Sequence[ExecutionStateFeatures], path: str | Path) -> None:
    """Save a list of ExecutionStateFeatures to a CSV file."""
    if not states:
        return
        
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Grab field names in schema definition order
    field_names = list(ExecutionStateFeatures.model_fields.keys())
    
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for state in states:
                # model_dump() returns a dictionary of the fields
                writer.writerow(state.model_dump())
    except OSError as e:
        raise FeatureExtractionError(f"Failed to write CSV dataset to {path}: {e}") from e


def load_dataset_csv(path: str | Path) -> list[ExecutionStateFeatures]:
    """Load a list of ExecutionStateFeatures from a CSV file."""
    path = Path(path)
    if not path.exists():
        raise FeatureExtractionError(f"CSV dataset not found: {path}")
        
    states = []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                states.append(ExecutionStateFeatures(**row))
    except (OSError, ValueError, TypeError) as e:
        raise FeatureExtractionError(f"Failed to read CSV dataset from {path}: {e}") from e
    
    return states


def save_dataset_jsonl(states: Sequence[ExecutionStateFeatures], path: str | Path) -> None:
    """Save a list of ExecutionStateFeatures to a JSON Lines file."""
    if not states:
        return
        
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            for state in states:
                f.write(state.model_dump_json() + "\n")
    except OSError as e:
        raise FeatureExtractionError(f"Failed to write JSONL dataset to {path}: {e}") from e


def load_dataset_jsonl(path: str | Path) -> list[ExecutionStateFeatures]:
    """Load a list of ExecutionStateFeatures from a JSON Lines file."""
    path = Path(path)
    if not path.exists():
        raise FeatureExtractionError(f"JSONL dataset not found: {path}")
        
    states = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    states.append(ExecutionStateFeatures.model_validate_json(line))
    except (OSError, ValueError, TypeError) as e:
        raise FeatureExtractionError(f"Failed to read JSONL dataset from {path}: {e}") from e
    
    return states
