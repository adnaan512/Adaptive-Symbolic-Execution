"""
Disk-based caching layer for LLM responses.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from backend.core.schemas import BranchPrediction

logger = logging.getLogger(__name__)


def compute_cache_key(
    source_code: str, 
    candidate_branches: list[int], 
    model_name: str, 
    prompt_version: str = "v1"
) -> str:
    """
    Compute a deterministic hash for an LLM prompt request to serve as a cache key.
    """
    m = hashlib.sha256()
    m.update(source_code.encode("utf-8"))
    m.update(str(sorted(candidate_branches)).encode("utf-8"))
    m.update(model_name.encode("utf-8"))
    m.update(prompt_version.encode("utf-8"))
    return m.hexdigest()


class LLMCache:
    """
    Handles saving and retrieving `BranchPrediction` responses to/from disk.
    This prevents burning API tokens and compute on identical source-code paths
    encountered multiple times during or across symbolic execution runs.
    """
    
    def __init__(self, cache_dir: str | Path = "results/llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, cache_key: str) -> Optional[BranchPrediction]:
        """
        Retrieve a BranchPrediction from the cache if it exists.
        Returns None if not found or if parsing fails (cache miss).
        """
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return BranchPrediction(**data)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupted cache file {cache_path}: {e}")
            return None

    def set(self, cache_key: str, prediction: BranchPrediction) -> None:
        """
        Save a BranchPrediction to the cache.
        """
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(prediction.model_dump_json(indent=2))
        except OSError as e:
            logger.error(f"Failed to write to LLM cache {cache_path}: {e}")
