"""
LLM Semantic Analyzer Module.
Coordinates with an OpenAI-compatible API to predict which branches to explore.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

from backend.core.schemas import BranchPrediction
from llm.cache import LLMCache, compute_cache_key
from llm.exceptions import LLMConnectionError, LLMResponseParseError
from llm.prompts import BRANCH_PREDICTION_SYSTEM_PROMPT, RETRY_PROMPT, build_branch_prediction_prompt

logger = logging.getLogger(__name__)


class LLMSemanticAnalyzer:
    """
    Interfaces with an LLM (OpenAI API or local equivalent via base_url)
    to perform semantic analysis on source code branches.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: str = "results/llm_cache",
        mock_mode: bool = False,
    ):
        self.model_name = model_name
        self.mock_mode = mock_mode
        self.cache = LLMCache(cache_dir=cache_dir)
        self._prompt_version = "v1"

        if not self.mock_mode:
            if OpenAI is None:
                raise ImportError("openai python package is required unless mock_mode is True.")
            # Auto-load from env if not provided
            api_key = api_key or os.environ.get("OPENAI_API_KEY", "mock-key")
            
            # Base URL allows swapping to local models like Ollama (e.g. http://localhost:11434/v1)
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
                
            self.client = OpenAI(**client_kwargs)

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        """Helper to invoke the OpenAI API."""
        if self.mock_mode:
            return self._mock_api_response()
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                temperature=0.0,    # Deterministic output for testing logic
                response_format={"type": "json_object"} if "gpt" in self.model_name else None,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMConnectionError("Received empty response from API.")
            return content
        except Exception as e:
            raise LLMConnectionError(f"Failed to communicate with LLM API: {e}") from e

    def _mock_api_response(self) -> str:
        """Returns a stable mock JSON response for testing without API keys."""
        return json.dumps({
            "branch": 42,
            "confidence": 0.85,
            "reason": "Mocked analysis indicates a potential null dereference."
        })

    def _parse_response(self, content: str) -> BranchPrediction:
        """Parses the raw JSON string into a BranchPrediction."""
        try:
            data = json.loads(content)
            return BranchPrediction(**data)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(f"LLM output was not valid JSON: {e}") from e
        except ValueError as e:
            # Pydantic validation error
            raise LLMResponseParseError(f"LLM JSON output did not match schema: {e}") from e

    def predict_branch(self, source_code: str, candidate_branches: list[int]) -> BranchPrediction:
        """
        Analyze the source code and return a prioritized branch prediction.
        Utilizes caching and handles exactly one retry for malformed output.
        """
        if not candidate_branches:
            raise ValueError("Candidate branches list cannot be empty.")

        # Check Cache
        cache_key = compute_cache_key(source_code, candidate_branches, self.model_name, self._prompt_version)
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"LLM cache hit for key: {cache_key}")
            return cached

        logger.debug(f"LLM cache miss for key: {cache_key}. Querying API...")
        
        # Build prompt
        messages = [
            {"role": "system", "content": BRANCH_PREDICTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_branch_prediction_prompt(source_code, candidate_branches)}
        ]

        # First Attempt
        raw_response = self._call_api(messages)
        try:
            prediction = self._parse_response(raw_response)
            self.cache.set(cache_key, prediction)
            return prediction
        except LLMResponseParseError as e:
            logger.warning(f"First LLM attempt failed parsing ({e}). Retrying...")

        # Second Attempt (Retry)
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": RETRY_PROMPT})
        
        raw_response_retry = self._call_api(messages)
        prediction = self._parse_response(raw_response_retry)
        self.cache.set(cache_key, prediction)
        return prediction

    def predict_batch(self, items: list[tuple[str, list[int]]]) -> list[BranchPrediction]:
        """
        Analyze a batch of source codes and their candidate branches.
        """
        # A true batch endpoint could use asyncio or thread pools.
        # For Phase 5, sequential prediction is sufficient to establish the schema boundary.
        return [self.predict_branch(src, branches) for src, branches in items]
