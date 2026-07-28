"""
Exceptions for the LLM Module.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for all LLM module errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when the LLM API cannot be reached or times out."""
    pass


class LLMResponseParseError(LLMError):
    """Raised when the LLM output cannot be parsed into a valid BranchPrediction."""
    pass
