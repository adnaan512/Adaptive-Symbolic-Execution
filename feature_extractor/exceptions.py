"""
Exceptions for the feature_extractor module.
"""

from __future__ import annotations


class FeatureExtractionError(Exception):
    """Base exception for feature extraction failures."""
    pass


class ValidationError(FeatureExtractionError):
    """Raised when the raw state data cannot be validated or cast to the correct types."""
    pass
