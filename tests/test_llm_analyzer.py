"""
Tests for Phase 5: LLM Semantic Analyzer Module.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from backend.core.schemas import BranchPrediction
from llm.analyzer import LLMSemanticAnalyzer
from llm.cache import compute_cache_key
from llm.exceptions import LLMResponseParseError


@pytest.fixture
def mock_analyzer(tmp_path):
    # Use mock_mode=True to avoid needing the openai package or real keys
    return LLMSemanticAnalyzer(
        model_name="test-model",
        cache_dir=str(tmp_path / "llm_cache"),
        mock_mode=True
    )


class TestLLMSemanticAnalyzer:
    def test_predict_branch_mock_mode(self, mock_analyzer):
        source = "int main() { if (x > 0) return 1; return 0; }"
        branches = [12, 13]
        
        prediction = mock_analyzer.predict_branch(source, branches)
        assert isinstance(prediction, BranchPrediction)
        assert prediction.branch == 42
        assert prediction.confidence == 0.85

    def test_cache_hit_avoids_api_call(self, mock_analyzer):
        source = "int foo() { return 0; }"
        branches = [1, 2]
        
        # Call once to populate cache
        first_pred = mock_analyzer.predict_branch(source, branches)
        
        # Now monkeypatch the mock response to ensure it isn't called again
        with patch.object(mock_analyzer, '_mock_api_response', side_effect=Exception("Should not be called!")):
            second_pred = mock_analyzer.predict_branch(source, branches)
            
        assert first_pred.branch == second_pred.branch
        assert first_pred.confidence == second_pred.confidence

    def test_empty_candidate_branches_raises(self, mock_analyzer):
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_analyzer.predict_branch("int x = 0;", [])

    def test_retry_logic_on_malformed_json(self, mock_analyzer):
        source = "int x = 0;"
        branches = [1]
        
        # Simulate LLM returning conversational text first, then valid JSON
        bad_response = "Here is the JSON:\n```json\n{'branch': 1, 'confidence': 0.9, 'reason': 'test'}\n```"
        good_response = json.dumps({"branch": 1, "confidence": 0.9, "reason": "test"})
        
        with patch.object(mock_analyzer, '_call_api', side_effect=[bad_response, good_response]) as mock_call:
            prediction = mock_analyzer.predict_branch(source, branches)
            
        assert mock_call.call_count == 2
        assert prediction.branch == 1
        assert prediction.confidence == 0.9

    def test_raises_parse_error_if_retry_fails(self, mock_analyzer):
        source = "int x = 0;"
        branches = [1]
        
        # LLM stubborn, returns bad text both times
        bad_response = "I cannot determine the branch."
        
        with patch.object(mock_analyzer, '_call_api', return_value=bad_response):
            with pytest.raises(LLMResponseParseError, match="LLM output was not valid JSON"):
                mock_analyzer.predict_branch(source, branches)

    def test_raises_parse_error_on_schema_violation(self, mock_analyzer):
        source = "int x = 0;"
        branches = [1]
        
        # Confidence out of bounds
        bad_schema_response = json.dumps({"branch": 1, "confidence": 1.5, "reason": "test"})
        
        with patch.object(mock_analyzer, '_call_api', return_value=bad_schema_response):
            with pytest.raises(LLMResponseParseError, match="did not match schema"):
                mock_analyzer.predict_branch(source, branches)

    def test_predict_batch(self, mock_analyzer):
        items = [
            ("int a = 1;", [1, 2]),
            ("int b = 2;", [3, 4])
        ]
        predictions = mock_analyzer.predict_batch(items)
        assert len(predictions) == 2
        assert all(isinstance(p, BranchPrediction) for p in predictions)


def test_compute_cache_key_is_deterministic():
    key1 = compute_cache_key("source", [1, 2], "gpt-4")
    key2 = compute_cache_key("source", [2, 1], "gpt-4")  # sorted branches
    assert key1 == key2
