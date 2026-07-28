"""
Tests for Phase 7: End-to-End Evaluation Pipeline.
"""

import pandas as pd
import pytest

from backend.core.schemas import SearchHeuristic
from evaluation.metrics import compute_metrics_table, significance_tests
from evaluation.runner import run_benchmark_suite


class TestEvaluationRunner:
    def test_run_benchmark_suite_mock(self):
        programs = ["prog1.bc", "prog2.bc"]
        heuristics = [SearchHeuristic.DFS, SearchHeuristic.AI_GUIDED]
        
        results = run_benchmark_suite(
            program_paths=programs, 
            heuristics=heuristics, 
            num_repetitions=2, 
            mock_execution=True
        )
        
        # 2 programs * 2 heuristics * 2 repetitions
        assert len(results) == 8
        assert results[0].program_name == "prog1"
        assert len(results[0].coverage_over_time) == 5


class TestEvaluationMetrics:
    @pytest.fixture
    def mock_results(self):
        return run_benchmark_suite(
            program_paths=["test_prog.bc"],
            heuristics=[SearchHeuristic.DFS, SearchHeuristic.AI_GUIDED],
            num_repetitions=5,
            mock_execution=True
        )

    def test_compute_metrics_table(self, mock_results):
        df = compute_metrics_table(mock_results)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "program_name" in df.columns
        assert "heuristic" in df.columns
        
        # Check that flattened columns exist (e.g. branch_coverage_mean)
        assert "branch_coverage_mean" in df.columns
        assert "branch_coverage_std" in df.columns
        assert len(df) == 2  # 1 program * 2 heuristics

    def test_significance_tests(self, mock_results):
        df = significance_tests(
            mock_results, 
            baseline=SearchHeuristic.DFS, 
            target=SearchHeuristic.AI_GUIDED
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "p_value" in df.columns
        assert "u_statistic" in df.columns
        assert "significant" in df.columns
        
        # AI heuristic in mock is designed to perform significantly better.
        # However, with N=5 it might barely hit p < 0.05. We just assert the struct is valid.
        assert not pd.isna(df.iloc[0]["p_value"])

    def test_metrics_empty_list(self):
        df1 = compute_metrics_table([])
        assert df1.empty
        
        df2 = significance_tests([], SearchHeuristic.DFS)
        assert df2.empty
