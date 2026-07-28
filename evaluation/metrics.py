"""
Metrics and Statistical Significance Testing for Phase 7.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
from scipy import stats

from backend.core.schemas import RunResult, SearchHeuristic

logger = logging.getLogger(__name__)


def compute_metrics_table(results: List[RunResult]) -> pd.DataFrame:
    """
    Computes aggregate metrics (mean and standard deviation) for each program and heuristic.
    
    Args:
        results: List of execution results from run_benchmark_suite.
        
    Returns:
        A pandas DataFrame indexed by (Program, Heuristic) with columns for coverage and bugs.
    """
    if not results:
        return pd.DataFrame()

    # Convert Pydantic models to dictionaries
    records = [res.model_dump() for res in results]
    df = pd.DataFrame(records)

    # Convert Enum to string for cleaner grouping
    df['heuristic'] = df['heuristic'].apply(lambda x: x.value if isinstance(x, SearchHeuristic) else str(x))

    # Group by Program and Heuristic, calculate mean and standard deviation
    groupby_cols = ['program_name', 'heuristic']
    metric_cols = ['branch_coverage', 'execution_time_seconds', 'unique_bugs', 'unique_paths']
    
    agg_df = df.groupby(groupby_cols)[metric_cols].agg(['mean', 'std']).reset_index()
    
    # Flatten multi-level columns
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    
    return agg_df


def significance_tests(results: List[RunResult], baseline: SearchHeuristic, target: SearchHeuristic = SearchHeuristic.AI_GUIDED) -> pd.DataFrame:
    """
    Performs the Mann-Whitney U test comparing the target heuristic against a baseline.
    Answers Research Question 1 and 3 (Does AI statistically improve coverage?).
    
    Args:
        results: List of execution results.
        baseline: The heuristic to compare against (e.g., DFS or BFS).
        target: The heuristic we hypothesize is better (default AI_GUIDED).
        
    Returns:
        A pandas DataFrame with the p-values per program.
    """
    if not results:
        return pd.DataFrame()

    records = [res.model_dump() for res in results]
    df = pd.DataFrame(records)
    
    programs = df['program_name'].unique()
    test_results = []
    
    for prog in programs:
        prog_df = df[df['program_name'] == prog]
        
        target_data = prog_df[prog_df['heuristic'] == target]['branch_coverage'].values
        baseline_data = prog_df[prog_df['heuristic'] == baseline]['branch_coverage'].values
        
        # We need at least two samples to compute standard deviations and run meaningul tests
        if len(target_data) < 2 or len(baseline_data) < 2:
            logger.warning(f"Not enough data for {prog} to run Mann-Whitney U test.")
            p_value = float('nan')
            statistic = float('nan')
        else:
            try:
                # alternative='greater' tests if target > baseline
                stat, p_value = stats.mannwhitneyu(target_data, baseline_data, alternative='greater')
                statistic = float(stat)
            except ValueError as e:
                logger.error(f"Mann-Whitney U failed for {prog}: {e}")
                p_value = float('nan')
                statistic = float('nan')
                
        test_results.append({
            'program_name': prog,
            'baseline': baseline.value,
            'target': target.value,
            'u_statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05 if not pd.isna(p_value) else False
        })
        
    return pd.DataFrame(test_results)
