"""
Evaluation Runner Framework (Phase 7).
Orchestrates benchmark executions across different heuristics.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import List, Optional

from backend.core.schemas import CoverageSnapshot, RunResult, SearchHeuristic

logger = logging.getLogger(__name__)


def _mock_klee_execution(
    program_path: str, heuristic: SearchHeuristic, seed: int
) -> RunResult:
    """
    Simulates KLEE execution for a specific heuristic to enable offline testing
    of the evaluation pipeline.
    
    In Phase 7, the AI-guided heuristic typically outperforms the baselines.
    We inject a synthetic bias here so the statistical significance tests 
    downstream have a clear signal to verify.
    """
    # Deterministic simulation based on seed and heuristic
    random.seed(seed + hash(heuristic))
    
    # Base performance metrics
    base_branch_cov = random.uniform(0.4, 0.7)
    base_time = random.uniform(10.0, 60.0)
    base_bugs = random.randint(0, 2)
    
    # Apply heuristic-specific biases
    if heuristic == SearchHeuristic.AI_GUIDED:
        # AI heuristic finds more coverage, faster, and more bugs
        branch_cov = min(1.0, base_branch_cov + random.uniform(0.15, 0.3))
        exec_time = base_time * random.uniform(0.5, 0.8)
        bugs = base_bugs + random.randint(1, 3)
    else:
        branch_cov = base_branch_cov
        exec_time = base_time
        bugs = base_bugs

    # Simulate coverage over time
    snapshots = []
    current_cov = 0.0
    for step in range(5):
        elapsed = (step + 1) * (exec_time / 5.0)
        current_cov += (branch_cov / 5.0) * random.uniform(0.8, 1.2)
        current_cov = min(branch_cov, current_cov)
        snapshots.append(
            CoverageSnapshot(
                elapsed_seconds=elapsed,
                branch_coverage=current_cov,
                instruction_coverage=min(1.0, current_cov * 1.1),
                num_states=int((step + 1) * 100 * random.uniform(0.9, 1.1))
            )
        )

    return RunResult(
        program_name=Path(program_path).stem,
        heuristic=heuristic,
        seed=seed,
        branch_coverage=branch_cov,
        instruction_coverage=min(1.0, branch_cov * 1.1),
        unique_paths=int(branch_cov * 5000),
        unique_bugs=bugs,
        solver_calls=int(exec_time * 50),
        execution_time_seconds=exec_time,
        memory_usage_mb=random.uniform(50.0, 200.0),
        state_explosion_count=random.randint(0, 5),
        avg_solver_time_ms=random.uniform(1.0, 5.0),
        coverage_over_time=snapshots,
        notes="Mocked execution"
    )


def run_benchmark_suite(
    program_paths: List[str],
    heuristics: List[SearchHeuristic],
    num_repetitions: int = 5,
    mock_execution: bool = True
) -> List[RunResult]:
    """
    Executes a grid of (programs x heuristics x repetitions) and collects results.
    
    Args:
        program_paths: List of file paths to the target binaries/bitcode.
        heuristics: List of KLEE SearchHeuristics to evaluate.
        num_repetitions: Number of times to run each configuration with different seeds.
        mock_execution: If True, uses the simulated KLEE runner instead of actual subprocess calls.
        
    Returns:
        List of RunResult objects detailing the performance of each run.
    """
    results: List[RunResult] = []
    
    total_runs = len(program_paths) * len(heuristics) * num_repetitions
    current_run = 0

    for program_path in program_paths:
        for heuristic in heuristics:
            for seed in range(num_repetitions):
                current_run += 1
                logger.info(
                    f"Run {current_run}/{total_runs} | "
                    f"Prog: {Path(program_path).stem} | "
                    f"Heuristic: {heuristic.value} | "
                    f"Seed: {seed}"
                )
                
                start_time = time.time()
                
                if mock_execution:
                    result = _mock_klee_execution(program_path, heuristic, seed)
                else:
                    # In a real environment, this would call Phase 2's KleeRunner
                    # and parse the output JSON/CSV into a RunResult.
                    raise NotImplementedError("Real KLEE execution not yet integrated in Phase 7 test harness.")
                    
                elapsed = time.time() - start_time
                logger.debug(f"Execution finished in {elapsed:.2f}s locally.")
                
                results.append(result)

    return results
