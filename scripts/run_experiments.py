"""
Command Line Interface to execute the End-to-End Evaluation Pipeline (Phase 7).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path so we can run this directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.schemas import SearchHeuristic
from evaluation.metrics import compute_metrics_table, significance_tests
from evaluation.runner import run_benchmark_suite

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Symbolic Execution Evaluation Benchmark")
    parser.add_argument(
        "--programs", 
        nargs="+", 
        required=True, 
        help="List of program paths or names to evaluate (e.g., bitcode/coreutils/ls.bc)"
    )
    parser.add_argument(
        "--heuristics", 
        nargs="+", 
        default=["dfs", "bfs", "nurs:covnew", "ai-guided"],
        help="List of heuristics to compare (default: dfs bfs nurs:covnew ai-guided)"
    )
    parser.add_argument(
        "--repetitions", 
        type=int, 
        default=5, 
        help="Number of times to run each configuration (default: 5)"
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        default="results/tables", 
        help="Directory to save the resulting tables and JSON (default: results/tables)"
    )
    
    args = parser.parse_args()
    
    # Parse heuristics
    try:
        heuristics_enum = [SearchHeuristic(h) for h in args.heuristics]
    except ValueError as e:
        logger.error(f"Invalid heuristic provided. Valid options are: {[h.value for h in SearchHeuristic]}")
        sys.exit(1)
        
    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Benchmark Suite...")
    results = run_benchmark_suite(
        program_paths=args.programs,
        heuristics=heuristics_enum,
        num_repetitions=args.repetitions,
        mock_execution=True  # Defaulting to mock for Phase 7 implementation proof
    )
    
    # Save raw results as JSON
    raw_results_file = out_path / "raw_results.json"
    with open(raw_results_file, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    logger.info(f"Saved raw JSON results to {raw_results_file}")
    
    # Generate and save metrics table
    logger.info("Computing aggregate metrics...")
    metrics_df = compute_metrics_table(results)
    metrics_file = out_path / "metrics_table.csv"
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Saved aggregate metrics to {metrics_file}")
    
    # Generate and save statistical significance tests
    logger.info("Running Mann-Whitney U tests vs DFS baseline...")
    # Using DFS as the default baseline for statistical comparison
    if SearchHeuristic.DFS in heuristics_enum and SearchHeuristic.AI_GUIDED in heuristics_enum:
        sig_df = significance_tests(results, baseline=SearchHeuristic.DFS, target=SearchHeuristic.AI_GUIDED)
        sig_file = out_path / "significance_dfs_vs_ai.csv"
        sig_df.to_csv(sig_file, index=False)
        logger.info(f"Saved significance tests to {sig_file}")
    else:
        logger.warning("Skipping statistical test: Requires 'dfs' and 'ai-guided' in heuristics list.")
        
    logger.info("Evaluation Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
