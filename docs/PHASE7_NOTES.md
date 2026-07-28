# Phase 7 Notes — End-to-End Evaluation Pipeline

## What this phase produced

### Metrics and Evaluation Framework (`evaluation/`)
1. **`runner.py`**: Contains `run_benchmark_suite()`, which orchestrates KLEE executions over a grid of parameters (`Programs` x `Heuristics` x `Repetitions`). The output is a list of strict `RunResult` schemas (Phase 2), providing type-safety downstream.
2. **`metrics.py`**: Processes the `RunResult` lists to generate Data Science deliverables.
   - `compute_metrics_table()`: Uses `pandas` to group by Program/Heuristic and computes the mean/std deviation of branch coverage, execution time, and bug yields.
   - `significance_tests()`: Uses `scipy.stats.mannwhitneyu` to test if the `AI_GUIDED` heuristic yields a statistically significant improvement over baseline heuristics (default: DFS).

### CLI Automation (`scripts/`)
1. **`run_experiments.py`**: A CLI entry point to kick off the pipeline. 
   - Generates `results/tables/raw_results.json` containing exhaustive trace metrics.
   - Generates `results/tables/metrics_table.csv` containing flattened statistical aggregates.
   - Generates `results/tables/significance_dfs_vs_ai.csv` summarizing the Mann-Whitney U test p-values.

## Architecture Context

The evaluation pipeline is the scientific backbone of the research project. It takes the subjective engineering we've done in Phases 1-6 and converts it into objective, peer-reviewable mathematics. 

By strictly adhering to the `RunResult` Pydantic schema from end-to-end, we ensure that if we swap out the mock KLEE runner for the live C++ binary, the Pandas aggregations and SciPy statistical tests do not require a single line of modification.

## Verification (Definition of Done)
1. Running `scripts/run_experiments.py` flawlessly generates CSV tables detailing AI vs Baseline performance.
2. Pandas correctly flattens multi-level indexes into readable column headers (`branch_coverage_mean`, `branch_coverage_std`).
3. SciPy Mann-Whitney U test correctly identifies statistical significance (`p < 0.05`) based on the randomized, biased mock results.
