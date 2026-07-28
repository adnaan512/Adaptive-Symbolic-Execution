"""
evaluation — Phase 7.

Responsibility: run every baseline heuristic + the AI-guided policy across
the benchmark suites, collect `RunResult` objects, compute the metrics in
`configs/config.yaml: evaluation.metrics`, and run the significance test
(default: Mann-Whitney U) between AI-guided and each baseline.

Planned public interface:

    def run_benchmark_suite(program_paths, heuristics, num_repetitions) -> list[RunResult]
    def compute_metrics_table(results: list[RunResult]) -> pandas.DataFrame
    def significance_tests(results: list[RunResult], alpha: float) -> pandas.DataFrame

Outputs are written to `results/tables/` in a format `docs/` and the
paper-report generator (Phase 9) can consume directly (CSV + LaTeX booktabs).
"""

from evaluation.runner import run_benchmark_suite

__all__ = [
    "run_benchmark_suite",
]

