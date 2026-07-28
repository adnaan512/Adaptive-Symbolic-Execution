"""
visualization — Phase 8.

Responsibility: turn `RunResult` / `CoverageSnapshot` data in `results/` into
(a) an interactive Plotly-based dashboard served by `backend/api`, rendered
by the `frontend/` React app, and (b) static publication-quality figures
(PDF/SVG) for the paper, generated headlessly via `plotly` + `kaleido`.

Planned components: coverage-over-time chart, execution-state tree
(NetworkX layout), coverage heatmap, RL learning curve, baseline comparison
bar/box charts, solver-call statistics.
"""
