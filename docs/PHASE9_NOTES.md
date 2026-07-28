# Phase 9 Notes — Report & Table Generation (The Paper Builder)

## What this phase produced

### Paper Builder (`scripts/generate_paper_artifacts.py`)
This script bridges the gap between raw experimental data and the final academic paper. It reads from `results/tables/` and generates assets directly into the `paper/` directory.

1. **LaTeX Tables (`paper/tables/`)**:
   - Uses `pandas.DataFrame.to_latex()` to automatically generate tables formatted with `booktabs` for double-column IEEE/ACM styling.
   - Outputs `coverage_results.tex` containing Branch Coverage and Execution Time (Mean $\pm$ Std).
   - Outputs `significance_results.tex` summarizing the Mann-Whitney U test p-values.

2. **High-DPI PDF Figures (`paper/figures/`)**:
   - Uses `matplotlib` and `seaborn` to parse the temporal snapshot data and generate vector-based PDF graphics. 
   - These graphics scale infinitely without pixelation, which is a hard requirement for top-tier software engineering conferences (ICSE, FSE, ASE, ISSTA).
   - Outputs `coverage_plot.pdf` illustrating how the AI heuristic accelerates coverage over time.

## Architecture Context

Why build this?
In empirical software engineering research, maintaining a clean pipeline from *code execution* directly to *paper compilation* is critical. If we tweak the LLM prompt or modify the Gym reward function, we can re-run `run_experiments.py` and `generate_paper_artifacts.py` to regenerate the entire paper's datasets and graphics in seconds, completely eliminating manual copy-pasting errors.

## Running the Builder
1. Install visualization dependencies:
   ```bash
   pip install pandas matplotlib seaborn jinja2
   ```
2. Run the artifact generator:
   ```bash
   python scripts/generate_paper_artifacts.py
   ```
3. The resulting `.tex` and `.pdf` files can now be included in your LaTeX manuscript!
