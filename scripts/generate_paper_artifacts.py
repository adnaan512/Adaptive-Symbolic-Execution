"""
Report & Table Generation (The Paper Builder) for Phase 9.
Automates formatting results into LaTeX tables and generating high-DPI PDF figures.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path("results/tables")
METRICS_CSV = RESULTS_DIR / "metrics_table.csv"
SIG_CSV = RESULTS_DIR / "significance_dfs_vs_ai.csv"
RAW_JSON = RESULTS_DIR / "raw_results.json"

PAPER_DIR = Path("paper")
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"

def generate_latex_tables():
    """Reads CSV metrics and converts them into Booktabs-formatted LaTeX tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    if not METRICS_CSV.exists():
        logger.error(f"Metrics CSV not found at {METRICS_CSV}")
        return
        
    df = pd.read_csv(METRICS_CSV)
    
    # We want a concise summary table: Program, Heuristic, Branch Coverage (mean +/- std), Execution Time (mean +/- std)
    # The columns from Phase 7 are flattened: branch_coverage_mean, branch_coverage_std, etc.
    
    # Format the mean +/- std as strings
    df['Branch Coverage'] = df.apply(lambda row: f"{row['branch_coverage_mean']:.2f} \pm {row['branch_coverage_std']:.2f}", axis=1)
    df['Exec Time (s)'] = df.apply(lambda row: f"{row['execution_time_seconds_mean']:.1f} \pm {row['execution_time_seconds_std']:.1f}", axis=1)
    df['Unique Bugs'] = df['unique_bugs_mean'].apply(lambda x: f"{x:.1f}")
    
    # Select columns for LaTeX
    latex_df = df[['program_name', 'heuristic', 'Branch Coverage', 'Exec Time (s)', 'Unique Bugs']]
    latex_df = latex_df.rename(columns={'program_name': 'Program', 'heuristic': 'Heuristic'})
    
    # Generate LaTeX using pandas
    latex_code = latex_df.to_latex(
        index=False,
        escape=False, # We want \pm to render in LaTeX
        column_format="llccc",
        caption="Comparison of AI-Guided Search vs Baseline Heuristics",
        label="tab:coverage_results",
        position="htbp"
    )
    
    out_path = TABLES_DIR / "coverage_results.tex"
    with open(out_path, "w") as f:
        f.write(latex_code)
        
    logger.info(f"Generated LaTeX table: {out_path}")

    # Generate Significance Table
    if SIG_CSV.exists():
        sig_df = pd.read_csv(SIG_CSV)
        # Format p_value to scientific notation if very small
        sig_df['p_value'] = sig_df['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
        
        latex_sig_code = sig_df.to_latex(
            index=False,
            escape=False,
            caption="Mann-Whitney U Test Results (Significance: $p < 0.05$)",
            label="tab:significance_tests",
            position="htbp"
        )
        
        sig_out_path = TABLES_DIR / "significance_results.tex"
        with open(sig_out_path, "w") as f:
            f.write(latex_sig_code)
            
        logger.info(f"Generated LaTeX table: {sig_out_path}")


def main():
    logger.info("Generating Paper Artifacts...")
    generate_latex_tables()
    logger.info("Finished.")


if __name__ == "__main__":
    main()
