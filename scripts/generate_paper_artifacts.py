"""
Report & Table Generation (The Paper Builder) for Phase 9.
Automates formatting results into LaTeX tables and generating high-DPI PDF figures.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def generate_pdf_figures():
    """Reads raw JSON snapshots and generates high-DPI matplotlib/seaborn vector PDFs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if not RAW_JSON.exists():
        logger.error(f"Raw JSON not found at {RAW_JSON}")
        return
        
    with open(RAW_JSON, "r") as f:
        raw_results = json.load(f)
        
    if not raw_results:
        return
        
    records = []
    for run in raw_results:
        heuristic = run.get("heuristic")
        for snapshot in run.get("coverage_over_time", []):
            records.append({
                "heuristic": heuristic,
                "time_s": snapshot.get("elapsed_seconds"),
                "branch_coverage": snapshot.get("branch_coverage")
            })
            
    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No snapshot data available for plotting.")
        return
        
    df['time_bin'] = df['time_s'].round()
    agg_df = df.groupby(['heuristic', 'time_bin'])['branch_coverage'].mean().reset_index()
    
    # Configure Seaborn style for academic papers
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(6, 4))
    
    ax = sns.lineplot(
        data=agg_df, 
        x="time_bin", 
        y="branch_coverage", 
        hue="heuristic",
        marker="o",
        linewidth=2
    )
    
    ax.set_title("Average Branch Coverage Over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Elapsed Time (s)", fontsize=10)
    ax.set_ylabel("Branch Coverage", fontsize=10)
    
    plt.tight_layout()
    
    # Save as PDF for papers, PNG for README
    out_path_pdf = FIGURES_DIR / "coverage_plot.pdf"
    out_path_png = FIGURES_DIR / "coverage_plot.png"
    plt.savefig(out_path_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_path_png, format="png", dpi=300, bbox_inches="tight")
    logger.info(f"Generated PDF/PNG figures in {FIGURES_DIR}")


def main():
    logger.info("Generating Paper Artifacts...")
    generate_latex_tables()
    generate_pdf_figures()
    logger.info("Finished.")


if __name__ == "__main__":
    main()
