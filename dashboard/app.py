"""
Main Streamlit Application for Phase 8.
Displays the evaluation results of the Adaptive Symbolic Execution framework.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Configure the page layout
st.set_page_config(
    page_title="Adaptive SE Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
RESULTS_DIR = Path("results/tables")
RAW_RESULTS_FILE = RESULTS_DIR / "raw_results.json"
METRICS_FILE = RESULTS_DIR / "metrics_table.csv"
SIGNIFICANCE_FILE = RESULTS_DIR / "significance_dfs_vs_ai.csv"

@st.cache_data
def load_raw_results():
    if not RAW_RESULTS_FILE.exists():
        return []
    with open(RAW_RESULTS_FILE, "r") as f:
        return json.load(f)

@st.cache_data
def load_metrics_csv():
    if not METRICS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_FILE)

@st.cache_data
def load_significance_csv():
    if not SIGNIFICANCE_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(SIGNIFICANCE_FILE)


def main():
    st.sidebar.title("🧠 Adaptive SE Pipeline")
    st.sidebar.markdown(
        "Welcome to the **Adaptive LLM-Guided Symbolic Execution** Dashboard."
    )
    
    st.title("Evaluation Metrics Overview")
    
    # Load Data
    raw_data = load_raw_results()
    metrics_df = load_metrics_csv()
    sig_df = load_significance_csv()
    
    if metrics_df.empty or not raw_data:
        st.warning("⚠️ No evaluation data found. Please run `python scripts/run_experiments.py` first.")
        st.stop()
        
    st.markdown("### Aggregated Benchmarks")
    st.dataframe(metrics_df, use_container_width=True)
    
    if not sig_df.empty:
        st.markdown("### Statistical Significance (Mann-Whitney U)")
        st.dataframe(sig_df, use_container_width=True)
        
    st.markdown("---")
    st.markdown("*(Detailed charts will be added in the next commit via modular components.)*")

if __name__ == "__main__":
    main()
