# Adaptive LLM-Guided Symbolic Execution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![KLEE](https://img.shields.io/badge/KLEE-Symbolic_Execution-green)](https://klee.github.io/)

> **A cutting-edge research framework for maximizing branch coverage in Symbolic Execution using Large Language Models (LLMs), Reinforcement Learning (RL), and traditional Machine Learning.**

Traditional symbolic execution engines (like KLEE) rely on static search heuristics (DFS, BFS, Random). This project introduces an adaptive, AI-guided meta-heuristic that dynamically learns which execution paths to explore next, prioritizing those most likely to yield new coverage or discover unique crashes.

This framework is built as a complete End-to-End Pipeline, ready for publication at top-tier software engineering conferences (ICSE, FSE, ASE, ISSTA).

---

## Project Phases & Architecture

This repository is modularly structured across 9 distinct phases:

1. **GitHub Repository & Structure**: Foundational directory layout (`backend/`, `models/`, `evaluation/`).
2. **KLEE Integration Harness**: Interacts with the KLEE C++ engine, orchestrating `.bc` (LLVM bitcode) files and generating execution traces.
3. **Execution State Feature Extraction**: Extracts robust AST, Call Graph, and execution metrics into strongly-typed Pydantic schemas.
4. **Machine Learning Ranking Models**: Offline XGBoost/RandomForest models that predict the coverage utility of a given state.
5. **LLM Prompt Strategy Module**: Zero-shot and Few-shot prompting mechanisms to ask GPT/Claude to semantically predict branch viability.
6. **Reinforcement Learning Agent**: A Gymnasium-compliant PPO/DQN agent that learns to prioritize the search queue dynamically over time.
7. **End-to-End Evaluation Pipeline**: A massive grid-search testbench comparing AI heuristics vs. DFS/BFS across benchmark programs.
8. **Real-Time Dashboard (Visualization)**: A native Streamlit app that visually tracks execution state, coverage growth, and heuristic performance.
9. **Report & Table Generation (The Paper Builder)**: Automates the transition from raw JSON metrics directly into `booktabs` LaTeX tables and high-DPI Vector PDFs for academic publication.

---

## Getting Started

### 1. Installation

Clone the repository and install the data science and ML dependencies:

```bash
git clone https://github.com/adnaan512/Adaptive-Symbolic-Execution.git
cd Adaptive-Symbolic-Execution
pip install -r requirements.txt
```

*(Note: Executing actual LLVM bitcode requires a Linux environment with KLEE installed. However, the evaluation and metrics pipeline can be run locally on any OS using the built-in mock simulator).*

### 2. Running the Evaluation Pipeline

To run the End-to-End benchmark suite (Phase 7) and simulate the KLEE engine across multiple heuristics:

```bash
python scripts/run_experiments.py --programs bitcode/coreutils/ls.bc bitcode/coreutils/cat.bc --heuristics dfs bfs nurs:covnew ai-guided --repetitions 5
```
*Outputs are saved to `results/tables/raw_results.json` and `results/tables/metrics_table.csv`.*

### 3. Visualizing Results (Real-Time Dashboard)

Once you have generated evaluation data, launch the Streamlit dashboard to explore the results interactively:

```bash
streamlit run dashboard/app.py
```
*This opens a local web server (typically `http://localhost:8501`) displaying coverage timelines and heuristic bar charts.*

### 4. Generating Publication Artifacts

To compile your metrics into publication-ready assets (LaTeX tables and PDF figures):

```bash
python scripts/generate_paper_artifacts.py
```
*Outputs are generated in the `paper/` directory, ready to be imported into your LaTeX manuscript!*

---

## Evaluation & Metrics

The framework mathematically tests the following Research Questions (RQs):
- **RQ1**: Does the AI-Guided heuristic achieve higher branch coverage than DFS/BFS?
- **RQ2**: Does the AI-Guided heuristic find bugs faster (Execution Time vs. Unique Crashes)?
- **RQ3**: Are the performance gains statistically significant?

The pipeline automatically calculates the **Mann-Whitney U** test for statistical significance (p < 0.05).

### Branch Coverage Velocity

![Coverage Over Time](paper/figures/coverage_plot.png)

*Figure: The AI-guided meta-heuristic achieves faster code coverage convergence compared to traditional DFS/BFS search strategies across the evaluation benchmark.*

---

## Repository Layout

```text
├── backend/                   # KLEE Harness & Core Pydantic Schemas
├── bitcode/                   # Target LLVM .bc files for analysis
├── configs/                   # Hyperparameter and execution configurations
├── dashboard/                 # Streamlit Real-Time Visualizer
├── docs/                      # Phase-by-Phase Developer Notes
├── evaluation/                # Benchmark Orchestrator & Stats Engine
├── models/                    # ML (XGBoost) and RL (PyTorch) Architectures
├── paper/                     # Generated LaTeX tables and PDF plots
├── reinforcement_learning/    # Gym Environment & RL Trainers
├── results/                   # Raw JSON traces and CSV aggregates
├── scripts/                   # CLI Entry points (run_experiments, etc)
└── tests/                     # Pytest suite validating all modules
```

## License
This project is open-source under the MIT License.
