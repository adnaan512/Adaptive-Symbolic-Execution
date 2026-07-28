# Adaptive LLM-Guided Symbolic Execution using Reinforcement Learning for Branch Coverage Maximization

Research-grade system that replaces KLEE's static search heuristics (DFS, BFS,
Random, Coverage-Optimized Search, NURS, MD2U, ...) with a learned
state-selection policy that combines:

- **Symbolic execution** (KLEE / LLVM bitcode)
- **Feature extraction** over live execution states
- **LLM semantic analysis** of source code to predict which branches likely
  expose unexplored behavior
- **ML ranking models** (Random Forest, XGBoost, LightGBM, NN, optional GNN)
  that turn features into a priority score
- **Reinforcement learning** (DQN / PPO) that learns a state-selection policy
  whose reward is the marginal increase in branch coverage

This repository is being built **in phases** (see `docs/ROADMAP.md`). Each
phase is independently testable, documented, and reviewed before moving to
the next, per standard research-software-engineering practice.

## Research Questions

- **RQ1** — Can AI-guided search improve branch coverage vs. traditional KLEE
  heuristics?
- **RQ2** — Can LLMs understand program semantics well enough to guide
  symbolic execution?
- **RQ3** — Can reinforcement learning continuously improve state-selection
  policies over time?

## Architecture

```
C Program
   │
   ▼
LLVM Bitcode  ──────────────┐
   │                        │
   ▼                        │
KLEE Symbolic Execution     │  (klee/)
   │                        │
   ▼                        │
Execution States            │
   │                        │
   ▼                        │
Feature Extraction ─────────┘  (feature_extractor/)
   │
   ▼
LLM Semantic Analyzer            (llm/)
   │
   ▼
ML Ranking Model                 (models/ml/)
   │
   ▼
RL Agent (DQN / PPO)              (reinforcement_learning/)
   │
   ▼
Priority Score → Best Execution State
   │
   ▼
Continue Symbolic Execution → Measure Branch Coverage
```

## Repository layout

```
Adaptive-Symbolic-Execution/
├── backend/                # FastAPI service: orchestrates KLEE runs, serves the dashboard API
│   ├── api/                 # HTTP routes
│   ├── core/                 # config, logging, orchestration engine
│   └── services/              # glue services (klee runner, dataset manager, ...)
├── frontend/                # React dashboard (coverage, state tree, RL learning curve, ...)
├── klee/                    # KLEE build assets, run wrappers, plugin/patches
├── llvm/                    # LLVM bitcode compilation helpers
├── feature_extractor/       # Extracts the 14-feature vector per execution state
├── llm/                     # Prompting + parsing layer for the LLM semantic analyzer
├── models/
│   ├── ml/                   # Random Forest / XGBoost / LightGBM / NN / GNN rankers
│   └── rl/                    # DQN / PPO agents, environment wrapper, replay buffer
├── reinforcement_learning/  # Training loops, reward shaping, RL experiment configs
├── dataset/                 # Benchmark corpora (Coreutils, BusyBox, SV-COMP, Juliet, LLVM test suite)
├── evaluation/               # Metrics, baselines, statistical significance testing
├── visualization/            # Plotly dashboard components, static figure generation
├── experiments/              # Experiment configs + run manifests (reproducibility)
├── results/                  # Generated tables/figures/logs (git-ignored except .gitkeep)
├── docs/                     # Roadmap, design decisions, installation guide
├── tests/                    # Unit + integration tests (pytest)
├── Docker/                   # Dockerfiles + docker-compose for reproducible builds
├── configs/                  # YAML configuration files (per-module, per-experiment)
└── scripts/                  # One-off / setup shell scripts
```

## Status

**Phase 1 — Environment setup & repository structure: in progress (this
commit).** See `docs/ROADMAP.md` for the full phase plan and
`docs/PHASE1_NOTES.md` for what has been verified so far and what still needs
to run in an environment with a real LLVM/KLEE toolchain (this sandbox has
restricted network egress and no GPU, so heavy builds are staged as
Dockerfiles/scripts rather than executed here — see notes for details).

## Quick start (once Phase 1 environment is built)

```bash
# Build the KLEE + LLVM toolchain image
docker compose -f Docker/docker-compose.yml build klee-env

# Install the Python side (host or dev container)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Sanity-check the toolchain
docker compose -f Docker/docker-compose.yml run --rm klee-env klee --version

# Run the test suite
pytest -q
```

## License

MIT (see `LICENSE`).
