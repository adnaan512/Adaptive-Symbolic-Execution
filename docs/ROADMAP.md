# Roadmap

The project is built in nine phases. Each phase must pass its "Definition of
Done" before the next begins. This keeps the codebase reviewable and keeps
every experimental result reproducible and attributable to a specific,
frozen version of the pipeline — important for a publication-quality
artifact.

| Phase | Name | Key outputs | Definition of Done |
|------|------|------------|---------------------|
| 1 | Environment setup | Repo skeleton, `Docker/`, `configs/`, `requirements*.txt`, CI stub | `docker compose build` succeeds; `klee --version` runs inside the container; `pytest` collects (even if empty) |
| 2 | Symbolic execution harness | `klee/run_klee.py`, state/coverage log parser | Running KLEE on a sample C program under a chosen search heuristic produces a structured JSON log of states + coverage over time |
| 3 | Feature extraction | `feature_extractor/` module + schema | 14-feature vector extracted per state, written to `dataset/*.parquet`, unit-tested against synthetic states |
| 4 | ML ranking model | `models/ml/` (RF, XGBoost, LightGBM, NN, optional GNN) | Models train on Phase 3 features, predict a priority score, cross-validated with reported metrics |
| 5 | LLM integration | `llm/` prompting + JSON-schema parsing + caching | LLM returns `{branch, confidence, reason}` reliably (schema-validated) for a batch of held-out functions |
| 6 | Reinforcement learning | `reinforcement_learning/` (DQN, PPO), Gym-style env wrapper around KLEE state selection | Agent trains end-to-end on at least one benchmark, learning curve trends upward, checkpoints saved |
| 7 | Evaluation | `evaluation/` metrics + baseline runners + significance tests | Full comparison table (AI-guided vs. DFS/BFS/Random/CovOpt/NURS/MD2U) across metrics, with Wilcoxon/Mann-Whitney significance |
| 8 | Visualization dashboard | `frontend/` (React) + `backend/api` (FastAPI) | Dashboard renders coverage-over-time, state tree, coverage heatmap, RL learning curve, solver stats, from `results/` |
| 9 | Documentation | `docs/`, README, installation guide, paper-style report | Reproducibility check: fresh clone + `docker compose up` + one command reproduces Table 1 |

## Current phase: Phase 1

See `docs/PHASE1_NOTES.md` for exactly what was created in this session, what
was verified in-sandbox, and what must be verified on a machine/container
with real LLVM+KLEE+Z3 installed (this development sandbox has restricted
network egress — only package registries — and cannot apt-install or build
KLEE's C/C++ toolchain live).

## Design principles carried through every phase

- **Clean architecture / SOLID** — each module (`feature_extractor`, `llm`,
  `models`, `reinforcement_learning`, `evaluation`) exposes a small, typed
  interface and does not import KLEE-specific internals from another module;
  all cross-module communication goes through typed dataclasses/Pydantic
  models defined in `backend/core/schemas.py`.
- **Reproducibility** — every experiment is described by a YAML config in
  `experiments/`, is given a run ID, and its full config + git commit hash +
  seed are written alongside its results.
- **Testability** — every module has unit tests using synthetic
  (non-KLEE-dependent) fixtures so CI can run without a KLEE build, plus a
  smaller set of integration tests gated behind a `klee-installed` marker.
