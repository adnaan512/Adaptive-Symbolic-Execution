# Phase 1 Notes — Environment Setup

## What this session produced

- Full repository skeleton matching the architecture in the README.
- `configs/config.yaml` — single source of truth for paths, feature list,
  model hyperparameters, RL hyperparameters, and baseline heuristic names.
- `Docker/Dockerfile.klee` — multi-stage build that compiles LLVM 13,
  Z3, STP-free (Z3-only) KLEE, and klee-uclibc, matching the versions KLEE's
  own CI uses. This is the artifact that should actually be built on a
  machine with real network access and enough disk (KLEE + LLVM needs
  ~15-20GB free and 30-60 min to build from source).
- `Docker/docker-compose.yml` — wires the `klee-env` build image and a
  `backend` service together.
- `requirements.txt` / `requirements-dev.txt` — pinned-family Python
  dependencies for every later phase (feature extraction, ML, LLM, RL,
  dashboard backend).
- `backend/core/config.py` + `backend/core/schemas.py` — typed config loader
  and the Pydantic schemas (`ExecutionStateFeatures`, `BranchPrediction`,
  `PriorityScore`, ...) that every later phase will import, so Phases 2-9
  share one contract instead of ad hoc dicts.
- `backend/core/logging.py` — structured logging setup used by every module.
- Stub packages for `feature_extractor/`, `llm/`, `models/ml/`, `models/rl/`,
  `reinforcement_learning/`, `evaluation/`, `visualization/` — each has an
  `__init__.py` with a module-level docstring describing its Phase-N
  responsibility and its public interface, so the shape of the whole system
  is visible even before each phase is implemented.
- `tests/test_config.py`, `tests/test_schemas.py` — smoke tests that pass
  with only `pyyaml`/`pydantic` installed (no KLEE/LLVM required), so CI is
  green from commit 1.
- `.gitignore`, `LICENSE` (MIT), `pyproject.toml` (black/ruff/mypy config).

## What was verified inside this sandbox

This development sandbox's outbound network is restricted to package
registries (PyPI, npm, crates, GitHub source, Ubuntu package *metadata*
mirrors) and does not currently reach the full package archive needed for a
from-scratch LLVM/KLEE apt install, so the following were verified logically
/ syntactically but **not** by actually invoking `klee`:

- `python -m pyflakes` / import-checked every `.py` file created.
- `docker build` syntax was hand-reviewed (not executed — no Docker daemon
  in this sandbox).
- YAML configs were parsed with `yaml.safe_load` to confirm they're valid.

## What you need to do on your machine (or a CI runner with full egress)

```bash
cd Adaptive-Symbolic-Execution
docker compose -f Docker/docker-compose.yml build klee-env   # ~30-60 min first time
docker compose -f Docker/docker-compose.yml run --rm klee-env klee --version
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q       # should pass fully offline
```

If `klee --version` prints a version string, Phase 1's Definition of Done is
met and we can move to Phase 2 (running KLEE on a real program and parsing
its state/coverage stream).

## Known risk items to revisit

- KLEE version pinning: the Dockerfile pins **KLEE 3.0 / LLVM 13**, which is
  the combination with the most active community documentation as of the
  last time this was checked. Confirm this is still the recommended pairing
  before your Phase-2 run — KLEE's supported LLVM version has moved before.
- Z3 vs STP: the Dockerfile builds KLEE against Z3 only, to avoid STP's
  extra build time; note this as a methodological choice in the eventual
  paper's threats-to-validity section, since solver choice can affect
  timing metrics.
- GPU: RL (Phase 6) and any NN/GNN ranking model (Phase 4) will want a CUDA
  image; `Docker/Dockerfile.klee` deliberately does *not* include CUDA to
  keep the symbolic-execution image lean. A separate `Dockerfile.ml` should
  be added in Phase 4 once model choices are locked in.
