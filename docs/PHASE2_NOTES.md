# Phase 2 Notes — Symbolic Execution Harness

## What this phase produced

### New modules

| File | Purpose |
|------|---------|
| `klee/exceptions.py` | `KleeError` hierarchy: `KleeCompilationError`, `KleeRuntimeError`, `KleeParseError` |
| `klee/compiler.py` | `compile_to_bitcode()` — wraps `clang -emit-llvm -c -g -O0` |
| `klee/runner.py` | `run_klee()` / `run_klee_blocking()` / `KleeRunHandle` — launches KLEE subprocess, maps `SearchHeuristic` enum → `--search=` flag |
| `klee/parser.py` | `parse_run_stats()`, `stream_coverage()`, `build_run_result()`, `save_run_result_json()`, and helper functions — reads KLEE 3.0's SQLite `run.stats` |
| `klee/run_klee.py` | `orchestrate_run()` — end-to-end pipeline (compile → run → parse → JSON), plus CLI entry point (`python -m klee.run_klee`) |
| `klee/__init__.py` | Updated with all public exports and `__all__` |

### Updated modules

| File | Change |
|------|--------|
| `backend/core/logging.py` | Added `setup_logging(level)` for CLI entry points |

### New test files

| File | Tests |
|------|-------|
| `tests/test_klee_parser.py` | 25+ synthetic-SQLite tests covering formulas, parsing, RunResult assembly, helpers, JSON round-trip |
| `tests/test_klee_compiler.py` | 11 mocked-subprocess tests covering command construction and error paths |
| `tests/test_klee_runner.py` | 16 mocked-Popen tests covering all 8 heuristic flags and handle lifecycle |

### New benchmark programs

| File | Description |
|------|-------------|
| `dataset/coreutils/samples/simple_branch.c` | 4-branch classify + sign function — verifies 100% coverage |
| `dataset/coreutils/samples/nested_loop.c` | Bubble-sort with `klee_assert` — exercises loop_depth feature |

### New experiment manifest

| File | Description |
|------|-------------|
| `experiments/phase2_baseline.yaml` | Runs all 7 heuristics on `simple_branch.c`, 3 repetitions, 60s timeout |

---

## KLEE output format (KLEE 3.0)

KLEE 3.0 writes a **SQLite database** to `<klee-out-N>/run.stats`.  
The table is named `stats`; one row is appended per flush interval.

Key columns used by `klee/parser.py`:

| Column | Type | Used for |
|--------|------|---------|
| `WallTime` | REAL | `CoverageSnapshot.elapsed_seconds` |
| `FullBranches` | INT | Branch coverage numerator (fallback) |
| `PartialBranches` | INT | Branch coverage denominator (fallback) |
| `BCovNew` | REAL | Branch coverage 0–1 (preferred, KLEE 3+) |
| `ICovNew` | REAL | Instruction coverage 0–1 (preferred) |
| `CoveredInstructions` | INT | Instruction coverage (fallback) |
| `UncoveredInstructions` | INT | Instruction coverage (fallback) |
| `NumStates` | INT | `CoverageSnapshot.num_states` |
| `NumQueries` | INT | `RunResult.solver_calls` |
| `QueryTime` | REAL | Microseconds; → `RunResult.avg_solver_time_ms` |
| `MallocUsage` | INT | Bytes; → `RunResult.memory_usage_mb` |

Unique paths = number of `.ktest` files in the output directory.  
Unique bugs = number of `.err` files in the output directory.

---

## KLEE search heuristic → `--search=` flag mapping

| `SearchHeuristic` | KLEE flag |
|-------------------|-----------|
| `DFS` | `--search=dfs` |
| `BFS` | `--search=bfs` |
| `RANDOM_STATE` | `--search=random-state` |
| `RANDOM_PATH` | `--search=random-path` |
| `NURS_COVNEW` | `--search=nurs:covnew` |
| `NURS_MD2U` | `--search=nurs:md2u` |
| `COV_OPT` | `--search=cov-opt` |
| `AI_GUIDED` | `--search=random-path` *(Phase 6 placeholder)* |

---

## What was verified (without KLEE)

All 52+ unit tests pass without a KLEE/LLVM installation:

```bash
pytest tests/test_klee_parser.py tests/test_klee_compiler.py tests/test_klee_runner.py -v
```

- **Parser**: synthetic SQLite databases verify coverage formulas, row ordering,
  missing-file errors, RunResult assembly, memory conversion, JSON round-trip.
- **Compiler**: mocked `subprocess.run` verifies clang command construction,
  custom flags, output paths, and error handling.
- **Runner**: mocked `subprocess.Popen` verifies all 8 heuristic flags, posix/kqueries/seed
  flag inclusion/exclusion, extra_args forwarding, and KleeRunHandle lifecycle.

---

## What to verify on a real machine (inside Docker)

```bash
# Build the Docker image (first time: ~30-60 min)
docker compose -f Docker/docker-compose.yml build klee-env

# Verify KLEE runs
docker compose -f Docker/docker-compose.yml run --rm klee-env klee --version

# Install Python dependencies on host
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Run all Phase 1 + Phase 2 offline tests
pytest -q

# Run the Phase 2 demo inside Docker
docker compose -f Docker/docker-compose.yml run --rm klee-env bash -c "
  python -m klee.run_klee \
    --source dataset/coreutils/samples/simple_branch.c \
    --heuristic bfs \
    --timeout 60 \
    --no-posix \
    --output-dir results/runs/phase2_demo
"
# → Produces results/runs/phase2_demo/simple_branch_bfs.json
```

**Phase 2 Definition of Done** is met when:
1. `pytest -q` is fully green (offline).
2. `klee --version` prints a version string inside the Docker container.
3. Running `orchestrate_run()` on `simple_branch.c` with `SearchHeuristic.BFS` produces
   a `RunResult` JSON with `branch_coverage ≥ 0.9` and `unique_paths ≥ 3`.

---

## Known risk items carried forward

- **State explosion** on larger programs (Coreutils) — the `experiments/phase2_baseline.yaml`
  uses `simple_branch.c` (tiny) as the Phase 2 DoD target; real Coreutils runs are
  deferred to Phase 7 where the full evaluation harness exists.
- **AI_GUIDED hook** — falls back to `random-path` for now; the real external-control
  design will be confirmed in Phase 6 after checking which KLEE 3.0 extension points
  are available inside the container.
- **QueryTime column name** — some KLEE builds use `SolverTime` instead of `QueryTime`.
  `parser.py` checks both (`row.get("QueryTime") or row.get("SolverTime")`).
