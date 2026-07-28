# Phase 10 Notes — Live C++ KLEE Integration

## What this phase produced

### The Capstone Architecture
This phase represents the final bridging between our modern Python AI stack and the legacy C++ LLVM infrastructure of KLEE.

1. **`backend/api/main.py` (FastAPI Inference Engine)**:
   - Exposes a live `POST /api/evaluate_state` endpoint.
   - Designed to receive a serialized JSON payload containing the active execution branches inside the KLEE virtual machine, rank them using our heuristic functions (or XGBoost/PPO in production), and return the optimal state ID to pursue.

2. **`integration/klee_ai_searcher.patch` (The C++ Patch)**:
   - This is a true compiler-level modification.
   - It patches KLEE's `lib/Core/Searcher.cpp` to introduce a new `AISearcher` class.
   - It injects `libcurl` to pause the C++ execution, serialize the states, make the HTTP request to our FastAPI backend, and resume execution down the AI's chosen branch.

3. **Docker Orchestration (`Dockerfile` & `docker-compose.yml`)**:
   - Because LLVM/KLEE compilation is extremely brittle on Windows, we containerize the entire architecture.
   - The orchestrator spins up two isolated environments on a virtual network:
     - `klee_ai_backend`: The Python server.
     - `klee_execution_engine`: The Linux C++ environment natively running the modified KLEE.

## Running the Live Architecture

If you have **Docker Desktop** installed on your system, you can boot the live architecture:

```bash
bash scripts/run_live_klee.sh
```

This will build the images and launch the background services. You can then attach to the C++ compiler container to run manual bitcode evaluations:
```bash
docker exec -it klee_execution_engine bash
```

## Academic Significance
By completely decoupling the search heuristic from the C++ binary and moving it to a Python API, this architecture allows researchers to dynamically swap out ML models, LLM prompts, and RL reward functions *without ever needing to recompile the LLVM toolchain*. This is a massive leap forward for empirical Symbolic Execution research.
