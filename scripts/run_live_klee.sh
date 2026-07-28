#!/bin/bash
# Phase 10: Live C++ KLEE Integration Trigger
# This script boots the docker-compose orchestrator.

echo "=========================================================="
echo " Starting Adaptive LLM-Guided KLEE Architecture (Live) "
echo "=========================================================="

# Check for docker
if ! command -v docker-compose &> /dev/null
then
    echo "[ERROR] Docker Compose is not installed."
    echo "Please install Docker Desktop to run the live C++ compiler backend."
    exit 1
fi

echo "[INFO] Building Docker Images (KLEE + Python)..."
docker-compose build

echo "[INFO] Booting FastAPI Server and KLEE Engine..."
docker-compose up -d

echo "[INFO] Architecture is LIVE."
echo "[INFO] FastAPI is listening on http://localhost:8000"
echo "[INFO] To attach to the KLEE container and run symbolic execution, use:"
echo "       docker exec -it klee_execution_engine bash"
echo ""
echo "To shut down the architecture, run: docker-compose down"
