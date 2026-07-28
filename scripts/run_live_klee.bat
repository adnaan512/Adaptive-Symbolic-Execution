@echo off
REM Phase 10: Live C++ KLEE Integration Trigger for Windows
REM This script boots the docker-compose orchestrator.

echo ==========================================================
echo  Starting Adaptive LLM-Guided KLEE Architecture (Live) 
echo ==========================================================

REM Check if docker is available
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please ensure Docker Desktop is running.
    exit /b 1
)

echo [INFO] Building Docker Images (KLEE + Python)...
docker compose build

echo [INFO] Booting FastAPI Server and KLEE Engine...
docker compose up -d

echo [INFO] Architecture is LIVE.
echo [INFO] FastAPI is listening on http://localhost:8000
echo [INFO] To attach to the KLEE container and run symbolic execution, use:
echo        docker exec -it klee_execution_engine bash
echo.
echo To shut down the architecture, run: docker compose down
pause
