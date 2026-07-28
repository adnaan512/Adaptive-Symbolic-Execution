# =========================================================================
# Phase 10: Dockerfile for KLEE + Python AI Integration
# Based on the official KLEE LLVM image
# =========================================================================

FROM klee/klee:2.3

# Switch to root to install dependencies
USER root

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libcurl4-openssl-dev \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /home/klee/project

# Copy Python requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /home/klee/project

# ---------------------------------------------------------
# Apply the KLEE AI Searcher Patch
# ---------------------------------------------------------
# In a full build, this requires downloading KLEE source,
# patching lib/Core/Searcher.cpp, and recompiling with cmake.
# For this Docker image, we place the patch in the root so 
# students/professors can inspect the integration architecture.
# ---------------------------------------------------------
RUN echo "KLEE AI Patch ready at /home/klee/project/integration/klee_ai_searcher.patch"

# Expose the FastAPI port
EXPOSE 8000

# Default command: Boot the FastAPI server
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
