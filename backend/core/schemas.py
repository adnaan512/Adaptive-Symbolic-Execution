"""
Shared data contracts between pipeline stages.

Design rationale: KLEE integration (Phase 2), feature extraction (Phase 3),
the LLM analyzer (Phase 5), the ML ranker (Phase 4), and the RL agent
(Phase 6) are separate, independently-testable modules. To keep them
decoupled (clean architecture / SOLID: each module depends on an
abstraction, not on another module's internals) they communicate only
through the typed models defined here. If a later phase needs a new field,
it's added here first and every producer/consumer updates against the same
contract, instead of silently drifting dict schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SearchHeuristic(str, Enum):
    """Baseline KLEE search heuristics we compare against (RQ1)."""

    DFS = "dfs"
    BFS = "bfs"
    RANDOM_STATE = "random-state"
    RANDOM_PATH = "random-path"
    NURS_COVNEW = "nurs:covnew"
    NURS_MD2U = "nurs:md2u"
    COV_OPT = "cov-opt"
    AI_GUIDED = "ai-guided"


class ExecutionStateFeatures(BaseModel):
    """The 14-dimensional feature vector collected per KLEE execution state.

    Field order matches ``configs/config.yaml: features.names`` — keep them
    in sync; ``tests/test_schemas.py`` asserts this.
    """

    state_id: str
    execution_depth: int
    num_path_constraints: int
    constraint_complexity: float
    num_symbolic_variables: int
    solver_time_ms: float
    instruction_count: int
    branch_count: int
    loop_depth: int
    distance_to_uncovered_branch: float
    current_branch_coverage: float = Field(ge=0.0, le=1.0)
    memory_object_count: int
    num_forks: int
    state_age: float
    path_length: int

    def to_vector(self) -> list[float]:
        """Return the 14 numeric features in canonical order (for ML/RL)."""
        return [
            float(self.execution_depth),
            float(self.num_path_constraints),
            float(self.constraint_complexity),
            float(self.num_symbolic_variables),
            float(self.solver_time_ms),
            float(self.instruction_count),
            float(self.branch_count),
            float(self.loop_depth),
            float(self.distance_to_uncovered_branch),
            float(self.current_branch_coverage),
            float(self.memory_object_count),
            float(self.num_forks),
            float(self.state_age),
            float(self.path_length),
        ]


class BranchPrediction(BaseModel):
    """Structured output of the LLM semantic analyzer (Phase 5)."""

    branch: int
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class PriorityScore(BaseModel):
    """Output of the ML ranking model (Phase 4) for one execution state."""

    state_id: str
    score: float
    model_name: str


class CoverageSnapshot(BaseModel):
    """A single (time, coverage) sample used to plot coverage-over-time."""

    elapsed_seconds: float
    branch_coverage: float
    instruction_coverage: float
    num_states: int


class RunResult(BaseModel):
    """Aggregated result of one symbolic-execution run under one heuristic.

    Produced by ``evaluation/`` (Phase 7) and consumed by the dashboard
    (Phase 8) and the paper-table generator (Phase 9).
    """

    program_name: str
    heuristic: SearchHeuristic
    seed: int
    branch_coverage: float
    instruction_coverage: float
    unique_paths: int
    unique_bugs: int
    solver_calls: int
    execution_time_seconds: float
    memory_usage_mb: float
    state_explosion_count: int
    avg_solver_time_ms: float
    coverage_over_time: list[CoverageSnapshot] = Field(default_factory=list)
    notes: Optional[str] = None
