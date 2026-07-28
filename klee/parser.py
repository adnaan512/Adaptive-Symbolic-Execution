"""
klee.parser — parse KLEE output directories into structured Python objects.

KLEE 3.0 writes its statistics to a **SQLite database** at
``<klee-out-N>/run.stats``.  The schema has a single table called
``stats`` with one row per time-sample (KLEE flushes a row periodically
during the run, giving us a time-series of coverage).

Key columns used by this parser
---------------------------------
WallTime            REAL   -- seconds since KLEE started
Instructions        INT    -- total instructions executed
FullBranches        INT    -- branches where both arms were taken
PartialBranches     INT    -- branches where only one arm was taken
NumStates           INT    -- currently active execution states
NumQueries          INT    -- SMT solver calls
QueryTime           REAL   -- cumulative solver time (microseconds)
SolverTime          REAL   -- synonym for QueryTime in some KLEE builds
CoveredInstructions INT    -- instructions covered so far
UncoveredInstructions INT  -- instructions not yet covered
MallocUsage         INT    -- heap memory in bytes
ICovNew             REAL   -- instruction coverage (0.0–1.0) [KLEE 3+]
BCovNew             REAL   -- branch coverage (0.0–1.0) [KLEE 3+]

Branch coverage formula (when ICovNew/BCovNew unavailable):
    branch_coverage = FullBranches / max(1, FullBranches + PartialBranches)

Instruction coverage formula (fallback):
    instr_coverage  = CoveredInstructions / max(1, CoveredInstructions
                                                   + UncoveredInstructions)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from backend.core.schemas import CoverageSnapshot, RunResult, SearchHeuristic
from klee.exceptions import KleeParseError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _latest_klee_out(base_dir: Path) -> Path:
    """Find the most recently created ``klee-out-<N>`` directory."""
    candidates = sorted(
        base_dir.glob("klee-out-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise KleeParseError(
            f"No klee-out-* directory found under {base_dir}. "
            "Has KLEE been run yet?"
        )
    return candidates[0]


def _open_run_stats(run_dir: Path) -> sqlite3.Connection:
    """Open ``run.stats`` (SQLite) inside *run_dir*, raising KleeParseError on failure."""
    db_path = run_dir / "run.stats"
    if not db_path.exists():
        raise KleeParseError(
            f"run.stats not found in {run_dir}. "
            "KLEE may have crashed before writing any stats."
        )
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.DatabaseError as exc:
        raise KleeParseError(f"Cannot open run.stats at {db_path}: {exc}") from exc


def _rows_to_dicts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cursor = conn.execute("SELECT * FROM stats ORDER BY WallTime ASC")
    return [dict(row) for row in cursor.fetchall()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _branch_coverage_from_row(row: dict[str, Any]) -> float:
    """Compute branch coverage from a single stats row."""
    # Prefer the native KLEE 3+ field
    if row.get("BCovNew") is not None:
        cov = _safe_float(row["BCovNew"])
        if 0.0 <= cov <= 1.0:
            return cov

    full = _safe_int(row.get("FullBranches"))
    partial = _safe_int(row.get("PartialBranches"))
    total = full + partial
    if total == 0:
        return 0.0
    return full / total


def _instr_coverage_from_row(row: dict[str, Any]) -> float:
    """Compute instruction coverage from a single stats row."""
    if row.get("ICovNew") is not None:
        cov = _safe_float(row["ICovNew"])
        if 0.0 <= cov <= 1.0:
            return cov

    covered = _safe_int(row.get("CoveredInstructions"))
    uncovered = _safe_int(row.get("UncoveredInstructions"))
    total = covered + uncovered
    if total == 0:
        return 0.0
    return covered / total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_run_dir(run_dir: Path) -> Path:
    """Resolve *run_dir* to an actual ``klee-out-<N>`` directory.

    If *run_dir* is already a ``klee-out-<N>`` directory it is returned
    as-is; otherwise the newest ``klee-out-*`` child is returned.
    """
    run_dir = Path(run_dir).resolve()
    if (run_dir / "run.stats").exists() or run_dir.name.startswith("klee-out-"):
        return run_dir
    return _latest_klee_out(run_dir)


def parse_run_stats(run_dir: Path) -> list[dict[str, Any]]:
    """Read all rows from ``run.stats`` and return them as plain dicts.

    Each dict corresponds to one KLEE stats flush and contains the raw
    column values from the ``stats`` table.  Downstream consumers
    (``feature_extractor``) should use :func:`stream_coverage` for the
    typed version.

    Parameters
    ----------
    run_dir:
        Path to the ``klee-out-<N>`` directory (or the parent directory
        containing it — :func:`resolve_run_dir` will find the newest run).

    Returns
    -------
    list[dict]
        One dict per stats row, keys are column names, values are raw
        Python scalars (int / float / None).

    Raises
    ------
    KleeParseError
        If ``run.stats`` is missing or corrupt.
    """
    run_dir = resolve_run_dir(run_dir)
    conn = _open_run_stats(run_dir)
    try:
        rows = _rows_to_dicts(conn)
    except sqlite3.DatabaseError as exc:
        raise KleeParseError(f"Failed to read stats rows: {exc}") from exc
    finally:
        conn.close()

    logger.debug("Parsed %d rows from %s/run.stats", len(rows), run_dir.name)
    return rows


def stream_coverage(run_dir: Path) -> Iterator[CoverageSnapshot]:
    """Yield :class:`~backend.core.schemas.CoverageSnapshot` objects, one per stats row.

    This is the primary API consumed by the evaluation module (Phase 7)
    and the dashboard (Phase 8) to draw coverage-over-time plots.

    Parameters
    ----------
    run_dir:
        Path to the ``klee-out-<N>`` directory (or its parent).

    Yields
    ------
    CoverageSnapshot
        One snapshot per flush interval recorded in ``run.stats``.
    """
    rows = parse_run_stats(run_dir)
    for row in rows:
        yield CoverageSnapshot(
            elapsed_seconds=_safe_float(row.get("WallTime")),
            branch_coverage=_branch_coverage_from_row(row),
            instruction_coverage=_instr_coverage_from_row(row),
            num_states=_safe_int(row.get("NumStates")),
        )


def parse_messages(run_dir: Path) -> list[str]:
    """Read ``messages.txt`` from the KLEE output directory.

    Returns an empty list if the file does not exist (KLEE only writes it
    when there are messages to report).

    Parameters
    ----------
    run_dir:
        Path to the ``klee-out-<N>`` directory (or its parent).

    Returns
    -------
    list[str]
        Lines from ``messages.txt``, stripped of trailing whitespace.
    """
    run_dir = resolve_run_dir(run_dir)
    messages_path = run_dir / "messages.txt"
    if not messages_path.exists():
        return []
    return [line.rstrip() for line in messages_path.read_text(encoding="utf-8").splitlines()]


def parse_warnings(run_dir: Path) -> list[str]:
    """Read ``warnings.txt`` from the KLEE output directory.

    Parameters
    ----------
    run_dir:
        Path to the ``klee-out-<N>`` directory (or its parent).

    Returns
    -------
    list[str]
        Lines from ``warnings.txt``, stripped of trailing whitespace.
    """
    run_dir = resolve_run_dir(run_dir)
    warnings_path = run_dir / "warnings.txt"
    if not warnings_path.exists():
        return []
    return [line.rstrip() for line in warnings_path.read_text(encoding="utf-8").splitlines()]


def count_ktest_files(run_dir: Path) -> int:
    """Count ``.ktest`` files in the KLEE output directory.

    Each ``.ktest`` file represents one unique execution path that
    reached a termination point (error, assertion, or normal exit).
    This gives the ``unique_paths`` metric.

    Parameters
    ----------
    run_dir:
        Path to the ``klee-out-<N>`` directory (or its parent).

    Returns
    -------
    int
        Number of ``.ktest`` files found.
    """
    run_dir = resolve_run_dir(run_dir)
    return len(list(run_dir.glob("*.ktest")))


def count_error_files(run_dir: Path) -> int:
    """Count ``.err`` files (unique bugs/errors) in the KLEE output directory.

    Parameters
    ----------
    run_dir:
        Path to the ``klee-out-<N>`` directory (or its parent).

    Returns
    -------
    int
        Number of ``.err`` files found.
    """
    run_dir = resolve_run_dir(run_dir)
    return len(list(run_dir.glob("*.err")))


def build_run_result(
    run_dir: Path,
    program_name: str,
    heuristic: SearchHeuristic,
    wall_time_seconds: float,
    seed: int = 42,
) -> RunResult:
    """Construct a :class:`~backend.core.schemas.RunResult` from a completed KLEE run.

    This is the primary output produced by Phase 2 and consumed by
    Phase 7 (evaluation) and Phase 8 (dashboard).

    Parameters
    ----------
    run_dir:
        Path to the ``klee-out-<N>`` directory (or its parent).
    program_name:
        Human-readable name of the program under test (e.g. ``"cat"``).
    heuristic:
        The :class:`~backend.core.schemas.SearchHeuristic` used for this run.
    wall_time_seconds:
        Total elapsed time measured by the wrapper (use
        ``KleeRunHandle.wall_time_seconds``).
    seed:
        Random seed used for this run (from config).

    Returns
    -------
    RunResult
        A fully-populated result object ready for evaluation and serialisation.
    """
    rows = parse_run_stats(run_dir)
    snapshots = list(stream_coverage(run_dir))

    if not rows:
        logger.warning("No stats rows found for %s — returning empty RunResult", program_name)
        return RunResult(
            program_name=program_name,
            heuristic=heuristic,
            seed=seed,
            branch_coverage=0.0,
            instruction_coverage=0.0,
            unique_paths=0,
            unique_bugs=0,
            solver_calls=0,
            execution_time_seconds=wall_time_seconds,
            memory_usage_mb=0.0,
            state_explosion_count=0,
            avg_solver_time_ms=0.0,
            coverage_over_time=snapshots,
        )

    last = rows[-1]

    # Aggregate solver stats across all rows (use last row's cumulative values)
    total_solver_calls = _safe_int(last.get("NumQueries"))
    total_solver_time_us = _safe_float(last.get("QueryTime") or last.get("SolverTime"))
    avg_solver_ms = (
        (total_solver_time_us / 1000.0 / total_solver_calls)
        if total_solver_calls > 0
        else 0.0
    )

    # Memory: convert bytes → MiB
    mem_bytes = _safe_int(last.get("MallocUsage"))
    mem_mb = mem_bytes / (1024 * 1024) if mem_bytes > 0 else 0.0

    # State explosion count heuristic: rows where NumStates > 10,000
    state_explosion_count = sum(
        1 for r in rows if _safe_int(r.get("NumStates")) > 10_000
    )

    return RunResult(
        program_name=program_name,
        heuristic=heuristic,
        seed=seed,
        branch_coverage=_branch_coverage_from_row(last),
        instruction_coverage=_instr_coverage_from_row(last),
        unique_paths=count_ktest_files(run_dir),
        unique_bugs=count_error_files(run_dir),
        solver_calls=total_solver_calls,
        execution_time_seconds=wall_time_seconds,
        memory_usage_mb=mem_mb,
        state_explosion_count=state_explosion_count,
        avg_solver_time_ms=avg_solver_ms,
        coverage_over_time=snapshots,
    )


def save_run_result_json(result: RunResult, output_path: Path) -> None:
    """Serialise a :class:`~backend.core.schemas.RunResult` to a JSON file.

    Parameters
    ----------
    result:
        The run result to save.
    output_path:
        Destination file path (parent directories are created if needed).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("RunResult saved → %s", output_path)
