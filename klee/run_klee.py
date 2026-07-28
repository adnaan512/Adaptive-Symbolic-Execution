"""
klee.run_klee — CLI entry point and high-level orchestrator.

This script ties together the compiler, runner, and parser into one
end-to-end pipeline: C source → LLVM bitcode → KLEE run → JSON result.

Usage (inside the klee-env Docker container)
---------------------------------------------
    python -m klee.run_klee \\
        --source dataset/coreutils/samples/simple_branch.c \\
        --heuristic bfs \\
        --timeout 300 \\
        --output-dir results/runs/

Or import the orchestrator function directly:

    from klee.run_klee import orchestrate_run
    result = orchestrate_run(c_source=Path("my.c"), heuristic=SearchHeuristic.DFS)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from backend.core.config import load_config
from backend.core.logging import setup_logging
from backend.core.schemas import RunResult, SearchHeuristic
from klee.compiler import compile_to_bitcode
from klee.exceptions import KleeCompilationError, KleeError, KleeRuntimeError
from klee.parser import build_run_result, save_run_result_json
from klee.runner import run_klee_blocking

logger = logging.getLogger(__name__)


def orchestrate_run(
    c_source: Path,
    heuristic: SearchHeuristic = SearchHeuristic.BFS,
    timeout_s: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
    output_base_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    posix_runtime: bool = True,
    write_kqueries: bool = True,
    save_json: bool = True,
) -> RunResult:
    """Full pipeline: compile C → run KLEE → parse → return RunResult.

    This is the primary entry point for Phase 2.  It reads defaults from
    ``configs/config.yaml`` and overrides them with any explicitly-passed
    arguments.

    Parameters
    ----------
    c_source:
        Path to the C source file under test.
    heuristic:
        KLEE search heuristic to use.
    timeout_s:
        Time limit in seconds.  Defaults to ``config.klee.max_time_seconds``.
    max_memory_mb:
        Memory cap in MiB.  Defaults to ``config.klee.max_memory_mb``.
    output_base_dir:
        Where to write the ``klee-out-<N>`` directory and the JSON result.
        Defaults to ``config.paths.results_dir / "runs" / <stem>``.
    seed:
        Random seed (for reproducibility).  Defaults to ``config.project.seed``.
    posix_runtime:
        Enable the POSIX model (required for Coreutils / BusyBox).
    write_kqueries:
        Save SMT queries to disk (needed for Phase 3 constraint features).
    save_json:
        If True, serialise the :class:`~backend.core.schemas.RunResult` to
        ``<output_base_dir>/<stem>_<heuristic>.json``.

    Returns
    -------
    RunResult
        Fully-populated result object.

    Raises
    ------
    KleeCompilationError
        If clang cannot compile the source.
    KleeRuntimeError
        If KLEE exits with an unexpected error code.
    KleeParseError
        If the KLEE output directory cannot be parsed.
    """
    cfg = load_config()

    c_source = Path(c_source).resolve()
    if not c_source.exists():
        raise FileNotFoundError(f"C source not found: {c_source}")

    timeout_s = timeout_s if timeout_s is not None else cfg.klee.max_time_seconds
    max_memory_mb = max_memory_mb if max_memory_mb is not None else cfg.klee.max_memory_mb
    seed = seed if seed is not None else cfg.project.seed

    if output_base_dir is None:
        output_base_dir = (
            Path(cfg.paths.results_dir) / "runs" / c_source.stem / heuristic.value
        )
    output_base_dir = Path(output_base_dir).resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "=== Phase 2 orchestrator | program=%s | heuristic=%s ===",
        c_source.stem,
        heuristic.value,
    )

    # Step 1: compile
    logger.info("[1/3] Compiling %s → LLVM bitcode", c_source.name)
    bc_path = compile_to_bitcode(
        c_source=c_source,
        flags=cfg.klee.bitcode_compile_flags,
        output_dir=output_base_dir,
    )

    # Step 2: run KLEE
    logger.info("[2/3] Running KLEE (heuristic=%s, timeout=%ds)", heuristic.value, timeout_s)
    handle = run_klee_blocking(
        bitcode=bc_path,
        heuristic=heuristic,
        timeout_s=timeout_s,
        max_memory_mb=max_memory_mb,
        output_base_dir=output_base_dir,
        posix_runtime=posix_runtime,
        write_kqueries=write_kqueries,
        seed=seed,
    )

    # Step 3: parse
    logger.info("[3/3] Parsing KLEE output from %s", output_base_dir)
    result = build_run_result(
        run_dir=output_base_dir,
        program_name=c_source.stem,
        heuristic=heuristic,
        wall_time_seconds=handle.wall_time_seconds,
        seed=seed,
    )

    if save_json:
        json_path = output_base_dir / f"{c_source.stem}_{heuristic.value}.json"
        save_run_result_json(result, json_path)

    logger.info(
        "=== Done | branch_cov=%.2f%% | instr_cov=%.2f%% | paths=%d | bugs=%d ===",
        result.branch_coverage * 100,
        result.instruction_coverage * 100,
        result.unique_paths,
        result.unique_bugs,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m klee.run_klee",
        description=(
            "Phase 2: Compile a C program and run KLEE under a chosen "
            "search heuristic, producing a structured JSON result."
        ),
    )
    p.add_argument(
        "--source", "-s",
        type=Path,
        required=True,
        help="Path to the C source file.",
    )
    p.add_argument(
        "--heuristic",
        type=str,
        default="bfs",
        choices=[h.value for h in SearchHeuristic],
        help="KLEE search heuristic (default: bfs).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Wall-clock time limit (default: from config.yaml).",
    )
    p.add_argument(
        "--memory",
        type=int,
        default=None,
        metavar="MB",
        help="Memory cap in MiB (default: from config.yaml).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory for KLEE artifacts and JSON result.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config.yaml).",
    )
    p.add_argument(
        "--no-posix",
        action="store_true",
        default=False,
        help="Disable POSIX runtime (not needed for simple programs).",
    )
    p.add_argument(
        "--no-kqueries",
        action="store_true",
        default=False,
        help="Skip writing SMT queries (saves disk space).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns exit code (0 = success, 1 = error)."""
    args = _build_parser().parse_args(argv)
    setup_logging(level=args.log_level)

    try:
        orchestrate_run(
            c_source=args.source,
            heuristic=SearchHeuristic(args.heuristic),
            timeout_s=args.timeout,
            max_memory_mb=args.memory,
            output_base_dir=args.output_dir,
            seed=args.seed,
            posix_runtime=not args.no_posix,
            write_kqueries=not args.no_kqueries,
        )
        return 0
    except (KleeCompilationError, KleeRuntimeError, KleeError) as exc:
        logger.error("KLEE pipeline failed: %s", exc)
        return 1
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
