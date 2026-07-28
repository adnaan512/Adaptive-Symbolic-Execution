"""
klee — Phase 2 (implemented).

Public API
----------
Compilation
    compile_to_bitcode   Compile a C source file → LLVM bitcode via clang.

Running KLEE
    run_klee             Launch KLEE non-blocking; returns KleeRunHandle.
    run_klee_blocking    Launch KLEE and wait for completion.
    KleeRunHandle        Handle to a live/completed KLEE process.

Parsing output
    parse_run_stats      Read all rows from run.stats (SQLite) → list[dict].
    stream_coverage      Yield CoverageSnapshot objects from run.stats.
    build_run_result     Assemble a RunResult from a completed KLEE run dir.
    save_run_result_json Serialise a RunResult to JSON.
    parse_messages       Read messages.txt.
    parse_warnings       Read warnings.txt.
    count_ktest_files    Count unique paths (.ktest files).
    count_error_files    Count unique bugs (.err files).

Orchestration
    orchestrate_run      End-to-end: C source → compile → KLEE → parse → RunResult.

Exceptions
    KleeError            Base exception.
    KleeCompilationError clang compilation failure.
    KleeRuntimeError     KLEE process failure.
    KleeParseError       Output directory parsing failure.

Notes
-----
AI_GUIDED heuristic (Phase 6): currently falls back to random-path as a
placeholder. The real hook design is documented in docs/design/klee_hook.md
(added in Phase 6 once KLEE version capabilities are confirmed).
"""

from klee.compiler import compile_to_bitcode
from klee.exceptions import KleeCompilationError, KleeError, KleeParseError, KleeRuntimeError
from klee.parser import (
    build_run_result,
    count_error_files,
    count_ktest_files,
    parse_messages,
    parse_run_stats,
    parse_warnings,
    save_run_result_json,
    stream_coverage,
)
from klee.run_klee import orchestrate_run
from klee.runner import KleeRunHandle, run_klee, run_klee_blocking

__all__ = [
    # compilation
    "compile_to_bitcode",
    # running
    "run_klee",
    "run_klee_blocking",
    "KleeRunHandle",
    # parsing
    "parse_run_stats",
    "stream_coverage",
    "build_run_result",
    "save_run_result_json",
    "parse_messages",
    "parse_warnings",
    "count_ktest_files",
    "count_error_files",
    # orchestration
    "orchestrate_run",
    # exceptions
    "KleeError",
    "KleeCompilationError",
    "KleeRuntimeError",
    "KleeParseError",
]
