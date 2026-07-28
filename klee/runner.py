"""
klee.runner — launch KLEE and manage the run lifecycle.

This module wraps the ``klee`` binary invocation, maps our
:class:`~backend.core.schemas.SearchHeuristic` enum to KLEE's
``--search`` flag syntax, and exposes a :class:`KleeRunHandle` that
lets callers wait for completion or stream live output.

KLEE CLI reference (v3.0)
--------------------------
  klee [options] <bitcode>

Key options used here:
  --search=<heuristic>      State-selection strategy
  --max-time=<sec>          Wall-clock timeout (seconds)
  --max-memory=<MB>         Memory cap
  --output-dir=<dir>        Where to write klee-out-*/
  --write-kqueries          Write SMT queries (enables Phase 3 constraint features)
  --use-forked-solver       Run Z3 in a forked process (more stable)
  --posix-runtime           Enable POSIX model (needed for Coreutils etc.)
  --libc=uclibc             Use klee-uclibc (required with --posix-runtime)
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.core.schemas import CoverageSnapshot, RunResult, SearchHeuristic
from klee.exceptions import KleeRuntimeError

logger = logging.getLogger(__name__)

# Maps our SearchHeuristic enum values to the exact --search= string KLEE expects.
_HEURISTIC_TO_FLAG: dict[str, str] = {
    SearchHeuristic.DFS.value: "dfs",
    SearchHeuristic.BFS.value: "bfs",
    SearchHeuristic.RANDOM_STATE.value: "random-state",
    SearchHeuristic.RANDOM_PATH.value: "random-path",
    SearchHeuristic.NURS_COVNEW.value: "nurs:covnew",
    SearchHeuristic.NURS_MD2U.value: "nurs:md2u",
    SearchHeuristic.COV_OPT.value: "cov-opt",
    # AI_GUIDED is handled specially in Phase 6 — use random-path as placeholder
    SearchHeuristic.AI_GUIDED.value: "random-path",
}


@dataclass
class KleeRunHandle:
    """Handle to a running (or completed) KLEE process.

    Attributes
    ----------
    run_dir:
        The ``klee-out-<N>`` directory produced by KLEE.
    bitcode:
        The ``.bc`` file that was given to KLEE.
    heuristic:
        The search heuristic used for this run.
    pid:
        OS process ID of the KLEE process (0 if the process has exited).
    returncode:
        Exit code of KLEE (``None`` while still running).
    stdout:
        Captured stdout (available after :meth:`wait`).
    stderr:
        Captured stderr (available after :meth:`wait`).
    wall_time_seconds:
        Elapsed wall-clock time measured by the wrapper (not KLEE's own timer).
    """

    run_dir: Path
    bitcode: Path
    heuristic: SearchHeuristic
    pid: int = 0
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    wall_time_seconds: float = 0.0
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)

    def is_running(self) -> bool:
        """Return True if the KLEE process is still alive."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def wait(self, timeout: Optional[float] = None) -> int:
        """Block until the KLEE process finishes and return its exit code.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.  Raises ``subprocess.TimeoutExpired``
            if the process does not finish in time.

        Returns
        -------
        int
            The KLEE process exit code.
        """
        if self._process is None:
            return self.returncode or 0
        try:
            stdout, stderr = self._process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            stdout, stderr = self._process.communicate()
            raise
        self.returncode = self._process.returncode
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        return self.returncode

    def kill(self) -> None:
        """Send SIGKILL to the KLEE process."""
        if self._process is not None and self.is_running():
            self._process.kill()
            logger.warning("KLEE process %d killed by wrapper", self.pid)


def run_klee(
    bitcode: Path,
    heuristic: SearchHeuristic = SearchHeuristic.BFS,
    timeout_s: int = 3600,
    max_memory_mb: int = 4096,
    output_base_dir: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
    klee_binary: str = "klee",
    posix_runtime: bool = True,
    write_kqueries: bool = True,
    seed: Optional[int] = None,
) -> KleeRunHandle:
    """Launch KLEE on a bitcode file and return a handle to the running process.

    This function is **non-blocking**: it starts KLEE in a subprocess and
    returns immediately.  Use :meth:`KleeRunHandle.wait` to wait for
    completion, or poll :meth:`KleeRunHandle.is_running`.

    Parameters
    ----------
    bitcode:
        Path to the ``.bc`` file (output of :func:`~klee.compiler.compile_to_bitcode`).
    heuristic:
        Which KLEE search strategy to use.
    timeout_s:
        Wall-clock time limit passed to KLEE via ``--max-time``.
    max_memory_mb:
        Memory cap in MiB passed to KLEE via ``--max-memory``.
    output_base_dir:
        Parent directory for the ``klee-out-<N>`` directory.
        Defaults to the directory containing the bitcode file.
    extra_args:
        Any additional raw flags forwarded verbatim to KLEE.
    klee_binary:
        Name / path of the ``klee`` executable.
    posix_runtime:
        Whether to enable the POSIX model (``--posix-runtime --libc=uclibc``).
        Required for Coreutils / BusyBox benchmarks.
    write_kqueries:
        Whether to save SMT queries (``--write-kqueries``).  Enables
        Phase 3 constraint-complexity features but increases disk usage.
    seed:
        Optional random seed passed via ``--seed-random=<seed>``.

    Returns
    -------
    KleeRunHandle
        A handle to the live KLEE subprocess.

    Raises
    ------
    FileNotFoundError
        If the bitcode file does not exist.
    """
    bitcode = Path(bitcode).resolve()
    if not bitcode.exists():
        raise FileNotFoundError(f"Bitcode file not found: {bitcode}")

    if output_base_dir is None:
        output_base_dir = bitcode.parent
    else:
        output_base_dir = Path(output_base_dir).resolve()
        output_base_dir.mkdir(parents=True, exist_ok=True)

    search_flag = _HEURISTIC_TO_FLAG.get(heuristic.value, "bfs")
    if heuristic == SearchHeuristic.AI_GUIDED:
        logger.warning(
            "AI_GUIDED heuristic is not yet implemented in Phase 2; "
            "falling back to random-path as placeholder."
        )

    cmd: list[str] = [
        klee_binary,
        f"--search={search_flag}",
        f"--max-time={timeout_s}",
        f"--max-memory={max_memory_mb}",
        f"--output-dir={output_base_dir}",
        "--use-forked-solver",
    ]

    if posix_runtime:
        cmd += ["--posix-runtime", "--libc=uclibc"]

    if write_kqueries:
        cmd.append("--write-kqueries")

    if seed is not None:
        cmd.append(f"--seed-random={seed}")

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(str(bitcode))

    logger.info(
        "Launching KLEE | heuristic=%s | timeout=%ds | binary=%s",
        heuristic.value,
        timeout_s,
        bitcode.name,
    )
    logger.debug("KLEE command: %s", " ".join(cmd))

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # KLEE names its output directory klee-out-<N> where N increments.
    # It prints the directory path to stderr; we capture it after a brief wait.
    # The actual directory is discovered by the parser from the output_base_dir.
    handle = KleeRunHandle(
        run_dir=output_base_dir,   # refined by parser once KLEE starts
        bitcode=bitcode,
        heuristic=heuristic,
        pid=proc.pid,
        _process=proc,
    )

    logger.info("KLEE started (pid=%d)", proc.pid)

    # Record wall time on a best-effort basis
    def _record_wall_time() -> None:
        handle.wall_time_seconds = time.monotonic() - t0

    handle._wall_time_start = t0  # type: ignore[attr-defined]
    handle._record_wall_time = _record_wall_time  # type: ignore[attr-defined]

    return handle


def run_klee_blocking(
    bitcode: Path,
    heuristic: SearchHeuristic = SearchHeuristic.BFS,
    timeout_s: int = 3600,
    max_memory_mb: int = 4096,
    output_base_dir: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
    klee_binary: str = "klee",
    posix_runtime: bool = True,
    write_kqueries: bool = True,
    seed: Optional[int] = None,
) -> KleeRunHandle:
    """Convenience wrapper: launch KLEE and block until it finishes.

    Raises :class:`~klee.exceptions.KleeRuntimeError` if KLEE exits
    with a non-zero return code (note: KLEE uses rc=0 even when it finds
    errors, so this only fires on hard failures like OOM or bad flags).

    Returns the completed :class:`KleeRunHandle` with ``returncode``,
    ``stdout``, and ``stderr`` populated.
    """
    handle = run_klee(
        bitcode=bitcode,
        heuristic=heuristic,
        timeout_s=timeout_s,
        max_memory_mb=max_memory_mb,
        output_base_dir=output_base_dir,
        extra_args=extra_args,
        klee_binary=klee_binary,
        posix_runtime=posix_runtime,
        write_kqueries=write_kqueries,
        seed=seed,
    )
    t0 = time.monotonic()
    try:
        rc = handle.wait(timeout=timeout_s + 30)  # +30s grace period
    except subprocess.TimeoutExpired:
        handle.kill()
        logger.error("KLEE exceeded timeout (%ds) — killed", timeout_s)
        rc = -1

    handle.wall_time_seconds = time.monotonic() - t0

    if rc not in (0, 1):  # KLEE rc=1 means it found errors (normal)
        raise KleeRuntimeError(
            f"KLEE exited with unexpected rc={rc}:\n{handle.stderr}",
            returncode=rc,
            stderr=handle.stderr,
        )

    logger.info(
        "KLEE finished | rc=%d | wall=%.1fs | heuristic=%s",
        rc,
        handle.wall_time_seconds,
        heuristic.value,
    )
    return handle
