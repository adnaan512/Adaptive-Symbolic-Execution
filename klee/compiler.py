"""
klee.compiler — compile C source files to LLVM bitcode.

This module wraps the ``clang`` invocation that turns a ``.c`` file into
a ``.bc`` (LLVM bitcode) file ready for KLEE symbolic execution.

Usage example
-------------
>>> from pathlib import Path
>>> from klee.compiler import compile_to_bitcode
>>> bc = compile_to_bitcode(Path("my_program.c"))
>>> print(bc)  # my_program.bc
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from klee.exceptions import KleeCompilationError

logger = logging.getLogger(__name__)

# Default flags that produce bitcode suitable for KLEE.
# -emit-llvm      → produce LLVM bitcode (.bc) instead of native object
# -c              → compile only (do not link)
# -g              → include debug symbols (KLEE uses them for coverage info)
# -O0             → no optimisation (KLEE works better with unoptimised IR)
# -Xclang -disable-O0-optnone → prevent clang from adding the optnone attribute
#                               that blocks KLEE's own IR passes
DEFAULT_CLANG_FLAGS: list[str] = [
    "-emit-llvm",
    "-c",
    "-g",
    "-O0",
    "-Xclang",
    "-disable-O0-optnone",
]


def compile_to_bitcode(
    c_source: Path,
    flags: list[str] | None = None,
    output_dir: Path | None = None,
    clang_binary: str = "clang",
) -> Path:
    """Compile a C source file to LLVM bitcode.

    Parameters
    ----------
    c_source:
        Path to the ``.c`` file to compile.
    flags:
        Clang flags to pass.  If *None*, :data:`DEFAULT_CLANG_FLAGS` are
        used (values from ``configs/config.yaml`` can be passed here via
        ``load_config().klee.bitcode_compile_flags``).
    output_dir:
        Directory in which to place the ``.bc`` file.  Defaults to the
        same directory as ``c_source``.
    clang_binary:
        Name / path of the ``clang`` executable.  Override for testing
        or when a versioned binary is needed (e.g. ``clang-13``).

    Returns
    -------
    Path
        Absolute path to the produced ``.bc`` file.

    Raises
    ------
    FileNotFoundError
        If ``c_source`` does not exist.
    KleeCompilationError
        If clang exits with a non-zero return code.
    """
    c_source = Path(c_source).resolve()
    if not c_source.exists():
        raise FileNotFoundError(f"C source not found: {c_source}")

    if flags is None:
        flags = DEFAULT_CLANG_FLAGS

    if output_dir is None:
        output_dir = c_source.parent
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    bc_path = output_dir / c_source.with_suffix(".bc").name

    cmd: list[str] = [clang_binary, *flags, str(c_source), "-o", str(bc_path)]
    logger.info("Compiling %s → %s", c_source.name, bc_path.name)
    logger.debug("clang command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise KleeCompilationError(
            f"clang failed (rc={result.returncode}) for {c_source.name}:\n{result.stderr}",
            returncode=result.returncode,
            stderr=result.stderr,
        )

    logger.info("Compiled successfully → %s (%d bytes)", bc_path.name, bc_path.stat().st_size)
    return bc_path
