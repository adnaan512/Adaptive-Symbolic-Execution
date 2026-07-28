"""
Custom exceptions for the klee module.

All exceptions derive from KleeError so callers can catch the whole
family with a single ``except KleeError`` if they don't care about
the specific failure mode.
"""

from __future__ import annotations


class KleeError(RuntimeError):
    """Base class for all KLEE-wrapper errors."""


class KleeCompilationError(KleeError):
    """Raised when ``clang`` cannot compile the C source to LLVM bitcode.

    Attributes
    ----------
    returncode:
        The process exit code returned by clang.
    stderr:
        The raw stderr output from clang.
    """

    def __init__(self, message: str, returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class KleeRuntimeError(KleeError):
    """Raised when the KLEE process exits with a non-zero code.

    Attributes
    ----------
    returncode:
        The process exit code returned by klee.
    stderr:
        The raw stderr output from klee.
    """

    def __init__(self, message: str, returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class KleeParseError(KleeError):
    """Raised when the KLEE output directory cannot be parsed.

    This usually means the run was interrupted before producing any
    stats, or the SQLite ``run.stats`` file is corrupted.
    """
