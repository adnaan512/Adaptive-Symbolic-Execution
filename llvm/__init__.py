"""
llvm — Phase 2 support.

Thin wrapper around `clang`/`opt` invocations used by `klee.compile_to_bitcode`.
Kept separate from `klee/` so the bitcode-compilation step can be unit
tested (given a real `clang` on PATH) independently of a KLEE installation.
"""
