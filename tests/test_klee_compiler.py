"""
Phase 2 tests for klee.compiler (using mocked subprocess).

Tests verify:
  - compile_to_bitcode builds the correct clang command
  - output path is derived correctly from input path
  - custom output_dir is used when provided
  - custom flags are forwarded
  - missing source file raises FileNotFoundError
  - non-zero clang exit code raises KleeCompilationError with correct attrs
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from klee.compiler import DEFAULT_CLANG_FLAGS, compile_to_bitcode
from klee.exceptions import KleeCompilationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def c_source(tmp_path) -> Path:
    """A real (empty) .c file so the existence check passes."""
    src = tmp_path / "my_program.c"
    src.write_text("int main(void){return 0;}", encoding="utf-8")
    return src


def _mock_success(bc_path: Path):
    """Return a mock subprocess.CompletedProcess that mimics a successful clang run."""
    result = MagicMock()
    result.returncode = 0
    result.stderr = ""
    result.stdout = ""
    # Simulate clang creating the output file
    bc_path.parent.mkdir(parents=True, exist_ok=True)
    bc_path.touch()
    return result


def _mock_failure(stderr: str = "clang: error: some error", returncode: int = 1):
    result = MagicMock()
    result.returncode = returncode
    result.stderr = stderr
    result.stdout = ""
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompileToBitcode:
    def test_returns_bc_path_in_same_dir(self, c_source, tmp_path):
        bc_path = c_source.parent / "my_program.bc"
        with patch("subprocess.run", return_value=_mock_success(bc_path)):
            result = compile_to_bitcode(c_source)
        assert result == bc_path

    def test_returns_bc_path_in_custom_output_dir(self, c_source, tmp_path):
        out_dir = tmp_path / "build"
        bc_path = out_dir / "my_program.bc"
        with patch("subprocess.run", return_value=_mock_success(bc_path)):
            result = compile_to_bitcode(c_source, output_dir=out_dir)
        assert result == bc_path
        assert out_dir.exists()  # mkdir was called

    def test_default_flags_used_when_none_given(self, c_source):
        bc_path = c_source.parent / "my_program.bc"
        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return _mock_success(bc_path)

        with patch("subprocess.run", side_effect=fake_run):
            compile_to_bitcode(c_source)

        for flag in DEFAULT_CLANG_FLAGS:
            assert flag in captured_cmd

    def test_custom_flags_forwarded(self, c_source):
        bc_path = c_source.parent / "my_program.bc"
        captured_cmd: list[str] = []
        custom_flags = ["-emit-llvm", "-c", "-g"]

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return _mock_success(bc_path)

        with patch("subprocess.run", side_effect=fake_run):
            compile_to_bitcode(c_source, flags=custom_flags)

        for flag in custom_flags:
            assert flag in captured_cmd

    def test_output_path_in_command(self, c_source):
        bc_path = c_source.parent / "my_program.bc"
        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return _mock_success(bc_path)

        with patch("subprocess.run", side_effect=fake_run):
            compile_to_bitcode(c_source)

        assert "-o" in captured_cmd
        assert str(bc_path) in captured_cmd

    def test_source_file_in_command(self, c_source):
        bc_path = c_source.parent / "my_program.bc"
        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return _mock_success(bc_path)

        with patch("subprocess.run", side_effect=fake_run):
            compile_to_bitcode(c_source)

        assert str(c_source) in captured_cmd

    def test_custom_clang_binary(self, c_source):
        bc_path = c_source.parent / "my_program.bc"
        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return _mock_success(bc_path)

        with patch("subprocess.run", side_effect=fake_run):
            compile_to_bitcode(c_source, clang_binary="clang-13")

        assert captured_cmd[0] == "clang-13"

    def test_missing_source_raises_file_not_found(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.c"
        with pytest.raises(FileNotFoundError, match="does_not_exist.c"):
            compile_to_bitcode(nonexistent)

    def test_clang_failure_raises_compilation_error(self, c_source):
        with patch(
            "subprocess.run",
            return_value=_mock_failure(stderr="error: unknown flag", returncode=1),
        ):
            with pytest.raises(KleeCompilationError) as exc_info:
                compile_to_bitcode(c_source)

        exc = exc_info.value
        assert exc.returncode == 1
        assert "error: unknown flag" in exc.stderr

    def test_compilation_error_message_contains_filename(self, c_source):
        with patch(
            "subprocess.run",
            return_value=_mock_failure(returncode=1),
        ):
            with pytest.raises(KleeCompilationError, match="my_program.c"):
                compile_to_bitcode(c_source)
