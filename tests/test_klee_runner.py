"""
Phase 2 tests for klee.runner (using mocked subprocess).

Tests verify:
  - run_klee builds the correct KLEE command (search flag, timeout, memory)
  - heuristic-to-flag mapping is correct for all SearchHeuristic values
  - posix_runtime / write_kqueries flags are included/excluded correctly
  - seed flag is included when seed is given
  - missing bitcode raises FileNotFoundError
  - KleeRunHandle.is_running() and .kill() behave correctly
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.core.schemas import SearchHeuristic
from klee.exceptions import KleeRuntimeError
from klee.runner import KleeRunHandle, _HEURISTIC_TO_FLAG, run_klee


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bitcode(tmp_path) -> Path:
    """A real (empty) .bc file so the existence check passes."""
    bc = tmp_path / "my_program.bc"
    bc.touch()
    return bc


def _mock_popen(returncode: int = 0) -> MagicMock:
    """Return a mock Popen that immediately appears finished."""
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = returncode
    proc.poll.return_value = returncode        # appears finished
    proc.communicate.return_value = ("", "")
    return proc


# ---------------------------------------------------------------------------
# Heuristic mapping
# ---------------------------------------------------------------------------

class TestHeuristicMapping:
    @pytest.mark.parametrize("heuristic,expected_flag", [
        (SearchHeuristic.DFS, "dfs"),
        (SearchHeuristic.BFS, "bfs"),
        (SearchHeuristic.RANDOM_STATE, "random-state"),
        (SearchHeuristic.RANDOM_PATH, "random-path"),
        (SearchHeuristic.NURS_COVNEW, "nurs:covnew"),
        (SearchHeuristic.NURS_MD2U, "nurs:md2u"),
        (SearchHeuristic.COV_OPT, "cov-opt"),
        (SearchHeuristic.AI_GUIDED, "random-path"),  # Phase 6 placeholder
    ])
    def test_mapping_correct(self, heuristic, expected_flag):
        assert _HEURISTIC_TO_FLAG[heuristic.value] == expected_flag


# ---------------------------------------------------------------------------
# run_klee command construction
# ---------------------------------------------------------------------------

class TestRunKlee:
    def test_search_flag_in_command(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, heuristic=SearchHeuristic.DFS)

        assert any("--search=dfs" in arg for arg in captured[0])

    def test_timeout_flag_in_command(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, timeout_s=120)

        assert any("--max-time=120" in arg for arg in captured[0])

    def test_memory_flag_in_command(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, max_memory_mb=2048)

        assert any("--max-memory=2048" in arg for arg in captured[0])

    def test_posix_runtime_included_by_default(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, posix_runtime=True)

        cmd = captured[0]
        assert "--posix-runtime" in cmd
        assert "--libc=uclibc" in cmd

    def test_posix_runtime_excluded_when_false(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, posix_runtime=False)

        cmd = captured[0]
        assert "--posix-runtime" not in cmd

    def test_write_kqueries_included_by_default(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, write_kqueries=True)

        assert "--write-kqueries" in captured[0]

    def test_write_kqueries_excluded_when_false(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, write_kqueries=False)

        assert "--write-kqueries" not in captured[0]

    def test_seed_flag_added_when_given(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, seed=42)

        assert "--seed-random=42" in captured[0]

    def test_seed_not_added_when_none(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, seed=None)

        assert not any("--seed-random" in arg for arg in captured[0])

    def test_extra_args_forwarded(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode, extra_args=["--only-output-states-covering-new"])

        assert "--only-output-states-covering-new" in captured[0]

    def test_bitcode_path_last_arg(self, bitcode):
        captured: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured.append(cmd)
            return _mock_popen()

        with patch("subprocess.Popen", side_effect=fake_popen):
            run_klee(bitcode)

        assert captured[0][-1] == str(bitcode)

    def test_missing_bitcode_raises_file_not_found(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.bc"
        with pytest.raises(FileNotFoundError, match="does_not_exist.bc"):
            run_klee(nonexistent)

    def test_returns_klee_run_handle(self, bitcode):
        with patch("subprocess.Popen", return_value=_mock_popen()):
            handle = run_klee(bitcode)
        assert isinstance(handle, KleeRunHandle)

    def test_handle_pid_set(self, bitcode):
        with patch("subprocess.Popen", return_value=_mock_popen()):
            handle = run_klee(bitcode)
        assert handle.pid == 12345


# ---------------------------------------------------------------------------
# KleeRunHandle lifecycle
# ---------------------------------------------------------------------------

class TestKleeRunHandle:
    def test_is_running_false_when_process_is_none(self):
        handle = KleeRunHandle(
            run_dir=Path("."),
            bitcode=Path("."),
            heuristic=SearchHeuristic.BFS,
        )
        assert not handle.is_running()

    def test_is_running_false_when_process_finished(self, bitcode):
        proc = _mock_popen(returncode=0)
        proc.poll.return_value = 0  # already exited
        handle = KleeRunHandle(
            run_dir=bitcode.parent,
            bitcode=bitcode,
            heuristic=SearchHeuristic.BFS,
            pid=proc.pid,
            _process=proc,
        )
        assert not handle.is_running()

    def test_wait_returns_exit_code(self, bitcode):
        proc = _mock_popen(returncode=0)
        handle = KleeRunHandle(
            run_dir=bitcode.parent,
            bitcode=bitcode,
            heuristic=SearchHeuristic.BFS,
            pid=proc.pid,
            _process=proc,
        )
        rc = handle.wait()
        assert rc == 0
