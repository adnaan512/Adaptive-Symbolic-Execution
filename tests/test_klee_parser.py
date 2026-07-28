"""
Phase 2 tests for klee.parser.

All tests use synthetic in-memory SQLite databases and temporary
directories so they pass without a KLEE installation.

Covered:
  - parse_run_stats: reads all rows from a synthetic run.stats
  - stream_coverage: produces correct CoverageSnapshot objects
  - _branch_coverage_from_row: formula correctness
  - _instr_coverage_from_row: formula correctness with/without ICovNew
  - build_run_result: assembles RunResult correctly from synthetic data
  - resolve_run_dir: finds klee-out-<N> directory automatically
  - parse_messages / parse_warnings: file-reading helpers
  - count_ktest_files / count_error_files: glob counting
  - save_run_result_json: round-trip serialisation
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from backend.core.schemas import RunResult, SearchHeuristic
from klee.exceptions import KleeParseError
from klee.parser import (
    _branch_coverage_from_row,
    _instr_coverage_from_row,
    build_run_result,
    count_error_files,
    count_ktest_files,
    parse_messages,
    parse_run_stats,
    parse_warnings,
    resolve_run_dir,
    save_run_result_json,
    stream_coverage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_stats_db(directory: Path, rows: list[dict]) -> Path:
    """Create a synthetic run.stats SQLite database in *directory*."""
    db_path = directory / "run.stats"
    conn = sqlite3.connect(str(db_path))
    # Build columns from the union of all row keys
    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    cols = ", ".join(f"{k} REAL" for k in all_keys)
    conn.execute(f"CREATE TABLE stats ({cols})")
    for row in rows:
        placeholders = ", ".join("?" for _ in all_keys)
        values = [row.get(k) for k in all_keys]
        conn.execute(f"INSERT INTO stats VALUES ({placeholders})", values)
    conn.commit()
    conn.close()
    return db_path


def _sample_rows() -> list[dict]:
    """Return three synthetic stats rows mimicking a short KLEE run."""
    return [
        {
            "WallTime": 1.0,
            "Instructions": 100,
            "FullBranches": 4,
            "PartialBranches": 2,
            "NumStates": 3,
            "NumQueries": 10,
            "QueryTime": 500_000.0,  # microseconds
            "CoveredInstructions": 80,
            "UncoveredInstructions": 20,
            "MallocUsage": 1024 * 1024 * 50,  # 50 MiB
            "ICovNew": None,
            "BCovNew": None,
        },
        {
            "WallTime": 2.0,
            "Instructions": 250,
            "FullBranches": 8,
            "PartialBranches": 2,
            "NumStates": 5,
            "NumQueries": 25,
            "QueryTime": 1_200_000.0,
            "CoveredInstructions": 140,
            "UncoveredInstructions": 10,
            "MallocUsage": 1024 * 1024 * 60,
            "ICovNew": None,
            "BCovNew": None,
        },
        {
            "WallTime": 3.0,
            "Instructions": 400,
            "FullBranches": 12,
            "PartialBranches": 2,
            "NumStates": 4,
            "NumQueries": 40,
            "QueryTime": 2_000_000.0,
            "CoveredInstructions": 150,
            "UncoveredInstructions": 0,
            "MallocUsage": 1024 * 1024 * 65,
            "ICovNew": None,
            "BCovNew": None,
        },
    ]


# ---------------------------------------------------------------------------
# Branch / instruction coverage formula tests
# ---------------------------------------------------------------------------

class TestCoverageFormulas:
    def test_branch_coverage_full_only(self):
        row = {"FullBranches": 10, "PartialBranches": 0}
        assert _branch_coverage_from_row(row) == pytest.approx(1.0)

    def test_branch_coverage_mixed(self):
        row = {"FullBranches": 8, "PartialBranches": 2}
        assert _branch_coverage_from_row(row) == pytest.approx(0.8)

    def test_branch_coverage_zero(self):
        row = {"FullBranches": 0, "PartialBranches": 0}
        assert _branch_coverage_from_row(row) == pytest.approx(0.0)

    def test_branch_coverage_prefers_bcovnew(self):
        row = {"FullBranches": 0, "PartialBranches": 10, "BCovNew": 0.75}
        assert _branch_coverage_from_row(row) == pytest.approx(0.75)

    def test_instr_coverage_from_counts(self):
        row = {"CoveredInstructions": 80, "UncoveredInstructions": 20}
        assert _instr_coverage_from_row(row) == pytest.approx(0.8)

    def test_instr_coverage_prefers_icovnew(self):
        row = {"CoveredInstructions": 0, "UncoveredInstructions": 100, "ICovNew": 0.9}
        assert _instr_coverage_from_row(row) == pytest.approx(0.9)

    def test_instr_coverage_zero(self):
        row = {"CoveredInstructions": 0, "UncoveredInstructions": 0}
        assert _instr_coverage_from_row(row) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# parse_run_stats
# ---------------------------------------------------------------------------

class TestParseRunStats:
    def test_reads_all_rows(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows())
        rows = parse_run_stats(klee_out)
        assert len(rows) == 3

    def test_rows_ordered_by_walltime(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        rows_in = _sample_rows()
        # Insert in reverse order
        _make_run_stats_db(klee_out, list(reversed(rows_in)))
        rows_out = parse_run_stats(klee_out)
        wall_times = [r["WallTime"] for r in rows_out]
        assert wall_times == sorted(wall_times)

    def test_missing_run_stats_raises(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        with pytest.raises(KleeParseError, match="run.stats not found"):
            parse_run_stats(klee_out)

    def test_no_klee_out_dir_raises(self, tmp_path):
        with pytest.raises(KleeParseError, match="No klee-out"):
            parse_run_stats(tmp_path)


# ---------------------------------------------------------------------------
# stream_coverage
# ---------------------------------------------------------------------------

class TestStreamCoverage:
    def test_yields_correct_count(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows())
        snapshots = list(stream_coverage(klee_out))
        assert len(snapshots) == 3

    def test_snapshot_branch_coverage_values(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows())
        snapshots = list(stream_coverage(klee_out))
        # Row 0: 4/(4+2) = 0.666...
        assert snapshots[0].branch_coverage == pytest.approx(4 / 6)
        # Row 2: 12/(12+2) ≈ 0.857
        assert snapshots[2].branch_coverage == pytest.approx(12 / 14)

    def test_snapshot_elapsed_seconds(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows())
        snapshots = list(stream_coverage(klee_out))
        assert snapshots[0].elapsed_seconds == pytest.approx(1.0)
        assert snapshots[1].elapsed_seconds == pytest.approx(2.0)
        assert snapshots[2].elapsed_seconds == pytest.approx(3.0)

    def test_snapshot_num_states(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows())
        snapshots = list(stream_coverage(klee_out))
        assert snapshots[0].num_states == 3
        assert snapshots[1].num_states == 5


# ---------------------------------------------------------------------------
# resolve_run_dir
# ---------------------------------------------------------------------------

class TestResolveRunDir:
    def test_resolves_parent_to_newest_klee_out(self, tmp_path):
        (tmp_path / "klee-out-0").mkdir()
        import time
        time.sleep(0.01)
        newest = tmp_path / "klee-out-1"
        newest.mkdir()
        # Place run.stats in newest
        _make_run_stats_db(newest, _sample_rows()[:1])
        resolved = resolve_run_dir(tmp_path)
        assert resolved == newest.resolve()

    def test_direct_klee_out_dir_returned_as_is(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows()[:1])
        resolved = resolve_run_dir(klee_out)
        assert resolved == klee_out.resolve()


# ---------------------------------------------------------------------------
# build_run_result
# ---------------------------------------------------------------------------

class TestBuildRunResult:
    def _setup(self, tmp_path) -> Path:
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows())
        # Fake ktest and error files
        (klee_out / "test000001.ktest").touch()
        (klee_out / "test000002.ktest").touch()
        (klee_out / "test000003.err").touch()
        return klee_out

    def test_returns_run_result_instance(self, tmp_path):
        klee_out = self._setup(tmp_path)
        result = build_run_result(
            run_dir=klee_out,
            program_name="simple_branch",
            heuristic=SearchHeuristic.BFS,
            wall_time_seconds=3.5,
        )
        assert isinstance(result, RunResult)

    def test_program_name_and_heuristic(self, tmp_path):
        klee_out = self._setup(tmp_path)
        result = build_run_result(
            run_dir=klee_out,
            program_name="simple_branch",
            heuristic=SearchHeuristic.DFS,
            wall_time_seconds=3.5,
        )
        assert result.program_name == "simple_branch"
        assert result.heuristic == SearchHeuristic.DFS

    def test_unique_paths_counts_ktest_files(self, tmp_path):
        klee_out = self._setup(tmp_path)
        result = build_run_result(
            run_dir=klee_out,
            program_name="p",
            heuristic=SearchHeuristic.BFS,
            wall_time_seconds=3.5,
        )
        assert result.unique_paths == 2

    def test_unique_bugs_counts_err_files(self, tmp_path):
        klee_out = self._setup(tmp_path)
        result = build_run_result(
            run_dir=klee_out,
            program_name="p",
            heuristic=SearchHeuristic.BFS,
            wall_time_seconds=3.5,
        )
        assert result.unique_bugs == 1

    def test_coverage_over_time_length(self, tmp_path):
        klee_out = self._setup(tmp_path)
        result = build_run_result(
            run_dir=klee_out,
            program_name="p",
            heuristic=SearchHeuristic.BFS,
            wall_time_seconds=3.5,
        )
        assert len(result.coverage_over_time) == 3

    def test_branch_coverage_from_last_row(self, tmp_path):
        klee_out = self._setup(tmp_path)
        result = build_run_result(
            run_dir=klee_out,
            program_name="p",
            heuristic=SearchHeuristic.BFS,
            wall_time_seconds=3.5,
        )
        # Last row: 12/(12+2) ≈ 0.857
        assert result.branch_coverage == pytest.approx(12 / 14, rel=1e-3)

    def test_memory_usage_converted_to_mib(self, tmp_path):
        klee_out = self._setup(tmp_path)
        result = build_run_result(
            run_dir=klee_out,
            program_name="p",
            heuristic=SearchHeuristic.BFS,
            wall_time_seconds=3.5,
        )
        # Last row MallocUsage = 1024*1024*65 bytes = 65 MiB
        assert result.memory_usage_mb == pytest.approx(65.0, rel=1e-3)

    def test_empty_run_returns_zeros(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, [])
        result = build_run_result(
            run_dir=klee_out,
            program_name="p",
            heuristic=SearchHeuristic.BFS,
            wall_time_seconds=0.0,
        )
        assert result.branch_coverage == pytest.approx(0.0)
        assert result.unique_paths == 0


# ---------------------------------------------------------------------------
# Helper file-reading functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_parse_messages_returns_lines(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows()[:1])
        (klee_out / "messages.txt").write_text("hello\nworld\n", encoding="utf-8")
        msgs = parse_messages(klee_out)
        assert msgs == ["hello", "world"]

    def test_parse_messages_missing_file(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows()[:1])
        assert parse_messages(klee_out) == []

    def test_parse_warnings_returns_lines(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows()[:1])
        (klee_out / "warnings.txt").write_text("warn1\n", encoding="utf-8")
        assert parse_warnings(klee_out) == ["warn1"]

    def test_count_ktest_files(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows()[:1])
        for i in range(5):
            (klee_out / f"test{i:06d}.ktest").touch()
        assert count_ktest_files(klee_out) == 5

    def test_count_error_files(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows()[:1])
        (klee_out / "test000001.ptr.err").touch()
        (klee_out / "test000002.assert.err").touch()
        assert count_error_files(klee_out) == 2


# ---------------------------------------------------------------------------
# JSON serialisation round-trip
# ---------------------------------------------------------------------------

class TestSaveRunResultJson:
    def test_file_created(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows())
        result = build_run_result(
            run_dir=klee_out,
            program_name="p",
            heuristic=SearchHeuristic.BFS,
            wall_time_seconds=3.0,
        )
        out_path = tmp_path / "result.json"
        save_run_result_json(result, out_path)
        assert out_path.exists()

    def test_json_round_trip(self, tmp_path):
        klee_out = tmp_path / "klee-out-0"
        klee_out.mkdir()
        _make_run_stats_db(klee_out, _sample_rows())
        result = build_run_result(
            run_dir=klee_out,
            program_name="simple_branch",
            heuristic=SearchHeuristic.DFS,
            wall_time_seconds=3.5,
        )
        out_path = tmp_path / "result.json"
        save_run_result_json(result, out_path)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["program_name"] == "simple_branch"
        assert data["heuristic"] == "dfs"
        assert "coverage_over_time" in data
        assert len(data["coverage_over_time"]) == 3
