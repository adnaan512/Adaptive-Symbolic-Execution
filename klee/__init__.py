"""
klee — Phase 2.

Responsibility: compile C sources to LLVM bitcode, run KLEE under a given
search heuristic (or the AI-guided searcher, once it exists), and parse
KLEE's `.klee-stats`, `run.stats`, and per-state ktest/log output into the
raw dicts that `feature_extractor.extract_features` consumes.

Planned public interface:

    def compile_to_bitcode(c_source: Path, flags: list[str]) -> Path
    def run_klee(bitcode: Path, heuristic: SearchHeuristic, timeout_s: int) -> KleeRunHandle
    def parse_run_stats(run_dir: Path) -> list[dict]   # -> feature_extractor
    def stream_coverage(run_dir: Path) -> Iterator[CoverageSnapshot]

For the AI-guided searcher specifically, Phase 6 will need KLEE's
`--search=random-path` mode combined with an external state-selection hook;
if KLEE's C++ side doesn't expose a clean per-step hook in the pinned
version, the fallback design is to run KLEE with `-only-output-states-covering-new`
and periodically checkpoint+fork via KLEE's `ktest` replay to approximate
external control. This tradeoff is documented in `docs/design/klee_hook.md`
once Phase 2 confirms which approach the installed KLEE version supports.
"""
