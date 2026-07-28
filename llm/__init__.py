"""
llm — Phase 5.

Responsibility: prompt an LLM (local CodeLlama/DeepSeek-Coder/Qwen-Coder, or
a hosted provider) with a C function and get back a schema-validated
:class:`backend.core.schemas.BranchPrediction`.

Planned public interface:

    class LLMSemanticAnalyzer:
        def predict_branch(self, source_code: str, candidate_branches: list[int]) -> BranchPrediction: ...
        def predict_batch(self, items: list[tuple[str, list[int]]]) -> list[BranchPrediction]: ...

Design notes:
- Responses are cached on (source_code_hash, model_name, prompt_version) in
  `results/llm_cache/` so repeated experiments don't re-spend tokens/compute.
- Malformed / non-JSON responses are retried once with a stricter
  "return ONLY JSON" instruction, then raise `LLMResponseParseError` — the
  RL/ML layers must be able to fall back to heuristic priority if the LLM is
  unavailable, since RQ2 explicitly studies how much the LLM signal helps
  (an ablation needs a "no-LLM" arm).
"""
