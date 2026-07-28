# Phase 5 Notes — LLM Semantic Analyzer

## What this phase produced

### New modules

| File | Purpose |
|------|---------|
| `llm/prompts.py` | Prompt engineering templates enforcing strict JSON output for Semantic Branch Analysis. |
| `llm/cache.py` | A deterministic hashing mechanism caching LLM responses to disk. Prevents burning API tokens on repeated symbolic execution loops. |
| `llm/analyzer.py` | Contains `LLMSemanticAnalyzer`, which manages the API connection (OpenAI or Local Ollama via `base_url`), implements parsing, and handles auto-retries when the LLM outputs malformed text. |
| `llm/exceptions.py` | `LLMConnectionError` and `LLMResponseParseError`. |

### New tests

| File | Tests |
|------|-------|
| `tests/test_llm_analyzer.py` | 7 unit tests mocking the LLM API to rigorously verify cache hits, JSON parsing, Pydantic schema validation, and the automated retry logic on conversational/bad outputs. |

---

## Architecture Context

Phase 5 introduces the Semantic Heuristic layer. Instead of purely mathematical features (Phase 3/4), this phase uses Large Language Models to evaluate the raw C code.

### The Problem with LLMs in Automated Loops
LLMs frequently hallucinate formatting (e.g. wrapping JSON in markdown ` ```json ` blocks) or hallucinate conversational text ("Sure! The best branch is..."). 

### Our Solution
1. **Pydantic Validation**: Every API string is immediately checked against `BranchPrediction`.
2. **Auto-Retry**: If parsing fails, the analyzer catches the `LLMResponseParseError`, appends the bad response to the conversation history, issues a strict `RETRY_PROMPT`, and calls the API one more time.
3. **Graceful Degradation**: If the second attempt fails, it raises the exception. Downstream callers (in RL or Orchestration layers) are expected to catch this and fall back to traditional heuristics (like `nurs:covnew`). This is essential for robust automated runs over thousands of test subjects.
4. **Mock Mode**: `LLMSemanticAnalyzer(mock_mode=True)` allows the entire system test suite to run in CI/CD without exposing OpenAI API keys.

---

## Verification (Definition of Done)

1. The test suite correctly bypasses the API call when a cache hit occurs.
2. The analyzer successfully parses valid JSON into `BranchPrediction` schemas and strictly bounds confidence values between `[0.0, 1.0]`.
3. The LLM connection supports swapping `base_url` to support local hosted open-source models (Llama-3, DeepSeek-Coder, etc.) natively out of the box.
