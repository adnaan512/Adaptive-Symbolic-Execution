# Phase 3 Notes — Feature Extraction

## What this phase produced

### New modules

| File | Purpose |
|------|---------|
| `feature_extractor/exceptions.py` | Contains `FeatureExtractionError` and `ValidationError` |
| `feature_extractor/extractor.py` | Contains `extract_features()` and `extract_batch()` for parsing raw dictionaries into `ExecutionStateFeatures`. |
| `feature_extractor/dataset.py` | Contains CSV and JSONL persistence logic (`save_dataset_csv`, `load_dataset_csv`, etc.) for building structured ML/RL training data. |

### New tests

| File | Tests |
|------|-------|
| `tests/test_feature_extractor.py` | 10+ unit tests covering valid parsing, type coercions, missing UUID generation, default fallbacks, schema validations (e.g. coverage bounds), and full CSV/JSONL round-tripping. |

---

## Architecture Context

The purpose of Phase 3 is to isolate **data wrangling** from downstream ML (Phase 4) and RL (Phase 6) algorithms.

In the real symbolic execution environment, KLEE will eventually produce state profiles. Rather than having our neural networks or Scikit-Learn models parse raw JSON directly (which causes silent failure if schemas drift), everything must pass through:

1. **`extract_features(raw_dict)`**
2. **`ExecutionStateFeatures` (Pydantic Model)**

If a field is missing, `extractor.py` provides sensible defaults (like 0 depth or 0 instructions) or computes it. If an invalid value is passed (e.g. branch coverage > 1.0), it strictly raises a `ValidationError`.

### Dataset Export
The `dataset.py` module exposes structured data saving.
ML pipelines (e.g. Pandas, PyTorch DataLoaders) heavily favor CSV or JSONL over single giant JSON files.
Our dataset module uses `.model_dump()` and `.model_validate_json()` to ensure that when we read a dataset back from disk, it is exactly the typed Pydantic models we expect.

For model inputs, `ExecutionStateFeatures.to_vector()` instantly produces the canonical 14-dimensional float array required by PyTorch or Scikit-Learn.

---

## Verification (Definition of Done)

1. The test suite covers missing fields, invalid bounds, string-to-int type casting, and serialization.
2. `pytest tests/test_feature_extractor.py` completes successfully.
3. The module has exactly zero dependencies on `klee`, `ml`, or `rl` directories, strictly adhering to Clean Architecture constraints.
