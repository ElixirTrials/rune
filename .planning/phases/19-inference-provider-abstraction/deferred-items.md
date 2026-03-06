# Deferred Items — Phase 19 Inference Provider Abstraction

## Pre-existing Ruff Violations (out of scope for 19-03)

Files created in Plan 19-01 have pre-existing ruff violations not caused by Plan 19-03 changes.

**Files:** `libs/inference/tests/test_ollama_provider.py`, `tests/test_provider.py`, `tests/test_vllm_provider.py`

**Violations:**
- `I001` — unsorted import blocks (fixable with `ruff check --fix`)
- `E501` — lines exceeding 88 character limit (long docstrings/assertions)
- `F401` — unused `pytest` import in `test_vllm_provider.py`

**Fix command:**
```bash
cd /Users/noahdolevelixir/Code/rune/libs/inference && uv run ruff check --fix tests/test_ollama_provider.py tests/test_provider.py tests/test_vllm_provider.py
```

Remaining E501 violations in long docstring lines may need manual shortening.
