---
phase: 19-inference-provider-abstraction
plan: 03
subsystem: infra
tags: [inference, vllm, ollama, factory, provider-pattern, python]

requires:
  - phase: 19-01
    provides: InferenceProvider ABC, VLLMProvider, OllamaProvider, GenerationResult, UnsupportedOperationError
provides:
  - factory.py with get_provider()/get_provider_for_step() and (type, url) instance cache
  - Updated __init__.py exporting 7 provider-based symbols
  - Deleted old stubs: adapter_loader.py, completion.py, vllm_client.py
  - Replaced old TDD stub tests with import smoke tests
affects:
  - agent-loop (imports from inference public API)
  - training-svc (uses InferenceProvider.generate())
  - phase 20+ (all downstream consumers now use get_provider factory)

tech-stack:
  added: []
  patterns:
    - Provider factory with (type, url) tuple cache key — env var reads inside function for monkeypatch testability
    - autouse conftest fixture for cache isolation between tests (clear_provider_cache)
    - Lazy imports inside if-blocks to avoid circular import risks

key-files:
  created:
    - libs/inference/src/inference/factory.py
    - libs/inference/tests/test_factory.py
  modified:
    - libs/inference/src/inference/__init__.py
    - libs/inference/tests/conftest.py
    - libs/inference/tests/test_adapter_loader.py
    - libs/inference/tests/test_completion.py
    - services/lora-server/tests/test_vllm_client.py
  deleted:
    - libs/inference/src/inference/adapter_loader.py
    - libs/inference/src/inference/completion.py
    - services/lora-server/vllm_client.py

key-decisions:
  - "os.environ.get() used instead of os.getenv() in factory.py — mypy cannot narrow os.getenv(key, str_default) to str even with a str default arg; os.environ.get has the same limitation but explicit type annotation resolves it"
  - "Env var reads placed inside get_provider() function body (not module level) — allows monkeypatch.setenv to work in tests; module-level reads are captured at import time"
  - "_clear_cache() helper retained in factory.py alongside conftest autouse fixture — belt-and-suspenders isolation; conftest fixture clears before and after each test"

patterns-established:
  - "Provider factory pattern: env var lookup + (type, url) cache key + lazy imports inside constructor branches"
  - "Autouse pytest fixture for shared mutable state isolation (provider cache)"

requirements-completed: [INF-04, INF-07]

duration: 5min
completed: 2026-03-05
---

# Phase 19 Plan 03: Factory + Cleanup Summary

**Provider factory with (type, url) instance cache and clean break from NotImplementedError stubs — inference library exports 7 provider-based symbols via unified public API**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-05T14:29:08Z
- **Completed:** 2026-03-05T14:33:48Z
- **Tasks:** 2
- **Files modified:** 10 (3 created, 4 modified, 3 deleted)

## Accomplishments
- factory.py: `get_provider()` and `get_provider_for_step()` with (provider_type, base_url) instance caching; env var reads inside function body for test monkeypatching
- `__init__.py` updated to export 7 provider-based symbols: InferenceProvider, GenerationResult, VLLMProvider, OllamaProvider, get_provider, get_provider_for_step, UnsupportedOperationError
- Deleted 3 old NotImplementedError stubs: adapter_loader.py, completion.py, vllm_client.py
- Replaced old TDD stub tests with import smoke tests; conftest.py now provides autouse cache-clearing fixture

## Task Commits

1. **Task 1: Provider factory with instance cache (TDD)** — `c528106` (feat)
2. **Task 2: Update __init__.py, delete stubs, replace tests** — `f482162` (feat)

## Files Created/Modified

- `libs/inference/src/inference/factory.py` — Provider factory with (type, url) cache, get_provider(), get_provider_for_step()
- `libs/inference/src/inference/__init__.py` — Updated to export 7 provider-based symbols
- `libs/inference/tests/test_factory.py` — 9 tests: type dispatch, caching, env var default, ValueError
- `libs/inference/tests/conftest.py` — Replaced mock_vllm_client with clear_provider_cache autouse fixture
- `libs/inference/tests/test_adapter_loader.py` — Replaced with import smoke test
- `libs/inference/tests/test_completion.py` — Replaced with import smoke test
- `services/lora-server/tests/test_vllm_client.py` — Replaced with VLLMProvider importability smoke test
- *(deleted)* `libs/inference/src/inference/adapter_loader.py`
- *(deleted)* `libs/inference/src/inference/completion.py`
- *(deleted)* `services/lora-server/vllm_client.py`

## Decisions Made

- Used `os.environ.get()` instead of `os.getenv()` in factory.py — both have the same mypy narrowing limitation but explicit `str` type annotation on the result resolves mypy's union-attr errors cleanly
- Env var reads placed inside `get_provider()` function body (not at module level) — module-level reads are captured at import time before monkeypatch.setenv() takes effect; function-body reads re-evaluate on each call
- `_clear_cache()` helper retained in factory.py alongside the conftest autouse fixture as belt-and-suspenders isolation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Env var read at module level prevented monkeypatch testability**
- **Found during:** Task 1 (factory TDD GREEN phase)
- **Issue:** `INFERENCE_PROVIDER = os.getenv(...)` at module level is evaluated at import time; monkeypatch.setenv() after import has no effect. Test `test_get_provider_uses_env_var_default` failed.
- **Fix:** Moved env var reads inside `get_provider()` function body using `os.environ.get(key, default)` with explicit type annotations
- **Files modified:** `libs/inference/src/inference/factory.py`
- **Verification:** 9/9 factory tests pass including env var default tests
- **Committed in:** `f482162` (Task 2 commit with mypy fix)

**2. [Rule 1 - Bug] mypy type errors from os.getenv() return type**
- **Found during:** Task 2 verification (mypy run)
- **Issue:** `os.getenv(key, str_default)` returns `str | None` in mypy's type stubs even when a str default is provided; caused union-attr and dict index type errors in factory.py
- **Fix:** Used `os.environ.get()` with explicit `str` variable annotations and extracted resolved values to typed locals before use
- **Files modified:** `libs/inference/src/inference/factory.py`
- **Verification:** `uv run mypy src/` reports "Success: no issues found in 6 source files"
- **Committed in:** `f482162`

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs)
**Impact on plan:** Both fixes necessary for correctness and type safety. No scope creep.

## Issues Encountered

- Pre-existing ruff violations (E501, I001, F401) in test files from Phase 19-01 are out of scope. Logged to `deferred-items.md` in phase directory. Files created/modified in this plan are ruff-clean.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Inference library public API complete: `from inference import InferenceProvider, VLLMProvider, OllamaProvider, get_provider, get_provider_for_step, GenerationResult, UnsupportedOperationError`
- All 34 inference tests + 3 lora-server tests passing
- Agent loop and training-svc can now import from the new provider-based API
- Ready for Phase 20 (agent loop integration)

---
*Phase: 19-inference-provider-abstraction*
*Completed: 2026-03-05*

## Self-Check: PASSED

- factory.py: FOUND
- test_factory.py: FOUND
- adapter_loader.py: DELETED (confirmed)
- completion.py: DELETED (confirmed)
- vllm_client.py: DELETED (confirmed)
- Commit c528106: FOUND
- Commit f482162: FOUND
