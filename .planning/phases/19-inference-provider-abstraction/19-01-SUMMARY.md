---
phase: 19-inference-provider-abstraction
plan: 01
subsystem: inference
tags: [python, abc, dataclass, asyncio, openai, httpx, vllm, ollama, lora]

# Dependency graph
requires:
  - phase: 18-adapter-registry
    provides: Registry infrastructure that adapter_id strings reference

provides:
  - InferenceProvider ABC with 4 async abstract methods in libs/inference/src/inference/provider.py
  - GenerationResult dataclass (text, model, adapter_id, token_count, finish_reason)
  - UnsupportedOperationError exception in libs/inference/src/inference/exceptions.py
  - VLLMProvider with full LoRA hot-loading via AsyncOpenAI + httpx in vllm_provider.py
  - OllamaProvider for base model inference via AsyncOpenAI compat endpoint in ollama_provider.py

affects:
  - 19-02 (factory + per-step config consuming these providers)
  - 20-agent-loop (InferenceProvider.generate() is the core generation call)

# Tech tracking
tech-stack:
  added: [pytest-asyncio, asyncio_mode=auto in inference lib pyproject.toml]
  patterns:
    - InferenceProvider ABC with async abstract methods as the provider contract
    - VLLMProvider routes adapter_id as model param to vLLM LoRA routing mechanism
    - Internal _loaded_adapters set for reliable adapter tracking (vLLM bug #11761 workaround)
    - OllamaProvider uses AsyncOpenAI with api_key="ollama" (non-empty required, value ignored)
    - UnsupportedOperationError raised explicitly for unsupported ops (no silent no-ops)
    - VLLM_BASE_URL defaults to port 8100 (not 8000 — api-service is on 8000)

key-files:
  created:
    - libs/inference/src/inference/provider.py
    - libs/inference/src/inference/exceptions.py
    - libs/inference/src/inference/vllm_provider.py
    - libs/inference/src/inference/ollama_provider.py
    - libs/inference/tests/test_provider.py
    - libs/inference/tests/test_vllm_provider.py
    - libs/inference/tests/test_ollama_provider.py
  modified:
    - libs/inference/pyproject.toml

key-decisions:
  - "asyncio_mode=auto added to inference lib pyproject.toml — lib-scoped pytest defaulted to strict mode despite root having auto"
  - "VLLMProvider.generate() passes adapter_id as model param — vLLM identifies loaded LoRA adapters by lora_name in the model field"
  - "Internal _loaded_adapters set used instead of querying vLLM endpoint — workaround for vLLM bug #11761 (unreliable list)"
  - "VLLM_BASE_URL default updated to http://localhost:8100/v1 — port 8100 locked in phase 19 context; api-service owns 8000"
  - "OllamaProvider.generate() always returns adapter_id=None — Ollama has no adapter concept"

patterns-established:
  - "Provider ABC pattern: all methods async, structured return type, explicit UnsupportedOperationError for unsupported ops"
  - "TDD flow maintained: failing test import error confirmed RED, then GREEN on all tests"

requirements-completed: [INF-01, INF-02, INF-03, INF-05, INF-06]

# Metrics
duration: 18min
completed: 2026-03-05
---

# Phase 19 Plan 01: InferenceProvider ABC, VLLMProvider, OllamaProvider Summary

**InferenceProvider ABC with GenerationResult dataclass, VLLMProvider (AsyncOpenAI + httpx LoRA hot-loading), and OllamaProvider (OpenAI-compat, raises UnsupportedOperationError for adapters) — 23 tests, mypy strict, ruff clean**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-03-05T12:00:00Z
- **Completed:** 2026-03-05T12:18:00Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- InferenceProvider ABC with 4 async abstract methods (generate, load_adapter, unload_adapter, list_adapters) providing a provider-agnostic contract for the agent loop
- VLLMProvider fully implementing all 4 methods: AsyncOpenAI for generation, httpx for vLLM proprietary LoRA endpoints, internal adapter tracking set for reliability
- OllamaProvider implementing generate via Ollama's OpenAI-compat endpoint, raising UnsupportedOperationError for adapter ops with descriptive messages
- 23 tests total (6 + 9 + 8) covering ABC constraints, dataclass construction, generate routing, adapter lifecycle, async verification

## Task Commits

Each task was committed atomically:

1. **Task 1: Create InferenceProvider ABC, GenerationResult, and UnsupportedOperationError** - `99ac7be` (feat)
2. **Task 2: Implement VLLMProvider with full LoRA hot-loading** - `1613667` (feat)
3. **Task 3: Implement OllamaProvider with UnsupportedOperationError for adapters** - `c70ff90` (feat)

_Note: TDD tasks have RED (import error) then GREEN (all pass) flow within each commit_

## Files Created/Modified

- `libs/inference/src/inference/provider.py` - InferenceProvider ABC and GenerationResult dataclass
- `libs/inference/src/inference/exceptions.py` - UnsupportedOperationError exception
- `libs/inference/src/inference/vllm_provider.py` - VLLMProvider with AsyncOpenAI + httpx LoRA management
- `libs/inference/src/inference/ollama_provider.py` - OllamaProvider via Ollama's OpenAI-compat endpoint
- `libs/inference/tests/test_provider.py` - 6 tests for ABC and dataclass behavior
- `libs/inference/tests/test_vllm_provider.py` - 9 tests for VLLMProvider generate and adapter lifecycle
- `libs/inference/tests/test_ollama_provider.py` - 8 tests for OllamaProvider behavior
- `libs/inference/pyproject.toml` - Added pytest-asyncio dev dep and asyncio_mode=auto

## Decisions Made

- asyncio_mode=auto added to the inference lib's own pyproject.toml — the lib-scoped pytest configuration overrides the root, so tests were running in strict mode and failing on async test methods
- VLLMProvider uses adapter_id as the model parameter when adapter is set — this is vLLM's documented mechanism for routing to loaded LoRA adapters (lora_name in model field)
- VLLM_BASE_URL default changed to port 8100 per locked decision (api-service owns port 8000)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added asyncio_mode=auto and pytest-asyncio to inference lib pyproject.toml**
- **Found during:** Task 2 (VLLMProvider implementation)
- **Issue:** The inference lib's own pyproject.toml has `[tool.pytest.ini_options]` which overrides the root config. It was missing `asyncio_mode = "auto"`, so pytest ran in strict mode and all async test methods failed with "async def functions are not natively supported"
- **Fix:** Added `asyncio_mode = "auto"` to `[tool.pytest.ini_options]` in `libs/inference/pyproject.toml` and added `pytest-asyncio>=1.0.0` to dev dependencies
- **Files modified:** `libs/inference/pyproject.toml`
- **Verification:** All 9 vllm_provider tests pass after fix; asyncio mode shows AUTO in test output
- **Committed in:** `1613667` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required for the async test suite to run. No scope creep — the fix directly unblocked planned async test execution.

## Issues Encountered

- Coverage data collision (INTERNALERROR in teardown) when running individual test files due to lib-scoped `--cov-branch` conflicting with root coverage data. Resolved by using `--no-cov` flag for isolated test runs. Final verification ran all 3 files together successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- InferenceProvider ABC is the contract; Phase 19 Plan 02 (factory + per-step config) can now implement the factory that creates VLLMProvider/OllamaProvider instances by name
- Phase 20 (agent loop) can consume InferenceProvider.generate() without knowing which backend is in use
- Both providers mypy strict and ruff clean — no technical debt

---
*Phase: 19-inference-provider-abstraction*
*Completed: 2026-03-05*
