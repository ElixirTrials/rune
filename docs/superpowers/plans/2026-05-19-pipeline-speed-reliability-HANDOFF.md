# Pipeline Speed & Reliability — Implementation Handoff

**Date:** 2026-05-19
**Branch:** `feat/training-speed-opts`
**Base SHA:** `ce87192` (before any implementation work)
**Plan:** `docs/superpowers/plans/2026-05-19-pipeline-speed-reliability.md`
**Spec:** `docs/superpowers/specs/2026-05-19-pipeline-speed-reliability-design.md`

## Status Summary

| Task | Status | Commit | Notes |
|------|--------|--------|-------|
| **Task 1: Eager Adapter Unload (P1)** | DONE | `1d179d7` | `_eager_unload()` at all 9 `run_iteration()` call sites |
| **Task 2: Decompose Prompt Improvement (P2)** | DONE | `6c23a35` | CoT suppression, few-shot examples, negative examples in 3 templates |
| **Task 3: Task-Complexity Gating (P3)** | DONE | `be71bb3` | `_should_skip_decompose()` + `DecomposeConfig` + wiring in pipeline |
| **Task 4: Thinking Token Budget (P4)** | NOT STARTED | — | Add `thinking_budget` param to provider ABC + all implementations |
| **Task 5: Runner-Managed Continuation (P5)** | NOT STARTED | — | Extract `_run_continuation_loop()` + wire into code phase |
| **Integration Verification** | BLOCKED | — | Blocked on Tasks 4 & 5 |

## What's Done (3/5 tasks)

### Task 1 — Eager Adapter Unload
- Added `_eager_unload()` helper at `scripts/rune_runner.py:554`
- Calls `provider.unload_adapter()` + `torch.cuda.empty_cache()` with INFRA-05 deferred imports
- Added after all 9 `run_iteration()` call sites (decompose, cycle-fix, plan loop, diagnose-in-retry, code, integrate, phase-5 diagnose, phase-5 repair, phase-5 reintegrate)
- `_cleanup_phase_adapters()` preserved as safety net
- Tests: `tests/test_eager_unload.py` (2 tests)

### Task 2 — Decompose Prompt Improvement
- `decompose.j2`: Added "Do NOT include your chain-of-thought", 3 examples (Web API, single function, merge lists), BAD/negative example, anti-patterns section
- `prompt_decompose.j2`: Added "No preamble, no analysis, no reasoning", example output, BAD example
- `prompt_decompose_concise.j2`: Added "Output ONLY", no-preamble directive, BAD inline example
- Tests: `libs/shared/tests/test_decompose_templates.py` (6 tests)

### Task 3 — Task-Complexity Gating
- `DecomposeConfig(skip_threshold=200)` frozen dataclass added to `pipeline_config.py`
- Wired into `PipelineConfig`, `_from_dict()`, and `override()` dotted-key support
- `_should_skip_decompose()` in `rune_runner.py`: checks word count < threshold AND single-function signals
- Skip logic in `run_phased_pipeline()`: injects synthetic single `implementation` subtask, gates evolution loop
- Tests: `libs/shared/tests/test_pipeline_config.py` (3 new tests), `tests/test_skip_decompose.py` (5 tests)

## What's Left (2/5 tasks)

### Task 4 — Thinking Token Budget (P4)
**Files to modify:**
- `libs/inference/src/inference/provider.py` — add `thinking_budget: int = 0` to ABC `generate()`
- `libs/inference/src/inference/transformers_provider.py` — `effective_max = max_tokens + thinking_budget`, fix `finish_reason` logic
- `libs/inference/src/inference/vllm_provider.py` — add param to signature
- `libs/inference/src/inference/ollama_provider.py` — add param to signature
- `libs/inference/src/inference/llamacpp_provider.py` — add param to signature
- `libs/shared/src/shared/pipeline_config.py` — add `thinking_budget: int = 512` to `GenerationConfig`
- `services/rune-agent/src/rune_agent/nodes.py` — `thinking_budget = 512 if enable_thinking else 0`
- Create `libs/inference/tests/test_thinking_budget.py`

**Key detail:** In `transformers_provider.py`, `finish_reason` must only fire "length" when *response* tokens (post-thinking-strip) exceed `max_tokens`, not when total tokens (including `<think>`) exceed.

### Task 5 — Runner-Managed Continuation (P5)
**Files to modify:**
- `scripts/rune_runner.py` — add `_run_continuation_loop()`, wire into code phase, remove `is_truncated` branch
- Create `tests/test_continuation.py`

**Key design decisions:**
- Continuation managed in runner, NOT via new LangGraph nodes (because `run_iteration()` resets state from scratch every call)
- `_run_continuation_loop()` is an extracted function with injectable dependencies (run_iteration_fn, render_trajectory_fn, etc.) for testability
- Tests use `AsyncMock` for the injected callables — 5 async tests covering: no-continuation, single-continuation, cap-at-3, accumulated-code-passed-to-trajectory, unload-called-each-time
- Three changes to code phase: (A) wire `_run_continuation_loop()` after `run_iteration()`, (B) remove `is_truncated` branch from retry logic, (C) remove old concat block
- Known limitation: after continuation, `tests_passed` reflects only last fragment — outer retry loop handles this correctly

### Integration Verification
After Tasks 4 & 5:
1. `uv run pytest -x -v` (776+ tests)
2. `uv run ruff check libs/ services/ scripts/`
3. `uv run mypy libs/ services/`
4. Compile check verifying `thinking_budget` in provider signature and `DecomposeConfig` defaults

## Execution Approach

Used **subagent-driven-development**: fresh subagent per task, spec compliance review after each. Task 1 had one spec fix (plan phase unload missing `empty_cache`). Tasks 2 and 3 were dispatched in parallel (independent files).

To resume: continue with subagent-driven-development from Task 4, or use inline execution.

## Test State

All 63 targeted tests pass (eager unload, decompose templates, pipeline config, skip decompose, rune-agent unit tests). No regressions introduced.
