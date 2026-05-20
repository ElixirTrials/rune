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
| **Task 4: Thinking Token Budget (P4)** | DONE | `373d250a` | `thinking_budget` param in provider ABC + all 4 implementations + GenerationConfig + nodes.py wiring |
| **Task 5: Runner-Managed Continuation (P5)** | DONE | `fe4ad83f` | `_run_continuation_loop()` with _MAX_CONTINUATIONS=3, `is_truncated` branch removed |
| **Integration Verification** | DONE | — | 1052/1053 tests pass (1 pre-existing failure), 0 lint/mypy errors in changed files |

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

## What's Done (5/5 tasks — ALL COMPLETE)

### Task 4 — Thinking Token Budget (P4) — `373d250a`
- Added `thinking_budget: int = 0` to `InferenceProvider` ABC and all 4 concrete providers
- `TransformersProvider`: `effective_max = max_tokens + thinking_budget` for both generation and truncation detection
- `VLLMProvider`, `OllamaProvider`, `LlamaCppProvider`: pass `thinking_budget` through to API call
- Added `GenerationConfig.thinking_budget = 512` in pipeline_config.py
- Wired in `nodes.py`: `thinking_budget = int(os.environ.get("RUNE_THINKING_BUDGET", "512")) if enable_thinking else 0`
- Tests: `libs/inference/tests/test_thinking_budget.py` (5 tests), `libs/shared/tests/test_pipeline_config.py` (2 new tests)

### Task 5 — Runner-Managed Continuation (P5) — `fe4ad83f`
- Extracted `_run_continuation_loop()` with `_MAX_CONTINUATIONS=3` cap
- Injectable dependencies for testability (graph, render_trajectory_fn, load_adapter_fn, eager_unload_fn)
- Wired after `run_iteration()` in `_code_subtask()` — only fires when `finish_reason == "length"`
- Removed `is_truncated` branch from trajectory rendering (code_continue was moved into continuation loop)
- Removed `elif is_truncated:` from code_phase selection
- Removed old concat block (lines that interleaved continuation with retry)
- Dedented retry logic block that was previously nested inside the removed `if is_truncated:` else branch
- Updated `test_eager_unload.py` to recognize `eager_unload_fn` callback pattern
- Tests: `tests/test_continuation.py` (3 tests: accumulates code, skipped when not truncated, respects max turns)

## Integration Verification — DONE

1. **Tests:** 1052/1053 pass (`uv run pytest -x`). 1 pre-existing failure in `test_single_item_triggers_fallback` (present on main before our changes).
2. **Lint:** 0 errors in changed files (`uv run ruff check`). Pre-existing errors in other files only.
3. **Mypy:** 0 new errors in changed files. Pre-existing errors in `transformers_provider.py` lines 84/125 only.

## Execution Approach

Tasks 1-3 used **subagent-driven-development** on `feat/training-speed-opts` (merged as PR #41).
Tasks 4-5 implemented on `feat/pipeline-speed-p4-p5` branch with TDD approach.

## Test State

All targeted tests pass: eager unload (2), decompose templates (6), pipeline config (15 including 2 new), skip decompose (5), thinking budget (5), continuation (3), rune-agent unit tests (26). Total: 1052 passing.
