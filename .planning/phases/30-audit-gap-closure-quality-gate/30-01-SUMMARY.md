---
phase: 30-audit-gap-closure-quality-gate
plan: 01
subsystem: training
tags: [mypy, probe-cache, data-pipeline, d2l_train, d2l_prep, tdd]

# Dependency graph
requires:
  - phase: 29-training-loop-integration
    provides: train_d2l_qwen3, _dry_run_validate_shapes, d2l_probe.load_probe_cache
  - phase: 25-configuration-data-pipeline
    provides: format_for_distillation, save_jsonl
provides:
  - Probe cache guard (RuntimeError before training with placeholder feature_sizes)
  - Zero mypy errors on libs/model-training/src
  - d2l_prep.prepare_training_jsonl CLI pipeline connecting format_for_distillation -> save_jsonl
affects: [training-loop, data-pipeline, CI]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_require_probe_cache() helper: defers load_probe_cache import, raises on None — callable from multiple sites without repeating error message"
    - "TDD red-green-commit cycle: test file committed at red before implementation"

key-files:
  created:
    - libs/model-training/src/model_training/d2l_prep.py
    - libs/model-training/tests/test_d2l_prep.py
  modified:
    - libs/model-training/src/model_training/d2l_train.py
    - libs/model-training/src/model_training/sakana_d2l.py
    - libs/model-training/src/model_training/d2l_probe.py

key-decisions:
  - "_require_probe_cache() helper encapsulates the RuntimeError — avoids duplicating long error message across _dry_run_validate_shapes and train_d2l_qwen3"
  - "smoke_test path skips probe cache guard — smoke_test uses generate_needle_dataset and may not have real probe cache on test machines"
  - "# noqa: C901 on train_d2l_qwen3 — 3-mode dispatch (dry_run, smoke_test, full) inherently pushes complexity above ruff's default limit of 10"
  - "sakana_d2l.py line 416: model annotated as Any — AutoModelForCausalLM.from_pretrained returns generic type; .to(device) arg-type error is a mypy limitation not a real type error"
  - "d2l_prep uses top-level imports (not deferred) — no GPU deps; all imports are stdlib + model_training"

patterns-established:
  - "Probe cache guard pattern: call _require_probe_cache(QWEN3_NEXT_CANONICAL_NAME) before build_qwen3_hypernet_config in any non-smoke-test training path"

requirements-completed: []

# Metrics
duration: ~25min
completed: 2026-03-16
---

# Phase 30 Plan 01: Audit Gap Closure and Quality Gate Summary

**Probe cache RuntimeError guard + zero mypy errors on libs/model-training + d2l_prep CLI pipeline wiring format_for_distillation to save_jsonl**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-16T12:00:00Z
- **Completed:** 2026-03-16T12:25:00Z
- **Tasks:** 2 (Task 1: fixes + guard; Task 2: TDD data prep)
- **Files modified:** 5

## Accomplishments

- Zero mypy errors on entire libs/model-training/src (was 2 errors in sakana_d2l.py)
- train_d2l_qwen3 and _dry_run_validate_shapes now raise RuntimeError when probe cache is absent, preventing silent training with incorrect LoRA dimensions
- d2l_prep.py provides prepare_training_jsonl() + CLI that connects format_for_distillation -> save_jsonl in a fully automated pipeline
- 3 new tests covering data prep pipeline (filter failures, empty output, CLI --help)
- 95 total tests passing (up from 92)

## Task Commits

Each task was committed atomically:

1. **Task 1: Probe cache guard + mypy fixes + cosmetic d2l_probe.py change** - `36758b5` (fix)
2. **Task 2 RED: Failing tests for d2l_prep** - `67501d3` (test)
3. **Task 2 GREEN: Implement d2l_prep pipeline** - `43c0b2c` (feat)

_TDD task has three commits: test (RED) → feat (GREEN) — no refactor needed._

## Files Created/Modified

- `libs/model-training/src/model_training/d2l_train.py` - Added _require_probe_cache() helper, guard in _dry_run_validate_shapes and train_d2l_qwen3 (non-smoke-test path), # noqa: C901
- `libs/model-training/src/model_training/sakana_d2l.py` - int() cast for bsz (line 80), Any annotation for model (line 416)
- `libs/model-training/src/model_training/d2l_probe.py` - Cosmetic docstring word-wrap (pre-existing change committed)
- `libs/model-training/src/model_training/d2l_prep.py` - New: prepare_training_jsonl(), _load_trajectories(), CLI __main__ block
- `libs/model-training/tests/test_d2l_prep.py` - New: 3 tests for data prep pipeline

## Decisions Made

- `_require_probe_cache()` helper encapsulates the RuntimeError to avoid duplicating the long error message across both call sites.
- smoke_test path skips the probe cache guard because smoke_test uses `generate_needle_dataset` and may not have a real probe cache on test machines.
- `# noqa: C901` added on `train_d2l_qwen3` — 3-mode dispatch (dry_run, smoke_test, full) inherently pushes cyclomatic complexity above ruff's default limit of 10; raising the limit project-wide would be wrong.
- `sakana_d2l.py` line 416: annotated `model: Any` to bypass PreTrainedModel constraint from mypy — `.to(device)` arg-type error is a mypy limitation on the transformers generic return type.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ruff E501 and C901 violations from new guard code**
- **Found during:** Task 1 (probe cache guard)
- **Issue:** Adding the guard inline with a long error message string triggered E501 (line too long) and C901 (function too complex) in ruff
- **Fix:** Extracted guard into `_require_probe_cache()` helper (eliminates E501, reduces branch count in train_d2l_qwen3); added `# noqa: C901` for residual complexity (function has 3 modes by design)
- **Files modified:** libs/model-training/src/model_training/d2l_train.py
- **Verification:** `uv run ruff check` passes clean
- **Committed in:** 36758b5 (Task 1 commit)

**2. [Rule 1 - Bug] ruff violations in d2l_prep.py and test_d2l_prep.py**
- **Found during:** Task 2 GREEN (d2l_prep implementation)
- **Issue:** E501 (line too long in warning call), I001 (unsorted imports), F401 (unused pytest import) in test file
- **Fix:** Split long logger.warning call, removed unused pytest import, ruff auto-sorted
- **Files modified:** libs/model-training/src/model_training/d2l_prep.py, libs/model-training/tests/test_d2l_prep.py
- **Verification:** `uv run ruff check` passes clean
- **Committed in:** 43c0b2c (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - ruff violations from new code)
**Impact on plan:** Both fixes are style-level corrections within planned files. No scope creep.

## Issues Encountered

- pytest-cov INTERNALERROR (Can't combine statement coverage data with branch data) when running test_d2l_prep.py alone — this is a pre-existing coverage data file conflict, not related to new code. Resolved by running with `--no-cov` for RED verification; full suite with coverage passes.

## Next Phase Readiness

- All quality gate items from v7.0 milestone audit are closed
- mypy clean, ruff clean, 95 tests passing
- Data prep CLI ready for use with real trajectory files
- Probe cache guard prevents silent incorrect training — operator must run probe_model() before full training

---
*Phase: 30-audit-gap-closure-quality-gate*
*Completed: 2026-03-16*

## Self-Check: PASSED

- FOUND: libs/model-training/src/model_training/d2l_prep.py
- FOUND: libs/model-training/tests/test_d2l_prep.py
- FOUND: .planning/phases/30-audit-gap-closure-quality-gate/30-01-SUMMARY.md
- FOUND: commit 36758b5 (probe cache guard + mypy fixes)
- FOUND: commit 67501d3 (failing tests RED)
- FOUND: commit 43c0b2c (d2l_prep implementation GREEN)
