---
phase: 15-new-reworked-library-wireframes
plan: "02"
subsystem: testing
tags: [evaluation, metrics, wireframe, tdd, notimplementederror, humaneval, pass-at-k]

# Dependency graph
requires:
  - phase: 14-core-library-wireframes
    provides: Component conftest.py pattern with pytest rootdir isolation handling
provides:
  - libs/evaluation/metrics.py with 6 public wireframed functions (run_humaneval_subset, calculate_pass_at_k, score_adapter_quality, compare_adapters, test_generalization, evaluate_fitness)
  - libs/evaluation/__init__.py exporting all 6 functions via __all__
  - libs/evaluation/tests/test_metrics.py with 6 TDD tests asserting NotImplementedError stubs
  - libs/evaluation fully wireframed as new module satisfying LIB-06
affects: [phase-16-implementation, phase-17-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [Google-style docstrings with Args/Returns/Raises/Example, NotImplementedError stubs, TDD red-phase test pattern, import alias for pytest collision avoidance]

key-files:
  created:
    - libs/evaluation/src/evaluation/metrics.py
    - libs/evaluation/tests/test_metrics.py
  modified:
    - libs/evaluation/src/evaluation/__init__.py
    - libs/evaluation/tests/conftest.py

key-decisions:
  - "Alias test_generalization import as _test_generalization in test file to prevent pytest from collecting the src function as a test (pytest collects all test_* names visible in test module namespace)"
  - "libs/evaluation conftest.py stays minimal — 6 TDD tests use only literal arguments, no factory fixtures needed"

patterns-established:
  - "Import alias pattern: when a function named test_* must be imported into a test module, alias it (e.g. test_generalization as _test_generalization) to avoid pytest collection conflict"
  - "NotImplementedError message includes full function name so pytest match= assertions work precisely"

requirements-completed: [LIB-06]

# Metrics
duration: 2min
completed: 2026-03-03
---

# Phase 15 Plan 02: libs/evaluation Wireframes Summary

**libs/evaluation wireframed from scratch with 6 evaluation metric functions (run_humaneval_subset, calculate_pass_at_k, score_adapter_quality, compare_adapters, test_generalization, evaluate_fitness), Google-style docstrings, and 6 passing TDD tests**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-03T20:54:11Z
- **Completed:** 2026-03-03T20:56:23Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created metrics.py with 6 fully-documented public functions, each raising NotImplementedError with its name in the error message
- Updated __init__.py to export all 6 functions with __all__ for clean public API
- Created test_metrics.py with 6 TDD tests, all passing (asserting expected NotImplementedError stubs)
- Resolved pytest collection conflict for `test_generalization` via import aliasing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create evaluation metrics module with 6 wireframed functions** - `3387142` (feat)
2. **Task 2: Update conftest.py and write failing TDD tests for all 6 evaluation functions** - `1f99847` (test)

**Plan metadata:** (docs commit — forthcoming)

## Files Created/Modified
- `libs/evaluation/src/evaluation/metrics.py` - 6 public functions with Google-style docstrings and NotImplementedError stubs (250 lines)
- `libs/evaluation/src/evaluation/__init__.py` - Package exports for all 6 functions with __all__
- `libs/evaluation/tests/test_metrics.py` - 6 TDD tests asserting NotImplementedError with match= on function names
- `libs/evaluation/tests/conftest.py` - Updated from placeholder to proper docstring (no fixtures needed for these tests)

## Decisions Made
- Aliased `test_generalization` import as `_test_generalization` in test_metrics.py to prevent pytest from collecting the src function as a test. Pytest discovers all `test_*` named objects in test module namespaces including those imported from source modules.
- Kept conftest.py minimal — the 6 TDD tests use only literal arguments and don't require factory fixtures.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Aliased test_generalization import to prevent pytest collection conflict**
- **Found during:** Task 2 (test_metrics.py creation)
- **Issue:** Importing `test_generalization` from `evaluation.metrics` into the test module caused pytest to collect the function as a test fixture, producing an ERROR (fixture 'adapter_id' not found) because pytest treated the function parameters as fixture names
- **Fix:** Changed `from evaluation.metrics import ... test_generalization` to `test_generalization as _test_generalization` and updated the test body to call `_test_generalization("adapter-001")`
- **Files modified:** libs/evaluation/tests/test_metrics.py
- **Verification:** `uv run pytest libs/evaluation/tests/ -v` — 6 passed, 0 errors
- **Committed in:** 1f99847 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Required fix for tests to run cleanly. The test_generalization function is still tested correctly under its proper name via the alias. No scope creep.

## Issues Encountered
- pytest collected `test_generalization` from the source module namespace in the test file, treating its parameters as fixture names. Fixed via import aliasing.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- libs/evaluation fully wireframed with LIB-06 satisfied
- All 6 evaluation metric functions ready for implementation in Phase 16
- No blockers

---
*Phase: 15-new-reworked-library-wireframes*
*Completed: 2026-03-03*
