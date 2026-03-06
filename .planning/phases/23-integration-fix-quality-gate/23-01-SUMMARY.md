---
phase: 23-integration-fix-quality-gate
plan: 01
subsystem: training
tags: [adapter-registry, hypernetwork, sqlmodel, mypy, ruff, training-svc]

# Dependency graph
requires:
  - phase: 22-kill-switch-gate
    provides: _run_hypernetwork_job function that saves adapters to disk via save_hypernetwork_adapter()
  - phase: 18-adapter-registry
    provides: AdapterRegistry.store(), AdapterRecord SQLModel, retrieve_by_id(), list_all()
provides:
  - Hypernetwork-generated adapters are stored in AdapterRegistry after each job completes
  - _run_hypernetwork_job wires AdapterRegistry.store() using training_svc.storage.engine
  - mypy passes clean on training-svc with model_training override in pyproject.toml
  - Integration test verifying registry storage of hypernetwork adapters
affects: inference-provider, evaluation, any component calling AdapterRegistry.list_all()

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Deferred import of training_svc.storage.engine inside function body for patch-testability (INFRA-05)
    - AdapterRecord construction with SHA-256 file_hash and stat().st_size for integrity metadata
    - sys.modules injection for GPU mock isolation in CI (established Phase 21, confirmed here)

key-files:
  created:
    - services/training-svc/tests/test_training.py (new test function added)
  modified:
    - services/training-svc/src/training_svc/routers/training.py
    - pyproject.toml

key-decisions:
  - "Use training_svc.storage.engine (not RUNE_DATABASE_URL) in _run_hypernetwork_job — service-level function shares service DB"
  - "model_training added to workspace-packages mypy override block (not GPU libs block)"
  - "Deferred imports for hashlib, datetime, AdapterRecord, AdapterRegistry, svc_engine inside try block"

patterns-established:
  - "Pattern: Service-level background jobs import storage.engine deferred for testability via patch"
  - "Pattern: AdapterRecord construction uses file SHA-256 hash + stat().st_size for integrity"

requirements-completed: [DTOL-04]

# Metrics
duration: 6min
completed: 2026-03-06
---

# Phase 23 Plan 01: Integration Fix Quality Gate Summary

**Hypernetwork-generated adapters registered in AdapterRegistry via training_svc.storage.engine, closing the v5.0 audit gap with clean mypy and full test coverage**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-06T14:30:58Z
- **Completed:** 2026-03-06T14:37:22Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `_run_hypernetwork_job` now calls `AdapterRegistry.store()` after `save_hypernetwork_adapter()` completes, using `training_svc.storage.engine` (not a separately-constructed engine) for DB consistency
- `model_training` and `model_training.*` added to workspace mypy overrides — `uv run mypy services/training-svc/src/` exits 0 with zero errors
- New test `test_hypernetwork_job_registers_adapter` exercises the full registry wiring: calls `_run_hypernetwork_job` directly, patches `training_svc.storage.engine` with shared in-memory SQLite, and verifies `source="hypernetwork"`, `rank=8`, 64-char `file_hash`, `file_size_bytes > 0`
- Full CI quality gate passes: ruff + mypy + 151 tests (10/10 training-svc, 0 regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire AdapterRegistry.store() into _run_hypernetwork_job and fix mypy overrides** - `9299cfb` (feat)
2. **Task 2: Add test verifying hypernetwork job registers adapter in registry** - `0d2c7b0` (test)

**Plan metadata:** (see final docs commit)

## Files Created/Modified

- `services/training-svc/src/training_svc/routers/training.py` - Added registry store block after save_hypernetwork_adapter(), deferred imports for hashlib, datetime, AdapterRecord, AdapterRegistry, svc_engine
- `pyproject.toml` - Added model_training and model_training.* to workspace mypy override block
- `services/training-svc/tests/test_training.py` - Added test_hypernetwork_job_registers_adapter with sys.modules GPU mocking, in-memory SQLite engine patching, and full AdapterRecord assertions

## Decisions Made

- Use `training_svc.storage.engine` (not `RUNE_DATABASE_URL` with a new engine) — `_run_hypernetwork_job` is a service-level function; using the service's shared engine ensures hypernetwork adapters land in the same DB as QLoRA adapters
- `model_training` belongs in the workspace-packages override block (not GPU libs block) — it is a local workspace package, not a GPU-only library
- All new imports deferred inside function body per INFRA-05 pattern — consistent with existing deferred imports and allows patch-testing via `patch("training_svc.storage.engine", ...)`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Import ordering violated ruff I001 (isort) rule**
- **Found during:** Task 1 (after writing registry store block)
- **Issue:** The block had stdlib (`hashlib`, `datetime`) and first-party (`adapter_registry`, `training_svc.storage`) without blank-line separators between import groups
- **Fix:** Used `uv run ruff check --fix` to auto-sort import groups; resulted in stdlib → (blank) → third-party/first-party → (blank) → first-party groupings
- **Files modified:** `services/training-svc/src/training_svc/routers/training.py`
- **Verification:** `uv run ruff check` exits 0 after fix
- **Committed in:** `9299cfb` (Task 1 commit)

**2. [Rule 1 - Bug] ruff E501 and N806 violations in test file**
- **Found during:** Task 2 (after writing test function)
- **Issue:** Docstring > 88 chars, variable name `MockHypernetworkClass` violated N806 (should be lowercase), and assert message line too long
- **Fix:** Shortened docstring, renamed to `mock_hypernetwork_class`, refactored assert message to use local `job` variable
- **Files modified:** `services/training-svc/tests/test_training.py`
- **Verification:** `uv run ruff check` exits 0 after fix
- **Committed in:** `0d2c7b0` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug, lint violations)
**Impact on plan:** Both auto-fixes were purely cosmetic lint compliance. No functional scope change.

## Issues Encountered

None beyond the two ruff lint issues auto-fixed above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All v5.0 audit gaps from 23-RESEARCH.md are now closed
- Hypernetwork adapters are discoverable via `AdapterRegistry.retrieve_by_id()` and `list_all()` after job completion
- CI quality gate (ruff + mypy + pytest) fully passing — ready for milestone completion

## Self-Check: PASSED

All files and commits verified:
- FOUND: services/training-svc/src/training_svc/routers/training.py
- FOUND: pyproject.toml
- FOUND: services/training-svc/tests/test_training.py
- FOUND: .planning/phases/23-integration-fix-quality-gate/23-01-SUMMARY.md
- FOUND commit: 9299cfb (Task 1)
- FOUND commit: 0d2c7b0 (Task 2)

---
*Phase: 23-integration-fix-quality-gate*
*Completed: 2026-03-06*
