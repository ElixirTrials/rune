---
phase: 22-kill-switch-gate
plan: 03
subsystem: api
tags: [hypernetwork, lora, peft, fastapi, background-tasks, deferred-import, cpu-ci]

requires:
  - phase: 22-01
    provides: DocToLoraHypernetwork, save_hypernetwork_adapter in model_training.hypernetwork

provides:
  - POST /train/hypernetwork endpoint dispatching _run_hypernetwork_job as background task
  - _run_hypernetwork_job with deferred GPU imports (INFRA-05 pattern)
  - Green tests for hypernetwork endpoint (test_train_hypernetwork_returns_job_id, _requires_trajectory_ids, _job_pollable)

affects: [training-svc, model-training]

tech-stack:
  added: []
  patterns: [deferred-gpu-import, background-tasks, job-store-polling]

key-files:
  created: []
  modified:
    - services/training-svc/src/training_svc/routers/training.py
    - services/training-svc/tests/test_training.py

key-decisions:
  - "_run_hypernetwork_job uses first trajectory_id from request.trajectory_ids — single trajectory drives the hypernetwork forward pass"
  - "Deferred GPU imports inside _run_hypernetwork_job body (torch, transformers, model_training.hypernetwork, model_training.trajectory) — INFRA-05 pattern keeps service importable without GPU deps"
  - "Mock _run_hypernetwork_job in tests (same pattern as _run_training_job) — prevents GPU import chain at test time"

patterns-established:
  - "Background task dispatch pattern: create JOB_STORE entry → add_task(_run_*_job) → return {job_id, status: queued}"

requirements-completed: [DTOL-04]

duration: ~8min
completed: 2026-03-06
---

# Phase 22 Plan 03: Hypernetwork HTTP Endpoint Summary

**POST /train/hypernetwork wired to DocToLoraHypernetwork via background task dispatch, replacing 501 stub with full deferred-import forward-pass job handler and 3 green tests.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-06T07:00:00Z
- **Completed:** 2026-03-06T07:08:00Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Replaced 501 Not Implemented stub with a working `POST /train/hypernetwork` endpoint that creates a `JOB_STORE` entry and dispatches `_run_hypernetwork_job` as a FastAPI `BackgroundTask`
- Implemented `_run_hypernetwork_job` with full deferred GPU import chain (torch, transformers, model_training.hypernetwork, model_training.trajectory) per INFRA-05 pattern
- Replaced single `xfail` stub test with 3 green tests covering job_id return, 422 validation, and GET /jobs/{job_id} pollability

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement POST /train/hypernetwork endpoint with background task dispatch** - `d37a8db` (feat)

**Plan metadata:** TBD (docs: complete plan)

## Files Created/Modified

- `services/training-svc/src/training_svc/routers/training.py` - Added `_run_hypernetwork_job()` function and replaced 501 stub endpoint; added `os`, `pathlib.Path` stdlib imports
- `services/training-svc/tests/test_training.py` - Replaced `test_train_hypernetwork_still_501` (xfail) with `test_train_hypernetwork_returns_job_id`, `test_train_hypernetwork_requires_trajectory_ids`, `test_train_hypernetwork_job_pollable`

## Decisions Made

- `_run_hypernetwork_job` uses `request.trajectory_ids[0]` — single trajectory drives the hypernetwork forward pass; multi-trajectory batching is a future concern
- All GPU imports deferred inside function body per INFRA-05: `import torch`, `from model_training.hypernetwork import ...`, `from model_training.trajectory import ...`, `from transformers import AutoTokenizer`
- Tests mock `_run_hypernetwork_job` (same pattern as `_run_training_job`) to prevent GPU import chain at test time; `clear_job_store` fixture prevents cross-test contamination

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ruff I001 import sort order in deferred import block**
- **Found during:** Task 1 (post-implementation ruff check)
- **Issue:** Deferred imports inside `_run_hypernetwork_job` were ordered torch → transformers → model_training, but ruff I001 requires alphabetical order within third-party import group: torch → model_training.hypernetwork → model_training.trajectory → transformers
- **Fix:** Ran `uv run ruff check --fix` to auto-sort; also expanded single-line `from model_training.trajectory import format_for_sft, load_trajectory` to multi-line parenthesized form
- **Files modified:** `services/training-svc/src/training_svc/routers/training.py`
- **Verification:** `uv run ruff check` passed with no errors
- **Committed in:** d37a8db (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - import order bug)
**Impact on plan:** Auto-fix necessary for lint compliance. No scope creep.

## Issues Encountered

None - implementation matched plan specification exactly.

## Next Phase Readiness

- All three Phase 22 plans complete: DocToLoraHypernetwork (22-01), kill-switch evaluation (22-02), hypernetwork HTTP endpoint (22-03)
- Full end-to-end path is wired: trajectory → tokenize → hypernetwork forward pass → PEFT adapter → JOB_STORE polling
- Runtime wiring (loading pre-trained hypernetwork weights, Sakana AI weight availability) remains a deployment concern, not a code concern

---
*Phase: 22-kill-switch-gate*
*Completed: 2026-03-06*
