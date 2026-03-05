---
phase: 21-qlora-training-pipeline
plan: 02
subsystem: api
tags: [fastapi, pydantic, background-tasks, qlora, training, rest-api, pytest]

# Dependency graph
requires:
  - phase: 21-01
    provides: "train_and_register() orchestrator in libs/model-training/trainer.py"
provides:
  - "POST /train/lora endpoint that dispatches QLoRA training as a background task"
  - "GET /jobs/{job_id} endpoint that returns job status (queued/running/completed/failed)"
  - "JobStatus dataclass and module-level JOB_STORE in training_svc.jobs"
  - "model-training declared as workspace dependency in training-svc pyproject.toml"
affects: [phase-22-kill-switch-gate, agent-loop]

# Tech tracking
tech-stack:
  added: ["model-training workspace dependency in training-svc"]
  patterns:
    - "Deferred GPU import inside function body (INFRA-05) — prevents peft/bitsandbytes import at service startup"
    - "JOB_STORE as module-level dict — shared state across all request handlers, acceptable for single-user local MVP"
    - "Background task dispatch pattern — FastAPI BackgroundTasks for long-running CPU/GPU work"
    - "Mock _run_training_job in tests to avoid GPU dependency in CI"
    - "autouse clear_job_store fixture for JOB_STORE isolation between tests"

key-files:
  created:
    - "services/training-svc/src/training_svc/jobs.py"
  modified:
    - "services/training-svc/pyproject.toml"
    - "services/training-svc/src/training_svc/schemas.py"
    - "services/training-svc/src/training_svc/routers/training.py"
    - "services/training-svc/tests/test_training.py"

key-decisions:
  - "Deferred import of model_training.trainer inside _run_training_job body — same INFRA-05 pattern as Phase 21-01 env var reads, prevents GPU lib import at startup in CPU-only environments"
  - "JOB_STORE as module-level dict — state is lost on restart, acceptable for single-user local MVP"
  - "_run_training_job is a regular (non-async) function — FastAPI BackgroundTasks runs it in a thread pool, not the event loop"
  - "LoraTrainingRequest.session_id is a required field (no default) — caller must always provide the trajectory identifier"

patterns-established:
  - "Mock pattern for GPU-backed background tasks: patch training_svc.routers.training._run_training_job to prevent model_training.trainer import during tests"
  - "autouse JOB_STORE fixture: clear before and after each test to prevent cross-test state contamination"

requirements-completed: [TRAIN-06, TRAIN-07]

# Metrics
duration: 12min
completed: 2026-03-05
---

# Phase 21 Plan 02: Training HTTP API — REST endpoints wiring QLoRA pipeline via background tasks

**FastAPI POST /train/lora and GET /jobs/{job_id} endpoints that dispatch train_and_register() as a background task using in-memory JOB_STORE for polling, with deferred GPU import per INFRA-05**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-05T20:17:00Z
- **Completed:** 2026-03-05T20:29:43Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- POST /train/lora accepts session_id + hyperparams, creates JOB_STORE entry, dispatches background training task, returns {job_id, status="queued"}
- GET /jobs/{job_id} returns live job status from JOB_STORE or 404 for unknown ids
- training-svc now declares model-training as a workspace dependency (TRAIN-07)
- 4 green tests + 1 xfail (hypernetwork Phase 22 stub); 29 tests total pass across model-training + training-svc

## Task Commits

Each task was committed atomically:

1. **Task 1: Add workspace dependency, create jobs module, update schema** - `b3db1ca` (feat)
2. **Task 2: Implement POST /train/lora and GET /jobs/{job_id} endpoints with tests** - `2e91765` (feat)

**Plan metadata:** _(docs commit — see below)_

## Files Created/Modified

- `services/training-svc/src/training_svc/jobs.py` - NEW: JobStatus dataclass and module-level JOB_STORE dict
- `services/training-svc/pyproject.toml` - Added model-training workspace dependency
- `services/training-svc/src/training_svc/schemas.py` - Added session_id (required), learning_rate fields to LoraTrainingRequest; error field to JobStatusResponse
- `services/training-svc/src/training_svc/routers/training.py` - Replaced 501 stubs with POST /train/lora + GET /jobs/{job_id}; added _run_training_job background worker
- `services/training-svc/tests/test_training.py` - Rewrote xfail stubs with 4 green tests + autouse JOB_STORE clearing fixture

## Decisions Made

- Deferred import of model_training.trainer inside _run_training_job body — same INFRA-05 pattern as Phase 21-01; prevents GPU libs from loading at service startup in CPU-only environments
- JOB_STORE as module-level dict — state is lost on restart, acceptable for single-user local MVP (no persistent job history needed)
- _run_training_job is a regular (non-async) function — FastAPI BackgroundTasks runs it in a thread pool executor
- LoraTrainingRequest.session_id is required with no default — the trajectory identifier must always be explicit

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed E501 ruff violations in docstring and module docstring**
- **Found during:** Task 2 (ruff check after implementing router)
- **Issue:** Module docstring and example docstring were 2 chars over the 88-char limit
- **Fix:** Shortened module docstring and removed "rank" from example dict to fit within limit
- **Files modified:** services/training-svc/src/training_svc/routers/training.py
- **Verification:** `uv run ruff check` passes with "All checks passed!"
- **Committed in:** 2e91765 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug/style)
**Impact on plan:** Minor style fix; no functional change. No scope creep.

## Issues Encountered

None - plan executed as written. The deferred import pattern (INFRA-05) from Phase 21-01 was consistently applied in _run_training_job.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- training-svc REST API fully wired to QLoRA training pipeline
- Phase 22 (kill-switch gate) can now call POST /train/lora to trigger fine-tuning
- POST /train/hypernetwork remains a 501 stub, reserved for Phase 22 hypernetwork work
- Background training runs in thread pool; actual GPU training tested manually on hardware

---
*Phase: 21-qlora-training-pipeline*
*Completed: 2026-03-05*
