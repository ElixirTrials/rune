---
phase: 06-service-scaffolds
plan: 03
subsystem: api
tags: [fastapi, sqlmodel, pydantic, training, evolution, workspace]

# Dependency graph
requires:
  - phase: 05-foundation-libraries
    provides: adapter-registry and shared libs as workspace members
provides:
  - training-svc FastAPI service with 3 stub 501 endpoints
  - evolution-svc FastAPI service with 4 stub 501 endpoints
  - TrainingJob SQLModel with __tablename__="training_jobs"
  - EvolutionJob SQLModel with __tablename__="evolution_jobs"
  - Both services registered as uv workspace members
affects: [06-04-PLAN, 07-config-quality-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FastAPI service scaffold with lifespan, 501 stub endpoints, and health check"
    - "SQLModel job table with explicit __tablename__ to avoid collisions"
    - "APIRouter with no prefix and full paths on each endpoint"

key-files:
  created:
    - services/training-svc/pyproject.toml
    - services/training-svc/src/training_svc/main.py
    - services/training-svc/src/training_svc/schemas.py
    - services/training-svc/src/training_svc/models.py
    - services/training-svc/src/training_svc/storage.py
    - services/training-svc/src/training_svc/dependencies.py
    - services/training-svc/src/training_svc/routers/training.py
    - services/evolution-svc/pyproject.toml
    - services/evolution-svc/src/evolution_svc/main.py
    - services/evolution-svc/src/evolution_svc/schemas.py
    - services/evolution-svc/src/evolution_svc/models.py
    - services/evolution-svc/src/evolution_svc/storage.py
    - services/evolution-svc/src/evolution_svc/dependencies.py
    - services/evolution-svc/src/evolution_svc/routers/evolution.py
  modified:
    - pyproject.toml

key-decisions:
  - "Used APIRouter with no prefix and full endpoint paths to avoid nesting issues"
  - "Followed api-service storage.py pattern with SQLite default and check_same_thread=False"

patterns-established:
  - "Service scaffold pattern: pyproject.toml + __init__.py + main.py + schemas.py + models.py + storage.py + dependencies.py + routers/"
  - "501 stub endpoint pattern: return JSONResponse(status_code=501, content={'detail': 'Not Implemented'})"

requirements-completed: [SVC-03, SVC-04]

# Metrics
duration: 3min
completed: 2026-03-03
---

# Phase 06 Plan 03: Training & Evolution Service Scaffolds Summary

**Two FastAPI services (training-svc, evolution-svc) with 7 combined 501 stub endpoints, Pydantic schemas, and SQLModel job tracking tables**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-03T07:18:41Z
- **Completed:** 2026-03-03T07:21:59Z
- **Tasks:** 3
- **Files modified:** 19

## Accomplishments
- Scaffolded training-svc with POST /train/lora, POST /train/hypernetwork, GET /jobs/{job_id} endpoints (all 501)
- Scaffolded evolution-svc with POST /evaluate, POST /evolve, POST /promote, POST /prune endpoints (all 501)
- Both services registered as uv workspace members with adapter-registry and shared as workspace deps
- TrainingJob and EvolutionJob SQLModel tables with non-colliding __tablename__ values
- uv lock && uv sync passes cleanly with both new members

## Task Commits

Each task was committed atomically:

1. **Task 1: Scaffold services/training-svc as workspace member** - `f121a47` (feat)
2. **Task 2: Scaffold services/evolution-svc as workspace member** - `73df1c9` (feat)
3. **Task 3: Register both services as workspace members and verify uv lock/sync** - `58183a0` (chore)

## Files Created/Modified
- `services/training-svc/pyproject.toml` - Workspace member config with hatchling, fastapi, sqlmodel, adapter-registry, shared deps
- `services/training-svc/src/training_svc/__init__.py` - Package init
- `services/training-svc/src/training_svc/main.py` - FastAPI app with training router and health check
- `services/training-svc/src/training_svc/schemas.py` - LoraTrainingRequest, HypernetworkTrainingRequest, JobStatusResponse
- `services/training-svc/src/training_svc/models.py` - TrainingJob SQLModel table
- `services/training-svc/src/training_svc/storage.py` - SQLite database setup
- `services/training-svc/src/training_svc/dependencies.py` - DB session dependency injection
- `services/training-svc/src/training_svc/routers/training.py` - Three 501 stub endpoints
- `services/evolution-svc/pyproject.toml` - Workspace member config with hatchling, fastapi, sqlmodel, adapter-registry, shared deps
- `services/evolution-svc/src/evolution_svc/__init__.py` - Package init
- `services/evolution-svc/src/evolution_svc/main.py` - FastAPI app with evolution router and health check
- `services/evolution-svc/src/evolution_svc/schemas.py` - EvaluationRequest, EvaluationResponse, EvolveRequest, PromoteRequest, PruneRequest
- `services/evolution-svc/src/evolution_svc/models.py` - EvolutionJob SQLModel table
- `services/evolution-svc/src/evolution_svc/storage.py` - SQLite database setup
- `services/evolution-svc/src/evolution_svc/dependencies.py` - DB session dependency injection
- `services/evolution-svc/src/evolution_svc/routers/evolution.py` - Four 501 stub endpoints
- `pyproject.toml` - Added both services to [tool.uv.workspace] members

## Decisions Made
- Used APIRouter with no prefix and full endpoint paths (/train/lora, /jobs/{job_id}) to avoid prefix nesting issues
- Followed api-service storage.py pattern with SQLite default and check_same_thread=False for consistency

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both training-svc and evolution-svc are scaffolded and workspace-registered
- Plan 04 (5-section synchronization) can now add these services to mypy overrides, pythonpath, coverage source, and testpaths
- No blockers

## Self-Check: PASSED

All 18 created files verified present. All 3 task commits (f121a47, 73df1c9, 58183a0) verified in git log.

---
*Phase: 06-service-scaffolds*
*Completed: 2026-03-03*
