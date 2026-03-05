---
phase: 18-adapter-registry
plan: 01
subsystem: database
tags: [sqlmodel, sqlalchemy, sqlite, wal, crud, adapter-registry]

# Dependency graph
requires: []
provides:
  - AdapterRegistry class with Engine-parameterized constructor and 4 CRUD methods
  - WAL mode activated via SQLAlchemy event hook on every connection
  - Idempotent SQLModel table creation on init
  - store(): persists AdapterRecord, raises AdapterAlreadyExistsError on duplicate
  - retrieve_by_id(): returns AdapterRecord or raises AdapterNotFoundError
  - query_by_task_type(): returns filtered list[AdapterRecord] by task_type
  - list_all(): returns list[AdapterRecord] where is_archived == False
affects:
  - 19-inference-abstraction
  - 20-agent-loop
  - 21-training-svc
  - 22-lora-server

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Session per method: each CRUD method creates and closes its own Session"
    - "Detach before return: expire_on_commit=False + session.expunge() prevents DetachedInstanceError"
    - "WAL before create_all: event hook registered before SQLModel.metadata.create_all() so every connection activates WAL"
    - "== False not is False: SQLModel where clause uses == for SQL translation (is False evaluates at Python level)"

key-files:
  created: []
  modified:
    - libs/adapter-registry/src/adapter_registry/registry.py
    - libs/adapter-registry/tests/test_registry.py

key-decisions:
  - "Used expire_on_commit=False + session.expunge() on returned records to prevent DetachedInstanceError after session close"
  - "Engine imported from sqlalchemy.engine (not sqlmodel) — Engine is not re-exported by SQLModel"
  - "Added # noqa: E712 to is_archived == False comparison to suppress ruff false positive"

patterns-established:
  - "Session-per-method pattern: no shared session state between CRUD calls"
  - "Explicit expunge: records must be detached from session before returning when using per-method sessions"

requirements-completed: [AREG-01, AREG-02, AREG-03, AREG-04, AREG-05]

# Metrics
duration: 12min
completed: 2026-03-05
---

# Phase 18 Plan 01: AdapterRegistry Implementation Summary

**SQLModel-backed AdapterRegistry with Engine constructor, WAL event hook, and 4 CRUD methods — store/retrieve/query/list — all passing mypy strict and ruff.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-05T10:24:55Z
- **Completed:** 2026-03-05T10:36:00Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Replaced all 4 NotImplementedError stubs with working SQLModel CRUD operations
- Wired WAL mode via SQLAlchemy `event.listens_for` hook registered before `create_all`
- Changed constructor to require an `Engine` parameter (engine stored as `self._engine`)
- All 11 tests pass; mypy strict and ruff clean with zero errors
- Updated test suite from red-phase stubs to full green-phase behavioral tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement AdapterRegistry with constructor and 4 CRUD methods** - `a900149` (feat)

**Plan metadata:** TBD (docs: complete plan)

_Note: TDD task committed in single green-phase commit after implementation._

## Files Created/Modified

- `libs/adapter-registry/src/adapter_registry/registry.py` - Full implementation replacing all NotImplementedError stubs with SQLModel CRUD
- `libs/adapter-registry/tests/test_registry.py` - 11 behavioral tests covering constructor, store, retrieve_by_id, query_by_task_type, and list_all

## Decisions Made

- **Engine from sqlalchemy.engine:** SQLModel does not re-export `Engine`; must import from `sqlalchemy.engine` directly. Plan listed `from sqlmodel import Engine` which fails — corrected to `from sqlalchemy.engine import Engine`.
- **expire_on_commit=False + session.expunge():** All read methods use this pattern to prevent `DetachedInstanceError` when callers access attributes after session close. The `store()` method also expunges the caller's record object to prevent the same issue.
- **# noqa: E712:** Added to `is_archived == False` comparison per plan instructions to suppress ruff E712 (comparison to False).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect Engine import from sqlmodel**
- **Found during:** Task 1 (implementation)
- **Issue:** Plan specified `from sqlmodel import Engine` but SQLModel does not re-export Engine; import fails with `ImportError: cannot import name 'Engine' from 'sqlmodel'`
- **Fix:** Changed to `from sqlalchemy.engine import Engine` in both registry.py and test_registry.py
- **Files modified:** `libs/adapter-registry/src/adapter_registry/registry.py`, `libs/adapter-registry/tests/test_registry.py`
- **Verification:** Import succeeds, mypy passes
- **Committed in:** a900149 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed DetachedInstanceError on returned SQLModel records**
- **Found during:** Task 1 (test execution)
- **Issue:** `retrieve_by_id()`, `query_by_task_type()`, and `list_all()` returned ORM objects that became detached after session close, causing `DetachedInstanceError` when callers accessed any attribute. `store()` also left the caller's record object in an expired/detached state.
- **Fix:** Added `expire_on_commit=False` to all Session constructors and explicit `session.expunge()` calls before returning records. This detaches objects cleanly with all attributes loaded in memory.
- **Files modified:** `libs/adapter-registry/src/adapter_registry/registry.py`
- **Verification:** All 11 tests pass including `test_store_persists_record` and `test_retrieve_by_id_returns_matching_record` which access attributes after session close
- **Committed in:** a900149 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes required for correctness. Engine import is a factual error in plan; session expunge is a necessary SQLAlchemy pattern when using per-method sessions. No scope creep.

## Issues Encountered

None beyond the two auto-fixed bugs above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- AdapterRegistry fully operational; all consumers (api-service, rune-agent, training-svc, lora-server) can instantiate with `AdapterRegistry(engine=engine)`
- Plan 02 can now update any remaining tests and validate integration
- No blockers for Phase 19 (inference abstraction)

## Self-Check: PASSED

- FOUND: libs/adapter-registry/src/adapter_registry/registry.py
- FOUND: .planning/phases/18-adapter-registry/18-01-SUMMARY.md
- FOUND: commit a900149

---
*Phase: 18-adapter-registry*
*Completed: 2026-03-05*
