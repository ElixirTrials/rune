---
phase: 04-cleanup
plan: 02
subsystem: infra
tags: [uv, workspace, monorepo, rename, rune-agent]

# Dependency graph
requires:
  - phase: 04-01
    provides: "Clean workspace with agent-b-service removed; no lockfile conflicts"
provides:
  - "services/rune-agent exists with rune-agent/rune_agent naming throughout"
  - "All root config, CI, and docs reference rune-agent instead of agent-a-service"
  - "uv lockfile regenerated with rune-agent package"
affects: [04-03, 05-01]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Four-location rename pattern: directory, src module, pyproject.toml name, hatch wheel packages"

key-files:
  created: []
  modified:
    - "services/rune-agent/pyproject.toml"
    - "services/rune-agent/src/rune_agent/__init__.py"
    - "services/rune-agent/mkdocs.yml"
    - "services/rune-agent/README.md"
    - "services/rune-agent/docs/api/index.md"
    - "pyproject.toml"
    - ".github/workflows/ci.yml"
    - ".github/copilot-instructions.md"
    - "Makefile"
    - "PROJECT_OVERVIEW.md"
    - "docs/onboarding.md"
    - "docs/components-overview.md"
    - "docs/architecture/monorepo-mapping.md"
    - "services/api-service/README.md"

key-decisions:
  - "Also updated Makefile typecheck target and copilot-instructions.md glob patterns (not in plan but contained stale references)"
  - "Cleaned monorepo-mapping.md directory tree comment to avoid leaving agent-a-service text in source"

patterns-established:
  - "Service rename order: git mv directory, mv src module, update pyproject.toml (4 locations), update all external refs, uv lock"

requirements-completed: [CLN-02]

# Metrics
duration: 5min
completed: 2026-03-02
---

# Phase 4 Plan 02: Rename agent-a-service to rune-agent Summary

**Renamed agent-a-service to rune-agent across directory, Python module, pyproject.toml, and all 14 config/doc files with zero stale references**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-02T19:49:39Z
- **Completed:** 2026-03-02T19:55:37Z
- **Tasks:** 2
- **Files modified:** 14 (9 renamed/modified in Task 1, 9 modified in Task 2, some overlap)

## Accomplishments
- Renamed services/agent-a-service directory to services/rune-agent via git mv
- Renamed src/agent_a_service Python module to src/rune_agent
- Updated all four pyproject.toml name locations: project name, description, wheel packages, pytest coverage
- Updated root pyproject.toml (workspace members, mypy overrides, pythonpath, coverage source)
- Updated CI workflow, Makefile, PROJECT_OVERVIEW.md, components-overview.md, onboarding.md, monorepo-mapping.md, api-service README, copilot-instructions.md
- Verified zero stale agent-a-service/agent_a_service references remain in source files
- uv lock && uv sync passes cleanly with rune-agent package

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename directory and module from agent-a-service to rune-agent** - `55aeea2` (feat)
2. **Task 2: Update all root config and docs references from agent-a-service to rune-agent** - `deaccbc` (feat)

## Files Created/Modified
- `services/rune-agent/pyproject.toml` - Project name, description, wheel packages, pytest coverage updated
- `services/rune-agent/src/rune_agent/` - Renamed Python module (4 files: __init__.py, graph.py, nodes.py, state.py)
- `services/rune-agent/mkdocs.yml` - site_name updated to rune-agent
- `services/rune-agent/README.md` - All agent_a_service references replaced with rune_agent
- `services/rune-agent/docs/api/index.md` - Title updated to rune-agent
- `pyproject.toml` - Workspace members, mypy overrides, pythonpath, coverage source updated
- `.github/workflows/ci.yml` - mypy command path updated
- `.github/copilot-instructions.md` - Key dirs, agent packages, canonical example paths updated
- `Makefile` - typecheck target path updated
- `PROJECT_OVERVIEW.md` - AI Agent Services table row updated
- `docs/onboarding.md` - Agent service reference updated
- `docs/components-overview.md` - Component table row updated
- `docs/architecture/monorepo-mapping.md` - Removed stale agent-a-service row, updated directory tree
- `services/api-service/README.md` - Import example updated from agent_a_service to rune_agent

## Decisions Made
- Updated Makefile typecheck target and copilot-instructions.md glob patterns which were not explicitly listed in the plan but contained stale agent-a-service references
- Cleaned the monorepo-mapping.md directory tree comment to avoid leaving "agent-a-service" text anywhere in source files

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Updated Makefile typecheck target**
- **Found during:** Task 2 (external reference updates)
- **Issue:** Makefile line 89 contained `services/agent-a-service/src` in the mypy command, which was not listed in the plan's file list
- **Fix:** Replaced with `services/rune-agent/src`
- **Files modified:** Makefile
- **Verification:** grep confirms zero matches for agent-a-service in Makefile
- **Committed in:** deaccbc (Task 2 commit)

**2. [Rule 2 - Missing Critical] Updated copilot-instructions.md glob patterns**
- **Found during:** Task 2 (external reference updates)
- **Issue:** copilot-instructions.md had `services/{api-service,agent-*}` and `services/agent-*/` glob patterns that no longer match any service after rename
- **Fix:** Replaced with explicit `services/{api-service,rune-agent}` and `services/rune-agent/` references
- **Files modified:** .github/copilot-instructions.md
- **Verification:** grep confirms zero stale references
- **Committed in:** deaccbc (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 missing critical)
**Impact on plan:** Both auto-fixes necessary for correctness. Makefile typecheck would fail; copilot-instructions would point to nonexistent directories. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- services/rune-agent exists with consistent rune-agent/rune_agent naming
- All config, CI, docs reference rune-agent
- uv lock && uv sync pass cleanly
- Ready for Plan 03 (root pyproject.toml hardcoded-path audit)

---
*Phase: 04-cleanup*
*Completed: 2026-03-02*

## Self-Check: PASSED
- 04-02-SUMMARY.md: FOUND
- Commit 55aeea2: FOUND
- Commit deaccbc: FOUND
- services/rune-agent/src/rune_agent/: FOUND
- services/rune-agent/pyproject.toml: FOUND
