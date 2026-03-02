---
phase: 04-cleanup
plan: 03
subsystem: infra
tags: [uv, pyproject, dependencies, mypy, glob, makefile, ci]

# Dependency graph
requires:
  - phase: 04-02
    provides: "Renamed rune-agent service; clean workspace for dependency audit"
provides:
  - "Lean root pyproject.toml without unused Google Cloud, GPU, or sentence-transformer packages"
  - "Self-maintaining mypy typecheck glob pattern in Makefile and CI"
  - "Python-file filter that skips TypeScript lib directories automatically"
affects: [05-01, 06-01]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Glob-based mypy with Python-file filter: find + grep to skip non-Python src dirs"

key-files:
  created: []
  modified:
    - "pyproject.toml"
    - "Makefile"
    - ".github/workflows/ci.yml"

key-decisions:
  - "Added Python-file filter around glob to skip TypeScript directories (events-ts, shared-ts) that cause mypy to error"
  - "Kept torch/transformers/peft in mypy ignore_missing_imports for future Phase 5 model-training lib"
  - "Removed langchain-huggingface along with GPU packages since it depends on sentence-transformers/torch"

patterns-established:
  - "Typecheck glob pattern: for d in services/*/src libs/*/src; filter by .py existence; pass to mypy"

requirements-completed: [CLN-03, CLN-04]

# Metrics
duration: 3min
completed: 2026-03-02
---

# Phase 4 Plan 03: Root Dependency Cleanup and Glob Typecheck Summary

**Removed 9 unused template dependencies (Google Cloud, GPU, sentence-transformers) from root pyproject.toml and replaced hardcoded mypy paths with self-maintaining glob patterns**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-02T19:58:41Z
- **Completed:** 2026-03-02T20:01:38Z
- **Tasks:** 2
- **Files modified:** 3 (pyproject.toml, Makefile, .github/workflows/ci.yml)

## Accomplishments
- Removed 9 unused packages from root dependencies: google-cloud-aiplatform, vertexai, google-genai, sentence-transformers, torch, transformers, accelerate, bitsandbytes, langchain-huggingface
- 38 transitive packages uninstalled from environment (Google Cloud stack, HuggingFace stack, PyTorch)
- Removed langchain_google_vertexai from mypy ignore_missing_imports overrides
- Removed stale google.genai filterwarnings entry from pytest config
- Replaced hardcoded mypy component paths with `services/*/src libs/*/src` glob pattern in Makefile and CI
- Added Python-file filter to skip TypeScript directories that have no .py files
- All 6 success criteria verified: no removed deps in pyproject.toml, glob patterns present, uv lock/sync passes, make typecheck passes

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove unused template dependencies from root pyproject.toml** - `4c06a63` (chore)
2. **Task 2: Replace hardcoded typecheck paths with glob patterns in Makefile and CI** - `988db27` (feat)

## Files Created/Modified
- `pyproject.toml` - Removed 9 unused deps, removed langchain_google_vertexai mypy override, removed google.genai filterwarning
- `Makefile` - Typecheck target uses glob pattern with Python-file filter
- `.github/workflows/ci.yml` - Type check step uses same glob pattern with filter

## Decisions Made
- Added Python-file filter around the glob pattern because `libs/*/src` expands to include TypeScript directories (events-ts, shared-ts) which have no .py files and cause mypy to error with "There are no .py[i] files in directory"
- Kept torch, transformers, peft in mypy ignore_missing_imports (will be needed by libs/model-training in Phase 5)
- Removed langchain-huggingface along with the GPU packages since it depends on sentence-transformers and torch, which are template-specific

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added Python-file filter to glob pattern**
- **Found during:** Task 2 (glob pattern implementation)
- **Issue:** `libs/*/src` glob expands to include `libs/events-ts/src` and `libs/shared-ts/src` which contain no Python files, causing mypy to fail with "There are no .py[i] files in directory"
- **Fix:** Wrapped the glob in a shell for-loop that uses `find` to check each directory has at least one .py file before passing it to mypy
- **Files modified:** Makefile, .github/workflows/ci.yml
- **Verification:** `make typecheck` passes (19 source files, no issues)
- **Committed in:** 988db27 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Auto-fix necessary for correctness. The raw glob pattern cannot work in a polyglot monorepo with TypeScript libs. The filter preserves the self-maintaining intent of the plan while handling non-Python directories. No scope creep.

## Issues Encountered
None beyond the deviation documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Root pyproject.toml is lean with only dependencies Rune actually uses
- Typecheck is self-maintaining: adding new services/libs with src directories automatically includes them
- Phase 4 (Cleanup) is complete; ready for Phase 5 (Foundation Libraries)

---
*Phase: 04-cleanup*
*Completed: 2026-03-02*

## Self-Check: PASSED
- 04-03-SUMMARY.md: FOUND
- pyproject.toml: FOUND
- Makefile: FOUND
- .github/workflows/ci.yml: FOUND
- Commit 4c06a63: FOUND
- Commit 988db27: FOUND
