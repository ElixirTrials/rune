---
phase: 20-agent-loop
plan: 01
subsystem: agent
tags: [trajectory, sft, langgraph, model-training, json, typeddict]

# Dependency graph
requires:
  - phase: 19-inference-provider-abstraction
    provides: InferenceProvider factory used by agent generate_node
provides:
  - trajectory JSON persistence library (record_trajectory, load_trajectory, format_for_sft)
  - RuneState.session_id field for trajectory linkage
  - model-training workspace dependency wired into rune-agent
affects: [20-agent-loop plan 02 (save_trajectory_node depends on these functions)]

# Tech tracking
tech-stack:
  added: [model-training workspace dep in rune-agent]
  patterns:
    - "TDD red-green: write failing tests first, then implement to pass"
    - "Env var reads inside function body (not module level) for monkeypatch testability — same pattern as Phase 19 factory.py"
    - "tmp_path + monkeypatch.setenv for file I/O test isolation"

key-files:
  created: []
  modified:
    - libs/model-training/src/model_training/trajectory.py
    - libs/model-training/src/model_training/__init__.py
    - libs/model-training/tests/test_trajectory.py
    - services/rune-agent/src/rune_agent/state.py
    - services/rune-agent/pyproject.toml

key-decisions:
  - "RUNE_TRAJECTORY_DIR env var read inside function body (not module level) for monkeypatch testability"
  - "record_trajectory() extended with keyword-only task_description, task_type, adapter_ids args — save_trajectory_node passes these from RuneState"
  - "format_for_sft() uses reversed(steps) to find last tests_passed=True step as assistant content"
  - "SYSTEM_PROMPT defined as module constant in trajectory.py — same prompt used by generate_node"

patterns-established:
  - "Trajectory persistence: ~/.rune/trajectories/{session_id}.json with json.dumps(indent=2)"
  - "SFT format: [system, user, assistant] 3-message list from successful trajectory"
  - "FileNotFoundError propagates naturally from load_trajectory for non-existent session_id"

requirements-completed: [AGENT-05, TRAIN-01, TRAIN-02]

# Metrics
duration: 3min
completed: 2026-03-05
---

# Phase 20 Plan 01: Trajectory Persistence Library Summary

**Trajectory JSON persistence (record/load/format_for_sft) with SFT chat formatting, session_id in RuneState, and model-training wired as rune-agent workspace dependency**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-05T15:52:35Z
- **Completed:** 2026-03-05T15:55:09Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Implemented record_trajectory(), load_trajectory(), format_for_sft() replacing all NotImplementedError stubs
- 9 green-phase behavior tests covering file I/O, round-trip, error cases, and SFT formatting — all passing
- Added session_id: str field to RuneState TypedDict for trajectory linkage
- Wired model-training as workspace dependency in rune-agent (pyproject.toml all 4 locations)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement trajectory library (record, load, format_for_sft)** - `099e59a` (feat)
2. **Task 2: Add session_id to RuneState and wire model-training dependency** - `abfaadd` (feat)

**Plan metadata:** (final docs commit, see below)

_Note: Task 1 used TDD pattern (RED then GREEN)_

## Files Created/Modified
- `libs/model-training/src/model_training/trajectory.py` - Full implementation: record_trajectory writes JSON, load_trajectory reads by ID, format_for_sft converts to SFT chat messages
- `libs/model-training/src/model_training/__init__.py` - Re-exports all 3 trajectory functions with __all__
- `libs/model-training/tests/test_trajectory.py` - 9 behavior tests (replaced 3 NotImplementedError tests)
- `services/rune-agent/src/rune_agent/state.py` - Added session_id: str field and docstring
- `services/rune-agent/pyproject.toml` - Added model-training to dependencies, mypy_path, pythonpath, uv.sources

## Decisions Made
- Env var RUNE_TRAJECTORY_DIR read inside function body (not module level) so monkeypatch.setenv() works in tests — same pattern established in Phase 19 factory.py
- record_trajectory() signature extended with keyword-only args (task_description, task_type, adapter_ids) — save_trajectory_node (Plan 02) will pass these directly from RuneState
- format_for_sft() uses reversed(steps) to find last step with tests_passed=True as assistant content, handling multi-attempt sessions correctly
- SYSTEM_PROMPT defined as module constant "You are a Python code generator. Output only code, no explanation." — same prompt used in generate_node

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Trajectory persistence infrastructure complete — save_trajectory_node (Plan 02) can call record_trajectory() and format_for_sft() directly
- RuneState.session_id field ready for UUID4 assignment in graph initialization
- model-training importable from rune-agent service code

---
*Phase: 20-agent-loop*
*Completed: 2026-03-05*
