---
phase: 20-agent-loop
plan: 02
subsystem: agent
tags: [langgraph, inference, subprocess, trajectory, tdd, nodes]

# Dependency graph
requires:
  - phase: 19-inference-provider-abstraction
    provides: get_provider() factory called in generate_node
  - phase: 20-agent-loop plan 01
    provides: record_trajectory() called in save_trajectory_node, session_id in RuneState
provides:
  - 4 fully implemented node functions (generate_node, execute_node, reflect_node, save_trajectory_node)
  - complete generate-execute-reflect-save agent loop operational
affects: [Phase 21 GPU/infra setup can now validate end-to-end agent loop]

# Tech tracking
tech-stack:
  added: [py.typed markers for inference and model-training libs]
  patterns:
    - "TDD red-green: tests written before implementation, confirmed RED then GREEN"
    - "Env var reads inside function body (not module level) for monkeypatch testability"
    - "re.search with re.DOTALL for python code block extraction with fallback to full text"
    - "subprocess.run with TimeoutExpired handling for sandboxed code execution"
    - "LangGraph immutable state: list concatenation (not .append()) in reflect_node"

key-files:
  created:
    - libs/inference/src/inference/py.typed
    - libs/model-training/src/model_training/py.typed
  modified:
    - services/rune-agent/src/rune_agent/nodes.py
    - services/rune-agent/tests/test_nodes.py

key-decisions:
  - "generate_node reads RUNE_MODEL env var inside function body — same pattern as Phase 19 factory.py for monkeypatch testability"
  - "execute_node reads RUNE_EXEC_TIMEOUT inside function body for monkeypatch testability"
  - "py.typed markers added to inference and model-training libs to satisfy mypy strict import-untyped check"
  - "reflect_node uses state['trajectory'] + [step] (new list) not .append() — LangGraph requires immutable state updates"

patterns-established:
  - "Code extraction: re.search(r'```python\\s*(.*?)```', text, re.DOTALL) with strip() + fallback to full text"
  - "Subprocess sandbox: tempfile.TemporaryDirectory() + subprocess.run with capture_output=True, text=True"
  - "Retry prompt: 'Your previous attempt produced the following errors:' prefix with prior code, stdout, stderr, exit_code"

requirements-completed: [AGENT-01, AGENT-02, AGENT-03, AGENT-04, AGENT-06]

# Metrics
duration: ~7min
completed: 2026-03-05
---

# Phase 20 Plan 02: Agent Loop Node Implementations Summary

**4 agent loop nodes fully implemented: generate_node (InferenceProvider via get_provider), execute_node (subprocess sandbox with timeout), reflect_node (immutable trajectory accumulation), save_trajectory_node (record_trajectory persistence)**

## Performance

- **Duration:** ~7 min
- **Completed:** 2026-03-05
- **Tasks:** 2
- **Files modified:** 4 (2 nodes files, 2 py.typed markers)

## Accomplishments

- Replaced all 4 NotImplementedError stubs in nodes.py with complete implementations
- Rewrote 4 placeholder tests as 11 green-phase behavior tests covering all node semantics
- generate_node: builds prompts with retry context, calls InferenceProvider.generate() via get_provider(), extracts code from python blocks with fallback
- execute_node: runs generated_code + test_suite concatenated in subprocess sandbox with configurable timeout and graceful TimeoutExpired handling
- reflect_node: increments attempt_count, appends step dict to trajectory using immutable list concatenation (LangGraph requirement)
- save_trajectory_node: determines 'success'/'exhausted' outcome, persists via record_trajectory()
- Added py.typed markers to inference and model-training libs to pass mypy strict mode
- All quality gates green: mypy strict, ruff clean, 30 tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement all 4 node functions (TDD green)** - `3c39e0a` (feat)
2. **Task 2: Quality gate — py.typed markers and ruff line length fix** - `7df42b4` (chore)

## Files Created/Modified

- `services/rune-agent/src/rune_agent/nodes.py` - 4 complete node implementations: generate_node, execute_node, reflect_node, save_trajectory_node with private helpers _build_prompt() and _extract_code()
- `services/rune-agent/tests/test_nodes.py` - 11 behavior tests: 4 generate_node (prompt construction, code extraction, adapter handling), 3 execute_node (pass, fail, timeout), 2 reflect_node (increment+append, immutability), 2 save_trajectory_node (success/exhausted outcomes)
- `libs/inference/src/inference/py.typed` - Added for mypy strict import-untyped compliance
- `libs/model-training/src/model_training/py.typed` - Added for mypy strict import-untyped compliance

## Decisions Made

- RUNE_MODEL and RUNE_EXEC_TIMEOUT env vars read inside function bodies (not module level) — follows Phase 19 factory.py pattern for monkeypatch testability in tests
- py.typed markers added to both libs rather than mypy ignore overrides — correct long-term solution, enables full type checking of imported packages
- reflect_node uses `state["trajectory"] + [step]` (new list) not `.append()` — LangGraph state must be immutable; in-place mutation causes undefined graph behavior

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Added py.typed markers to inference and model-training libs**
- **Found during:** Task 2 (mypy quality gate)
- **Issue:** mypy strict mode reported `import-untyped` errors for `inference` and `model_training.trajectory` imports despite mypy_path being configured; packages lacked py.typed markers
- **Fix:** Created empty `py.typed` files in both lib packages — standard PEP 561 solution
- **Files modified:** `libs/inference/src/inference/py.typed`, `libs/model-training/src/model_training/py.typed`
- **Commit:** `7df42b4`

**2. [Rule 1 - Bug] Fixed ruff E501 line-too-long in nodes.py**
- **Found during:** Task 2 (ruff quality gate)
- **Issue:** Comment in reflect_node was 89 characters (limit is 88)
- **Fix:** Shortened comment text by 1 character
- **Files modified:** `services/rune-agent/src/rune_agent/nodes.py`
- **Commit:** `7df42b4`

## Agent Loop End-to-End Status

The full generate -> execute -> reflect -> [retry|save] cycle is now operational:
- `create_graph()` compiles the StateGraph with all 4 nodes
- `should_retry()` routes based on tests_passed and attempt_count vs max_attempts
- All nodes are async and return partial state dicts for LangGraph merging
- Phase 20 (agent loop) is complete

## Next Phase Readiness

- Agent loop fully operational — can be invoked with a RuneState dict containing task_description, test_suite, adapter_ids, session_id, max_attempts
- Phase 21 (GPU/infra) can validate end-to-end loop against a live vLLM instance
- Trajectory files are saved to ~/.rune/trajectories/{session_id}.json for Phase 22 LoRA training

---
*Phase: 20-agent-loop*
*Completed: 2026-03-05*
