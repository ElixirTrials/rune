---
phase: 20-agent-loop
verified: 2026-03-05T17:00:00Z
status: passed
score: 15/15 must-haves verified
re_verification: false
---

# Phase 20: Agent Loop Verification Report

**Phase Goal:** Users can invoke the Rune agent on a coding task and observe a complete generate → execute → reflect → save_trajectory cycle through the InferenceProvider interface, with trajectory data persisted to disk
**Verified:** 2026-03-05
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

All truths are drawn from Plan 01 and Plan 02 must_haves frontmatter.

**Plan 01 Truths (AGENT-05, TRAIN-01, TRAIN-02):**

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | RuneState TypedDict includes session_id field accessible as state['session_id'] | VERIFIED | `state.py` line 36: `session_id: str` declared in TypedDict body with docstring annotation |
| 2 | record_trajectory() writes a JSON file to configured trajectory directory with all session metadata | VERIFIED | `trajectory.py` lines 52-70: builds dict with all 7 fields, calls `file_path.write_text(json.dumps(...))` |
| 3 | load_trajectory() reads a previously recorded trajectory by session_id and returns the full dict | VERIFIED | `trajectory.py` lines 73-88: reads from `{trajectory_dir}/{trajectory_id}.json`, returns `json.loads()` |
| 4 | format_for_sft() converts a successful trajectory to system/user/assistant message list | VERIFIED | `trajectory.py` lines 91-123: returns 3-message list when `outcome == "success"` and a passing step exists |
| 5 | format_for_sft() returns empty list for non-success trajectories | VERIFIED | `trajectory.py` line 104-105: `if trajectory.get("outcome") != "success": return []` |
| 6 | rune-agent pyproject.toml declares model-training as workspace dependency with mypy and pytest paths | VERIFIED | `pyproject.toml` lines 10, 33, 35, 43: all 4 locations correctly configured |

**Plan 02 Truths (AGENT-01 through AGENT-04, AGENT-06):**

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 7 | generate_node calls InferenceProvider.generate() via get_provider() and returns extracted code | VERIFIED | `nodes.py` line 103: `provider = get_provider()`, line 107: `await provider.generate(...)`, line 122: `return {"generated_code": extracted}` |
| 8 | generate_node constructs prompts with task_description and test_suite on first attempt, and appends prior failure context on retries | VERIFIED | `nodes.py` lines 40-62: `_build_prompt()` returns base prompt at attempt_count==0; includes "Your previous attempt produced the following errors:" prefix on retry |
| 9 | generate_node extracts code from python code blocks in LLM response, with fallback to full text | VERIFIED | `nodes.py` lines 74-77: `re.search(r'```python\s*(.*?)```', text, re.DOTALL)` with fallback `return text.strip()` |
| 10 | execute_node runs generated_code + test_suite in a subprocess sandbox with configurable timeout | VERIFIED | `nodes.py` lines 143-165: concatenates scripts, writes to tmpdir, runs `subprocess.run(["python", ...], timeout=timeout)` |
| 11 | execute_node returns stdout, stderr, exit_code, and tests_passed (exit_code == 0) | VERIFIED | `nodes.py` lines 177-182: returns all 4 fields; `tests_passed = proc.returncode == 0` |
| 12 | execute_node handles subprocess timeout gracefully with error message | VERIFIED | `nodes.py` lines 165-170: `except subprocess.TimeoutExpired:` returns `stderr = f"Execution timed out after {timeout}s"` |
| 13 | reflect_node increments attempt_count and appends step data to trajectory without LLM call | VERIFIED | `nodes.py` lines 205-226: no LLM call; increments count and uses list concatenation (not append) |
| 14 | save_trajectory_node determines outcome (success/exhausted), calls record_trajectory(), and returns outcome | VERIFIED | `nodes.py` lines 249-265: `outcome = "success" if state["tests_passed"] else "exhausted"`, calls `record_trajectory()`, returns `{"outcome": outcome}` |
| 15 | Agent loop closes end-to-end: graph invocation with all 4 nodes produces a terminal outcome | VERIFIED | `graph.py` lines 57-66: all 4 nodes wired with `add_node`, connected via `add_edge`/`add_conditional_edges`; `should_retry()` routes to either generate or save_trajectory |

**Score:** 15/15 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `services/rune-agent/src/rune_agent/state.py` | RuneState TypedDict with session_id field | VERIFIED | 49 lines; `session_id: str` present at line 36 |
| `libs/model-training/src/model_training/trajectory.py` | Trajectory recording, loading, and SFT formatting | VERIFIED | 124 lines; exports record_trajectory, load_trajectory, format_for_sft — all fully implemented, no stubs |
| `libs/model-training/src/model_training/__init__.py` | Public API exports for trajectory functions | VERIFIED | 5 lines; `from model_training.trajectory import format_for_sft, load_trajectory, record_trajectory` + `__all__` |
| `libs/model-training/tests/test_trajectory.py` | Green-phase behavior tests | VERIFIED | 166 lines (>50 min); 9 tests, all pass |
| `services/rune-agent/src/rune_agent/nodes.py` | 4 implemented node functions | VERIFIED | 267 lines (>80 min); exports generate_node, execute_node, reflect_node, save_trajectory_node — all implemented, zero NotImplementedError stubs |
| `services/rune-agent/tests/test_nodes.py` | Green-phase behavior tests for all 4 nodes | VERIFIED | 295 lines (>80 min); 11 tests, all pass |
| `libs/inference/src/inference/py.typed` | PEP 561 marker for mypy strict | VERIFIED | Exists (added in Plan 02 deviation fix) |
| `libs/model-training/src/model_training/py.typed` | PEP 561 marker for mypy strict | VERIFIED | Exists (added in Plan 02 deviation fix) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `libs/model-training/src/model_training/trajectory.py` | `~/.rune/trajectories/{session_id}.json` | `file_path.write_text(json.dumps(...))` | WIRED | Line 68: `file_path.write_text(json.dumps(trajectory, indent=2))` — exact pattern confirmed |
| `libs/model-training/src/model_training/__init__.py` | `libs/model-training/src/model_training/trajectory.py` | re-export of trajectory functions | WIRED | Line 3: `from model_training.trajectory import format_for_sft, load_trajectory, record_trajectory` |
| `services/rune-agent/pyproject.toml` | `libs/model-training` | workspace dependency declaration | WIRED | Line 43: `model-training = { workspace = true }` |
| `services/rune-agent/src/rune_agent/nodes.py` | `libs/inference/src/inference/factory.py` | `get_provider()` call in generate_node | WIRED | Line 10: `from inference import GenerationResult, get_provider`; line 103: `provider = get_provider()` |
| `services/rune-agent/src/rune_agent/nodes.py` | `libs/model-training/src/model_training/trajectory.py` | `record_trajectory()` call in save_trajectory_node | WIRED | Line 11: `from model_training.trajectory import record_trajectory`; line 251: called with all required kwargs |
| `services/rune-agent/src/rune_agent/nodes.py` | `subprocess.run` | execute_node sandboxed execution | WIRED | Line 154: `subprocess.run(["python", script_path], capture_output=True, text=True, timeout=timeout, cwd=tmpdir)` |
| `services/rune-agent/src/rune_agent/graph.py` | all 4 node functions | `add_node` + `add_edge` wiring | WIRED | Lines 57-66: generate, execute, reflect, save_trajectory all wired; conditional routing on reflect via `should_retry` |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| AGENT-01 | Plan 02 | generate_node calls InferenceProvider.generate() with task description and optional adapter, backend-agnostic | SATISFIED | `nodes.py` calls `get_provider()` (factory abstraction) not a concrete provider; adapter_id passed from state |
| AGENT-02 | Plan 02 | execute_node runs generated code in sandboxed subprocess with timeout, returns stdout/stderr/exit_code/tests_passed | SATISFIED | `nodes.py` execute_node: tempfile sandbox, configurable RUNE_EXEC_TIMEOUT, TimeoutExpired handling, all 4 return fields |
| AGENT-03 | Plan 02 | reflect_node accumulates trajectory data (attempt count, code, results) without LLM call | SATISFIED | `nodes.py` reflect_node: no LLM import or call; increments count + appends step via list concat |
| AGENT-04 | Plan 02 | save_trajectory_node persists trajectory via record_trajectory() and sets outcome | SATISFIED | `nodes.py` save_trajectory_node calls record_trajectory with all state fields; returns outcome |
| AGENT-05 | Plan 01 | RuneState includes session_id field for trajectory persistence | SATISFIED | `state.py` line 36: `session_id: str` in TypedDict |
| AGENT-06 | Plan 02 | Agent loop closes end-to-end: generate → execute → reflect → retry/save with should_retry routing | SATISFIED | `graph.py` wires all 4 nodes; should_retry conditional edge routes on tests_passed and attempt_count vs max_attempts |
| TRAIN-01 | Plan 01 | User can record a coding trajectory as structured JSON via record_trajectory(session_id, steps, outcome) | SATISFIED | `trajectory.py` record_trajectory: writes JSON with full metadata; 4 passing tests confirm behavior |
| TRAIN-02 | Plan 01 | User can convert trajectory data to SFT chat format via format_for_sft() | SATISFIED | `trajectory.py` format_for_sft: returns [system, user, assistant] for success trajectories; 3 passing tests confirm behavior |

**Orphaned requirements check:** REQUIREMENTS.md confirms AGENT-01 through AGENT-06 and TRAIN-01, TRAIN-02 all mapped to Phase 20. No orphaned IDs.

---

### Anti-Patterns Found

No anti-patterns detected.

| File | Pattern Searched | Result |
|------|-----------------|--------|
| `nodes.py` | TODO/FIXME/NotImplementedError | None found |
| `trajectory.py` | TODO/FIXME/NotImplementedError | None found |
| `state.py` | TODO/FIXME/NotImplementedError | None found |
| `__init__.py` | TODO/FIXME/NotImplementedError | None found |

---

### Test Execution Results

All tests pass as of verification:

```
libs/model-training/tests/test_trajectory.py: 9/9 passed
services/rune-agent/tests/test_graph.py: 4/4 passed
services/rune-agent/tests/test_importability.py: 1/1 passed
services/rune-agent/tests/test_nodes.py: 11/11 passed
Total: 25/25 passing
```

Quality gate results:
- `uv run mypy services/rune-agent/src/`: Success (no issues found)
- `uv run mypy libs/model-training/src/`: Success (no issues found)
- `uv run ruff check services/rune-agent/src/ libs/model-training/src/`: All checks passed

Commits verified in git log:
- `099e59a` — feat(20-01): implement trajectory library
- `abfaadd` — feat(20-01): add session_id to RuneState and wire model-training dependency
- `3c39e0a` — feat(20-02): implement all 4 agent loop node functions with tests
- `7df42b4` — chore(20-02): add py.typed markers and fix ruff line length

---

### Human Verification Required

None. All behaviors are unit-tested and verifiable programmatically.

One item that requires a live vLLM instance to verify at runtime (not a gap — expected to be validated in Phase 21):
- **End-to-end graph invocation against a real InferenceProvider**: tests mock `get_provider()`. Actual network call to vLLM is deferred to Phase 21 GPU/infra setup.

---

### Summary

Phase 20 fully achieves its goal. The complete generate → execute → reflect → save_trajectory cycle is operational:

- All 4 node functions are implemented with zero stubs in `nodes.py`
- The LangGraph graph in `graph.py` wires all 4 nodes with correct edge routing, including the `should_retry` conditional that enables retry loops
- `record_trajectory()` writes structured JSON to `~/.rune/trajectories/{session_id}.json` (configurable via `RUNE_TRAJECTORY_DIR`)
- `format_for_sft()` converts successful trajectories to the SFT 3-message chat format required for Phase 22 LoRA training
- `RuneState.session_id` is present, typed, and documented
- `model-training` is wired as a workspace dependency in all 4 required locations of `rune-agent/pyproject.toml`
- All 8 requirement IDs (AGENT-01 through AGENT-06, TRAIN-01, TRAIN-02) are satisfied with test coverage

---

_Verified: 2026-03-05_
_Verifier: Claude (gsd-verifier)_
