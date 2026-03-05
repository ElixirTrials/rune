# Phase 20: Agent Loop - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Backend-agnostic generate → execute → reflect → save_trajectory cycle. Agent invokes InferenceProvider.generate() for code generation, runs code in a sandboxed subprocess, accumulates trajectory data, and persists structured JSON trajectories to disk. Also implements trajectory recording and SFT format conversion in model-training lib. Graph topology (should_retry, create_graph) is already implemented — this phase fills in the node implementations and trajectory persistence.

</domain>

<decisions>
## Implementation Decisions

### Subprocess sandbox
- Simple `subprocess.run()` with configurable timeout — no file/network restrictions at this stage
- Docker container isolation deferred to ADV-03
- Default timeout: 30 seconds, configurable via RuneState or env var
- Exit code determines tests_passed: `tests_passed = (exit_code == 0)`
- Concatenate generated_code + test_suite into a single .py file, run it — mirrors HumanEval evaluation pattern
- Run in a fresh temp directory (tmpdir) to prevent accidental writes to the project tree

### Prompt construction
- First attempt prompt includes both task_description AND test_suite — model sees the tests it needs to pass
- System prompt sets the role (e.g., "You are a Python code generator. Output only code, no explanation.") + user prompt contains task + tests
- On retry attempts, append prior generated_code + stdout + stderr + exit_code to the prompt — classic self-repair pattern
- Extract code from ```python ... ``` blocks in the response; fallback: treat entire response as code if no blocks found

### Trajectory storage
- Storage directory: `~/.rune/trajectories/` — consistent with `~/.rune/registry.db` from Phase 18
- Configurable via `RUNE_TRAJECTORY_DIR` env var
- File naming: `{session_id}.json` — one file per session, session_id is UUID so no collisions
- Session ID created by caller (uuid4 at agent start), passed through RuneState — record_trajectory() receives it, doesn't generate it
- Session-level metadata stored: session_id, task_description, task_type, adapter_ids, outcome, timestamp, steps[]
- Each step contains: generated_code, stdout, stderr, exit_code, tests_passed — mirrors RuneState fields

### SFT format mapping
- Task as user message, successful code as assistant message — standard SFT chat pattern
- System prompt included in output (role: "system") — trains model with same prompt structure used at inference
- Only successful trajectories formatted (outcome="success") — failed trajectories are noise for SFT
- Final successful attempt only — single user→assistant turn, not full retry chain
- format_for_sft() returns `list[dict[str, str]]` with role/content keys per existing signature

### Claude's Discretion
- Exact system prompt wording for generate_node
- Code block extraction regex details
- Temp directory cleanup strategy
- Error message formatting in trajectory steps
- Whether to add model-training as workspace dep for rune-agent or import via shared path

</decisions>

<specifics>
## Specific Ideas

- Concatenate + run pattern mirrors HumanEval evaluation — generated function definitions followed by assert-based test suite in one script
- Self-repair on retry: prompt includes "Your previous attempt produced the following errors:" + stderr/stdout — standard approach from self-repair literature
- Trajectory JSON should be human-readable (json.dumps with indent=2) for debugging

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `InferenceProvider.generate()`: async, returns `GenerationResult(text, model, adapter_id, token_count, finish_reason)` — generate_node calls this
- `get_provider()` / `get_provider_for_step()`: factory with (type, url) cache — agent obtains providers through these
- `should_retry()` + `create_graph()`: already implemented in graph.py — graph topology is locked
- `RuneState` TypedDict: has all fields except `session_id` — needs one field addition (AGENT-05)
- `make_coding_session` fixture in root conftest.py: factory with session_id defaults — usable for trajectory tests
- `record_trajectory()`, `load_trajectory()`, `format_for_sft()` stubs in model-training/trajectory.py — signatures defined, need implementation

### Established Patterns
- Async providers, sync DB ops — generate_node is async (calls InferenceProvider), trajectory persistence is sync (file I/O)
- Env var for service discovery: `INFERENCE_PROVIDER`, `VLLM_BASE_URL`, `DATABASE_URL` — follow with `RUNE_TRAJECTORY_DIR`
- Google-style docstrings, mypy strict, ruff
- TDD: existing red-phase tests in test_nodes.py (4 tests) and test_trajectory.py (3 tests) — pivot to green

### Integration Points
- `rune-agent/nodes.py` → `inference.get_provider()` for code generation
- `rune-agent/nodes.py` → `model_training.record_trajectory()` for trajectory persistence
- `rune-agent/pyproject.toml` needs `model-training` as workspace dependency (currently missing)
- `rune-agent/state.py` → add `session_id: str` field to RuneState

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 20-agent-loop*
*Context gathered: 2026-03-05*
