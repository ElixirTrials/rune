# Prompt Layer Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the Jinja2 prompt/trajectory layer with the xgrammar schemas, finish the half-migrated continuation path, and establish a single shared rendering path so the mined training corpus matches what the engine renders at serve time — without silently dropping the SFT completion target.

**Architecture:** All work lands on a feature branch off `fix/pr45-review-correctness`. Phase 1 (P0) is the smoke-gated merge unit: it is the only phase whose changes can move engine/training quality, so the GPU smoke gate runs (directly, on this GPU instance) after Phase 1 and merge is blocked until it shows no regression. Phases 2–3 (hardening, polish) are CPU-verifiable and merge on green CI.

**Runtime:** This is a GPU instance. Per `CLAUDE.md`, GPU/long-running ops (engine runs, smoke tests, distillation, benchmarks) are executed directly here — capture logs, and prefer background runs for multi-minute jobs. The "deferred GPU imports" rule still holds (the code must stay importable in CPU-only CI).

**Tech Stack:** Python 3.12, `uv`, Jinja2 (`StrictUndefined`), Pydantic v2, pytest, ruff, mypy (strict). Engine = LangGraph single-loop. Training = trl `SFTTrainer` via `DiffAwareSFTTrainer`.

---

## Why this ordering (read before starting)

Two findings from the review drive the structure:

1. **The CPU gate (`ruff`/`mypy`/`pytest unit`) is blind to the riskiest change (train/serve, Task 4).** Task 4 is therefore built to be *shape-verifiable on CPU* first (producer → miner → trainer column contract is unit-tested), and then confirmed end-to-end with a real GPU run on this instance (smoke engine run → mine → tiny distillation) before merge.

2. **The engine performs TWO distinct serve renders per action**, both at `src/rune/engine/graph.py:174-175`:
   - `trajectory_text = render_template(action.trajectory_template, **ctx)` → fed to `model.generate_adapter()` (conditions the hypernetwork).
   - `prompt_text = render_template(action.prompt_template, **ctx)` → fed to `model.generate()` (the model's user prompt).
   Faithful train/serve capture therefore needs **both renders plus the output**. The original review plan captured only `trajectory` + `output` and sourced output from `result.text`; both are wrong (see Task 4).

**Hard rules from `CLAUDE.md`:** GPU/long-running ops run directly on this instance (capture logs; background multi-minute jobs); GPU imports stay deferred inside function bodies; no backward-compat shims; `uv run` for all Python; diff-style edits; no comments unless the *why* is non-obvious.

---

## File Structure

**Created:**
- `src/rune/mining/session_log.py` — `write_session()` producer: turns a final `RunState` into `session.jsonl` + `metadata.json`. The producer mining never had.
- `src/rune/templates/prompt_code_continue.j2` — continuation user-prompt template (replaces the inline `cont_user` string).
- `tests/unit/test_session_log.py` — producer + corpus round-trip / column-contract tests.
- `tests/unit/test_templates.py` — parametrized render tests for all 13 templates.

**Modified:**
- `src/rune/templates/decompose.j2`, `prompt_decompose_concise.j2` — describe the enforced JSON contract (Task 1).
- `src/rune/engine/continuation.py` — `CONT_SYSTEM_PROMPT` constant (Task 2).
- `src/rune/engine/graph.py` — reference the constant + new prompt template; remove `resume_tail`; delete debug block; populate new `StepRecord` fields; single truncation layer (Tasks 2,3,4,8).
- `src/rune/model/hypernetwork.py` — delete all `# #region agent log` blocks (defs + 4 call sites) (Task 3).
- `src/rune/engine/state.py` — `StepRecord` gains `trajectory_text`/`prompt_text`/`output_text` (Task 4).
- `src/rune/mining/miner.py` — `_render_trajectory`/`extract_trajectories` consume stored renders and emit the `prompt`/`completion` contract (Task 4).
- `src/rune/training/d2l_train.py` — `run_distillation` builds the dataset from the `prompt`/`completion` columns (Task 4).
- `src/rune/bench/runner.py` — `run_benchmark` writes one session dir per task (Task 4).
- `tools/cont_probe.py` — add a `"production"` trajectory flavor calling `render_template("code_continue", ...)` (Task 4).
- `tools/smoke_test_engine.py` — add `--dump-sessions DIR` so the smoke run emits a real corpus (Task 4 / smoke gate).
- `tests/unit/test_state.py`, `tests/unit/test_miner.py`, `tests/unit/test_parse.py` — fix stale ref, update fixtures, add schema-key assertions (Tasks 1,4,5).
- `src/rune/engine/policy.py` — cap `integration_doc` (Task 7).
- `src/rune/config.py` — `prompt_version` field (Task 9).

---

## Phase 0 — Branch setup

### Task 0: Create the feature branch

- [ ] **Step 1: Branch off the integration branch**

```bash
git checkout fix/pr45-review-correctness
git pull --ff-only
git checkout -b fix/prompt-layer-remediation
```

- [ ] **Step 2: Confirm a clean baseline**

Run: `uv sync && uv run pytest tests/unit/ -q && uv run ruff check . && uv run mypy src/`
Expected: all green (this is the pre-change baseline; if anything is red, stop and report).

---

## Phase 1 — P0 fixes + cleanup (smoke-gated merge unit)

### Task 1: Schema alignment for decompose (#1)

The decompose templates instruct the model to emit a "numbered list", but `DecomposeResult` (`src/rune/engine/parse.py:29`) forces xgrammar JSON `{"subtasks":[...]}`. The instructions contradict the grammar. `decompose.j2` is the *trajectory* template (conditions the adapter); `prompt_decompose_concise.j2` is the *prompt* template. Both must describe the JSON contract.

**Files:**
- Modify: `src/rune/templates/decompose.j2`
- Modify: `src/rune/templates/prompt_decompose_concise.j2`
- Test: `tests/unit/test_parse.py` (extend the existing `TestRender` block near line 16)

- [ ] **Step 1: Write the failing test (schema keys must appear in the render)**

A pure-mechanics render test (StrictUndefined + undeclared-vars) cannot catch "numbered list" copy — the template renders fine. Assert the rendered text actually names the JSON schema keys. Add to `tests/unit/test_parse.py`:

```python
    def test_decompose_describes_json_schema(self) -> None:
        text = render_template("decompose", project="build a calculator", subtasks=[])
        assert "subtasks" in text
        assert "depends_on" in text
        assert "numbered list" not in text.lower()

    def test_decompose_prompt_describes_json_schema(self) -> None:
        text = render_template(
            "prompt_decompose_concise", task_description="build a calculator"
        )
        assert "subtasks" in text
        assert "depends_on" in text
        assert "numbered list" not in text.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_parse.py -k decompose_describes -v`
Expected: FAIL — both assert `"numbered list" not in ...` fail (current templates say "numbered list").

- [ ] **Step 3: Rewrite `src/rune/templates/decompose.j2`**

Replace the entire file with:

```jinja
ROLE: project-decomposer
PROJECT: {{ project[:1200] }}
METHODOLOGY: Decompose into independent subtasks that can be implemented
and tested in isolation. Each subtask should:
- Have a clear, focused scope (one layer or feature)
- Produce testable output independently
- Minimize dependencies on other subtasks
Order by dependency: data/models first, then logic, then interface, then integration.

Output a JSON object: {"subtasks": [{"name": ..., "description": ..., "depends_on": [...]}]}
- name: short identifier for the subtask
- description: one-line description of what to implement
- depends_on: list of subtask names that must complete first (use [] for none)
No preamble, no analysis, no reasoning outside the JSON.

ANTI-PATTERNS (do NOT produce these):
- A single monolithic subtask covering everything
- 10+ micro-subtasks with heavy overlap
- Chain-of-thought or reasoning steps disguised as subtask entries

Example:
{"subtasks": [
  {"name": "models", "description": "data structures", "depends_on": []},
  {"name": "logic", "description": "core algorithm", "depends_on": ["models"]},
  {"name": "cli", "description": "command-line interface", "depends_on": ["logic"]}
]}
```

- [ ] **Step 4: Rewrite `src/rune/templates/prompt_decompose_concise.j2`**

Replace the entire file with:

```jinja
Decompose into subtasks with dependencies. Output a JSON object:
{"subtasks": [{"name": ..., "description": ..., "depends_on": [...]}]}
Use depends_on: [] when a subtask has no prerequisites.
No preamble, no analysis, no reasoning outside the JSON.

BAD: a "subtask" named "Analyze the request" — that is reasoning, NOT a subtask.

[USER TASK]
{{ task_description }}
[/USER TASK]
```

(The `[USER TASK] … [/USER TASK]` delimiters also satisfy the injection-delimiting goal #4 for this template — see Task 6.)

- [ ] **Step 5: Run to verify it passes + no other render test broke**

Run: `uv run pytest tests/unit/test_parse.py -k decompose -v`
Expected: PASS (new schema tests pass; existing `test_renders_jinja2` still passes).

- [ ] **Step 6: Audit the other 11 templates (no edits expected)**

Run: `uv run pytest tests/unit/ -q`
Read each of `plan.j2`, `code.j2`, `integrate.j2`, `diagnose.j2`, `code_repair.j2`, `prompt_*` against its Pydantic schema in `src/rune/engine/parse.py`. `PlanResult`/`CodeResult`/`IntegrateResult` are free-form `{...:str}` and `DiagnoseResult` copy already matches. Expected: no changes needed. If a mismatch is found, note it and stop for review (out of this task's scope).

- [ ] **Step 7: Commit**

```bash
git add src/rune/templates/decompose.j2 src/rune/templates/prompt_decompose_concise.j2 tests/unit/test_parse.py
git commit -m "fix(templates): describe enforced JSON schema in decompose prompts"
```

---

### Task 2: Complete the continuation path (#3, #11)

The continuation system string is hardcoded inline (`graph.py:244-249`), there is no prompt template for the continuation user turn, and `resume_tail` is computed (`graph.py:260`) but unused (`code_continue.j2` uses `accumulated_code[-3500:]`).

**Files:**
- Modify: `src/rune/engine/continuation.py`
- Create: `src/rune/templates/prompt_code_continue.j2`
- Modify: `src/rune/engine/graph.py:244-261`
- Test: `tests/unit/test_continuation.py` (create if absent) and `tests/unit/test_parse.py`

- [ ] **Step 1: Write the failing test for the constant + template**

Create/extend `tests/unit/test_continuation.py`:

```python
from rune.engine.continuation import CONT_SYSTEM_PROMPT
from rune.engine.parse import render_template


def test_cont_system_prompt_is_code_only() -> None:
    assert "Output only Python code" in CONT_SYSTEM_PROMPT
    assert "markdown" in CONT_SYSTEM_PROMPT.lower()


def test_prompt_code_continue_renders() -> None:
    text = render_template("prompt_code_continue", task_description="build a parser")
    assert "build a parser" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_continuation.py -k "cont_system or code_continue" -v`
Expected: FAIL — `ImportError: cannot import name 'CONT_SYSTEM_PROMPT'` and missing template.

- [ ] **Step 3: Add the constant to `src/rune/engine/continuation.py`**

After the imports (below line 8), add:

```python
CONT_SYSTEM_PROMPT = (
    "Output only Python code. No commentary, no explanations, "
    "no markdown fences. Continue exactly from where the code "
    "left off."
)
```

- [ ] **Step 4: Create `src/rune/templates/prompt_code_continue.j2`**

```jinja
{% if task_description %}{{ task_description[:200] }}{% endif %}
```

- [ ] **Step 5: Wire `graph.py` to the constant + template; drop `resume_tail`**

In `src/rune/engine/graph.py`, add `CONT_SYSTEM_PROMPT` to the existing import from `rune.engine.continuation` (line 15-19):

```python
from rune.engine.continuation import (
    CONT_SYSTEM_PROMPT,
    degeneration_score,
    extract_partial_code,
    validate_syntax,
)
```

Replace the inline `cont_sys`/`cont_user` block (lines 244-249) with:

```python
            cont_sys = CONT_SYSTEM_PROMPT
            cont_user = render_template("prompt_code_continue", **ctx)
```

Remove the unused `resume_tail` key from `cont_ctx` (lines 257-261) so it reads:

```python
                cont_ctx = {
                    **ctx,
                    "accumulated_code": accumulated_code,
                }
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/unit/test_continuation.py tests/unit/test_parse.py -q`
Expected: PASS.

- [ ] **Step 7: Lint + type check**

Run: `uv run ruff check . && uv run mypy src/`
Expected: clean (confirms `resume_tail` removal left no dangling reference).

- [ ] **Step 8: Commit**

```bash
git add src/rune/engine/continuation.py src/rune/templates/prompt_code_continue.j2 src/rune/engine/graph.py tests/unit/test_continuation.py
git commit -m "fix(engine): single-owner continuation system prompt + template; drop unused resume_tail"
```

---

### Task 3: Remove debug log blocks (cleanup)

Two debug instrumentation areas write to `.cursor/debug-88deb7.log`. The `graph.py` block is self-contained (an inline `try/except` at lines 177-215). The `hypernetwork.py` block is **not** self-contained: deleting only the `_dbg`/`_cuda_mem_mb` definitions (lines 13-60) leaves 4 live call sites (lines 92/100, 407, 430, 456) → `NameError`. All blocks are fenced with `# #region agent log` / `# #endregion`; remove **every** fenced region. mypy is the guard.

**Files:**
- Modify: `src/rune/engine/graph.py:177-215`
- Modify: `src/rune/model/hypernetwork.py` (all `# #region agent log` regions)

- [ ] **Step 1: Delete the `graph.py` debug block**

In `src/rune/engine/graph.py`, delete the entire region from the `# #region agent log` comment (line 177) through `# #endregion` (line 215) inclusive — the whole `try: ... except OSError: pass` instrumentation inside the `for action in actions:` loop.

- [ ] **Step 2: Delete all `hypernetwork.py` debug regions**

In `src/rune/model/hypernetwork.py`, delete every `# #region agent log` … `# #endregion` region:
- the definition region (lines ~13-60: `_DEBUG_LOG`, `_dbg`, `_cuda_mem_mb`),
- and each call-site region (around lines 92, 407, 430, 456 — the `_dbg(...)` calls).

Verify none remain:

Run: `grep -rn "agent log\|_dbg\|_cuda_mem_mb\|_DEBUG_LOG\|debug-88deb7" src/`
Expected: no output.

- [ ] **Step 3: Type check catches any dangling reference**

Run: `uv run mypy src/`
Expected: clean. (If mypy reports an undefined name, a call site was missed — remove it.)

- [ ] **Step 4: Run unit tests + lint**

Run: `uv run pytest tests/unit/ -q && uv run ruff check .`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/engine/graph.py src/rune/model/hypernetwork.py
git commit -m "chore: remove agent-log debug instrumentation"
```

---

### Task 4: Train/serve alignment — shared rendering path with an SFT contract (#2)

This is the riskiest change and is built to be **shape-verifiable on CPU**. The contract: the producer captures the engine's actual serve renders; the miner emits a corpus record carrying the hypernetwork-conditioning text **and** an SFT `prompt`/`completion` pair; the trainer consumes `prompt`/`completion` natively. **Do not** strip the existing output target without adding the `completion` column — that would silently leave the trainer with no target.

**Corpus record contract (v2):**
```json
{
  "task_id": "<benchmark>/<problem_id>",
  "trajectory": "<trajectory_template render — conditions the hypernetwork>",
  "prompt":     "<prompt_template render — the model user prompt>",
  "completion": "<the code/text the model produced — SFT target>",
  "metadata": {"phase": "<action>", "target": "<subtask|null>", "benchmark": ..., "problem_id": ...}
}
```

**Session line shape (`session.jsonl`):** `{step, action, target, trajectory, prompt, output, feedback}`.

#### Task 4a: Capture all three serve renders in `StepRecord`

**Files:**
- Modify: `src/rune/engine/state.py:65-72`
- Modify: `src/rune/engine/graph.py` (`step_node`, record construction at lines 372-382)
- Test: `tests/unit/test_graph_records.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_graph_records.py`:

```python
from rune.engine.state import StepRecord


def test_step_record_carries_renders() -> None:
    rec = StepRecord(
        step=0,
        action_name="code",
        target_subtask="_main",
        adapter_id="a",
        feedback=None,
        generated_code="print(1)",
        trajectory_text="ROLE: coder",
        prompt_text="write code",
        output_text="print(1)",
    )
    assert rec.trajectory_text == "ROLE: coder"
    assert rec.prompt_text == "write code"
    assert rec.output_text == "print(1)"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_graph_records.py -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'trajectory_text'`.

- [ ] **Step 3: Extend `StepRecord` in `src/rune/engine/state.py`**

Replace the `StepRecord` dataclass (lines 65-72) with:

```python
@dataclass(frozen=True)
class StepRecord:
    step: int
    action_name: str
    target_subtask: str | None
    adapter_id: str | None
    feedback: Feedback | None
    generated_code: str | None = None
    trajectory_text: str = ""
    prompt_text: str = ""
    output_text: str = ""
```

- [ ] **Step 4: Populate the fields in `step_node`**

The renders and outputs are already computed per action. Thread them into `results` and the `StepRecord` construction.

In `src/rune/engine/graph.py`, change the `results` accumulator type/append. The tuple currently is `(action, target_name, raw_text, adapter_id)` (line 167, 328). Extend it to also carry the two render strings and the per-action output text. Update line 167:

```python
    results: list[tuple[Action, str, str, str | None, str, str, str]] = []
```

Update the append (line 327-328) to include `trajectory_text`, `prompt_text`, and the action's output. For code/repair/integrate the SFT output is the assembled code (`raw_text` is `json.dumps({"code": accumulated})` after continuation); use the extracted code so it is exactly what ran. Replace lines 327-328 with:

```python
        target_name = action.target_subtask or ""
        output_text = extract_partial_code(raw_text) if action.executes_code else raw_text
        results.append(
            (action, target_name, raw_text, adapter_id, trajectory_text, prompt_text, output_text)
        )
```

Update the two later unpacks that iterate `results`:
- The code-map loop (line 344): `for a, name, text, _ in results:` → `for a, name, text, _, _traj, _prompt, _out in results:`
- The parse loop (line 366): `for action, target_name, raw, _ in results:` → `for action, target_name, raw, _, _traj, _prompt, _out in results:`
- `current_adapter` (line 387): `results[-1][3]` stays correct (index 3 is still `adapter_id`).

Replace the `records` comprehension (lines 372-382) with:

```python
    records = [
        StepRecord(
            step=state["step"],
            action_name=a.name,
            target_subtask=name,
            adapter_id=aid,
            feedback=feedback_map.get(name),
            generated_code=code_map.get(name) or None,
            trajectory_text=traj,
            prompt_text=prompt,
            output_text=out,
        )
        for a, name, _, aid, traj, prompt, out in results
    ]
```

- [ ] **Step 5: Run to verify it passes + nothing broke**

Run: `uv run pytest tests/unit/test_graph_records.py -q && uv run mypy src/ && uv run ruff check .`
Expected: PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add src/rune/engine/state.py src/rune/engine/graph.py tests/unit/test_graph_records.py
git commit -m "feat(engine): record trajectory/prompt/output renders on StepRecord"
```

#### Task 4b: Add the `write_session` producer

**Files:**
- Create: `src/rune/mining/session_log.py`
- Test: `tests/unit/test_session_log.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_session_log.py`:

```python
import json
from pathlib import Path

from rune.engine.state import Feedback, StepRecord
from rune.mining.session_log import write_session


def _state() -> dict:
    return {
        "trajectory": [
            StepRecord(
                step=0,
                action_name="code",
                target_subtask="_main",
                adapter_id="a0",
                feedback=Feedback(stdout="", stderr="", exit_code=0),
                generated_code="print(1)",
                trajectory_text="ROLE: coder",
                prompt_text="write a printer",
                output_text="print(1)",
            )
        ]
    }


def test_write_session_emits_jsonl_and_metadata(tmp_path: Path) -> None:
    out = write_session(
        _state(),
        {"benchmark": "mbpp", "problem_id": "7"},
        tmp_path / "sess",
    )
    lines = (out / "session.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["action"] == "code"
    assert rec["target"] == "_main"
    assert rec["trajectory"] == "ROLE: coder"
    assert rec["prompt"] == "write a printer"
    assert rec["output"] == "print(1)"
    assert rec["feedback"]["exit_code"] == 0
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["benchmark"] == "mbpp"
    assert meta["problem_id"] == "7"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_session_log.py -v`
Expected: FAIL — `ModuleNotFoundError: rune.mining.session_log`.

- [ ] **Step 3: Implement `src/rune/mining/session_log.py`**

```python
"""Session producer: serialize a final RunState into the mining input format."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def write_session(
    final_state: dict[str, Any],
    metadata: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Write session.jsonl + metadata.json from a finished engine run.

    Each StepRecord becomes one line carrying the exact serve-time renders
    (trajectory + prompt) and the produced output, so the mined corpus
    matches what the engine generated by construction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for rec in final_state.get("trajectory", []):
        feedback = asdict(rec.feedback) if rec.feedback is not None else None
        lines.append(
            json.dumps(
                {
                    "step": rec.step,
                    "action": rec.action_name,
                    "target": rec.target_subtask,
                    "trajectory": rec.trajectory_text,
                    "prompt": rec.prompt_text,
                    "output": rec.output_text,
                    "feedback": feedback,
                }
            )
        )
    (out_dir / "session.jsonl").write_text("\n".join(lines) + "\n" if lines else "")
    (out_dir / "metadata.json").write_text(json.dumps(metadata))
    return out_dir
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_session_log.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rune/mining/session_log.py tests/unit/test_session_log.py
git commit -m "feat(mining): write_session producer emitting serve-time renders"
```

#### Task 4c: Hook the producer into the benchmark runner

`run_benchmark` is the corpus producer mining never had. The main `rune run` CLI path is **not** wired here (a deliberate scope choice: benchmark runs are the controlled training-data source). Note this in the commit so it is explicit.

**Files:**
- Modify: `src/rune/bench/runner.py:97-163`
- Test: `tests/unit/test_bench_runner_sessions.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_bench_runner_sessions.py`:

```python
import asyncio
from pathlib import Path

from rune.bench.runner import BenchTask, run_benchmark
from rune.engine.state import Feedback, StepRecord


class _FakeEngine:
    async def ainvoke(self, state, config):
        return {
            **state,
            "integrated_code": "print(1)",
            "code_results": {"_main": "print(1)"},
            "trajectory": [
                StepRecord(
                    step=0,
                    action_name="code",
                    target_subtask="_main",
                    adapter_id="a0",
                    feedback=Feedback(stdout="", stderr="", exit_code=0),
                    generated_code="print(1)",
                    trajectory_text="ROLE: coder",
                    prompt_text="p",
                    output_text="print(1)",
                )
            ],
        }


def test_run_benchmark_writes_one_session_per_task(tmp_path: Path) -> None:
    tasks = [BenchTask(task_id="t1", description="print 1", test_code="assert True")]
    config = {"run_config": {"max_phase_iterations": 3}, "benchmark": "mbpp"}
    asyncio.run(
        run_benchmark(tasks, _FakeEngine(), config, sessions_dir=tmp_path)
    )
    assert (tmp_path / "t1" / "session.jsonl").exists()
    assert (tmp_path / "t1" / "metadata.json").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_bench_runner_sessions.py -v`
Expected: FAIL — `run_benchmark() got an unexpected keyword argument 'sessions_dir'`.

- [ ] **Step 3: Add the optional `sessions_dir` param + producer call**

In `src/rune/bench/runner.py`, add the import at the top (after line 12):

```python
from rune.mining.session_log import write_session
```

Change the signature (lines 97-101) to:

```python
async def run_benchmark(
    tasks: list[BenchTask],
    engine: Any,
    config: dict[str, Any],
    sessions_dir: Path | None = None,
) -> BenchResult:
```

After `final_state` is obtained successfully (immediately after line 125, inside the `try` success path but after the `except` — place it right after the `final_state` assignment block, before `generated_code = ...` at line 135):

```python
        if sessions_dir is not None:
            write_session(
                final_state,
                {
                    "benchmark": config.get("benchmark", "unknown"),
                    "problem_id": task.task_id,
                },
                sessions_dir / task.task_id,
            )
```

- [ ] **Step 4: Run to verify it passes + existing runner tests**

Run: `uv run pytest tests/unit/test_bench_runner_sessions.py tests/unit/ -k bench -q && uv run mypy src/`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/bench/runner.py tests/unit/test_bench_runner_sessions.py
git commit -m "feat(bench): emit session dirs from run_benchmark (corpus producer); rune-run path intentionally not wired"
```

#### Task 4d: Miner consumes stored renders and emits the SFT contract

Replace the `Input:/Output:` concat with the stored renders, and emit `prompt`/`completion` so the trainer keeps a target.

**Files:**
- Modify: `src/rune/mining/miner.py:34-89`
- Modify: `tests/unit/test_miner.py` fixtures (lines 27+)

- [ ] **Step 1: Update the fixtures to the v2 session shape**

In `tests/unit/test_miner.py`, replace the `_STEPS` fixture entries so each step carries `trajectory`, `prompt`, `output` (not `input`/`output`). For example the first two entries become:

```python
_STEPS = [
    {
        "step": 0,
        "action": "decompose",
        "target": None,
        "trajectory": "ROLE: project-decomposer",
        "prompt": "implement binary search",
        "output": '{"subtasks": []}',
        "feedback": None,
    },
    {
        "step": 1,
        "action": "plan",
        "target": "a",
        "trajectory": "ROLE: planner",
        "prompt": "plan subtask a",
        "output": "do the thing",
        "feedback": None,
    },
]
```

(Apply the same field rename to any remaining `_STEPS` entries.)

- [ ] **Step 2: Add the failing assertion for the contract**

In `tests/unit/test_miner.py`, add to the `extract_trajectories` test (or create one):

```python
def test_extract_emits_prompt_completion_contract() -> None:
    metadata = {"benchmark": "mbpp", "problem_id": "1"}
    records = extract_trajectories(_STEPS, metadata)
    rec = records[0]
    assert rec["trajectory"] == "ROLE: project-decomposer"
    assert rec["prompt"] == "implement binary search"
    assert rec["completion"] == '{"subtasks": []}'
    assert rec["metadata"]["phase"] == "decompose"
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/unit/test_miner.py -k "prompt_completion or extract" -v`
Expected: FAIL — current records have no `prompt`/`completion` keys; `_render_trajectory` reads `input`.

- [ ] **Step 4: Rewrite `_render_trajectory` and `extract_trajectories`**

In `src/rune/mining/miner.py`, replace `_render_trajectory` (lines 34-50) with a helper that joins the stored renders per `(action, target)`:

```python
def _join_field(
    steps: list[dict],  # type: ignore[type-arg]
    action: str,
    target: str | None,
    field: str,
) -> str:
    """Join one stored field across steps matching (action, target) with '---'."""
    parts = [
        step.get(field, "")
        for step in steps
        if step.get("action") == action and step.get("target") == target
    ]
    return "\n---\n".join(p for p in parts if p)
```

Replace the body of `extract_trajectories` (the loop at lines 67-89) so each record carries the contract:

```python
    for step in steps:
        action = step.get("action", "unknown")
        target = step.get("target")
        key = (action, target)
        if key in seen:
            continue
        seen.add(key)

        records.append(
            {
                "task_id": task_id,
                "trajectory": _join_field(steps, action, target, "trajectory"),
                "prompt": _join_field(steps, action, target, "prompt"),
                "completion": _join_field(steps, action, target, "output"),
                "metadata": {
                    "phase": action,
                    "target": target,
                    "benchmark": benchmark,
                    "problem_id": problem_id,
                },
            }
        )

    return records
```

- [ ] **Step 5: Run to verify it passes + full miner suite**

Run: `uv run pytest tests/unit/test_miner.py -q && uv run mypy src/`
Expected: PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add src/rune/mining/miner.py tests/unit/test_miner.py
git commit -m "feat(mining): emit trajectory/prompt/completion contract from stored renders"
```

#### Task 4e: Trainer consumes the `prompt`/`completion` contract

`run_distillation` (plain SFT, `completion_only_loss=True`) needs a target. trl's `SFTTrainer` natively handles a dataset with `prompt` + `completion` columns. Build the dataset from those columns so the completion is masked correctly; drop the others before handing to trl.

**Files:**
- Modify: `src/rune/training/d2l_train.py:86-132`
- Test: `tests/unit/test_d2l_train.py` (extend) — CPU-only, no GPU

- [ ] **Step 1: Write the failing test for the record→column mapping**

Add a pure helper (no GPU) and test it. In `tests/unit/test_d2l_train.py`:

```python
from rune.training.d2l_train import to_sft_columns


def test_to_sft_columns_maps_prompt_completion() -> None:
    records = [
        {"trajectory": "T", "prompt": "P", "completion": "C", "metadata": {}},
        {"trajectory": "T2", "prompt": "P2", "completion": "C2", "metadata": {}},
    ]
    rows = to_sft_columns(records)
    assert rows == [
        {"prompt": "P", "completion": "C"},
        {"prompt": "P2", "completion": "C2"},
    ]


def test_to_sft_columns_skips_empty_completion() -> None:
    records = [{"prompt": "P", "completion": ""}]
    assert to_sft_columns(records) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_d2l_train.py -k to_sft_columns -v`
Expected: FAIL — `cannot import name 'to_sft_columns'`.

- [ ] **Step 3: Add the helper and use it in `run_distillation`**

In `src/rune/training/d2l_train.py`, add a module-level function (CPU-safe, no deferred imports needed):

```python
def to_sft_columns(records: list[dict[object, object]]) -> list[dict[str, str]]:
    """Project corpus records onto trl's prompt/completion SFT columns.

    Records with an empty completion carry no training target and are dropped
    so completion-only-loss never sees an all-masked example.
    """
    rows: list[dict[str, str]] = []
    for rec in records:
        completion = str(rec.get("completion", ""))
        if not completion:
            continue
        rows.append({"prompt": str(rec.get("prompt", "")), "completion": completion})
    return rows
```

In `run_distillation`, replace the dataset construction (line 132) `dataset = hf_datasets.Dataset.from_list(records)` with:

```python
    sft_rows = to_sft_columns(records)
    logger.info("run_distillation: %d records with completion target", len(sft_rows))
    dataset = hf_datasets.Dataset.from_list(sft_rows)
```

- [ ] **Step 4: Run to verify it passes + type check**

Run: `uv run pytest tests/unit/test_d2l_train.py -q && uv run mypy src/`
Expected: PASS / clean.

- [ ] **Step 5: GPU contract check — run a tiny real distillation**

Confirm trl actually consumes the `prompt`/`completion` columns and masks the prompt (no all-masked-batch warning). Write a 3-record corpus and run a 2-step distillation on this GPU:

```bash
python - <<'PY'
import json, pathlib
rows = [
    {"task_id": "t/1", "trajectory": "ROLE: coder", "prompt": "write add(a,b)", "completion": "def add(a,b):\n    return a+b", "metadata": {}},
    {"task_id": "t/2", "trajectory": "ROLE: coder", "prompt": "write sub(a,b)", "completion": "def sub(a,b):\n    return a-b", "metadata": {}},
    {"task_id": "t/3", "trajectory": "ROLE: coder", "prompt": "write mul(a,b)", "completion": "def mul(a,b):\n    return a*b", "metadata": {}},
]
p = pathlib.Path("/tmp/contract_corpus.jsonl")
p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(p)
PY
uv run python - <<'PY'
from rune.training.d2l_train import D2LTrainConfig, run_distillation
cfg = D2LTrainConfig(
    corpus_path="/tmp/contract_corpus.jsonl",
    checkpoint_dir="/tmp/contract_ckpt",
    num_epochs=1,
    batch_size=1,
    max_seq_length=256,
)
run_distillation(cfg)
PY
```

Expected: log line `run_distillation: 3 records with completion target`, training runs to completion, and **no** `all-masked batch` / `denom = 0` warnings (those would mean the completion column wasn't recognized). If warnings appear, the column contract is wrong — stop and fix before merge.

- [ ] **Step 6: Commit**

```bash
git add src/rune/training/d2l_train.py tests/unit/test_d2l_train.py
git commit -m "feat(training): consume prompt/completion contract in run_distillation"
```

#### Task 4f: Add a `production` continuation flavor to the probe + corpus dump to the smoke test

So probe sweeps are comparable to production, and so the smoke gate can emit a real corpus.

**Files:**
- Modify: `tools/cont_probe.py` (`TRAJECTORY_FLAVORS` dict, ~line 503)
- Modify: `tools/smoke_test_engine.py` (add `--dump-sessions DIR`)

- [ ] **Step 1: Add the production flavor in `tools/cont_probe.py`**

Above the `TRAJECTORY_FLAVORS` dict, add a builder that calls the real template:

```python
def _traj_production(task: str, accumulated: str, window: int) -> str:
    from rune.engine.parse import render_template  # noqa: PLC0415

    return render_template(
        "code_continue",
        project=task,
        subtask=None,
        accumulated_code=accumulated,
    )
```

Add it to the dict:

```python
TRAJECTORY_FLAVORS: dict[str, Any] = {
    "sliding_window": _traj_sliding_window,
    "minimal_goal_code": _traj_minimal,
    "with_attempt_counter": _traj_with_counter,
    "with_structural_summary": _traj_with_structure,
    "code_template": _traj_code_template,
    "production": _traj_production,
}
```

- [ ] **Step 2: Add `--dump-sessions` to `tools/smoke_test_engine.py`**

After `final_state` is computed (after line 131), add:

```python
    dump_dir = None
    for i, arg in enumerate(sys.argv):
        if arg == "--dump-sessions" and i + 1 < len(sys.argv):
            dump_dir = Path(sys.argv[i + 1])
    if dump_dir is not None:
        from rune.mining.session_log import write_session  # noqa: PLC0415

        write_session(
            final_state,
            {"benchmark": "smoke", "problem_id": "linkedlist"},
            dump_dir / "linkedlist",
        )
        log.info("Wrote session corpus to %s", dump_dir)
```

- [ ] **Step 3: Lint, then verify the production flavor renders on GPU**

Run: `uv run ruff check tools/`
Expected: ruff clean.

Then exercise the new flavor against the real model on this GPU instance (sweep just the `production` flavor on one truncation case):

```bash
uv run python tools/cont_probe.py --trajectory-flavor production --cut medium 2>&1 | tee /tmp/probe_production.log
```

Expected: the run completes and `production` appears in the results table (confirms `render_template("code_continue", ...)` drives a real continuation). If `cont_probe.py` uses different CLI flags, run `uv run python tools/cont_probe.py --help` first and adapt.

- [ ] **Step 4: Commit**

```bash
git add tools/cont_probe.py tools/smoke_test_engine.py
git commit -m "feat(tools): production continuation flavor + smoke corpus dump"
```

---

### Task 5: Fix the stale template reference in tests (cosmetic)

`tests/unit/test_state.py:24` constructs an `Action` with `prompt_template="prompt_decompose"`, a template that does not exist (only `prompt_decompose_concise.j2` does). It is a string in a dataclass ctor, not a render — cosmetic, but misleading.

**Files:**
- Modify: `tests/unit/test_state.py:24`

- [ ] **Step 1: Fix the reference**

In `tests/unit/test_state.py`, change `prompt_template="prompt_decompose",` to `prompt_template="prompt_decompose_concise",`.

- [ ] **Step 2: Run + commit**

Run: `uv run pytest tests/unit/test_state.py -q`
Expected: PASS.

```bash
git add tests/unit/test_state.py
git commit -m "test: correct stale prompt_decompose template reference"
```

---

### Phase 1 verification + SMOKE GATE (merge blocker)

- [ ] **Step 1: Full CPU gate**

Run: `uv run ruff check . && uv run mypy src/ && uv run pytest tests/unit/ -q`
Expected: all green.

- [ ] **Step 2: GPU smoke gate — run directly on this instance**

Run this procedure here (GPU is available). Merge to `fix/pr45-review-correctness` is blocked until it passes. Use background runs for the engine invocations (each is multi-minute) and tee logs.

**Baseline (on `fix/pr45-review-correctness`, before the branch changes):**
```bash
git checkout fix/pr45-review-correctness
uv run python tools/smoke_test_engine.py 2>&1 | tee /tmp/smoke_base.log
uv run python tools/smoke_test_engine.py --eos 2>&1 | tee /tmp/smoke_base_eos.log
```
Record from the logs: count of `decompose output failed validation; re-decomposing` warnings, whether subtask names look like real subtasks vs chain-of-thought, and EOS pass/fail.

**After (on `fix/prompt-layer-remediation`):**
```bash
git checkout fix/prompt-layer-remediation
uv run python tools/smoke_test_engine.py --dump-sessions /tmp/smoke_corpus 2>&1 | tee /tmp/smoke_new.log
uv run python tools/smoke_test_engine.py --eos 2>&1 | tee /tmp/smoke_new_eos.log
uv run rune mine --sessions-dir /tmp/smoke_corpus --output-dir /tmp/smoke_shards
```

**Merge gate — all must hold:**
1. `/tmp/smoke_new.log` shows **zero** `decompose output failed validation` warnings (was ≥0 baseline; must not increase, target zero).
2. Decompose subtasks are coherent (real names, no "Analyze the request"-style reasoning entries).
3. `--eos` still prints `=== EOS SMOKE PASSED ===`.
4. `/tmp/smoke_shards` contains shards whose records have non-empty `trajectory`, `prompt`, and `completion` (spot-check one JSONL line).
5. No regression in assembled-code coherence vs baseline (`=== INTEGRATED CODE ===` non-empty when baseline was).

If any gate fails, stop and report — do not merge.

---

## Phase 2 — P1 hardening (CPU-verifiable; merges on green CI)

### Task 6: Parametrized render tests for all 13 templates (#5)

**Files:**
- Create: `tests/unit/test_templates.py`

- [ ] **Step 1: Write the test**

```python
"""Render-mechanics tests for every engine template."""

from __future__ import annotations

import jinja2
import pytest

from rune.engine.parse import _env, render_template
from rune.engine.state import Subtask

_SUBTASK = Subtask(name="_main", description="do it", depends_on=[])


def _ctx() -> dict[str, object]:
    """Superset of keys produced by state_to_ctx, for branch coverage."""
    return {
        "project": "build a thing",
        "task_description": "build a thing",
        "project_label": "build a thing",
        "subtask_count": 1,
        "subtask": _SUBTASK,
        "subtask_name": "_main",
        "subtask_index": 1,
        "total_subtasks": 1,
        "plan": "the plan",
        "target_subtask": "_main",
        "existing_code": "print(1)",
        "error_summary": "boom",
        "fix_guidance": "fix it",
        "repair_history": ["err"],
        "code_trajectory": [
            {"step": 0, "action": "code", "code": "x=1", "error": "", "passed": True}
        ],
        "integration_doc": "- _main: do it",
        "skeletons": {"_main": "print(1)"},
        "code_outputs": {"_main": "print(1)"},
        "integration_error": "",
        "accumulated_code": "print(1)",
    }


_TEMPLATES = [
    "decompose", "prompt_decompose_concise", "plan", "prompt_plan",
    "code", "prompt_code", "code_repair", "prompt_code_repair",
    "integrate", "prompt_integrate", "diagnose", "prompt_diagnose",
    "code_continue", "prompt_code_continue",
]


@pytest.mark.parametrize("name", _TEMPLATES)
def test_renders_without_undefined(name: str) -> None:
    render_template(name, **_ctx())


@pytest.mark.parametrize("name", _TEMPLATES)
def test_declared_vars_are_supplied(name: str) -> None:
    source = _env.loader.get_source(_env, f"{name}.j2")[0]
    declared = jinja2.meta.find_undeclared_variables(_env.parse(source))
    missing = declared - set(_ctx())
    assert not missing, f"{name}.j2 needs unsupplied vars: {missing}"


@pytest.mark.parametrize("name", ["diagnose", "prompt_diagnose", "code_continue"])
def test_renders_with_no_target(name: str) -> None:
    ctx = _ctx()
    ctx.update({"subtask": None, "target_subtask": None})
    render_template(name, **ctx)
```

(Note: `_TEMPLATES` lists 14 entries — the 13 `.j2` files plus `prompt_code_continue` added in Task 2. Confirm the count matches `ls src/rune/templates/`.)

- [ ] **Step 2: Run; fix any template/ctx mismatch surfaced**

Run: `uv run pytest tests/unit/test_templates.py -q`
Expected: PASS. If `test_declared_vars_are_supplied` fails, either the template uses a stray variable (fix the template) or `_ctx()` is missing a key that `state_to_ctx` really supplies (add it to `_ctx()` to mirror `state_to_ctx`).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_templates.py
git commit -m "test(templates): parametrized render + undeclared-var coverage for all templates"
```

### Task 7: Injection caps for unbounded user content (#4)

`prompt_decompose_concise.j2` already gained `[USER TASK]` delimiters in Task 1. The remaining unbounded field is `integration_doc`, built in `state_to_ctx` (`graph.py:131`) by joining every subtask's full description.

**Files:**
- Modify: `src/rune/engine/graph.py:131`
- Test: `tests/unit/test_state_to_ctx.py` (create or extend)

- [ ] **Step 1: Write the failing test**

```python
from rune.engine.graph import state_to_ctx, _INTEGRATION_DOC_LINE_CAP
from rune.engine.state import Subtask, make_initial_state


def test_integration_doc_caps_line_length() -> None:
    state = make_initial_state("t", 5)
    state["subtasks"] = [Subtask(name="a", description="x" * 5000, depends_on=[])]
    ctx = state_to_ctx(state)
    line = ctx["integration_doc"].splitlines()[0]
    assert len(line) <= _INTEGRATION_DOC_LINE_CAP + len("- a: ")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_state_to_ctx.py -k integration_doc -v`
Expected: FAIL — `cannot import name '_INTEGRATION_DOC_LINE_CAP'`.

- [ ] **Step 3: Add the cap in `graph.py`**

Near the top-level constants (after line 48), add:

```python
_INTEGRATION_DOC_LINE_CAP = 200
```

Replace line 131:

```python
    ctx["integration_doc"] = "\n".join(
        f"- {s.name}: {s.description[:_INTEGRATION_DOC_LINE_CAP]}" for s in subtasks
    )
```

- [ ] **Step 4: Run + commit**

Run: `uv run pytest tests/unit/test_state_to_ctx.py -q && uv run mypy src/`
Expected: PASS / clean.

```bash
git add src/rune/engine/graph.py tests/unit/test_state_to_ctx.py
git commit -m "fix(engine): cap integration_doc line length against prompt injection"
```

### Task 8: Single truncation layer (#7)

Move template-inline slices (`project[:1200]` in `decompose.j2`, `project[:300]` in `code_continue.j2`, `accumulated_code[-3500:]`) into named constants applied once in `state_to_ctx`. Keep `decompose.j2` readable but stop double-slicing.

**Files:**
- Modify: `src/rune/engine/graph.py` (`state_to_ctx`)
- Modify: `src/rune/templates/decompose.j2`, `code_continue.j2`
- Test: `tests/unit/test_state_to_ctx.py`

- [ ] **Step 1: Write the failing test**

```python
from rune.engine.graph import state_to_ctx, _PROJECT_CAP
from rune.engine.state import make_initial_state


def test_project_is_pre_sliced() -> None:
    state = make_initial_state("z" * 5000, 5)
    ctx = state_to_ctx(state)
    assert len(ctx["project"]) == _PROJECT_CAP
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_state_to_ctx.py -k pre_sliced -v`
Expected: FAIL — `cannot import name '_PROJECT_CAP'`.

- [ ] **Step 3: Apply caps once in `state_to_ctx`**

Add constants (after line 48):

```python
_PROJECT_CAP = 1200
_ACCUMULATED_CODE_CAP = 3500
```

Replace the `ctx` dict head (lines 66-71) so `project`/`task_description` are pre-sliced and the redundant `project_label` is removed:

```python
    ctx: dict[str, Any] = {
        "project": task[:_PROJECT_CAP],
        "task_description": task[:_PROJECT_CAP],
        "subtask_count": len(subtasks),
    }
```

- [ ] **Step 4: De-slice the templates**

In `src/rune/templates/decompose.j2`, change `PROJECT: {{ project[:1200] }}` → `PROJECT: {{ project }}`.
In `src/rune/templates/code_continue.j2`, change `{{ project[:300] }}` → `{{ project }}` and `{{ accumulated_code[-3500:] }}` → `{{ accumulated_code }}`.
In `src/rune/engine/graph.py`, slice `accumulated_code` once where `cont_ctx` is built (Task 2 left it as `"accumulated_code": accumulated_code`):

```python
                cont_ctx = {
                    **ctx,
                    "accumulated_code": accumulated_code[-_ACCUMULATED_CODE_CAP:],
                }
```

Remove any remaining reference to `project_label` (grep to confirm it is unused after removal):

Run: `grep -rn "project_label" src/ tests/`
Expected: no output (delete any stragglers; the `_ctx()` fixture in Task 6 may list it — drop it there too).

- [ ] **Step 5: Run the relevant suites + commit**

Run: `uv run pytest tests/unit/test_state_to_ctx.py tests/unit/test_templates.py tests/unit/test_parse.py -q && uv run mypy src/`
Expected: PASS / clean.

```bash
git add src/rune/engine/graph.py src/rune/templates/decompose.j2 src/rune/templates/code_continue.j2 tests/unit/test_state_to_ctx.py tests/unit/test_templates.py
git commit -m "refactor(engine): single truncation layer in state_to_ctx; drop project_label"
```

---

## Phase 3 — P2 polish (CPU-verifiable)

### Task 9: Whitespace control + prompt_version (#10, #P2)

**Files:**
- Modify: `src/rune/engine/parse.py:16`
- Modify: `src/rune/config.py`
- Test: `tests/unit/test_parse.py`, `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test for prompt_version**

In `tests/unit/test_config.py` (create if absent):

```python
from rune.config import PipelineConfig


def test_prompt_version_default() -> None:
    assert PipelineConfig().prompt_version == "v1"
    assert "prompt_version" in PipelineConfig().to_dict()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -k prompt_version -v`
Expected: FAIL — `PipelineConfig` has no `prompt_version`.

- [ ] **Step 3: Add the field**

In `src/rune/config.py`, add to `PipelineConfig` (after line 27):

```python
    prompt_version: str = "v1"
```

Confirm `to_dict()` (line 31) serializes all dataclass fields (if it enumerates explicitly, add `prompt_version`).

- [ ] **Step 4: Enable whitespace trimming**

In `src/rune/engine/parse.py`, change line 16 to:

```python
_env = Environment(
    loader=PackageLoader("rune", "templates"),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
```

- [ ] **Step 5: Run all render + config tests; re-snapshot if any literal whitespace assertions break**

Run: `uv run pytest tests/unit/test_parse.py tests/unit/test_templates.py tests/unit/test_config.py -q`
Expected: PASS. If a test asserts exact whitespace, update the expected string to the trimmed output (the templates' meaning is unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/rune/engine/parse.py src/rune/config.py tests/unit/test_config.py tests/unit/test_parse.py
git commit -m "feat(config): prompt_version field; chore(templates): trim/lstrip blocks"
```

### Task 10: DRY — drop the duplicate `skeletons` key (#6/#9)

`state_to_ctx` exposes `code_results` twice as both `skeletons` and `code_outputs` (lines 132-133). Keep `code_outputs`; drop `skeletons` only if no template references it.

**Files:**
- Modify: `src/rune/engine/graph.py:132`
- Modify: `tests/unit/test_templates.py` `_ctx()` (drop `skeletons`)

- [ ] **Step 1: Confirm `skeletons` is unused by templates**

Run: `grep -rn "skeletons" src/rune/templates/`
Expected: no output. If a template uses it, switch that template to `code_outputs` instead, then proceed.

- [ ] **Step 2: Remove the duplicate key**

In `src/rune/engine/graph.py`, delete line 132 (`ctx["skeletons"] = code_results`). Remove `skeletons` from the Task 6 `_ctx()` fixture.

- [ ] **Step 3: Run + commit**

Run: `uv run pytest tests/unit/ -q && uv run mypy src/ && uv run ruff check .`
Expected: PASS / clean.

```bash
git add src/rune/engine/graph.py tests/unit/test_templates.py
git commit -m "refactor(engine): drop duplicate skeletons ctx key"
```

> **Deliberately skipped (YAGNI, per the source review):** full `TemplateContext` TypedDict + runtime contract validation (#9), CI `from_string` grep guard (#P2 — static templates make SSTI non-exploitable), `SandboxedEnvironment`/autoescape/bytecode cache, the probe rewrite (we *added* the production flavor instead of gutting the A/B harness), header/`error_type` macro extraction (low value vs. churn across already-touched files), and git-hash plumbing (templates are git-tracked; `prompt_version` suffices).

---

## Final integration

- [ ] **Step 1: Full CPU gate on the branch**

Run: `uv run ruff check . && uv run mypy src/ && uv run pytest tests/ -q`
Expected: all green.

- [ ] **Step 2: Re-confirm the Phase 1 smoke gate still holds** — Phases 2–3 touched templates (Task 8/9), so re-run the after-side smoke from the Phase 1 gate directly on this GPU instance and re-check the 5 merge conditions.

- [ ] **Step 3: Re-mine note** — because template *trajectory* files changed (`decompose.j2`, `code_continue.j2`), any previously-mined decompose/continuation corpus and oracle adapters are stale. State explicitly in the merge PR: "re-mine after merge; prior decompose/cont shards are invalidated by the template rewrites." (No backward-compat shim per project policy.)

- [ ] **Step 4: Merge to the integration branch (after the smoke gate passes)**

```bash
git checkout fix/pr45-review-correctness
git merge --no-ff fix/prompt-layer-remediation
```

---

## Self-review notes (coverage map)

- #1 schema drift → Task 1 (+ schema-key assertion guards the P0 that render-mechanics tests miss).
- #2 train/serve → Tasks 4a–4f (producer → miner → trainer contract, all CPU shape-tested; `output` sourced from extracted code, **not** `result.text`; **both** serve renders captured).
- #3/#11 continuation → Task 2.
- #4 injection caps → Task 1 (decompose delimiters) + Task 7 (`integration_doc`).
- #5 tests → Task 6 (+ Task 5 stale ref).
- #6 DRY → Task 10.
- #7 truncation → Task 8.
- #10 whitespace, #P2 prompt_version → Task 9.
- Cleanup (dead debug) → Task 3 (all `# #region agent log` regions, defs + call sites; mypy-guarded).
- **Known residual gap (acknowledged, not fixed here):** continuation rounds regenerate the adapter from the `code_continue` render, but mining captures only the *initial* action render — so continued code/integrate trajectories still under-represent the continuation-phase conditioning. Flag in the PR; address only if the smoke gate shows continuation-related train/serve drift.

---

## Phase 4 — Conditioning-strategy experiment (speculative quality; smoke-gated + HPO-validated)

> **Added 2026-05-29.** Independent follow-on to Phases 0–3. Branches off the merged remediation. This phase is **pure pass@1 quality work, not a bug fix** — right-size effort accordingly: exploratory, default-safe, decided by benchmark numbers, not by exhaustive TDD.

### Provenance and why this is *not* a memory fix

A separate, already-landed change (`_chunk_gated_mlp` in `src/rune/model/hypernetwork.py`, with `tests/unit/test_hypernetwork_chunking.py`) fixed the continuation-OOM: the perceiver `modality_projection` MLP intermediate is sized by `n_layers(32) × seq_len`, and the old chunk guard only looked at `seq_len > 2048` so it never fired. That fix bounds the transient regardless of trajectory length. **Commit that fix first** — it is the prerequisite that makes this phase safe to run.

With OOM solved, the original architectural question ("the hypernetwork re-encodes the full growing trajectory every step into a fresh adapter, discarding the last") reduces to a *quality* question, not a memory one. Two premises were checked against the failing run's debug log before writing this phase:

- **Truncation never fires.** Over 77 `generate_adapter` calls, the context token `seq_len` maxed at **1387** against the `max_length=2048` cap (median 386). So `extract_activations_with_model`'s tokenizer truncation is dormant on representative (MBPP-scale) tasks — a truncation-*side* fix would be a no-op. (If a longer benchmark is later added, re-check this distribution; if it overflows, set `tokenizer.truncation_side` so the freshest tail survives.)
- **Adapters do not accumulate.** `hotswap_adapter` calls `set_peft_model_state_dict`, overwriting one slot; per-step baseline GPU `alloc` was flat (~18.8→19.2 GB across 65 steps). No leak.

So Phase 4 tests one hypothesis the user raised directly: *is re-encoding the full history optimal — and is per-step regeneration optimal?* Both knobs default to **exactly current behavior**, so the default serve path cannot regress; the gate is whether any HPO-selected configuration *beats* baseline held-out `validation_pass_at_1`.

### Task 0 (Phase 4): Branch + prerequisite

- [ ] Commit the chunking fix (`hypernetwork.py` + `tests/unit/test_hypernetwork_chunking.py`) if not already on the integration branch.
- [ ] Branch off the merged remediation: `git checkout fix/pr45-review-correctness && git pull --ff-only && git checkout -b feat/conditioning-experiment`.
- [ ] Clean baseline: `uv sync && uv run pytest tests/unit/ -q && uv run ruff check . && uv run mypy src/`.

---

### Task 11 — Stage A: trajectory content-curation knobs (the bigger, safer lever)

**Hypothesis:** the kitchen-sink trajectory (every sibling skeleton + full repair/code history) is not necessarily the best conditioning for the hypernetwork. Parameterize *content breadth* and let HPO find the sweet spot. This builds directly on Task 8's `state_to_ctx` single-truncation layer.

**Knobs** (each defaults to current behavior → no-op by default):
- `traj_sibling_k` (int, default = large/all): cap on sibling `skeletons`/`code_outputs` entries fed into the trajectory, ordered by relevance — the current subtask's `depends_on` first, then the rest. `0` = no siblings.
- `traj_repair_depth` (int, default 2): recent repair attempts retained (current code already uses `[-2:]`).
- `traj_code_history` (int, default = all): `code_trajectory` entries retained.

**Files:**
- Modify: `src/rune/config.py` — add the three fields + `hpo` ranges (default `hpo` dict entries with `low`/`high`/`step`).
- Modify: `src/rune/engine/graph.py` — `state_to_ctx(state, action, run_config=None)`: apply the caps when building `skeletons`/`code_outputs`/`repair_history`/`code_trajectory`. Thread `run_config` from the call site (`graph.py:173`, available as `configurable["run_config"]`).
- Modify: `src/rune/bench/hpo.py` — add `_suggest(trial, "traj_sibling_k", int_param=True)` etc. to `objective`'s `override(...)`.
- Test: `tests/unit/test_state_to_ctx.py` — caps applied; dependency-first ordering; `run_config=None` reproduces today's full content exactly (the no-op guarantee).

- [ ] **Step 1: Failing test** — `state_to_ctx` with `run_config={"traj_sibling_k": 1}` yields ≤1 sibling, dependency picked first; with `run_config=None` yields all siblings (unchanged).
- [ ] **Step 2:** Add config fields + thread `run_config` into `state_to_ctx`; apply caps. (`run_config=None` / missing key ⇒ no cap.)
- [ ] **Step 3:** Wire the three knobs into `hpo.py objective` + add `hpo` ranges to the default config.
- [ ] **Step 4:** `uv run pytest tests/unit/ -q && uv run mypy src/ && uv run ruff check .` → green.
- [ ] **Step 5: Commit** — `feat(engine): tunable trajectory content-curation knobs (default no-op)`.

---

### Task 12 — Stage B: slow adapter EMA across same-action steps (exploratory spike — generalizes issue #46)

> **Treat as a bet, not a sound operation.** A LoRA adapter `ΔW = B·A` has gauge freedom (`BA = (BR)(R⁻¹A)` for any invertible `R`), so A and B live in an arbitrary basis; per-tensor EMA of A/B is only meaningful if the (deterministic) hypernetwork emits a *stable gauge* on similar inputs — plausible but unproven. So this is HPO-validated and **gated behind Stage A** (pursue only if Stage A's ceiling is too low). Do not write line-by-line TDD for it; the real verdict is the benchmark.

**Connection to issue #46** (see `project_issue46_kv_cache` memory): with `ema_alpha = α`,
- `α = 1.0` ≡ today's regenerate-a-fresh-adapter-every-step,
- `α = 0.0` ≡ #46's "frozen adapter" (carry the previous one unchanged),
- `0 < α < 1` ≡ the "slow" evolution the user asked for.

The densest same-action case is the **continuation sub-loop**, which regenerates the code adapter every round — exactly what #46 analyzed. Phase 4 EMA therefore **subsumes** #46's frozen-adapter-incremental recommendation rather than competing with it; do not open a separate #46 design.

**Design:**
- `ModelWrapper` holds `_ema_state: dict | None` and `_ema_key: tuple | None`. Lineage key = `(action_name, target_subtask)`; on key change, reset (`_ema_state = new`).
- Blend the **raw generated** state dict (pre-scaling) per tensor: `ema = (1-α)·prev + α·new`; then apply the existing `scale_lora_b` on the blended dict before hotswap, so scaling semantics (incl. `cont_multiplier`) are unchanged.
- `ema_alpha` knob (PipelineConfig + HPO range `[0.3, 1.0]`); **default `1.0` ⇒ exact current behavior** (no-op, gate-safe).

**Kill criterion (run before any HPO sweep):** one GPU sanity run with `ema_alpha=0.5` on the smoke task — does generation stay coherent? If output is garbage, the gauge is unstable: restrict the knob to `α ∈ {0.0, 1.0}` (frozen vs fresh — the #46 binary, which sidesteps the averaging-basis problem) or drop Stage B. Record the verdict in the PR either way.

**Files:** `src/rune/model/wrapper.py` (EMA state + blend), `src/rune/engine/graph.py` (pass `ema_alpha` lineage to the hotswap calls in both the main step and the continuation loop), `src/rune/config.py` + `src/rune/bench/hpo.py` (knob + range), `tests/unit/test_wrapper_ema.py` (blend shapes; `α=1.0` is identity; `α<1` moves toward `new`; lineage-change resets).

- [ ] **Step 1:** Unit test for the blend + reset semantics (CPU, plain dicts of tensors — no model load).
- [ ] **Step 2:** Implement EMA state/blend in `ModelWrapper`; wire lineage + `ema_alpha` through `graph.py`.
- [ ] **Step 3:** Knob + HPO range; `uv run pytest tests/unit/ -q && uv run mypy src/ && uv run ruff check .`.
- [ ] **Step 4: Kill-criterion GPU sanity run** (`ema_alpha=0.5`). Coherent → proceed; garbage → restrict/abandon (record).
- [ ] **Step 5: Commit** — `feat(model): slow adapter EMA across same-action steps (default no-op; generalizes #46)`.

---

### Phase 4 gate (no merge without this)

- [ ] **CPU gate:** `uv run ruff check . && uv run mypy src/ && uv run pytest tests/ -q` → green.
- [ ] **Smoke (no-regression / no-op proof):** with default knobs (`traj_sibling_k`=all, `ema_alpha`=1.0), re-run the Phase 1 smoke procedure on this GPU instance; all 5 Phase-1 merge conditions must still hold — defaults must reproduce baseline.
- [ ] **Benchmark A/B (the actual decision):** run `rune bench` HPO including the new knobs on this GPU instance. The best config's **held-out `validation_pass_at_1`** must beat (or at least not trail) the pre-Phase-4 baseline.
  - If a curated/EMA config wins → keep, set the winning values as new defaults, document the lift.
  - If nothing beats baseline → **negative result: remove the knobs** (no dead config surface left behind) and record that full-history + per-step-regeneration was, empirically, already near-optimal. That is a valid and informative outcome.
