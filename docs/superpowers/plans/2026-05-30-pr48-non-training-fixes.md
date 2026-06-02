# PR 48 Non-Training Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get PR 48 (`fix/pr45-review-correctness`) to green CI and close every outstanding review finding that can be fixed without retraining the hypernetwork, on this branch.

**Architecture:** Three concerns, in order: (1) make CI green (ruff format), (2) close the one external reviewer comment + low-risk dead-dep cleanup, (3) land the "solvable-without-retraining" roadmap the author filed (bench self-test scoring fix, gateable/seeded pass@1, over-decomposition curb, and the one high-value DRY extraction). Adapter-application correctness (issue #49 §D) is deliberately **deferred** — it is code-writable now but unverifiable until a non-collapsed checkpoint exists.

**Tech Stack:** Python 3.12, `uv`, ruff, mypy (strict), pytest. Engine = LangGraph; model = transformers + PEFT + xgrammar.

---

## Context the engineer needs first

- **Read `PRODUCT.md`** before starting (CLAUDE.md hard rule). It is present and non-stub.
- You are on branch `fix/pr45-review-correctness`. All work lands here (the user chose to fold the roadmap into PR 48). Do **not** branch off.
- Run everything with `uv run`. Sync once before starting: `uv sync --extra gpu` (plain `uv sync` prunes the gpu extra — but for these CPU tasks `uv sync` is fine; CI uses `uv sync`).
- The fast gate you will run after most tasks:
  ```bash
  uv run ruff format --check . && uv run ruff check . && uv run mypy src/ && uv run pytest tests/unit/ -q
  ```
- **Findings already fixed by the prompt-layer merge `685c7ca4`** (verified against HEAD — do NOT re-do): `.cursor` agent debug-logging blocks (gone from `graph.py`/`hypernetwork.py`), `tools/cont_probe.py`/`capacity_sweep.py` broken imports (now import `extract_partial_code`), `tests/unit/test_state.py:24` (already `prompt_decompose_concise`), and `skip_completion_retry` (now live — 5 `tools/diag_*` probes set it `True`; keep it).
- **Moot finding:** "CUDA mem reporting repeated 3 ways → `cuda_mem_snapshot()`". Verified: the only GB-formatting site is `tools/smoke_test_engine.py._mem()`; `graph.py`/`hypernetwork.py` only call `empty_cache()`. Nothing to dedup — skip it.

## File map (what each task touches)

| Task | Files |
|------|-------|
| 1 — CI format | 15 drifted files (formatter only, no hand edits) |
| 2 — bare except | `tools/smoke_test_engine.py` |
| 3 — dead-dep prune | `pyproject.toml` |
| 4 — bench self-test scoring | `src/rune/bench/runner.py`, `tests/unit/test_bench_runner_scoring.py` (new) |
| 5 — seeded pass@1 | `src/rune/config.py`, `src/rune/bench/runner.py`, `tests/unit/test_bench_runner_scoring.py` |
| 6 — over-decomposition curb | `src/rune/engine/parse.py`, `src/rune/templates/prompt_decompose_concise.j2`, `tests/unit/test_parse_decompose.py` (new) |
| 7 — DRY: `extract_code_from_raw` | `src/rune/engine/parse.py`, `src/rune/engine/continuation.py`, `tests/unit/test_parse_extract.py` (new) |
| 8 — repo-wide mypy (tools/ duplicate) | `pyproject.toml` |
| 9 — final gate + push | — |

---

## Task 1: Fix CI format check (BLOCKING)

CI `lint-and-type-check` fails on `ruff format --check .` (15 files drifted after the prompt-layer merge). `test` is gated `needs: lint-and-type-check` (ci.yml:49), so it shows `skipping`; this fix unblocks it. `ruff check` and `mypy` already pass. No semantic changes — the formatter is deterministic.

**Files:**
- Modify (formatter-only): `src/rune/engine/continuation.py`, `src/rune/mining/miner.py`, `src/rune/model/hypernetwork.py`, `tests/unit/test_bench_runner_sessions.py`, `tests/unit/test_hypernetwork_chunking.py`, `tests/unit/test_miner.py`, `tests/unit/test_templates.py`, `tools/diag_adapter_scaling_probe.py`, `tools/diag_continuation_probe.py`, `tools/diag_format_probe.py`, `tools/diag_inference_probe.py`, `tools/diag_recall_probe.py`, `tools/diag_retrieval_probe.py`, `tools/diag_scaling_mode_probe.py`, `tools/smoke_test_engine.py`

- [ ] **Step 1: Confirm the drift (should list 15 files)**

Run: `uv run ruff format --check .`
Expected: `15 files would be reformatted, 64 files already formatted`

- [ ] **Step 2: Apply the formatter**

Run: `uv run ruff format .`
Expected: `15 files reformatted, 64 files left unchanged`

- [ ] **Step 3: Verify format + lint + types now pass**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy src/`
Expected: format prints `79 files already formatted`; ruff prints `All checks passed!`; mypy prints `Success: no issues found`.

- [ ] **Step 4: Verify unit tests still pass (no semantic change)**

Run: `uv run pytest tests/unit/ -q`
Expected: all pass (≥239 passed, per the merge's gate note).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "style: apply ruff format repo-wide (unblock CI format check)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Replace bare `except` in `smoke_test_engine._mem()`

The only external reviewer inline comment (github-code-quality bot, `tools/smoke_test_engine.py:44`): the `except (ImportError, RuntimeError): pass` swallows silently. Keep the `return ""` fallback but make the swallow intentional and diagnosable. `log` is already defined in this file (no new import).

**Files:**
- Modify: `tools/smoke_test_engine.py:36-46` (the `_mem()` function)

- [ ] **Step 1: Edit `_mem()` to log the swallowed exception at debug level**

Replace:
```python
def _mem() -> str:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            return f"GPU alloc={a:.1f}GB reserved={r:.1f}GB"
    except (ImportError, RuntimeError):
        pass
    return ""
```
with:
```python
def _mem() -> str:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            return f"GPU alloc={a:.1f}GB reserved={r:.1f}GB"
    except (ImportError, RuntimeError):
        # torch missing or CUDA unavailable: memory line is best-effort.
        log.debug("GPU memory probe failed", exc_info=True)
    return ""
```

- [ ] **Step 2: Verify lint/format/types**

Run: `uv run ruff format --check tools/smoke_test_engine.py && uv run ruff check tools/smoke_test_engine.py`
Expected: clean (format already formatted; checks pass).
Note: `tools/` is excluded from `mypy src/`, so no mypy step needed here.

- [ ] **Step 3: Sanity-import the module on CPU (no GPU needed)**

Run: `uv run python -c "import importlib.util, pathlib; importlib.util.spec_from_file_location('s', pathlib.Path('tools/smoke_test_engine.py'))" && uv run python -c "import ast; ast.parse(open('tools/smoke_test_engine.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 4: Commit**

```bash
git add tools/smoke_test_engine.py
git commit -m "fix(tools): log swallowed GPU mem-probe error instead of bare pass

Addresses github-code-quality review comment on smoke_test_engine._mem().

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Prune unused deps (`httpx`, `tree-sitter*`) + stale mypy overrides

Review finding (Orphan/Low): `httpx` has no usage in `src/`/`tests/`; `tree-sitter`/`tree-sitter-python` are unused (`continuation.validate_syntax` uses stdlib `ast`, not tree-sitter). Removing the deps makes their `tool.mypy` overrides stale, so drop those too. Keep `offload_base` (it's the documented RAM-OOM knob in CLAUDE.md) and `current_adapter` (now self-referential, harmless) — those are NOT in scope here.

**Files:**
- Modify: `pyproject.toml` — `dependencies` (remove L14-15 `tree-sitter*`, L22 `httpx`) and `tool.mypy.overrides.module` (remove L103-104 `tree_sitter*`)

- [ ] **Step 1: Prove the deps are unused in shipped code**

Run: `grep -rnE "import httpx|from httpx|import tree_sitter|from tree_sitter" src/ tests/ tools/`
Expected: no output (no usages).

- [ ] **Step 2: Remove the three dependency lines**

In `pyproject.toml` `dependencies`, delete these three lines:
```toml
    "tree-sitter>=0.24.0",
    "tree-sitter-python>=0.23.0",
```
```toml
    "httpx>=0.28.0",
```

- [ ] **Step 3: Remove the stale mypy overrides**

In `pyproject.toml` `[tool.mypy.overrides].module`, delete these two lines:
```toml
    "tree_sitter", "tree_sitter.*",
    "tree_sitter_python", "tree_sitter_python.*",
```

- [ ] **Step 4: Regenerate the lockfile and re-sync**

Run: `uv lock && uv sync`
Expected: lock updates (httpx/tree-sitter removed from resolution); sync succeeds.

- [ ] **Step 5: Verify the full gate (nothing imported the removed deps)**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy src/ && uv run pytest tests/unit/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: drop unused deps (httpx, tree-sitter*) and stale mypy overrides

No usages in src/tests/tools; validate_syntax uses stdlib ast.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Fix the bench self-test scoring flaw (HIGH — real correctness)

`bench/runner.py:142` does `full_code = generated_code + "\n\n" + task.test_code`. The model's own self-authored tests (including a `if __name__ == "__main__":` block) execute when the held-out tests run, so a **correct** implementation can fail because the model's *wrong* self-test fires (confirmed on mbpp/279: `assert is_num_decagonal(10) == 380`, actual 370). The engine already strips self-tests for its in-loop sandbox run (`graph.py:326`), but the **scored** path does not. Fix: strip self-tests from `generated_code` before appending held-out tests, reusing the existing `strip_self_tests` (continuation.py). This does not touch the recorded artifact — only the scoring input.

**Files:**
- Test: `tests/unit/test_bench_runner_scoring.py` (new)
- Modify: `src/rune/bench/runner.py` (import + line 142)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_bench_runner_scoring.py`:
```python
"""Bench scoring: the model's own self-tests must not contaminate held-out scoring."""

from __future__ import annotations

import asyncio
from typing import Any

from rune.bench.runner import BenchTask, run_benchmark


class _FakeEngine:
    """Returns a fixed final_state from ainvoke, ignoring the input state."""

    def __init__(self, integrated_code: str) -> None:
        self._code = integrated_code

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return {"integrated_code": self._code, "code_results": {}}


def _run(tasks: list[BenchTask], engine: Any) -> Any:
    config = {"run_config": {"max_phase_iterations": 3}}
    return asyncio.run(run_benchmark(tasks, engine, config))


def test_correct_impl_with_wrong_self_test_passes_after_strip() -> None:
    # Correct impl, but the model appended a WRONG __main__ self-test.
    code = (
        "def is_num_decagonal(n):\n"
        "    return 4 * n**2 - 3 * n\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    assert is_num_decagonal(10) == 380  # wrong: actual is 370\n"
    )
    # Held-out test is correct: is_num_decagonal(7) == 175.
    task = BenchTask(
        task_id="decagonal",
        description="nth decagonal number",
        test_code="assert is_num_decagonal(7) == 175\n",
        entry_point="is_num_decagonal",
    )
    result = _run([task], _FakeEngine(code))
    assert result.passed_tasks == 1
    assert result.per_task[0].passed is True


def test_genuinely_wrong_impl_still_fails() -> None:
    # No self-tests; impl is wrong. Must still fail (stripping changes nothing).
    code = "def is_num_decagonal(n):\n    return n\n"
    task = BenchTask(
        task_id="decagonal-bad",
        description="nth decagonal number",
        test_code="assert is_num_decagonal(7) == 175\n",
        entry_point="is_num_decagonal",
    )
    result = _run([task], _FakeEngine(code))
    assert result.passed_tasks == 0
    assert result.per_task[0].passed is False
```

- [ ] **Step 2: Run the test, verify the first case FAILS**

Run: `uv run pytest tests/unit/test_bench_runner_scoring.py -q`
Expected: `test_correct_impl_with_wrong_self_test_passes_after_strip` FAILS (the wrong `__main__` assert fires → exit 1 → passed=False). `test_genuinely_wrong_impl_still_fails` PASSES.

- [ ] **Step 3: Add the import to `runner.py`**

In `src/rune/bench/runner.py`, after the existing imports (around line 14), add:
```python
from rune.engine.continuation import strip_self_tests
```

- [ ] **Step 4: Strip self-tests before appending held-out tests**

In `src/rune/bench/runner.py`, replace line 142:
```python
        full_code = generated_code + "\n\n" + task.test_code
```
with:
```python
        # Strip the model's own self-tests (incl. __main__ asserts) before
        # appending the held-out tests: otherwise a wrong self-test fails a
        # correct implementation. The recorded `code` below stays full-length.
        full_code = strip_self_tests(generated_code) + "\n\n" + task.test_code
```

- [ ] **Step 5: Run the test, verify both cases PASS**

Run: `uv run pytest tests/unit/test_bench_runner_scoring.py -q`
Expected: both tests PASS.

- [ ] **Step 6: Verify no import cycle / full gate**

Run: `uv run mypy src/ && uv run pytest tests/unit/ -q`
Expected: mypy clean; all unit tests pass. (`runner.py` importing `continuation.py` is fine — `continuation` imports only `parse`/`json_repair`, no back-edge to `bench`.)

- [ ] **Step 7: Commit**

```bash
git add src/rune/bench/runner.py tests/unit/test_bench_runner_scoring.py
git commit -m "fix(bench): strip model self-tests before scoring held-out tests

A correct impl could fail when the model's own __main__ self-test was
wrong (e.g. mbpp/279). Reuses strip_self_tests; recorded artifact unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Seed generation to *enable* gateable pass@1

Generation is `do_sample=True`, unseeded — a single bench run can't characterize quality or prove a regression. Add an optional `seed` to config and seed the global torch RNG before each task in the runner (torch's RNG is process-global, so seeding propagates to the whole in-engine generation sequence). Pure harness change; defaults to `None` (current behavior). Lands with Task 4 because stripping self-tests changes the pass@1 numbers — only with a seed can the new numbers be reproduced.

**Scope/framing:** this task adds the *seed*, not a gate. Seeding **enables** reproducibility but does not by itself guarantee bit-identical CUDA results (kernel non-determinism is possible). Whether pass@1 is actually deterministic on this GPU must be **measured** (Step 8 below) before anyone builds a regression gate on it. No gate is created here.

**Files:**
- Modify: `src/rune/config.py` (add `seed` field)
- Modify: `src/rune/bench/runner.py` (add `_seed_rng` helper + per-task seeding)
- Test: `tests/unit/test_bench_runner_scoring.py` (extend)

- [ ] **Step 1: Add `seed` to `PipelineConfig`**

In `src/rune/config.py`, in the `PipelineConfig` field block (after `checkpoint_path: str = ""`, around line 27), add:
```python
    seed: int | None = None
```

- [ ] **Step 2: Write the failing test for per-task seeding**

In `tests/unit/test_bench_runner_scoring.py`, add at the end:
```python
def test_runner_seeds_rng_per_task(monkeypatch: Any) -> None:
    import rune.bench.runner as runner_mod

    calls: list[int] = []
    monkeypatch.setattr(runner_mod, "_seed_rng", lambda s: calls.append(s))

    tasks = [
        BenchTask(task_id=f"t{i}", description="d", test_code="assert True\n")
        for i in range(3)
    ]
    config = {"run_config": {"max_phase_iterations": 1, "seed": 100}}
    asyncio.run(run_benchmark(tasks, _FakeEngine("x = 1\n"), config))

    assert calls == [100, 101, 102]


def test_runner_does_not_seed_when_seed_absent(monkeypatch: Any) -> None:
    import rune.bench.runner as runner_mod

    calls: list[int] = []
    monkeypatch.setattr(runner_mod, "_seed_rng", lambda s: calls.append(s))

    task = BenchTask(task_id="t", description="d", test_code="assert True\n")
    config = {"run_config": {"max_phase_iterations": 1}}  # no seed
    asyncio.run(run_benchmark([task], _FakeEngine("x = 1\n"), config))

    assert calls == []
```

- [ ] **Step 3: Run the test, verify it FAILS**

Run: `uv run pytest tests/unit/test_bench_runner_scoring.py::test_runner_seeds_rng_per_task -q`
Expected: FAIL with `AttributeError: module 'rune.bench.runner' has no attribute '_seed_rng'`.

- [ ] **Step 4: Add `_seed_rng` and per-task seeding to `runner.py`**

In `src/rune/bench/runner.py`, add the helper after `logger = logging.getLogger(__name__)` (around line 16):
```python
def _seed_rng(seed: int) -> None:
    """Seed the global torch RNG so in-engine generation is reproducible.

    torch's RNG is process-global, so seeding here propagates to every
    model.generate() call the engine makes for the task.
    """
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Then in `run_benchmark`, in the `for task in tasks:` loop, change the loop header to enumerate and seed before `make_initial_state`. Replace:
```python
    for task in tasks:
        initial_state = make_initial_state(task.description, budget)
```
with:
```python
    seed = config["run_config"].get("seed")
    for i, task in enumerate(tasks):
        if seed is not None:
            _seed_rng(seed + i)
        initial_state = make_initial_state(task.description, budget)
```

- [ ] **Step 5: Run the new tests, verify PASS**

Run: `uv run pytest tests/unit/test_bench_runner_scoring.py -q`
Expected: all (4) tests PASS.

- [ ] **Step 6: Verify config round-trips the new field + full gate**

Run: `uv run python -c "from rune.config import PipelineConfig; c=PipelineConfig(seed=7); assert c.to_dict()['seed']==7; assert PipelineConfig().seed is None; print('ok')" && uv run mypy src/ && uv run pytest tests/unit/ -q`
Expected: `ok`; mypy clean; all unit tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/rune/config.py src/rune/bench/runner.py tests/unit/test_bench_runner_scoring.py
git commit -m "feat(bench): optional seed for reproducible/gateable pass@1

PipelineConfig.seed (default None). When set, the runner seeds the global
torch RNG per task so the engine's generation sequence is deterministic.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 8: Verify determinism on GPU (the claim, not an assertion)**

Seeding only *enables* reproducibility — confirm it empirically on this GPU before treating pass@1 as gateable. Run the same seeded bench twice and diff the result:
```bash
uv run rune bench --tasks-file benchmarks/mbpp_smoke.json --config <(printf 'seed: 1234\nmax_phase_iterations: 8\n') 2>&1 | tee /tmp/run_a.txt
uv run rune bench --tasks-file benchmarks/mbpp_smoke.json --config <(printf 'seed: 1234\nmax_phase_iterations: 8\n') 2>&1 | tee /tmp/run_b.txt
diff <(grep -E "pass@1|\[PASS\]|\[FAIL\]" /tmp/run_a.txt) <(grep -E "pass@1|\[PASS\]|\[FAIL\]" /tmp/run_b.txt) && echo "DETERMINISTIC" || echo "NON-DETERMINISTIC — do not build a gate on pass@1"
```
Expected: identical pass@1 and per-task PASS/FAIL across both runs (`DETERMINISTIC`). Use an existing tasks JSON (e.g. generate one via `rune gen-tasks --out benchmarks/mbpp_smoke.json --limit 4`) if `mbpp_smoke.json` doesn't exist. Record the outcome in the PR thread. This is a GPU run (RAM watchdog rules in CLAUDE.md apply); it requires **no retraining**. If runs differ, the seed still helps but a pass@1 gate is not yet safe — note that and stop short of gating.

> **Optional extension (not required, omit unless asked):** N-sample pass@k — run each task `k` times with seeds `seed + i*k + j` and report `pass@k`. This multiplies bench runtime by `k`; keep it out of this PR unless explicitly requested.

---

## Task 6: Curb over-decomposition of trivial tasks

Single-function MBPP tasks get split into 5–6 subtasks ("Add documentation", "Write unit tests", "Handle edge cases", "Define function signature"), inflating step counts, runtime, and the integration-failure surface. Two changes: (a) a deterministic, unit-testable parse-level filter that drops pure-chore subtasks (never emptying the list), and (b) sharpened prompt guidance naming the observed anti-patterns. The filter is the gateable part; the prompt is validated by GPU smoke (not unit-testable, low risk).

**Files:**
- Test: `tests/unit/test_parse_decompose.py` (new)
- Modify: `src/rune/engine/parse.py` (decompose case)
- Modify: `src/rune/templates/prompt_decompose_concise.j2`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_parse_decompose.py`:
```python
"""Decompose parsing: drop pure-chore subtasks without emptying the plan."""

from __future__ import annotations

import json

from rune.engine.parse import parse_output
from rune.engine.policy import ACTIONS


def _decompose(subtasks: list[dict]) -> list:
    raw = json.dumps({"subtasks": subtasks})
    out = parse_output(ACTIONS["decompose"], raw, None, {})
    return out["subtasks"]


def test_drops_chore_subtasks_when_real_work_remains() -> None:
    result = _decompose(
        [
            {"name": "implement", "description": "core algorithm", "depends_on": []},
            {
                "name": "Write unit tests",
                "description": "test the function",
                "depends_on": ["implement"],
            },
            {
                "name": "Add documentation",
                "description": "write docstrings",
                "depends_on": ["implement"],
            },
        ]
    )
    assert [s.name for s in result] == ["implement"]


def test_chore_dep_references_are_dropped() -> None:
    # A surviving subtask must not depend on a removed chore subtask.
    result = _decompose(
        [
            {"name": "models", "description": "data structures", "depends_on": []},
            {
                "name": "Add type hints",
                "description": "annotate signatures",
                "depends_on": [],
            },
            {
                "name": "logic",
                "description": "core algorithm",
                "depends_on": ["models", "Add type hints"],
            },
        ]
    )
    names = {s.name for s in result}
    assert names == {"models", "logic"}
    logic = next(s for s in result if s.name == "logic")
    assert logic.depends_on == ["models"]


def test_keeps_all_when_every_subtask_is_chore() -> None:
    # Never empty the plan — degrade to keeping everything.
    result = _decompose(
        [
            {"name": "Add documentation", "description": "docstrings", "depends_on": []},
            {"name": "Write unit tests", "description": "tests", "depends_on": []},
        ]
    )
    assert len(result) == 2
```

- [ ] **Step 2: Run the tests, verify they FAIL**

Run: `uv run pytest tests/unit/test_parse_decompose.py -q`
Expected: all three FAIL (chores are currently kept).

- [ ] **Step 3: Add the chore filter to `parse.py`**

First add `re` to the top stdlib import block in `src/rune/engine/parse.py`. Replace:
```python
import logging
from typing import Any
```
with:
```python
import logging
import re
from typing import Any
```

Then add the filter as module-level definitions before `_FIX_GUIDANCE_CAP` (around line 60):
```python
# Subtasks that are project chores, not implementation units. The model tends
# to split trivial single-function tasks into these, inflating step counts and
# the integration-failure surface. Dropped at decompose-time (but never if it
# would empty the plan).
_CHORE_RE = re.compile(
    r"\b("
    r"documentation|docstrings?|"
    r"unit tests?|write tests?|add tests?|test cases?|testing|"
    r"edge cases?|"
    r"function signature|type hints?|annotations?|"
    r"comments?"
    r")\b",
    re.IGNORECASE,
)


def _is_chore_subtask(s: SubtaskSchema) -> bool:
    # Match the NAME only. Matching the description would drop legitimate
    # implementation subtasks whose subject is docs/tests/annotations
    # (e.g. a docstring parser, a type-hint linter) — a false positive that
    # would silently nuke real work on every `rune run`. Conservative by design.
    return bool(_CHORE_RE.search(s.name))
```

Then in `parse_output`, in the `case "decompose":` branch, replace:
```python
            names = {s.name for s in result.subtasks}
            return {
                "subtasks": [
                    Subtask(
                        name=s.name,
                        description=s.description,
                        # Drop phantom (typo'd/unknown) and self dependencies so
                        # readiness checks and the DAG can never softlock.
                        depends_on=[
                            d for d in s.depends_on if d in names and d != s.name
                        ],
                    )
                    for s in result.subtasks
                ]
            }
```
with:
```python
            # Drop pure-chore subtasks (docs/tests/edge-cases/signatures) — but
            # never empty the plan; degrade to keeping everything if all are chores.
            kept = [s for s in result.subtasks if not _is_chore_subtask(s)]
            if not kept:
                kept = list(result.subtasks)
            names = {s.name for s in kept}
            return {
                "subtasks": [
                    Subtask(
                        name=s.name,
                        description=s.description,
                        # Drop phantom (typo'd/unknown), self, and dropped-chore
                        # dependencies so readiness checks and the DAG never softlock.
                        depends_on=[
                            d for d in s.depends_on if d in names and d != s.name
                        ],
                    )
                    for s in kept
                ]
            }
```

- [ ] **Step 4: Run the tests, verify PASS**

Run: `uv run pytest tests/unit/test_parse_decompose.py -q`
Expected: all three PASS.

- [ ] **Step 5: Sharpen the decompose prompt**

Replace the body of `src/rune/templates/prompt_decompose_concise.j2` with:
```jinja
Decompose into subtasks with dependencies. Output a JSON object:
{"subtasks": [{"name": ..., "description": ..., "depends_on": [...]}]}
Use depends_on: [] when a subtask has no prerequisites.
No preamble, no analysis, no reasoning outside the JSON.

A subtask is a unit of IMPLEMENTATION, not a project chore. For a single
self-contained function, ONE subtask is correct.

BAD subtasks (never emit these):
- "Analyze the request" — that is reasoning, not a subtask.
- "Add documentation" / "Write docstrings" — part of the implementation.
- "Write unit tests" / "Handle edge cases" — part of the implementation.
- "Define the function signature" / "Add type hints" — part of the implementation.

[USER TASK]
{{ task_description }}
[/USER TASK]
```

- [ ] **Step 6: Verify the template still renders + full gate**

Run: `uv run python -c "from rune.engine.parse import render_template; print(bool(render_template('prompt_decompose_concise', task_description='add two numbers')))" && uv run mypy src/ && uv run pytest tests/unit/ -q`
Expected: prints `True`; mypy clean; all unit tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/rune/engine/parse.py src/rune/templates/prompt_decompose_concise.j2 tests/unit/test_parse_decompose.py
git commit -m "feat(engine): curb over-decomposition of trivial tasks

Drop pure-chore subtasks (docs/tests/edge-cases/signatures) at decompose
time without emptying the plan; sharpen the decompose prompt with the
observed anti-patterns.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> **Validation note:** the filter is unit-verified (CPU). The prompt change's effect (fewer subtasks on real MBPP tasks) needs a GPU smoke run to confirm — capture before/after subtask counts on a single-function task. This does NOT require retraining.

---

## Task 7: DRY — shared `extract_code_from_raw` (the one high-value extraction)

The pattern "`<Model>.model_validate_json(raw).code`, fall back to `extract_code_value(raw)`" is repeated in `continuation.extract_partial_code`, `parse._parse_code_action`, and `parse` integrate. Extract one helper in `parse.py` (continuation already imports from parse — no cycle). `continuation` needs the `or raw` fallback; expose it via a flag.

> The other DRY/duplication findings are deliberately deferred (YAGNI / stale) — see "Deliberately skipped" below. This is the only one worth doing now.

**Files:**
- Test: `tests/unit/test_parse_extract.py` (new)
- Modify: `src/rune/engine/parse.py` (add helper; use it in code + integrate)
- Modify: `src/rune/engine/continuation.py` (use helper in `extract_partial_code`)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_parse_extract.py`:
```python
"""Shared code extraction: validate-then-fallback behavior."""

from __future__ import annotations

from rune.engine.parse import CodeResult, extract_code_from_raw


def test_valid_json_returns_code() -> None:
    raw = '{"code": "def f():\\n    return 1"}'
    assert extract_code_from_raw(raw, CodeResult) == "def f():\n    return 1"


def test_invalid_json_falls_back_to_lenient_extraction() -> None:
    # Not valid CodeResult JSON; lenient extractor recovers the code field.
    raw = 'garbage {"code": "x = 1"} trailing'
    assert "x = 1" in extract_code_from_raw(raw, CodeResult)


def test_fallback_to_raw_when_requested_and_nothing_extracted() -> None:
    raw = "def f():\n    return 2"  # plain python, not JSON at all
    assert extract_code_from_raw(raw, CodeResult, fallback_to_raw=True) == raw


def test_no_raw_fallback_returns_empty_when_nothing_extracted() -> None:
    raw = "def f():\n    return 2"
    assert extract_code_from_raw(raw, CodeResult, fallback_to_raw=False) == ""
```

- [ ] **Step 2: Run the test, verify it FAILS**

Run: `uv run pytest tests/unit/test_parse_extract.py -q`
Expected: FAIL with `ImportError: cannot import name 'extract_code_from_raw'`.

- [ ] **Step 3: Add the helper to `parse.py`**

In `src/rune/engine/parse.py`, add after `_FIX_GUIDANCE_CAP = 150` (around line 61, after Task 6's additions):
```python
def extract_code_from_raw(
    raw: str, model: type[BaseModel], *, fallback_to_raw: bool = False
) -> str:
    """Parse *raw* as *model* and return its ``code`` field.

    On validation failure, fall back to lenient extraction; if that yields
    nothing and ``fallback_to_raw`` is set, return *raw* unchanged.
    """
    try:
        return model.model_validate_json(raw).code  # type: ignore[attr-defined]
    except Exception:
        extracted = extract_code_value(raw)
        if not extracted and fallback_to_raw:
            return raw
        return extracted
```

- [ ] **Step 4: Use it in `_parse_code_action` and integrate**

In `_parse_code_action`, replace:
```python
    if code is None:
        try:
            code = CodeResult.model_validate_json(raw).code
        except Exception:
            code = extract_code_value(raw)
```
with:
```python
    if code is None:
        code = extract_code_from_raw(raw, CodeResult)
```

In the `case "integrate":` branch, replace:
```python
            if code is None:
                try:
                    code = IntegrateResult.model_validate_json(raw).code
                except Exception:
                    code = extract_code_value(raw)
```
with:
```python
            if code is None:
                code = extract_code_from_raw(raw, IntegrateResult)
```

- [ ] **Step 5: Use it in `continuation.extract_partial_code`**

In `src/rune/engine/continuation.py`, change the import line:
```python
from rune.engine.parse import CodeResult
```
to:
```python
from rune.engine.parse import CodeResult, extract_code_from_raw
```
Then replace `extract_partial_code`:
```python
def extract_partial_code(raw: str) -> str:
    """Extract code from a possibly-truncated CodeResult JSON string.

    Falls back to *raw* when input isn't JSON at all (e.g. continuation
    rounds that emit plain Python).
    """
    try:
        return CodeResult.model_validate_json(raw).code
    except Exception:
        return extract_code_value(raw) or raw
```
with:
```python
def extract_partial_code(raw: str) -> str:
    """Extract code from a possibly-truncated CodeResult JSON string.

    Falls back to *raw* when input isn't JSON at all (e.g. continuation
    rounds that emit plain Python).
    """
    return extract_code_from_raw(raw, CodeResult, fallback_to_raw=True)
```

- [ ] **Step 6: Remove the now-unused `extract_code_value` import from continuation.py (if unused)**

Run: `grep -n "extract_code_value" src/rune/engine/continuation.py`
If the only remaining reference is the import line, remove it:
```python
from rune.engine.json_repair import extract_code_value
```
(If other references remain, leave the import.)

- [ ] **Step 7: Run new + existing tests, verify PASS**

Run: `uv run pytest tests/unit/test_parse_extract.py tests/unit/ -q -k "parse or continuation or extract"`
Then full: `uv run pytest tests/unit/ -q`
Expected: all pass — behavior is unchanged (existing `test_parse`/`test_continuation` cover the callers).

- [ ] **Step 8: Verify lint + types**

Run: `uv run ruff check . && uv run mypy src/`
Expected: clean. (If ruff flags an unused import in continuation.py, complete Step 6.)

- [ ] **Step 9: Commit**

```bash
git add src/rune/engine/parse.py src/rune/engine/continuation.py tests/unit/test_parse_extract.py
git commit -m "refactor(engine): share extract_code_from_raw across parse + continuation

Single validate-then-fallback helper replaces three copies of the
model_validate_json -> extract_code_value pattern.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Fix repo-wide mypy (`tools/` duplicate-module abort)

Running `mypy .` (repo-wide, as opposed to CI's `mypy src/`) hard-aborts:
`tools/cont_probe.py: Source file found twice under different module names: "cont_probe" and "tools.cont_probe"`. Cause: `tools/capacity_sweep.py:22` imports `from tools.cont_probe import ...` while `tools/` has no `__init__.py`, so mypy discovers the same file both as top-level `cont_probe` and as `tools.cont_probe`. It's a hard error ("errors prevented further checking") — nothing else gets type-checked.

`tools/` is throwaway diagnostic probes, never part of the typed surface (CI runs `mypy src/`). The proportionate fix is to exclude `tools/` from mypy discovery so repo-wide `mypy .` matches CI intent. **Verified empirically:** with `tools/` excluded, `mypy .` → `Success: no issues found in 34 source files`.

> **Alternative (only if you want `tools/` type-checked):** add `tools/__init__.py` + `explicit_package_bases = true`. This unmasks strict-mode checking of every diag probe — an unbounded set of new errors to triage. Not recommended for throwaway scripts; raise with the user before taking this path.

**Files:**
- Modify: `pyproject.toml` (`[tool.mypy]` `exclude`)

- [ ] **Step 1: Reproduce the abort**

Run: `uv run mypy . 2>&1 | tail -3`
Expected: `Found 1 error in 1 file (errors prevented further checking)` (the duplicate-module error).

- [ ] **Step 2: Add `tools` to the mypy exclude**

In `pyproject.toml` `[tool.mypy]`, change:
```toml
exclude = "^(site|tests)/"
```
to:
```toml
exclude = "^(site|tests|tools)/"
```

- [ ] **Step 3: Verify repo-wide mypy is clean**

Run: `uv run mypy .`
Expected: `Success: no issues found in 34 source files`.

- [ ] **Step 4: Verify CI-scoped mypy still passes (unchanged)**

Run: `uv run mypy src/`
Expected: `Success: no issues found`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore(mypy): exclude tools/ so repo-wide mypy doesn't abort

tools/ are diag probes outside the typed surface (CI runs mypy src/).
The tools.cont_probe import made mypy find cont_probe.py under two module
names, hard-aborting repo-wide runs. Excluding tools/ -> clean mypy .

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Final gate, format, and push

- [ ] **Step 1: Run the full local gate exactly as CI does**

Run:
```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy src/ && uv run pytest tests/unit/ -q
```
Expected: all green. If `ruff format --check` flags anything from Tasks 4–7, run `uv run ruff format .` and amend the relevant commit (or add a follow-up format commit).

- [ ] **Step 2: Push and confirm CI goes green**

```bash
git push
gh pr checks 48 --watch
```
Expected: `lint-and-type-check` ✅ and `test` ✅ (no longer skipped).

- [ ] **Step 3: Post a PR comment summarizing the non-training remediation**

```bash
gh pr comment 48 --body "$(cat <<'EOF'
## Non-training remediation (this push)

- **CI green:** applied `ruff format` repo-wide (post-merge drift); `test` job unblocked.
- **Reviewer comment:** `smoke_test_engine._mem()` bare-except → debug log (github-code-quality).
- **Dead deps:** dropped unused `httpx` + `tree-sitter*` and their stale mypy overrides.
- **Repo-wide mypy:** excluded `tools/` (diag probes, outside the typed surface) so `mypy .` no longer hard-aborts on the `tools.cont_probe` duplicate-module error; `mypy src/` (CI) unchanged.
- **Bench scoring flaw (correctness):** `runner.py` now strips model self-tests before appending held-out tests (reuses `strip_self_tests`); recorded artifact byte-identical. Unit-verified (mbpp/279-style case).
- **Gateable pass@1:** `PipelineConfig.seed` (default None) seeds the global torch RNG per task for reproducibility.
- **Over-decomposition:** decompose drops pure-chore subtasks (docs/tests/edge-cases/signatures) without emptying the plan; sharpened prompt. Filter unit-verified; prompt effect needs a GPU smoke.
- **DRY:** single `extract_code_from_raw` helper across parse + continuation.

**Deliberately skipped (with reasons):** `cuda_mem_snapshot` (moot — single site), `load_yaml_model`/`tools/_model_session`/`evaluate_code`/policy+inference builder extractions (YAGNI/low-value), `offload_base` (documented RAM knob — kept), `current_adapter` (self-referential, harmless).

**Deferred to issue #49 (code-ready, NOT verifiable without a non-collapsed checkpoint):** adapter-application correctness (§D — `combine_lora` + head bias, `disable_adapter()` activation extraction, adapter-scaling re-tune). Writable now but unverifiable until retrain, so held to avoid landing silent bugs indistinguishable from training issues.
EOF
)"
```

---

## Out of scope (do not attempt here)

- **Deferred — code-ready but unverifiable (issue #49 §D):** adapter-application correctness (`combine_lora` + `get_head_bias()`/`bias_A`, un-contaminating activation extraction via `disable_adapter()`, re-tuning adapter scaling). These need no retraining to *write*, but their effect cannot be validated until a non-collapsed checkpoint exists — landing them now risks silent bugs that surface only at the next train, indistinguishable from training issues. Hold for the #49 work.
- **Requires retraining:** the `scaler_B ≈ 0` adapter collapse itself (issue #49). Not code-fixable.

## Low-priority findings — dispositions (no task; recorded for the PR thread)

- `current_adapter` (`state.py`): now read as its own fallback in `graph.py:365` — effectively self-referential, harmless. Leave.
- `offload_base` (`wrapper.py`/`hypernetwork.py`): the documented RAM-OOM knob (CLAUDE.md). Keep, never wired to `True` by design.
- `skip_completion_retry`: now live (5 `tools/diag_*` probes). Keep.
- `config.from_env`/`save`: operator/test API; keep (documented use).
- `continuation.validate_syntax` raising `NotImplementedError` for non-Python: by design (Python-only engine). Keep.
- `parse.py` returning `{}` on validation failure: by design (re-issue the action). Keep.
- No `tests/gpu/` dir despite `@pytest.mark.gpu`: marker is for future GPU tests; out of scope.

## Self-review notes

- **Spec coverage:** CI format (T1), only external reviewer comment (T2), dead deps (T3), bench scoring flaw (T4), seeded pass@1 (T5), over-decomposition (T6), highest-value DRY (T7), repo-wide mypy `tools/` duplicate (T8, user-reported), final gate + push (T9). All "solvable-without-retraining" roadmap items + the verified-still-live findings are covered; already-fixed and moot findings are explicitly excluded with evidence.
- **Type consistency:** `extract_code_from_raw(raw, model, *, fallback_to_raw)` used identically in T7 across `parse.py` and `continuation.py`; `_seed_rng(seed: int)` name matches between definition (T5 Step 4) and test monkeypatch (T5 Step 2); `_is_chore_subtask`/`_CHORE_RE` consistent in T6.
- **No placeholders:** every code step shows complete code; every run step shows the exact command and expected output.
