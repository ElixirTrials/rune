# Powered Strict-Superset Evaluation — Implementation Plan

> **STATUS (executed).** All four code tasks landed (significance module, temp-0 floor, oracle-gated
> best-of-k, resumable runner — verified in `src/rune/bench/significance.py`, `src/rune/engine/graph.py`,
> `src/rune/config.py`, `src/rune/bench/runner.py`). The run protocol was applied to **HumanEval+**
> → strict superset base 134/164 ⊇ c3 135/164, 0 regressions
> ([issue52-humaneval-regression-rca-fix-2026-06-22.md](../../issue52-humaneval-regression-rca-fix-2026-06-22.md));
> the LCB N=63 k=8 `c≥5` significance endpoint was not reached. Checkboxes below are left as the
> historical implementation record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the rune engine reach statistical significance over base on a fixed task set by (a) a strict-superset-by-construction at temperature 0 and (b) oracle-gated best-of-k escalation, scored with a one-sided exact McNemar test.

**Architecture:** Three engine/runner changes plus one pure analysis module. The escalation floor's zero-shot attempt is forced to greedy decoding so it is byte-identical to the base arm (⇒ `b=0` by construction). Escalation steps sample k candidates and keep the first that passes the *trusted public* oracle. A pure `significance` module computes the exact McNemar p from two per-task pass/fail maps. The benchmark runner becomes resumable so a killed run can be finished without losing completed tasks.

**Tech Stack:** Python 3.12, `uv`, pytest, LangGraph engine (`src/rune/engine/`), transformers/PEFT inference, no new third-party deps (exact McNemar via `math.comb`).

## Global Constraints

- Always launch Python via `uv run` (GPU runs: `uv run --extra gpu` with `UV_NO_SYNC=1`).
- GPU imports stay deferred inside function bodies (CPU-only CI must import modules).
- Never hardcode a model id; base model is single-sourced via `config.yaml` / `load_rune_config()`.
- No new third-party dependency for statistics — exact McNemar uses stdlib `math.comb`.
- Style: no emoji; comments only where the *why* is non-obvious; diff-style edits.
- Quality gates that must pass before each commit: `uv run ruff check .`, `uv run mypy src/`, and the touched tests.
- Pre-registered run parameters (do not change after looking at results): one-sided McNemar, strict-superset audit, `escalation_best_of_k = 8`, LCB task set N=63, temperature-0 floor.

---

### Task 1: Significance module (exact McNemar, paired compare)

Pure, no GPU, no engine. Computes regression/gain counts and exact one- & two-sided McNemar p from two `{task_id: bool}` maps.

**Files:**
- Create: `src/rune/bench/significance.py`
- Test: `tests/unit/test_significance.py`

**Interfaces:**
- Produces:
  - `mcnemar_exact(base_only: int, c3_only: int) -> tuple[float, float]` returning `(p_one_sided, p_two_sided)`.
  - `@dataclass(frozen=True) PairedResult` with fields `n: int, both_pass: int, both_fail: int, base_only: int, c3_only: int, strict_superset: bool, p_one_sided: float, p_two_sided: float`.
  - `paired_compare(base: dict[str, bool], c3: dict[str, bool]) -> PairedResult` (intersects keys).
  - `format_report(r: PairedResult, alpha: float = 0.05) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_significance.py
from __future__ import annotations

from rune.bench.significance import mcnemar_exact, paired_compare


def test_mcnemar_strict_superset_thresholds() -> None:
    # b=0: one-sided p = 0.5**c, two-sided = 2*0.5**c (capped at 1).
    assert mcnemar_exact(0, 4) == (0.0625, 0.125)
    one5, two5 = mcnemar_exact(0, 5)
    assert round(one5, 5) == 0.03125 and round(two5, 5) == 0.0625
    one6, _ = mcnemar_exact(0, 6)
    assert round(one6, 5) == 0.01563


def test_mcnemar_no_discordants_is_one() -> None:
    assert mcnemar_exact(0, 0) == (1.0, 1.0)


def test_mcnemar_with_regressions() -> None:
    # b=1, c=2: n_d=3; P(X>=2)=4/8=0.5 one-sided; two-sided=min(1,2*0.5)=1.0
    one, two = mcnemar_exact(1, 2)
    assert round(one, 5) == 0.5 and two == 1.0


def test_paired_compare_counts_and_superset() -> None:
    base = {"a": True, "b": True, "c": False, "d": False}
    c3 = {"a": True, "b": False, "c": True, "d": True}  # b=1 (lost b), c=2 (gained c,d)
    r = paired_compare(base, c3)
    assert (r.n, r.both_pass, r.both_fail, r.base_only, r.c3_only) == (4, 1, 0, 1, 2)
    assert r.strict_superset is False

    base2 = {"a": True, "b": False, "c": False}
    c3_2 = {"a": True, "b": True, "c": True}  # b=0, c=2
    r2 = paired_compare(base2, c3_2)
    assert r2.strict_superset is True and r2.c3_only == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_significance.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rune.bench.significance'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rune/bench/significance.py
"""Exact McNemar significance for paired pass/fail benchmark outcomes (issue #52).

For paired binary results, significance is governed by the discordant pairs, not
by N. With a strict superset (no regressions, ``base_only == 0``) the one-sided
exact McNemar p collapses to ``0.5 ** gains`` — so significance is reachable at a
fixed N. No scipy dependency: the exact binomial tail uses stdlib ``math.comb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


def mcnemar_exact(base_only: int, c3_only: int) -> tuple[float, float]:
    """Exact McNemar p-values from the discordant counts.

    ``base_only`` = regressions (base passed, c3 failed); ``c3_only`` = gains.
    Returns ``(p_one_sided, p_two_sided)`` where the one-sided alternative is
    "c3 better" (more gains than regressions). Under H0 each discordant is a gain
    with probability 0.5.
    """
    b, c = base_only, c3_only
    n = b + c
    if n == 0:
        return 1.0, 1.0
    half_n = 0.5**n
    p_ge_c = sum(comb(n, i) for i in range(c, n + 1)) * half_n  # P(X >= c)
    p_le_c = sum(comb(n, i) for i in range(0, c + 1)) * half_n  # P(X <= c)
    p_one_sided = p_ge_c
    p_two_sided = min(1.0, 2.0 * min(p_ge_c, p_le_c))
    return p_one_sided, p_two_sided


@dataclass(frozen=True)
class PairedResult:
    n: int
    both_pass: int
    both_fail: int
    base_only: int  # regressions
    c3_only: int  # gains
    strict_superset: bool
    p_one_sided: float
    p_two_sided: float


def paired_compare(base: dict[str, bool], c3: dict[str, bool]) -> PairedResult:
    """Compare two per-task pass/fail maps on their shared task ids."""
    keys = set(base) & set(c3)
    both_pass = sum(1 for k in keys if base[k] and c3[k])
    both_fail = sum(1 for k in keys if not base[k] and not c3[k])
    base_only = sum(1 for k in keys if base[k] and not c3[k])
    c3_only = sum(1 for k in keys if not base[k] and c3[k])
    p_one, p_two = mcnemar_exact(base_only, c3_only)
    return PairedResult(
        n=len(keys),
        both_pass=both_pass,
        both_fail=both_fail,
        base_only=base_only,
        c3_only=c3_only,
        strict_superset=(base_only == 0),
        p_one_sided=p_one,
        p_two_sided=p_two,
    )


def format_report(r: PairedResult, alpha: float = 0.05) -> str:
    """One-line headline + a transparency line (exact counts and both p-values)."""
    verdict = "SIGNIFICANT" if r.p_one_sided <= alpha else "n.s."
    superset = "strict superset" if r.strict_superset else f"{r.base_only} regression(s)"
    return (
        f"base+{r.c3_only} / -{r.base_only} (n={r.n}, {superset}); "
        f"McNemar one-sided p={r.p_one_sided:.4f} [{verdict} @ alpha={alpha}]\n"
        f"  transparency: both_pass={r.both_pass} both_fail={r.both_fail} "
        f"base_only={r.base_only} c3_only={r.c3_only} two-sided p={r.p_two_sided:.4f}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_significance.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/rune/bench/significance.py tests/unit/test_significance.py
uv run mypy src/rune/bench/significance.py
git add src/rune/bench/significance.py tests/unit/test_significance.py
git commit -m "feat(bench): exact McNemar significance module for paired outcomes (#52)"
```

---

### Task 2: Temperature-0 escalation floor

Force the zero-shot floor attempt to greedy so it is byte-identical to the base single-shot arm (strict superset by construction). Mirrors the existing `_effective_scaling`.

**Files:**
- Modify: `src/rune/engine/graph.py` (add `_effective_temperature` next to `_effective_scaling` near line 247; use it in `step_node`'s generate + continuation calls)
- Test: `tests/unit/test_effective_temperature.py`

**Interfaces:**
- Consumes: `_is_zeroshot_attempt(prompt_mode, action, code_results)` (existing, `graph.py`), `Action` (existing).
- Produces: `_effective_temperature(prompt_mode: str, action: Action, code_results: Mapping[str, str], base_temperature: float) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_effective_temperature.py
from __future__ import annotations

from rune.engine.graph import _effective_temperature
from rune.engine.policy import ACTIONS, _with_target


def test_zeroshot_floor_is_greedy() -> None:
    code_action = _with_target("code", "f")  # first code attempt for subtask "f"
    # escalate mode, no prior code for "f" => zero-shot floor => temperature 0
    assert _effective_temperature("escalate", code_action, {}, 0.8) == 0.0


def test_escalation_uses_configured_temperature() -> None:
    code_action = _with_target("code", "f")
    # "f" already has code => this is an escalation re-code => keep configured temp
    assert _effective_temperature("escalate", code_action, {"f": "def f(): ..."}, 0.8) == 0.8


def test_non_escalate_mode_unchanged() -> None:
    code_action = _with_target("code", "f")
    assert _effective_temperature("full", code_action, {}, 0.3) == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_effective_temperature.py -q`
Expected: FAIL with `ImportError: cannot import name '_effective_temperature'`.

- [ ] **Step 3: Add the helper**

In `src/rune/engine/graph.py`, immediately after the `_effective_scaling` function (ends ~line 257), add:

```python
def _effective_temperature(
    prompt_mode: str,
    action: Action,
    code_results: Mapping[str, str],
    base_temperature: float,
) -> float:
    """Greedy (temperature 0) for the zero-shot floor candidate so it is
    byte-identical to the base single-shot arm — a strict-superset-by-construction
    (issue #52 powered-eval). Greedy decoding is argmax and does not draw from the
    RNG, so the floor matches base regardless of prior decompose/plan steps. All
    other attempts (escalation re-code/repair) use the configured temperature."""
    if _is_zeroshot_attempt(prompt_mode, action, code_results):
        return 0.0
    return base_temperature
```

- [ ] **Step 4: Use it in `step_node`**

In `src/rune/engine/graph.py`, inside the `for action in actions:` loop, right after the existing `eff_scaling = _effective_scaling(...)` line (~line 885), add:

```python
        eff_temperature = _effective_temperature(
            prompt_mode, action, state.get("code_results", {}), temperature
        )
```

Then in the `result = await model.generate(...)` call in that loop (~line 889), change `temperature=temperature` to `temperature=eff_temperature`. In the continuation block's `model.generate_continuation(...)` call (~line 933), also change `temperature=temperature` to `temperature=eff_temperature` (a continued floor stays greedy, so it remains ≥ base, never worse).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_effective_temperature.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Full unit suite + gates + commit**

```bash
uv run pytest tests/unit/ -q
uv run ruff check src/rune/engine/graph.py tests/unit/test_effective_temperature.py
uv run mypy src/rune/engine/graph.py
git add src/rune/engine/graph.py tests/unit/test_effective_temperature.py
git commit -m "feat(engine): temperature-0 escalation floor (strict superset vs base, #52)"
```

Expected: full suite still green (same pass count as before plus 3 new).

---

### Task 3: Oracle-gated best-of-k escalation

Adds an `escalation_best_of_k` config and a pure orchestration helper that returns the first candidate passing the trusted public oracle, else the best by quality. The pure helper is dependency-injected (generate + evaluate callables) so it is unit-testable without a model.

**Files:**
- Modify: `src/rune/config.py` (add `escalation_best_of_k: int = 1` to `PipelineConfig`; add env mapping)
- Modify: `src/rune/engine/graph.py` (add `oracle_gated_best_of_k` helper; import `candidate_quality`; wire into `step_node`)
- Test: `tests/unit/test_best_of_k.py`

**Interfaces:**
- Consumes: `build_code_probe`, `apply_oracle_fail_closed`, `apply_episodic_adapter`, `extract_partial_code`, `run_in_sandbox`, `Feedback` (existing in `graph.py`); `candidate_quality` (from `rune.engine.parse`).
- Produces:
  - `PipelineConfig.escalation_best_of_k: int` (default 1 = today's single-shot escalation).
  - `async oracle_gated_best_of_k(k: int, generate_one: Callable[[], Awaitable[Any]], evaluate: Callable[[Any], Awaitable[tuple[bool, int]]]) -> Any` — returns the first result whose `evaluate` reports `passed=True`, else the highest-`quality` result; returns `None` only if `k <= 0`.

- [ ] **Step 1: Write the failing test (pure orchestration)**

```python
# tests/unit/test_best_of_k.py
from __future__ import annotations

import asyncio

from rune.engine.graph import oracle_gated_best_of_k


def _run(coro):
    return asyncio.run(coro)


def test_returns_first_oracle_passing_candidate() -> None:
    seq = iter(["c0", "c1_PASS", "c2"])
    calls = {"gen": 0, "eval": 0}

    async def gen():
        calls["gen"] += 1
        return next(seq)

    async def evaluate(r):
        calls["eval"] += 1
        return (r.endswith("PASS"), 1)

    out = _run(oracle_gated_best_of_k(8, gen, evaluate))
    assert out == "c1_PASS"
    assert calls["gen"] == 2  # stopped at the first passing candidate


def test_falls_back_to_best_quality_when_none_pass() -> None:
    results = iter([("a", 1), ("b", 3), ("c", 2)])

    async def gen():
        return next(results)

    async def evaluate(r):
        return (False, r[1])  # never passes; quality is the second element

    out = _run(oracle_gated_best_of_k(3, gen, evaluate))
    assert out == ("b", 3)  # highest quality
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_best_of_k.py -q`
Expected: FAIL with `ImportError: cannot import name 'oracle_gated_best_of_k'`.

- [ ] **Step 3: Add the config field**

In `src/rune/config.py`, add to `PipelineConfig` (near the escalation/repair fields, after `max_repairs`):

```python
    escalation_best_of_k: int = 1  # >1: sample k escalation candidates, keep first
    #                                that passes the trusted public oracle (#52)
```

And in the env-mapping dict (where `RUNE_MAX_PHASE_ITERATIONS` etc. are listed):

```python
            "RUNE_ESCALATION_BEST_OF_K": ("escalation_best_of_k", int),
```

- [ ] **Step 4: Add the pure helper and import**

In `src/rune/engine/graph.py`, add to the existing `from rune.engine.parse import (...)` block the name `candidate_quality`. Add `from collections.abc import Awaitable, Callable` to the typing imports (the module already imports `from collections.abc import Mapping`). Add the helper after `_effective_temperature`:

```python
async def oracle_gated_best_of_k(
    k: int,
    generate_one: Callable[[], Awaitable[Any]],
    evaluate: Callable[[Any], Awaitable[tuple[bool, int]]],
) -> Any:
    """Sample up to *k* escalation candidates; return the first whose ``evaluate``
    reports ``passed=True`` (it passed the trusted public oracle), else the
    highest-quality candidate (issue #52 powered-eval, B). ``evaluate`` returns
    ``(passed, quality)``. Generation/oracle deps are injected so this is testable
    without a model."""
    best: Any = None
    best_quality = -2
    for _ in range(max(k, 0)):
        result = await generate_one()
        passed, quality = await evaluate(result)
        if passed:
            return result
        if quality > best_quality:
            best, best_quality = result, quality
    return best
```

- [ ] **Step 5: Run the pure test to verify it passes**

Run: `uv run pytest tests/unit/test_best_of_k.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Wire into `step_node`**

In `src/rune/engine/graph.py` `step_node`, replace the single generation (the `eff_scaling = ...`, `adapter_id = apply_episodic_adapter(...)`, `result = await model.generate(...)` sequence, ~lines 885–900) with a branch that uses best-of-k for escalation code/repair steps. Use this block:

```python
        eff_scaling = _effective_scaling(
            prompt_mode, action, state.get("code_results", {}), adapter_scaling
        )
        eff_temperature = _effective_temperature(
            prompt_mode, action, state.get("code_results", {}), temperature
        )
        best_of_k = int(run_config.get("escalation_best_of_k", 1))
        is_escalation = (
            action.executes_code
            and action.name in ("code", "repair")
            and not _is_zeroshot_attempt(
                prompt_mode, action, state.get("code_results", {})
            )
        )
        gen_kwargs: dict[str, Any] = dict(
            prompt=prompt_text,
            system_prompt=action.system_prompt,
            output_schema=action.output_schema,
            max_tokens=run_config.get("max_tokens", 2048),
            temperature=eff_temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            no_repeat_ngram_size=run_config.get("no_repeat_ngram_size", 0),
            presence_penalty=presence_penalty,
            thinking_budget=thinking_budget,
        )
        adapter_id: str | None = None

        if is_escalation and best_of_k > 1:
            _cand_target = action.target_subtask or ""

            async def _generate_one() -> Any:
                nonlocal adapter_id
                adapter_id = apply_episodic_adapter(
                    model, trajectory_text, scaling=eff_scaling
                )
                return await model.generate(**gen_kwargs)

            async def _evaluate(result: Any) -> tuple[bool, int]:
                cand = extract_partial_code(result.text)
                probe, fired, resolved = build_code_probe(_cand_target, cand, state)
                raw = await asyncio.to_thread(run_in_sandbox, probe)
                fb = apply_oracle_fail_closed(
                    fired,
                    resolved,
                    Feedback(stdout=raw.stdout, stderr=raw.stderr, exit_code=raw.exit_code),
                )
                passed = bool(fired and fb.exit_code == 0)
                return passed, candidate_quality(cand, fb)

            result = await oracle_gated_best_of_k(best_of_k, _generate_one, _evaluate)
        else:
            adapter_id = apply_episodic_adapter(
                model, trajectory_text, scaling=eff_scaling
            )
            result = await model.generate(**gen_kwargs)
```

This preserves the existing downstream code unchanged: `raw_text = result.text`, the continuation sub-loop, `output_text`, the post-loop probe/feedback computation, and `results.append((action, target_name, raw_text, adapter_id, ...))` all keep working on the selected `result` and the `adapter_id` set during generation. The post-loop oracle evaluation re-runs once on the selected candidate (the authoritative feedback), so best-of-k only changes *which* candidate is selected, never how it is finally scored.

- [ ] **Step 7: Full unit suite + gates + commit**

```bash
uv run pytest tests/unit/ -q
uv run ruff check src/rune/config.py src/rune/engine/graph.py tests/unit/test_best_of_k.py
uv run mypy src/
git add src/rune/config.py src/rune/engine/graph.py tests/unit/test_best_of_k.py
git commit -m "feat(engine): oracle-gated best-of-k escalation (escalation_best_of_k, #52)"
```

Expected: full suite green; `escalation_best_of_k` defaults to 1 so existing tests/behaviour are unchanged.

---

### Task 4: Resumable benchmark runner

A killed run (the 2026-06-22 HumanEval+ SIGTERM at task 152/164) must be finishable without re-running completed tasks. `run_benchmark` already writes a per-task `metadata.json` with `pass_at_1`; add a `resume` flag that reuses it. Wire `resume=True` into both harnesses.

**Files:**
- Modify: `src/rune/bench/runner.py` (`run_benchmark` signature + per-task resume check)
- Modify: `tools/_lcb_run.py` (pass `resume=True`), `tools/_he_run.py` (pass `resume=True`)
- Test: `tests/unit/test_run_benchmark_resume.py`

**Interfaces:**
- Consumes: `BenchTask`, `TaskResult`, `BenchResult`, `run_benchmark` (existing, `rune.bench.runner`).
- Produces: `run_benchmark(tasks, engine, config, sessions_dir=None, resume=False)` — when `resume=True` and `sessions_dir/<task_id>/metadata.json` exists, that task is *not* re-run; its `TaskResult.passed` is read from the metadata's `pass_at_1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_run_benchmark_resume.py
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rune.bench.runner import BenchTask, run_benchmark


class _ExplodingEngine:
    """Fails the test if the engine is invoked for a task that should be resumed."""

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("engine must not run a resumed task")


def test_resume_skips_completed_task(tmp_path: Path) -> None:
    task = BenchTask(task_id="HumanEval/0", description="d", test_code="assert True", entry_point="f")
    sess = tmp_path / "sessions"
    (sess / task.task_id).mkdir(parents=True)
    (sess / task.task_id / "metadata.json").write_text(json.dumps({"pass_at_1": True}))

    result = asyncio.run(
        run_benchmark(
            [task],
            _ExplodingEngine(),
            {"run_config": {"max_phase_iterations": 3}},
            sessions_dir=sess,
            resume=True,
        )
    )
    assert result.passed_tasks == 1
    assert result.per_task[0].passed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_run_benchmark_resume.py -q`
Expected: FAIL — either `TypeError: run_benchmark() got an unexpected keyword argument 'resume'`, or `AssertionError: engine must not run a resumed task`.

- [ ] **Step 3: Implement resume in `run_benchmark`**

In `src/rune/bench/runner.py`, change the signature:

```python
async def run_benchmark(
    tasks: list[BenchTask],
    engine: Any,
    config: dict[str, Any],
    sessions_dir: Path | None = None,
    resume: bool = False,
) -> BenchResult:
```

At the very top of the `for i, task in enumerate(tasks):` loop (before the seeding line), add:

```python
        if resume and sessions_dir is not None:
            meta_path = sessions_dir / task.task_id / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    meta = None
                if meta is not None and "pass_at_1" in meta:
                    results.append(
                        TaskResult(
                            task_id=task.task_id,
                            passed=bool(meta["pass_at_1"]),
                            code="",
                            stderr="resumed from session",
                        )
                    )
                    continue
```

(`json` is already imported at the top of `runner.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_run_benchmark_resume.py -q`
Expected: PASS.

- [ ] **Step 5: Wire `resume=True` into the harnesses**

In `tools/_he_run.py`, change the engine-arm call (~line 148) from:

```python
            result = asyncio.run(run_benchmark(tasks, engine, config, sessions_dir=sess))
```
to:
```python
            result = asyncio.run(
                run_benchmark(tasks, engine, config, sessions_dir=sess, resume=True)
            )
```

In `tools/_lcb_run.py`, change the call (~line 330) from:

```python
                run_benchmark(tasks, engine, config, sessions_dir=sessions)
```
to:
```python
                run_benchmark(tasks, engine, config, sessions_dir=sessions, resume=True)
```

- [ ] **Step 6: Full suite + gates + commit**

```bash
uv run pytest tests/unit/ -q
uv run ruff check src/rune/bench/runner.py tools/_he_run.py tools/_lcb_run.py tests/unit/test_run_benchmark_resume.py
uv run mypy src/rune/bench/runner.py
git add src/rune/bench/runner.py tools/_he_run.py tools/_lcb_run.py tests/unit/test_run_benchmark_resume.py
git commit -m "feat(bench): resumable run_benchmark (skip tasks with a written session, #52)"
```

Expected: green; existing runner tests unaffected (`resume` defaults False).

---

## Execution / run protocol (pre-registered — not a code task)

After all four tasks land and the suite is green, run the LCB significance attempt. base is re-run at temperature 0; c3 at temp-0 floor + best-of-k.

> **HARD PRECONDITION (the byte-identical floor==base property is operator-enforced, not machine-enforced).** `_effective_temperature` forces only the *c3 floor* to greedy; the *base arm* reads `temperature` from config (default 0.3). The base arm MUST run with `RUNE_TEMPERATURE=0` (greedy) — otherwise base is stochastic, the floor is not byte-identical to base, and the strict-superset (`b=0`) guarantee silently breaks. Pin `thinking_budget` identically on both arms too (config.yaml already sets `thinking_budget: 0`; do not override one arm). **Backstop:** the `b=0` audit below catches any divergence — a non-zero `base_only` means the arms diverged (or a real oracle false-negative); investigate before claiming one-sided significance.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000 UV_NO_SYNC=1 TMPDIR=/tmp

# base arm: single-shot greedy (== the c3 floor). RUNE_TEMPERATURE=0 is REQUIRED
# (see HARD PRECONDITION above) — without it base samples at 0.3 and b=0 breaks.
RUNE_TEMPERATURE=0 uv run --extra gpu python tools/_lcb_run.py --arm base \
  --seed 0 --out /tmp/lcbout/lcb_base_t0.json --experiment issue52-powered

# c3 arm: greedy floor (auto), best-of-k=8 escalation at temp 0.8
RUNE_TEMPERATURE=0.8 RUNE_ESCALATION_BEST_OF_K=8 \
  uv run --extra gpu python tools/_lcb_run.py --arm c3 \
  --seed 0 --out /tmp/lcbout/lcb_c3_powered.json --experiment issue52-powered
# if killed, re-run the SAME command — resume=True skips completed tasks
```

Then score and audit:

```bash
uv run python -c "
import json
from rune.bench.significance import paired_compare, format_report
base = {k: bool(v) for k, v in json.load(open('/tmp/lcbout/lcb_base_t0.json')).items()}
c3   = {k: bool(v) for k, v in json.load(open('/tmp/lcbout/lcb_c3_powered.json')).items()}
r = paired_compare(base, c3)
print(format_report(r))
print('STRICT SUPERSET' if r.strict_superset else 'AUDIT REGRESSIONS:', 
      [k for k in base if base[k] and not c3.get(k)])
"
```

Pre-registered decision rule: success = `strict_superset` (audited `base_only == 0`) and `c3_only >= 5` ⇒ one-sided p ≤ 0.031. If `base_only > 0`, report two-sided p and treat each regression as an oracle false-negative bug (do not claim one-sided significance).

(`*.json` for the LCB harness store `{task_id: generation}`; convert to pass/fail via the official grader output the harness already logs, or use `result.per_task` — wire the per-task pass map the same way `_he_run.py` does. If the LCB harness does not already emit a `{task_id: bool}` map, add a one-line dump of `{tr.task_id: bool(tr.passed) for tr in result.per_task}` beside the existing `gens` dump.)

---

## Self-review notes (addressed)

- **base→bool conversion for LCB:** the LCB harness emits generations, not pass/fail. The execution section flags wiring a `{task_id: bool}` dump from `result.per_task` if absent — fold that one-liner into the run step.
- **RepoBench:** out of scope for this plan; the significance module + temp-0 floor transfer unconditionally, best-of-k only if RepoBench tasks have an executable public oracle (per spec).
