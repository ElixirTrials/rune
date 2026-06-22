# HumanEval+ regression — root cause + fix (2026-06-22)

Supersedes the "rune is difficulty-dependent / hurts easy tasks" conclusion in
`issue52-lcb-durable-findings-2026-06-19.md` §7/§9. **That conclusion was an
artifact.** The HumanEval+ "−16, 20 regressions" was not a capability regression;
it was the engine shipping an unvalidated escalation over a correct zero-shot it
had already generated, because the harness fed `public_checks=""`.

## Root cause

The engine implements an escalation **floor** (`escalate` mode): the first `code`
attempt per subtask runs the base model on a clean single-shot prompt with the
adapter OFF (scaling 0) — i.e. base capability — and the design intent is
"keep-best floors the engine at base". On LCB this holds: c3 is a strict
superset of base, 0 regressions. The floor depends on a trustworthy public
signal (`public_checks`), and HumanEval+ was built with `public_checks=""`,
which broke it on two layers:

1. **In-loop (escalate uselessly):** with no `public_checks`, the in-loop oracle
   falls back to the decompose model's `acceptance_check`, which is frequently
   garbage — `debug(...)` (NameError), malformed/unparseable (fail-closed:
   "oracle checks configured but probe did not fire"), or simply wrong. The
   correct zero-shot is marked failing, so the engine escalates through its whole
   budget.
2. **Ship-time (lose the floor):** `resolve_shipped_code` gates candidates through
   `_benchmark_shippable`, which returns `True` for **any** code when
   `public_checks` is empty. So `integrated_code` (or a repair) is shipped
   unconditionally and the retained-correct zero-shot is never selected.

The base arm grades the clean zero-shot; the c3 arm grades the engine output with
the floor switched off → not apples-to-apples.

**Evidence:** of the 20 regressed tasks, **19/20 — the engine's own zero-shot
(adapter off) passes the held-out test, but the shipped escalation fails.**
0/20 were over-decomposed (all single-subtask). The engine generated the correct
answer and discarded it.

## Fix (root cause, no band-aids)

An earlier ship-time "zero-shot floor" in `resolve_shipped_code` was **reverted**:
it patched the symptom (ship-selection) and was a fallback. The root cause is the
in-loop oracle trusting model-authored checks, so the fix is there. Three changes:

1. **`engine/graph.py` `resolve_in_loop_check`** — the graded **entry point** is
   gated **only** by trusted public examples (wired benchmark checks or the spec's
   doctests). The decompose model's `acceptance_check` is never the entry's
   correctness oracle (it emitted undefined-`debug`/unparseable checks that
   rejected correct base zero-shots). With no public signal the entry has no
   in-loop gate (module-load only) and the engine ships the base zero-shot.
   Non-entry **helper** subtasks (no public examples of their own) still use their
   model-authored check.
2. **`engine/oracle.py` `extract_public_checks`** — extract docstrings via AST
   (`_doctest_source`) before running `DocTestParser`, so a HumanEval prompt's
   closing `"""` is no longer absorbed into the last example's expected value
   (was producing `repr(14\n""")` → `SyntaxError`, e.g. `fib4`).
3. **`bench/runner.py` `run_benchmark`** — derive `public_checks` from the spec's
   doctests even when the task ships none (the merge no longer requires a non-empty
   wired set). HumanEval/MBPP carry their examples in the docstring, so this gives
   the engine a trustworthy in-loop + ship-time signal.

Net behaviour: trusted signal present → escalate and **verify**, keeping gains and
flooring at the doctest-passing base via keep-best (as on LCB); no trusted signal
→ no in-loop gate → ship the base zero-shot. The engine works without a
special-case floor or fallback, and nothing is swallowed.

Tests: `tests/unit/test_oracle_signal_fixes.py` (A: no triple-quote in extracted
checks + runnable; B: `run_benchmark` derives public_checks from spec doctests;
C: entry ignores an untrusted model `acceptance_check`, helpers still use theirs).

## Verification (end-to-end re-run — completed 2026-06-22)

Both arms re-run at engine `efa7b9e` (seed 0, budget 24, judge off) through the
fixed harness (the escalation-floor RCA above **plus** the import-preamble fix in
the addendum below). The apparent −16 regression is **fully resolved**.

| arm | pass@1 |
|---|---|
| base, pre-import-fix `db48504` | 116/164 (0.707) |
| c3 BEFORE fixes `5954c81` | 100/164 (0.610, −16) |
| **base, post-import-fix `efa7b9e`** | **134/164 (0.817)** |
| **c3 AFTER fixes `efa7b9e`** | **135/164 (0.823)** — strict superset of base: +1 (`HumanEval/10`), 0 regressions |

The base arm rose **+18** (116→134) purely from the import-preamble grading fix
(addendum; within the 19 typing-signature tasks — the base arm does not use the
engine). c3 then adds **one** engine gain (`HumanEval/10`, a multi-step
`make_palindrome` the zero-shot gets subtly wrong and the escalate→repair loop
fixes) with **zero** regressions — c3 (135) ⊇ base (134). MLflow
`issue52-humanevalplus`: `he-base-seed0` / `he-c3-seed0` @ `efa7b9e`.

Trace-based soundness pre-check (fixed extractor): of the 69 doctest-bearing
tasks, the held-out-correct zero-shot fails the derived doctests in only **1**
case (low false-negative/regression risk), while **21** zero-shots pass the public
doctests yet fail the hidden EvalPlus tests — the engine early-stops on those (it
cannot know they are wrong), so the upside over base is bounded by escalation
headroom on the remaining doctest-failing tasks.

## Addendum (2026-06-22): grading dropped the prompt's imports

A pre-launch smoke (HumanEval/0, /1) caught a **second, independent** grading
defect affecting BOTH arms: the graded program was `solution + test`, omitting the
prompt's imports. A correct solution whose generated body does not echo
`from typing import List` then fails with `NameError: List` at definition (the
signature annotation is evaluated at def-time). Confirmed: the shipped, correct
`has_close_elements` failed grading as-is; `+ "from typing import List"` → pass.
Blast radius: **19/164** tasks use a typing name in the signature.

Fix (`bench/runner.py`): `BenchTask.grading_preamble` (default `""`) + a single
`build_graded_program(task, code)` assembler now used by both the c3 (`run_benchmark`)
and base (`_gen_base`) paths; `tools/_he_run.py::load_he_tasks` populates the
preamble with the prompt's import lines. Verified end-to-end: HumanEval/0+1 base
**0/2 → 2/2** and c3 **0/2 → 2/2** (the final-grade fix suffices — the engine
already ships the correct floored zero-shot; no in-loop probe change needed).
Both arms are affected symmetrically, so prior base-vs-c3 comparisons stay valid,
but absolute pass@1 was understated; the full re-run uses the fixed grader.
Tests: `tests/unit/test_grading_preamble.py`.
