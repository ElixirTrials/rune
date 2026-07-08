<!-- Produced by an automated multi-agent investigation during the C4 Stage-1 campaign (2026-07-08); shipped generations re-graded official-equivalently on-box. Companion to issue52-lcb-durable-findings-2026-06-19.md. -->

# LCB 6-task failure-mode investigation — run of 2026-07-08 (C4/I0, `issue52-c4`)

**Run under investigation:** `lcb-c3-full-seed0`, MLflow exp 81 run `8bfc80c43cad`, engine commit `6f12b40`,
checkpoint c3 (`53e24af2…`), seed 0, `max_phase_iterations=24`, `prompt_mode=full`, `temperature=0.3`,
`max_tokens=2048`, `thinking_budget=0`, `model_judge=False`, `escalation_best_of_k=1`, `max_repairs=4`
(→ retry cap 8, `src/rune/engine/policy.py:19-20`), launched per `docs/publication/c4_implementation_plan.md`
Task 2 with **`--no-grade`** (its purpose was regenerating sessions for the I0 symbol-reuse audit, not pass@1).
Recorded result: rune-internal pass@1 = 1/6 (3777 only). **No official LCB grade was produced** — `--no-grade`
was passed AND the official harness is gone from this box (`/tmp/lcbenv`, `/tmp/LiveCodeBench` missing; only
`/tmp/lcb/test6.jsonl` survives).

**Headline finding:** the internal 1/6 is *right by coincidence and wrong per task, twice over*. Grading
today's shipped `i0_gens.jsonl` against the full public+private suites from `test6.jsonl` with
official-equivalent call-based semantics (star-import preamble, 6 s/test, json-normalized comparison):

| qid | entry_point | internal pass@1 | official-equivalent verdict (this investigation) |
|---|---|---|---|
| 3753 | maxDifference | **false** | WRONG at test 3/33 (`"mwm"` → got 1, want −1) |
| 3754 | maxDistance | false | WRONG at test 1/42 (returns `None`) |
| 3748 | sortMatrix | false | WRONG at test 1/43 |
| 3801 | beautifulNumbers | **false** | TLE at test 3/42 |
| 3777 | maxProduct | **true** | **TLE at test 4/43** — internal pass is an official fail |
| 3799 | totalNumbers | **false** | **PASS 35/35** — internal fail is an official pass |

3799 was solved today — correct algorithm, verified both against the full official suite and against an
independent brute force (0/300 random mismatches) — **and no historical arm ever solved it**. The internal
harness mislabeled it, and with `--no-grade` nobody saw it.

---

## 1. Per-task autopsy

Sessions: `/tmp/c4/i0_sessions/<qid>/session.jsonl`. Run order and wall time (session-dir mtimes):
3753 ~1 min → 3754 **~62 min** → 3748 ~12 min → 3801 ~5 min → 3777 ~9 min → 3799 ~6 min (total ~95 min vs
the plan's 20–30 min estimate; 3754 alone is the "retry-exhaustion slowness" finding made flesh).

### 3753 maxDifference — near-miss algorithm + ship-form grading defect (the requested discrepancy)
- **What happened:** 3 records (decompose/plan/code). Step-2 code passed the in-loop oracle
  (`feedback: {'stdout': '', 'stderr': '', 'exit_code': 0}`) so `select_action` returned `[]`
  (single-subtask done, `policy.py:212-215`) and the session ended believing success. Metadata says
  `pass_at_1: false`.
- **Why the discrepancy — two independent layers:**
  1. **Grading-form mismatch (engine defect).** Step-2 output is `class Solution:` form. `best_code`
     stores the raw extracted code; `resolve_shipped_code` validates the *normalized* form
     (`_passes_public_checks`, `runner.py:174-182` calls `extract_entry_function`) but **ships the raw
     class-form candidate** (`runner.py:279-283`). `build_graded_program` (`runner.py:106-117`) then appends
     `test_code` whose asserts call the *bare* function (`build_public_assert_checks`, `lcb.py:10-18`:
     "Matches the engine's top-level ``def entry_point`` contract (not ``Solution()``)"). Reproduced:
     class-form graded → `NameError: name 'maxDifference' is not defined` → `pass_at_1=false`; the same code
     normalized → exit 0. So internally 3753 "failed" on a NameError, not on its logic.
  2. **abs() near-miss (model limit + weak public oracle).** The logic is wrong anyway: it computes
     `max(|odd−even|)` instead of `max(odd)−min(even)`. Both public tests pass
     (`'aaaaabbc'→3`, `'abcabcab'→1`); official-equivalent grading fails at test 3/33: `"mwm"` → got 1,
     want −1 (answers can be negative; `abs` can never return one). The 2-check public oracle cannot see
     this, and `model_judge=False` today, so nothing routed it to repair.

### 3754 maxDistance — prose degeneration + truncated headless function; 62 minutes burned
- 11 records, 9 code/repair attempts, outputs of 6.5k–30k chars each. Every attempt starts a plausible
  `def maxDistance(s, k):` then drifts into thinking-out-loud prose, e.g. step-3 tail:
  `"Given the ambiguity, and since the user input is empty, I will output:\n\n0"`; step-9 contains corrupted
  text mid-function (`“nwknee” contains ‘n’ and ‘nw’.`). The continuation sub-loop
  (`graph.py:1002-1077`, `cont_budget=5` × 2048 tokens, `cont_scaling = 1.0×1.53`) pumped these blobs;
  the degeneration guard fired only twice all run (log: `Degeneration detected (0.57/0.79)`) because most
  prose scores under the 0.5 threshold (`graph.py:1050`).
- Every probe result was identical: `AssertionError: maxDistance(*['NWSE', 1]) -> None, want 3` — the
  salvage extractor (`lcb.py:82-113`) recovers the largest parseable prefix, which is a function **with no
  return statement**; shipped code in `i0_gens.jsonl` is that headless 260-char prefix. The engine spent
  8 retries + ~62 min re-generating the same `-> None` failure and never produced a complete function.
- Underlying model limit: it never approaches the correct O(n) "flip up to k wrong-direction moves,
  max over prefixes" algorithm (same verdict as June: PR #55 "q3754 … ignores `k` … model limit").

### 3748 sortMatrix — clean model-capability failure (oracle worked as designed)
- 11 records, 9 attempts, all quality-2 near-misses. The oracle fired every time with a specific
  actual-vs-expected message, e.g. step 2:
  `AssertionError: sortMatrix(*[[[1,7,3],[9,8,2],[4,5,6]]]) -> [[1,8,6],[4,7,3],[1,5,2]], want [[8,2,3],[9,6,7],[4,5,1]]`.
- The model never indexes the diagonals correctly (reads column/row segments, writes diagonals; step-5
  "repair" emitted only a 52-char grid literal → NameError). Shipped = step-10 attempt (last of the
  quality-2 ties wins, `parse.py:368`), officially WRONG at test 1/43. Same failure mode as June
  ("model misread spec … wrong sort — model limit").

### 3801 beautifulNumbers — correct-on-publics brute force; complexity-capability wall + ship-form defect
- 11 records. The model oscillates between a *correct* brute force (`for i in range(l, r+1)`) that the
  complexity oracle rejects (`constraint_scale: static analysis indicates O(n); Constraints allow
  n≤1000000000 — need O(log n) or better`, steps 2/4/5/7/8/10) and *wrong* "fixes" (digit-0 → not
  beautiful) that fail the public oracle (steps 3/6/9). It never attempts digit-DP. Retries exhausted.
- Constraint-scale-only failures rank quality 3 (`parse.py:322-323`), so the last brute-force candidate
  shipped — in `class Solution` form → internally graded to `NameError: beautifulNumbers is not defined`
  (reproduced). Normalized, it passes publics but is an official TLE at test 3/42. So: internal label
  (NameError) ≠ actual failure (TLE); both are fails, for different reasons.

### 3777 maxProduct — identical story to 3801, but it won the "bare-def lottery"
- 11 records; every attempt is the same O(2^n) subset enumeration; the complexity oracle rejected each
  (`constraint_scale: assessed O(2^n·n) … Constraints allow n≤150 — need O(n³) or better`;
  `complexity_failed/maxProduct=1` in MLflow). Retries exhausted; last quality-3 brute force shipped.
- Because this model happened to emit **bare `def maxProduct(...)`** (no class wrapper), the ship-time
  sandbox pass ran the asserts successfully → internal `pass_at_1: true`. Officially it is a TLE at test
  4/43 (private n up to 150) — exactly June's verdict ("Public 3/3 pass; official TLE … complexity-class
  error"). **Today's single internal "pass" is a false positive w.r.t. official grading.**

### 3799 totalNumbers — SOLVED by the engine; failed only by the internal grading harness
- 6 records. Steps 2–4 fail the public oracle with specific wrong counts; step-5 repair returns
  `exit_code: 0` on all 4 public checks. The algorithm (Counter-based distinct-3-digit-even counting) is
  correct: official-equivalent **35/35 pass**, and 0/300 mismatches vs an independent
  `itertools.permutations` brute force. This is real diagnose→repair value — and a task no arm ever solved.
- Internally marked failed because the shipped raw code is `class Solution` form (+ module imports) →
  bare-call asserts NameError. Even the *normalized* bare form fails the internal gate for a second
  reason: `build_graded_program` has **no import preamble**, so the retained `List[int]` annotation raises
  `NameError: name 'List' is not defined` (reproduced) — the exact bug class fixed for the *in-loop* probe
  in `oracle.py:212-229` (`with_probe_imports`: "Mirrors the grader, not a behaviour change") and for
  HumanEval+ in the 06-22 RCA, but never applied to the LCB ship-time gate. The official harness injects
  `from typing import *` / `from collections import *`, so officially the code passes.

---

## 2. Failure-mode taxonomy

**A. Grading/harness defects (engine measurement, not generation)** — corrupted 3 of 6 internal labels:
- **A1. Ship-form mismatch:** `resolve_shipped_code` validates the normalized entry form but returns the
  raw `class Solution` blob (`runner.py:279-283` vs `:174-182`); `build_graded_program` appends bare-call
  asserts → NameError regardless of correctness. Hit 3753, 3801, 3799. Whether a task passes internal
  grading depends on whether the model happened to emit a bare def (3777) or a class (the others).
- **A2. Missing grader-mirror imports at ship time:** `build_graded_program` lacks
  `with_probe_imports`; typing/collections annotations NameError (hit 3799's normalized form).
  Also `normalize_lcb_submission`/`extract_entry_function` (`lcb.py:144-199`) drop the model's own
  module-level imports when unparsing the bare function — survivable officially only because the official
  grader star-imports.
- **A3. In-loop oracle ≠ ship gate:** the loop terminates on oracle pass (3753 at step 2, 3799 at step 5)
  but the ship gate re-grades under different rules, so "the engine thinks it's done" and "the metric says
  it failed" coexist. `metadata.json` then persists the wrong verdict, which `resume=True` will re-serve
  (`runner.py:393-409`).
- **A4. No official grade in the loop today:** `--no-grade` + missing lcbenv. June history shows official
  and internal verdicts disagree in both directions (exp 72 run `cc753c11…`: internal 0/6, official 1/6).

**B. Model-capability limits (4B ceiling)** — the dominant cause of *true* failures:
- **B1. Cannot derive the required algorithm:** 3748 (diagonal indexing), 3754 (k-flip prefix max).
- **B2. Complexity wall:** 3777, 3801 — the model reliably finds the brute force, the complexity oracle
  correctly rejects it 6+ times with precise guidance ("need O(n³)/O(log n) or better"), and the model
  re-submits the same brute force every time. Feedback quality is not the limiter; capability is
  (consistent with June's E-oracle-rootcause: perfect oracle 11/11 fires, 0/11 solves).
- **B3. Near-miss logic invisible to weak publics:** 3753's `abs()` variant passes 2/2 publics.
- **B4. Prose degeneration on hard tasks:** 3754's 27–30k-char rambles; continuation loop amplifies.

**C. Configuration choices:**
- `prompt_mode=full`, not `escalate` — today's run lacks the zero-shot floor and adapter-on-repair
  escalation used by the pre-registered June arms (the only historical 3753 solve came from an escalate
  arm). Also `model_judge=False`, `escalation_best_of_k=1`, `thinking_budget=0`.
- Retry cap 8 (`max_repairs=4`) × continuation budget 5 × 2048 tokens with no same-failure dedup →
  3754's hour. Degeneration threshold 0.5 too permissive for prose tails.

---

## 3. Historical answer — were these six ever solved?

Sources: durable 06-19 campaign per-qid artifacts recovered from S3
(`s3://elixirtrials-949678234935-us-east-1-artifacts/mlflow/artifacts/78/…/{base,scale0,c3}_perqid.json`,
`attribution_summary.json`; the MLflow DB rows for that experiment were lost in the snapshot restore —
the surviving DB has no `issue52-lcb-durable` experiment, but the S3 artifacts are complete);
`docs/issue52-lcb-durable-findings-2026-06-19.md`; `docs/issue52-experimentation-log.md` §3.7; PR #55
comments; MLflow exps 72/73/74.

| qid | base (06-19) | scale0 (06-19) | c3 (06-19) | ever solved before today? |
|---|---|---|---|---|
| 3748 | wrong 0/1 | wrong 0/1 | wrong 0/1 | **Never.** |
| 3753 | wrong 2/3 | **PASS 33/33** | wrong 2/3 | **Yes — once, scale0 (escalate, adapter-off), 06-19** (base→scale0 gain; c3 lost it again — the doc's "adapter churn"). Also officially passed in the 06-08 replay after the ship-best fix (PR #55: "3753 … 0/1 wrong → 1/1 pass"), and exp-72 run `cc753c11…` (internal 0/6, official 1/6) is consistent with 3753 being that 1. Caveat: the 06-09 "10/49" run solved it via hard-coded answer injection in `repair_brief.py`, retracted in E-deoverfit. |
| 3754 | wrong 0/1 | wrong 1/2 | wrong 0/1 | **Never.** June verdict: model limit (ignores `k`). |
| 3777 | TLE 3/4 | TLE 3/4 | TLE 3/4 | **Never.** Always public-pass → private TLE. |
| 3799 | wrong 1/2 | wrong 0/1 | wrong 0/1 | **Never before today.** June failure was full problem substitution (`TypeError`). **Today the engine produced a fully correct solution (35/35)** — recorded as a failure by the internal harness. |
| 3801 | TLE 2/3 | TLE 2/3 | TLE 2/3 | **Never.** Always brute-force TLE. |

Provenance note: these six are the `lcb_engine_fixes` regression fixtures precisely *because* they exposed
engine bugs in the June 0/49 overnight run (vendored at `af06a5b`; tests cover decompose-collapse,
ship-best-over-regressing-integrate, wrong-signature integrate, exhaustion shipping —
`tests/unit/test_lcb_engine_fixes.py`). They are an adversarial slice, not a representative sample; 1/6
internal ≈ the June per-slice expectation (historical arms scored 0–1/6 on this slice officially).

What is *missing* from the record: the durable experiment's params/metrics rows (DB loss, exps 78–86);
per-qid grades for the pre-06-19 exp-72 6-task runs (only step artifacts survive, no gens); nothing else
material.

---

## 4. What would convert failures to passes — ranked

1. **Fix the internal ship-time grading harness (A1+A2) — highest impact, small effort (S).**
   In `run_benchmark`/`resolve_shipped_code`, grade the *normalized* entry form (the same
   `extract_entry_function` output that `_passes_public_checks` already validates and that
   `normalize_lcb_submission` ships to the official grader), and build the graded program with the same
   import preamble the in-loop probe uses (`with_probe_imports`). Today that flips 3799 to a recorded pass
   (true official-equivalent 1/6, correct task identity) and makes internal pass@1 trustworthy.
   *Pre-registration-safe:* this changes measurement, not generation; saved gens can be re-graded with
   both old and new gates for a documented delta, and official grading is unaffected (it already
   normalizes). The six fixtures are the natural regression tests.
2. **Restore official grading in the loop (A4) — S/M (environment).** Re-provision `/tmp/lcbenv` +
   `/tmp/LiveCodeBench`, stop passing `--no-grade` on runs whose sessions feed analysis, and treat the
   internal metric as advisory. June's record already shows internal↔official disagreement in both
   directions; today it flipped *which* task passed. Zero provenance cost.
3. **Stop paying for hopeless repair loops (B2/B4 budget waste) — M.**
   Two cheap policy guards: (a) same-failure dedup — if the probe stderr and the approach signature
   (`_approach_signature`, `graph.py:163`) are unchanged for N consecutive attempts, stop or force replan
   (3777/3801 re-submitted the identical brute force 6+ times; 3748 similar); (b) constraint-scale-only
   failures already ship at quality 3, so cap complexity-driven repairs at ~2 attempts. Frees ~half the
   retry budget. *Provenance cost:* changes engine behavior mid-campaign — gate behind a config flag,
   document, and don't compare against pre-change runs.
4. **Tame the continuation loop (3754's hour) — M.** Lower/complement the 0.5 degeneration threshold with
   a structural check ("does the new chunk parse as a continuation of the code?"), and abort continuation
   when the accumulated blob's salvaged function is complete-but-prose-tailed. Converts no failures by
   itself (3754 is a capability failure) but recovers ~60 min/run of GPU time and reduces
   context/adapter-conditioning pollution in later repairs (step-6+ outputs visibly quote the garbage
   back). Same provenance caveat as (3).
5. **Strengthen the near-miss oracle (B3) — S config / M for new checks.** Turn `model_judge=True`
   (exists, off today) or synthesize 1–2 extra small-input checks by brute-force reference on tiny inputs.
   Expected yield is modest at 4B capability (June: perfect oracle converted 0/11), but 3753-class
   near-misses are exactly its target. Judge adds GPU cost per step.
6. **Run the pre-registered arm (C) — S.** If the goal is comparability with June, use `--prompt-mode
   escalate` (zero-shot floor + adapter escalation). The only historical solve on this slice (3753,
   scale0) came from an escalate arm; today's `full` mode is a different treatment. No engine change.

Not recommended as a lever: bigger retry caps or budgets — every exhausted task today failed on capability
or complexity, and more of the same attempts would not have changed any verdict.

---

## Appendix: evidence trail

- Today's traces: `/tmp/c4/i0_sessions/{3748,3753,3754,3777,3799,3801}/session.jsonl`, `/tmp/c4/i0_gens.jsonl`
  (single pretty-printed JSON array, not JSONL), `/tmp/c4/i0_run.log`.
- Reproductions run during this investigation (CPU sandbox only):
  - 3753 class-form + `build_graded_program` → `NameError: name 'maxDifference' is not defined`; bare form → exit 0.
  - 3753 abs-bug counterexamples: `'aabbc'` → 1 (want −1); official test 3/33 `"mwm"` → 1 (want −1).
  - 3801 class-form → `NameError: name 'beautifulNumbers' is not defined`; bare form passes publics.
  - 3799 bare form + graded program → `NameError: name 'List' is not defined`; with probe imports → exit 0;
    full official-equivalent suite 35/35; brute-force cross-check 0/300 mismatches.
  - Official-equivalent grades (star-import preamble, 6 s/test): table in header.
- Engine code: `src/rune/bench/runner.py:106-117` (build_graded_program), `:174-182` / `:279-283`
  (normalize-validate vs raw-ship), `:393-409` (resume trusts stored pass_at_1), `:499` (pass gate);
  `src/rune/engine/oracle.py:212-245` (probe imports + build_probe); `src/rune/bench/lcb.py:10-37`
  (bare-call asserts), `:144-199` (normalization drops imports); `src/rune/engine/graph.py:1002-1077`
  (continuation, degen>0.5), `:1236-1255` (complexity oracle); `src/rune/engine/parse.py:303-328`
  (quality; constraint-scale-only = 3); `src/rune/engine/policy.py:19-29`, `:158-209` (retry caps,
  exhaustion shipping), `:212-215` (single-subtask done); `tools/_lcb_run.py:25-34` (arms), `:196-212`
  (defaults incl. `--grade`).
- History: `docs/issue52-lcb-durable-findings-2026-06-19.md` (§2-3 per-arm, §7 HumanEval RCA pointer);
  `docs/issue52-experimentation-log.md` §3.7 (E-lcb49-arc 0/49→9/49, E-deoverfit answer-injection);
  `docs/issue52-humaneval-regression-rca-fix-2026-06-22.md`; PR #55/#57 comment archives; MLflow exps
  72/73/74 (surviving) and S3 `mlflow/artifacts/78/` (durable per-qid, DB rows lost);
  `tests/unit/test_lcb_engine_fixes.py` + `tests/fixtures/lcb_engine_fixes/` (fixture provenance, `af06a5b`).
