# Powered strict-superset evaluation — design (2026-06-22)

## Goal

Achieve a **statistically significant** improvement of the rune engine over the
base model on a **fixed task set** (no new tasks collected). Immediate target:
LiveCodeBench-v6 functional (N=63), currently base 12 → c3 16 (+4, strict
superset, McNemar two-sided p=0.125 — underpowered). The methodology must be
reusable for RepoBench.

## Core insight

For **paired** binary outcomes, significance is governed by the **discordant
pairs** (McNemar), not by N. With `b` = regressions (base✓/c3✗) and `c` = gains
(base✗/c3✓), the exact McNemar p when `b = 0` (a strict superset) is:

| net gains `c` (with `b=0`) | one-sided p | two-sided p |
|---|---|---|
| 4 (LCB today) | 0.0625 | 0.125 |
| **5** | **0.031 ✓** | 0.0625 |
| 6 | 0.016 ✓ | 0.031 ✓ |

So at the locked N, significance reduces to: **achieve `b=0` and `c ≥ 5`** (one-sided).

## Statistical design (pre-registered)

- **Hypothesis (directional, fixed before looking at final numbers):**
  `pass@1(c3) ≥ pass@1(base)` — the engine never hurts, may help.
- **Test:** McNemar **exact, one-sided** on the paired per-task outcomes.
- **Strict-superset property (`b=0`) by construction** — see Run protocol: at
  temperature 0 the escalation floor is byte-identical to the base arm, so the
  engine can only *add* tasks, not lose them.
- **`b=0` audit (the one place it can break):** `b>0` is only possible if the
  in-loop oracle false-negatives a base-correct floor (rejects a correct
  zero-shot → escalates → ships worse). Fixed-oracle false-negative rate ≈ 1/69.
  Pre-registered rule: after each run, audit every base✓ task. If `b>0`, report
  the actual `b,c` with **two-sided** McNemar (no one-sided claim) and treat the
  false-negative as a bug.
- **Transparency:** always report `b`, `c`, and **both** one- and two-sided p
  alongside the headline. The one-sided claim is justified by the engine's
  no-regress design, not chosen post-hoc.

## Run protocol

- **Floor = greedy (temperature 0), adapter off.** Greedy decoding is argmax and
  does not draw from the RNG, so the escalation floor's first code attempt
  (`_is_zeroshot_attempt`, same `prompt_zeroshot` as the base arm) is
  byte-identical to base's output. Where base passes, the oracle accepts it and
  the engine early-stops → ships the *same* output → c3 passes too. This makes
  `b=0` structural and removes the temp-0.3 RNG divergence (which produced the
  HE/140 churn regression). The base arm also runs at temperature 0.
- **Escalation = oracle-gated best-of-k.** Fires *only* where the floor fails the
  **trusted public** oracle (wired `public_test_cases` / spec doctests). Draw `k`
  candidates (temperature > 0 for diversity), ship the first that passes that
  public oracle. The floor stays greedy; only the escalation samples.
- **No leakage:** candidate *selection* uses public tests only. The held-out
  official tests (EvalPlus "plus" / LCB private) are the final arbiter and are
  never seen in-loop. Consequences: a best-of-k candidate counts as a gain only
  if it *also* passes held-out, so best-of-k can **only add or be neutral, never
  regress**, and never inflates the held-out-measured gain count.

## Build scope

1. **Significance module** — `src/rune/bench/significance.py`, pure-Python,
   unit-tested. Input: two `{task_id: bool}` maps (base, c3). Output: `b`, `c`,
   strict-superset flag, exact McNemar one-sided + two-sided p (via `math.comb`
   binomial — no scipy dependency). One report function emitting the headline +
   transparency line.

2. **Engine change** (in `rune`, no parallel harness):
   - **Temp-0 floor:** an `_effective_temperature` companion to the existing
     `_effective_scaling` in `engine/graph.py`, so the zero-shot floor attempt
     generates greedily regardless of the configured escalation temperature.
     Confirm `model.generate` / inference maps temperature 0 to greedy
     (`do_sample=False`). Base arm config set to temperature 0.
   - **Oracle-gated best-of-k escalation:** new `escalation_best_of_k` config
     (default 1 = current behaviour; pre-registered **k=8** for the significance
     runs). On an escalation step (floor already failed the trusted public
     oracle), sample `k` candidates and keep the first that passes the in-loop
     public oracle. Reuses the existing keep-best / `public_checks` gating; the
     new part is sampling `k` instead of 1.

3. **Hardened, resumable run harness** — `tools/_lcb_run.py` / `tools/_he_run.py`
   skip tasks that already have a written session (resume after a kill) and write
   the final per-task JSON from the sessions, so a late kill never loses the
   result. Investigate the per-task memory creep that SIGTERM-killed the
   2026-06-22 HumanEval+ run at task 152/164.

## Execution order

1. Land the significance module + engine changes + resumable harness (with tests).
2. **LCB re-run** (the first significance attempt): base @ temp 0, c3 @ temp-0
   floor + oracle-gated best-of-k (k=8), N=63 locked, official LCB grader.
   Report `b, c`, one/two-sided p, strict-superset audit.
3. RepoBench reuses the protocol once its track is ready (see dependency below).

HumanEval+ remains a **separate correctness track** (not a significance vehicle;
base is already 71% with little headroom).

## Risks & mitigations

1. **Temp-0 raises base's own pass@1 → less headroom.** Measure base@temp0
   first; the strict-superset property holds regardless, and best-of-k targets
   the residual base-failures.
2. **Best-of-k overfits the public oracle.** A public-pass / held-out-fail
   candidate is not counted (held-out arbiter) and cannot regress (fires only on
   base-failures) — no false gains, only wasted compute. `k` is pre-registered to
   prevent k-hacking.
3. **`b=0` not perfectly structural** (oracle false-negative ≈ 1/69). Audit; if
   `b>0`, report two-sided only. Prediction to verify: temp-0 makes HE/140's
   regression vanish (floor == base).
4. **One-sided pushback.** Pre-registered directional hypothesis + two-sided
   reported alongside.
5. **Method honesty.** Oracle-gated best-of-k is a standard public-test reranking
   technique (à la CodeT); the claim is stated plainly as "rune with
   oracle-gated escalation ≥ base single-shot."
6. **Memory creep / kills.** Resumable harness + memory investigation (above).

## RepoBench reuse

The **statistical framing** (one-sided McNemar, strict superset, temp-0 floor)
transfers unconditionally. The **best-of-k gain lever requires an executable
public oracle** for in-loop gating — it transfers if RepoBench tasks ship runnable
tests, but **not if RepoBench scores by exact-match completion**. Open dependency:
confirm RepoBench's evaluation signal with that track before relying on best-of-k
there; if exact-match, the gain lever must be redesigned (the strict-superset
floor still applies).

## Success criteria

- LCB: `b=0` (audited) and `c ≥ 5` ⇒ one-sided McNemar p ≤ 0.031, at N=63.
- The strict-superset + significance result is reproducible (temp 0 ⇒ deterministic
  floor; best-of-k `k` and seed pinned).
- Methodology + pre-registration documented so RepoBench can adopt it.

## Out of scope (YAGNI)

- Increasing the task set / collecting new benchmarks.
- Multi-seed confidence intervals (temp-0 determinism makes the floor exact;
  best-of-k variance is bounded by pinning `k` and the escalation seed).
- Changing the base model, adapter checkpoint, or training.
