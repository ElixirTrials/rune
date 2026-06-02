# HPO Trace Findings — experiment 38 / run `ec397bf9…`

Analysis of the 76 traces accumulated under MLflow experiment 38 (6 FAILED + 1 RUNNING
`bench-hpo` attempts, sharing `optuna_bench_hpo.db`). OOMs set aside on purpose — this
focuses on *logical* failure modes and on the search space itself.

## Failure taxonomy (76 traces)

| Outcome | Count | Where | Cause |
|---|---|---|---|
| OK | 31 | — | — |
| OutOfMemoryError | 25 | `generate_adapter` | resource (out of scope here) |
| **ValidationError** | **11** | `step` → decompose (5) / plan (6) | **truncated structured output** |
| UndefinedError | 5 | `code.j2:12` | template bug — **already fixed on current branch** |
| CancelledError | 4 | `step` | collateral async teardown |

## Where things actually go wrong

1. **Truncated JSON at decompose/plan (the only live logical failure — 11 traces).**
   All are `Invalid JSON: EOF while parsing…` on `DecomposeResult`/`PlanResult`. The
   payloads show the model **degenerating into repetition and never closing the object**:
   `"plan": "1. 1. 1. 1. …"`, `… 141, 142, 143, 144 …`, `{ \t\t\t\t\t…`. xgrammar keeps
   the prefix valid, the model loops, hits `max_tokens` mid-object, pydantic gets a
   truncated string. Proximate trigger = the token cap; real driver = **decoding
   degeneration**, concentrated in the two longest-output steps. `code` is clean once
   the template bug is excluded.

2. **`code_trajectory is undefined` (5) — already remediated.** From old commit
   `1e2f000` (`fix/pr45-review-correctness`). Current `graph.py:127/134` always sets
   `ctx["code_trajectory"]`. Historical, not live.

3. **CancelledError (4) — collateral**, not a distinct bug.

## Hyperparameter critique

Current search space: `{adapter_scaling, temperature, max_tokens, max_phase_iterations, cont_multiplier}`.

**`max_tokens` (512–4096) — remove from the search.** Weakly monotone against
correctness, no interior optimum. Raising it *reduces* the 11 truncations and *causes*
the OOMs, so the objective just rewards "largest value that didn't OOM this trial." Pure
confound. **Fix it high (~3072) and drop it.**

**`max_phase_iterations` (3–10) — keep it, but fix the objective.** It's also monotone
against pass@1 alone, so under the current objective its optimum is trivially the
ceiling. The fix is *not* to drop it but to **add a latency (or token) cost term to the
objective** so more repair passes are traded against their wall-clock cost — then the
search has a genuine trade-off and the interior optimum is meaningful. Keep the 3–10
range.

**Knobs with real interior optima — keep, but narrow (see below):**
`adapter_scaling`, `cont_multiplier`, `temperature`.

**The actual miss:** the lever that would fix the dominant non-OOM failure is
**`presence_penalty`**, and it is **held fixed (1.5) and never searched**. Note the code
detail: `repetition_penalty` and `presence_penalty` are independent processors
(`inference.py:155` vs `160`), but the HPO base disables `repetition_penalty` (1.0) and
relies on `presence_penalty` (per the Qwen note at `inference.py:17`) — so presence
penalty is the *active* anti-repetition control. And `no_repeat_ngram_size=12` only
applies when `output_schema is None` (`inference.py:170`); it does **nothing** for the
grammar-constrained decompose/plan generations where every truncation occurs. So
`presence_penalty` is the one knob that bites on these failures, and HPO leaves it fixed.

## Narrow the search space around known-good values

Ranges are too wide; we already know roughly where the good region is (HPO base config:
`adapter_scaling=0.49`, `temperature=0.7`, `cont_multiplier=1.53`). Tighten to:

| Param | Current | Proposed | Note |
|---|---|---|---|
| `adapter_scaling` | 0.1–1.0 (log) | **0.35–0.65** | around 0.49 |
| `temperature` | 0.1–1.0 | **0.5–0.8** | around 0.7; avoid low temps (degeneration) |
| `cont_multiplier` | 1.0–2.5 | **1.3–1.8** | around 1.53 |
| `max_tokens` | 512–4096 | **drop → fix 3072** | budget knob, not quality |
| `max_phase_iterations` | 3–10 | **keep 3–10** | needs latency term in objective (below) |
| `presence_penalty` | not tuned | **add ~1.2–2.0** | active lever for the truncation failures |

**Objective change (required for `max_phase_iterations`):** move from raw pass@1 to a
latency-aware objective, e.g. `pass@1 − λ·mean_latency` (or wall-clock / token budget),
so extra repair passes are charged for. Without this term `max_phase_iterations` has no
interior optimum and the search just pins it to 10.

Net: drop the one pure-budget confound (`max_tokens`), add the lever that actually fixes
the failures (`presence_penalty`), and make the objective cost-aware so the kept budget
knob (`max_phase_iterations`) is genuinely tunable — all over ranges centered on
known-good values. Smaller, non-confounded space converges far faster than the current
30-trial sweep.

## Two operational notes

- **No "current best" exists.** `optuna_bench_hpo.db` holds 1 RUNNING trial, 0 completed
  (`study.best_trial` → *"Record does not exist."*); the study was reset between attempts.
  The only in-flight trial: `adapter_scaling=0.446, temperature=0.423, max_tokens=512,
  max_phase_iterations=9, cont_multiplier=2.312`. The `0.49/0.7/3072/…` on the parent run
  are base-config defaults, not a tuned result.
- **Traces don't record the per-trial tuned config** (spans carry only `task`/`code`).
  Log trial params into trace tags so failures can be correlated with the config that
  produced them.
