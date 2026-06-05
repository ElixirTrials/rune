# Hard-task failure diagnosis — why the adapter helps on easy but not hard tasks (2026-06-05)

**Observation:** on the easy MBPP pool, the c3 adapter lifted reference-mode pass@1
**0.33 → 0.58 (+0.25)**. On 8 hard multistep tasks (calculate, decode_string,
int_to_roman, merge_intervals + 4 opaque-named), **c3 (0.25, 2/8) == scale0 floor
(0.25, 2/8)** — no memory gain. Thorough analysis below (prompts, adapter
conditioning, hyperparameters, failure modes).

## What is in the prompts (per engine action, reference_a mode)

| action | prompt contains | spec in prompt? |
|---|---|---|
| decompose | "Decide how many subtasks…" + decision rule | **no** (spec only in adapter) |
| plan | `Project: """{FULL SPEC incl. doctest}"""` | **YES — leaks** |
| code | "implement mission `X` — the Task is in your context" | no (spec only in adapter) |

`project_label = task[:1200]` regardless of `prompt_mode` — so the **plan step leaks
the full spec into the prompt in BOTH arms**, and the resulting plan is carried into
the code step's context. ⇒ `reference_a` is **not** a clean "spec only in adapter"
regime; the spec reaches the model at plan time even with the adapter off. The memory
comparison therefore measures only the *code-step* adapter contribution on top of a
plan that already saw the spec — not pure spec-carrying.

## What is passed to the adapter (conditioning / trajectory)

`render_training_format_trajectory`: `## Task\n{FULL SPEC}\n\n## Current Code\n\n\n##
Review Feedback\n` (code/repair use `render_reference_adapter(reference_a)` = `## Task\n
{spec}`). The multistep specs are ~280–340 chars — **well under the 1200 cap, no
truncation**. So the adapter *does* carry the full spec; **capacity/truncation is NOT
the failure.**

## Hyperparameters (c3 hard run)

`adapter_scaling=0.627` (HPO-tuned on the **easy** pool), `max_phase_iterations=6`,
`temperature=0.3`, `presence_penalty=0.0`, `top_p=0.9`, `thinking_budget=0`,
`cont_multiplier=1.53`, `cont_budget=5`, `max_tokens=2048`, `repetition_penalty=1.1`,
`no_repeat_ngram_size=12`, `model_judge=OFF`.

## Failure modes (with evidence)

**1. Oracle insufficiency — the primary cause.** The in-loop stop signal is the spec's
single public doctest example. On hard algorithms the public→held-out gap is large.
- c3 reached the public example on **attempt-1 for 5/8** tasks (3-step finishes) vs
  scale0 **1/8** — the adapter clearly helps produce public-passing code.
- But only **2/8 pass held-out**. The engine **stops the moment the public example
  passes** and ships code that is wrong on hidden cases.
- e.g. c3 `int_to_roman`: code passes `int_to_roman(9)=="IX"` (public) → engine stops at
  3 steps → fails the **first** held-out assert `int_to_roman(3)`. scale0's
  `int_to_roman` passes **all 7** held-out asserts.

**2. Adapter over-perturbation at scaling 0.627.** The adapter **trades tasks**: c3
*gained* `merge_intervals` but *lost* `int_to_roman` (scale0 solved it correctly, 7/7;
c3 broke it). 0.627 was tuned on easy tasks; on hard tasks it perturbs enough to
degrade some. Net held-out neutral (2 vs 2) via gain/loss churn — and n=8 is noisy.

**3. Plan-step spec leakage** (table above) — muddies the memory measurement; the floor
is not a true "no spec anywhere".

**4. Over-decomposition (secondary).** `calculate`/`calculate_opaque` split into 2–3
subtasks (`parse_expression`+`evaluate_tokens`), don't complete within budget=6, and
the subtasks don't define the `calculate` entry_point → fail regardless of adapter.

## Why easy ≠ hard

- **Easy:** public example ≈ held-out (few edge cases) → c3's public-passing help shows
  in pass@1 (+0.25); adapter perturbation is tolerable.
- **Hard:** public example ≪ held-out (many edge cases) → c3's public-passing help is
  invisible in held-out pass@1; adapter perturbation breaks some tasks → net neutral.

**The held-out pass@1 metric hides the adapter's real contribution** (5/8 vs 1/8
public-example attempt-1 success).

## Fixes (ranked)

1. **Stronger in-loop signal.** One public example is too weak a stop criterion on hard
   tasks. Use ALL public examples (LiveCodeBench problems have several), the fixed
   reason-first judge, or model-generated tests — so the engine doesn't stop at a
   wrong-but-public-passing solution and repair engages on held-out-style bugs.
2. **Re-tune `adapter_scaling` for hard tasks** (0.627 over-perturbs; sweep lower).
3. **Fix plan-step spec leakage** — minimal prompt at ALL steps for a clean memory test.
4. **Guard decompose** against over-decomposing single-function tasks (or raise budget).
5. Report the adapter effect on the **in-loop** signal, not only held-out pass@1.

## Caveats

n=8, single seed, scaling untuned-for-hard. The directional finding (oracle stops too
early on hard tasks; adapter helps reach it but not held-out; over-perturbation churns)
is consistent across the 8, but the magnitudes are noisy.
