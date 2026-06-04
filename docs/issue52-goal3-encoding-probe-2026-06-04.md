# Goal-3: is the recitation an ENCODING problem or a TRAINING problem?

**Date:** 2026-06-04 · **Branch:** issue52-bf16-body-contrastive · **Checkpoint:** c3
(`c3_t07_lp2_lg1.pt`, `body_recall_guarded`, distilled with EMPTY `## Current Code` /
`## Review Feedback`).

**Question (owner).** Before spending GPU on RL, test whether the substrate *already* carries
useful episode information and we just embed it wrong. If delineating the failure as something to
AVOID (not as `## Current Code`) plus a prompt that says "avoid these failure modes" widens the
fix-vs-failure logprob gap, then the fix is an encoding change, not retraining.

**Metric.** `gap = logprob(FIX) − logprob(FAILURE)` under the c3 adapter conditioned on a given
"episode template", measured on two repair scenarios (`add`: `a-b`→`a+b`; `largest`:
`min`→`max`). Higher gap = the correct fix is more accessible than repeating the failure.

## Episode probe (`tools/_episode_probe.py`) — prior step

- **SENSITIVITY ✓** — the adapter *responds* to episode content: max|Δ| 0.11–0.26 between
  task-only, episode(fail1), episode(fail2) conditioning.
- **USEFULNESS ✗** — conditioning on the episode (failing code in `## Current Code`) *raised*
  `logprob(FAILURE)` by ~+0.55, shrinking the gap from **+0.69 (task-only)** to **+0.27
  (episode)**. The adapter makes the failing code *more* accessible — recitation.

## Embed probe (`tools/_embed_probe.py`) — this step

Six (template, prompt) conditions. `task_only` = no failure info (baseline); `current_code` = the
recitation baseline; `failed_attempts`/`failure_summary`/`empty_plus_avoid` = failure delineated as
"do NOT repeat" or summarized as a mode, paired with the AVOID prompt.

| template | prompt | GAP (add) | GAP (largest) |
|---|---|---|---|
| **task_only** | default | **+0.800** | **+0.422** |
| current_code | default | +0.437 | +0.258 |
| current_code | avoid | +0.303 | +0.216 |
| failed_attempts | avoid | +0.543 | +0.158 |
| failure_summary | avoid | +0.620 | +0.161 |
| empty_plus_avoid | avoid | +0.441 | +0.169 |

`task_only` (NO failure info) has the widest gap in both — but that is the **wrong baseline**: it
can't host a repair turn (repair *requires* the failure in context) and its `lp(FAIL)` is low only
because the model never saw that specific bug. The **honest baseline is `current_code`** (a real
repair turn). Against it, the owner's hypothesis went **1-1**: `failure_summary/avoid` beat
`current_code/default` on `add` (+0.620 vs +0.437) but lost on `largest` (+0.161 vs +0.258).
**Toy tasks are also saturated** — base `lp(FIX) ≈ −0.1` (≈0.9 prob/token), no headroom for the
adapter to raise the fix, so only `lp(FAIL)` can move and *everything* looks like recitation. The
gap is positive in all 12 cells, so "recites" overstates it. → **inconclusive; need hard tasks.**

## Embed probe — HARD tasks (`tools/_embed_probe_hard.py`)

Same grid on `int_to_roman` (no→with subtractive pairs), `decode_string` (single→multi-digit k),
`merge_intervals` (unsorted→sorted) — multi-line, non-obvious fixes. Added a `base` row (adapter
**off**, `disable_adapter()`) to confirm headroom.

| task | base GAP | base lp(FIX) | current_code/default | failed_attempts/avoid | failure_summary/avoid |
|---|---|---|---|---|---|
| int_to_roman | +0.487 | −0.631 | +0.109 | +0.095 | +0.142 |
| decode_string | +0.185 | −1.048 | +0.083 | +0.084 | +0.079 |
| merge_intervals | +0.093 | −1.111 | +0.061 | +0.090 | +0.106 |

Three findings settle the **encoding** question:

1. **Headroom is real and the adapter DOES help the fix** — base `lp(FIX)` is −0.6 to −1.1 (not
   saturated), and the adapter raises it by 0.5–0.85 nats (merge_intervals −1.111 → ≈−0.25). So this
   is **not pure recitation**; the substrate genuinely makes the correct fix more accessible.
2. **The margin collapse is caused by task-conditioning, not by the failure.** The key control is
   `task_only`, which has **no failure in context at all** — yet it collapses the gap just as much
   as `current_code` (int_to_roman base +0.487 → task_only +0.137 → current_code +0.109). The
   adapter boosts *all* plausible code, and the fix-vs-failure margin collapses toward a small
   constant regardless of whether the failure is present. So how we embed the failure **cannot** be
   the lever — the failure's presence is nearly irrelevant to the margin in the first place.
3. The owner's delineate+avoid templates move the gap by only ±0.01–0.04 vs `current_code/default`,
   inconsistently (`failure_summary` helps on 2/3, hurts on 1/3). The encoding choice is
   second-order.

## Conclusion (on the ENCODING question)

The substrate is a **recall/accessibility booster**: it raises the probability of *any* code
resembling its conditioning, and (control #2) the margin collapse happens even with no failure in
context — so it is driven by task-conditioning, not by reciting the failure. *Plausible mechanism
(hypothesis, untested):* a buggy near-miss and its fix are token-neighbors, so boosting one boosts
the other; the achievable margin is capped by how token-distinct the fix is from the bug.
Regardless of mechanism, **encoding/prompt changes do not move the margin** (the prompt can't change
the weights; relabeling the section barely does) — a property of the **recall distillation
objective**, not of the episode template.

→ **Answering the owner's question: no, this is not an encoding fix.** Producing an "avoid the
failure" preference is not something the current substrate can be prompted or re-templated into.

## What this does NOT settle — repair success

This probe measures the margin against **one specific** prior bug (a recitation proxy). It does
**not** measure whether repair *succeeds*: under sampling the fix competes with *all* wrong programs,
not just this near-miss, and the adapter already raises `lp(FIX)` substantially. So "collapsed
margin" ≠ "repair is broken." Before committing GPU to RL, run the cheap discriminator:
**actual repair pass@1 under `current_code` conditioning on these three hard tasks**, via the
existing multi-turn rune harness.
- Repairs well despite the collapsed margin → RL's premise (repair is broken) is false; don't run it.
- Fails to repair → inspect *what* it samples instead: if a different wrong program (not this bug),
  a contrastive-on-the-prior-bug term wouldn't help — that redesigns the RL objective.

→ **RL is _indicated, pending the repair-pass@1 check_** — not yet justified. The encoding finding
above is solid and final; the RL decision waits on measuring repair, not a logprob proxy.

## UPDATE — the repair-pass@1 check exposed an upstream codegen bug (confound)

Ran the real engine (c3, `prompt_mode=full`, 4 hard tasks). pass@1 = 1/4 (only
`merge_intervals`). Inspecting `session.jsonl` per step + the raw MLflow `output.txt`:

- The model emits code as `{"code": "..."}`. For `int_to_roman` and `decode_string` it
  **over-escaped the newlines** (`\\n` instead of `\n`). Both are valid JSON, so the
  xgrammar JSON-schema constraint cannot reject it; `json_repair.loads` decodes `\\n` →
  literal backslash-n, the code collapses to **one line**, and the sandbox throws a
  phantom `SyntaxError L1: unexpected character after line continuation character`.
- `diagnose` then faithfully reports a "syntax error / unclosed quote" (it *is* reading the
  real stderr, line 1), writes that hallucinated-looking guidance into `## Review Feedback`,
  the adapter is conditioned on a **misleading failure**, and repair re-emits `\\n` code —
  an infinite phantom loop. 2 of 4 hard "failures" are this, not logic.
- `compile()`/`ast` detects the broken decode cleanly; `code.encode().decode("unicode_escape")`
  recovers valid, compilable code for **both** over-escaped tasks (verified). `calculate`
  failed on a *genuine* `}` typo (L34, real newlines); `merge_intervals` was clean.

**Consequence:** the repair-pass@1 number and the RL framing above were **confounded** by this
serialization bug — for the over-escaped tasks the repair loop never saw a real logic error.

**Fix + de-confounded re-run (validated).** Code actions now emit freeform Python (a ```python
fence or bare code), de-fenced with markdown-it; the JSON `{"code": ...}` wrapping (and the
over-escape class) is gone. Probe (`tools/_codegen_format_probe.py`) on the broken tasks: freeform
3/3 compile with real newlines vs the JSON path stochastically collapsing to one line. Re-ran the
real engine (same c3, 4 hard tasks):

- **0** code/repair outputs with literal `\n` (was 2/4); **0** phantom line-1 diagnose errors.
- `int_to_roman` flipped **FAIL → PASS** — it was a pure over-escape phantom.
- **Repair pass@1: 1/4 → 2/4.** The two remaining failures (`calculate`, `decode_string`) are now
  *genuine* logic failures (real assertion/typo errors), not serialization artifacts — the repair
  loop working honestly.

**Net for the RL question.** With the confound removed, repair is *not* trivially broken (2/4 hard
tasks repair cleanly) and the failures that remain are real. The RL question — does a contrastive /
anti-recitation training signal improve repair beyond the de-confounded baseline — is now
well-posed and can be measured on the clean path. Still **indicated, not yet justified**: the next
step is to measure repair-pass@1 with vs without such training, not to assume it.

## Limits

- 2 toy + 3 hard scenarios, single seed, mean-token frozen-probe logprob (not sampled pass@1) — a
  mechanistic margin signal, not an end-to-end benchmark. Per-cell magnitude is noisy; the robust
  signals are the *encoding-invariance of the gap* and the *task_only control collapsing it without
  any failure present*.
- Both probe tools are REMOVE-BEFORE-MERGE.
