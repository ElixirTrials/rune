# Why the runner generates poorly — root-cause diagnosis (2026-06-04)

After fixing the three engine miswirings (fence / diagnose-livelock / mlflow — see
`issue52-goal3-pipeline-traceThrough-2026-06-04.md`), the engine *works*: on a 5-task smoke the
now-functional repair loop drives **scale0 5/5 and warm 5/5 final pass@1**. But **attempt-1 quality is
poor**, and the repair loop masks it at the cost of extra turns + wall-clock. Owner asked to diagnose
the prompts / adapter templates / langgraph. Evidence below is from the verify + verify2 smokes
(15 task-runs, real prompts/outputs/sandbox stderr).

## Attempt-1 outcome (15 runs): 8 PASS / 5 SyntaxError / 2 NameError

Three distinct failure modes, by cause:

### 1. Prompt-induced OVER-GENERATION → truncation → SyntaxError  (the dominant attempt-1 killer)
`templates/prompt_code.j2` ends with **"Write tests FIRST, then implement to pass them."** On a
single-function MBPP task this backfires: the instruct model writes a whole test module / multiple
functions / walls of comments, blows past `max_tokens=2048`, the continuation sub-loop extends it
further, and it still truncates mid-statement → `SyntaxError`.
- mbpp/113 (scale0): output began `# test_check_integer.py\nimport pytest\n\ndef check_integer(s):`
  and grew to **13,835 chars** → `SyntaxError: unterminated string literal (line 210)`.
- 4/15 outputs defined **>1 function** (unasked extras); 4/15 emitted module-level self-test asserts
  (which `strip_self_tests` discards anyway = pure wasted tokens + truncation risk).
- Median output length PASS=394 chars vs FAIL=908 (max FAIL 13,835). Longer ⇒ failure.

### 2. Adapter OVER-CONDITIONING → degenerate generation  (the "adapter templates" issue)
The adapter arms apply the checkpoint at **effective scaling 45.25** (the un-divided `lora_alpha`
contract — correct for logprob *probing*, but very strong for free *generation*). At that strength,
conditioned on the training-format trajectory, generation intermittently **collapses to garbage**:
- warm, mbpp/108 & mbpp/115: attempt-1 `code` output = literally **`"error"`** (5 chars) →
  `NameError`. Confirmed a real model output (no such pipeline fallback). Repair recovered it.
- This matches the project's prior note "spec-divergence at adapter scaling ≥0.49": a high-magnitude
  adapter can push the base into a degenerate/recitation mode that overrides the task.
- Hypothesis (thesis-relevant): **c3 (trained on this exact trajectory format) should degenerate less
  than warm (doc-to-lora, never saw the format).** c3/mbpp106 attempt-1 was concise + correct (69
  chars) vs scale0 362 / warm 394 — suggestive but n=1; c3 arm still completing.

### 3. Missing `entry_point` in the prompt → NameError
`prompt_code.j2` never states the required function name; the model infers it from the docstring
example and sometimes picks a different name (e.g. defines `tuple_to_int` while the tests call
`string_to_tuple_list`) → `NameError`. `BenchTask.entry_point` exists but is **not threaded into the
engine** (`make_initial_state(task.description, budget)` drops it; `state_to_ctx` has no
`entry_point`).

### Secondary: `project_label` truncated at 200 chars
`state_to_ctx` sets `project_label = task[:_PROJECT_LABEL_CAP=200]`. For tasks with a long docstring
example the expected-output assert is cut off mid-line (mbpp/108: `...19, 20, ` then truncated), so
the model loses part of the spec.

## What is NOT the problem
- **LangGraph wiring:** fine. Post-fix, `code→diagnose→repair` advances and terminates correctly; the
  repair loop recovers attempt-1 failures (scale0/warm 5/5 final). The earlier "livelock" was the
  diagnose-key bug, now fixed.
- **xgrammar / pydantic / structured output:** working (raw is valid JSON).
- **Adapter conditioning FORMAT** (`render_training_format_trajectory`): well-formed
  (`## Task / ## Current Code / ## Review Feedback`). The issue is conditioning *strength*, not format.

## Recommendations (owner-review — these change runner behavior + the experiment)
1. **Prompt (`prompt_code.j2` / `prompt_code_repair.j2`): stop inducing over-generation.** Drop "Write
   tests FIRST"; instruct "implement **only** the function `{{ entry_point }}`; no tests, no extra
   functions, no prose." Expected: far fewer truncations/SyntaxErrors on attempt-1.
2. **Thread `entry_point` into the engine** (BenchTask → make_initial_state → state → ctx) and name it
   in the prompt. Kills the NameError class.
3. **Raise/remove `_PROJECT_LABEL_CAP`** (or pass the full short MBPP spec) so the expected-output
   example isn't truncated.
4. **Adapter generation scaling:** investigate a *generation-time* scaling lower than the 45.25
   probe-calibrated value (the degeneration suggests 45.25 is too strong for free decoding), OR treat
   "c3 degenerates less than warm" as a thesis result. Needs the c3 vs warm degeneration-rate
   comparison (verify2 c3 pending). Do NOT silently change the established scaling contract.

These are attempt-1 quality levers; the multi-turn repair loop already recovers final pass@1, so for
the GOAL-3 experiment the recovery-gap metric still works — but (1)–(3) would cut wall-clock and
de-confound attempt-1, and (4) is itself a finding about the adapter as a generation substrate.
