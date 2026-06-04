# Issue #52 — Goal-1 (the decisive experiment): adapter-as-memory recall, scale=0 control

**Status: COMPLETE (2026-06-04 UTC).** One CI-clean positive result (k=1). The multi-task "capacity"
sweep (k>1) tested an **off-design** scenario — see the correction below.
Companion: `issue52-phase1-results-2026-06-04.md` (the c3 checkpoint this builds on).

> **CORRECTION (project owner, 2026-06-04) — the k>1 sweep mismodels the thesis.** The hypernetwork is
> **always single-step**: it is never trained on (and is not meant to be trained on) "multi-step" or
> multi-task conditioning. The **runner** is what is multistep — across its iterations on a *single
> evolving task* the hypernet re-encodes the current trajectory each step, acting as the **substrate
> that carries stepwise context** (long-running repair, code-continuation, development) so that
> history need not live in the prompt. So "adapter memory growing" means *one coherent run's evolving
> state carried across steps*, **not** k independent tasks packed into one adapter. This doc's k>1
> sweep (concatenated disjoint tasks, recall-by-name) therefore tests something the system does not do
> and was never designed to do. **What survives is the k=1 result** (single-conditioning recall beats
> scale=0, in-distribution). The earlier "future work: train the hypernet on multi-task conditioning"
> recommendation is **withdrawn** — see the corrected Future-work section. The genuinely decisive
> multistep test is *iterative single-task repair/continuation with context carried in the adapter
> substrate vs scale=0* — which uses the runner, not task-packing.

## What "the decisive experiment" actually is

The handoff framed goal 1 as a **multi-step engine** eval (engine runs N steps, state evicted into
the adapter, prompt held FIXED, adapter memory growing) vs a **scale=0** (no-adapter) control —
"a result without the scale=0 control is not a result."

> **RETRACTED PREMISE (2026-06-04).** This section originally claimed "the LangGraph engine is the
> wrong vehicle because it puts prior state in the prompt, so scale=0 is dirty." **That is false** —
> verified against the Jinja2 templates. The engine's *generation* prompt (`Action.prompt_template`,
> the `prompt_*` family) is deliberately **minimal** and defers to the adapter: `prompt_code` says
> "Follow the architecture plan **in your context**", `prompt_plan` "based on the spec **in your
> context**", `prompt_integrate` "Combine all implementations **from your context**". The rich
> history fields (`existing_code`, `repair_history`, `code_outputs`, `accumulated_code`) live in the
> `trajectory_template` slot — which the engine never even renders — while the live adapter
> conditioning is `render_training_format_trajectory(task, current_code, feedback)`. So **the runner is
> already prompt-minimal / adapter-as-memory by design, and scale=0 on it is essentially clean** (minor
> caveat: `fix_guidance[:150]` in repair and `error_summary[:300]` in diagnose are short derived hints,
> not raw trajectory). The standing rule (project owner): **always use the rune runner; never build a
> parallel one** — change rune itself if needed. The probe below was therefore built on a wrong
> premise; its **k=1 result still stands** (in-distribution single-step recall), but the correct
> vehicle for the multistep test is the rune runner, swapping the adapter — see Future work.

The probe used here (built on the now-retracted premise) lives on the **`_pass1_probe` ABSENT harness**
(spec absent from the prompt; `--scale0` is a native no-adapter floor) extended to **multi-task
accumulation** (`tools/_recall_capacity_probe.py`):

- Partition the 24 held-out MBPP-recall tasks into disjoint blocks of `k` tasks.
- Condition **one** adapter on the concatenation of the `k` task descriptions ("study" → adapter
  memory, NOT the prompt).
- Query each of the `k` tasks **name-cued**, spec absent: *"Write the Python function named
  `{entry_point}` that you have just studied."* Score pass@1 against the real MBPP 3-test suite.
- The query prompt is name-only → **flat token length** regardless of `k`, while the adapter
  conditioning grows. Flat-prompt + growing-memory is the adapter-as-memory thesis surface.

## Results (pass@1 / 24, held-out, name-cued, spec absent)

| k | scale=0 (floor) | warm-start | c3 (trained) |
|---|-----------------|------------|--------------|
| 1 | 5 (0.21)        | 9 (0.38)   | **12 (0.50)** |
| 2 | 5 (0.21)        | 7 (0.29)   | 10 (0.42)    |
| 4 | 5 (0.21)        | 6 (0.25)   | 9 (0.38)     |
| 8 | 5 (0.21)        | 8 (0.33)   | 6 (0.25)     |

Paired bootstrap CIs (10k resamples, paired by `task_id` at fixed `k`):

| pair | k=1 | k=2 | k=4 | k=8 |
|------|-----|-----|-----|-----|
| **c3 − scale0** | **+0.292 [+0.083,+0.500]** ✓ | +0.208 [0.000,+0.417] | +0.167 [0.000,+0.375] | +0.042 [−0.125,+0.250] |
| c3 − warm | +0.125 [−0.042,+0.292] | +0.125 [−0.042,+0.292] | +0.125 [0.000,+0.250] | −0.083 [−0.250,+0.083] |
| warm − scale0 | +0.167 [0.000,+0.375] | +0.083 [0.000,+0.208] | +0.042 [−0.083,+0.208] | +0.125 [0.000,+0.250] |

Instrument controls: cross-task **interference = 0** at every k/arm (no adapter ever emitted a
*different* studied task's name). Stronger check (`emitted_def != entry_point` over all 96 rows/arm):
the adapter arms (warm, c3) emit the queried name **0/96 wrong-or-missing** — failures are wrong-
*body*, not wrong-function; scale=0 had 12/96 with **no valid `def`** (name-only, no adapter). So the
name cue disambiguates perfectly *for the adapter arms*. prompt_tokens **27–34 (flat)**; study_tokens
**36–498 (grows with k)**. scale=0 is **flat at 5/24 across all k** — exactly
as it must be, since the study material never reaches the model (k-invariant floor).

## VERDICT

1. **The adapter-as-memory channel is real at k=1 (single item).** c3 beats the no-adapter floor by
   **+0.292, CI [+0.083, +0.500] excludes 0**. This is in-distribution conditioning (at k=1 the
   "concatenation" is just the single description the hypernet was trained on) and is the result the
   scale=0 control was for. The handoff's decisive control is satisfied **and passed at k=1.**

2. **The k>1 sweep is OFF-DESIGN — it does not measure a real system property** (see Correction).
   c3's per-k point estimates decay (12→10→9→6), but packing k *independent* tasks into one adapter
   is not what the runner does: the hypernet is single-step and the multistep value is *within-run*
   context (repair/continuation), not cross-task storage. The decay conflates "memory capacity" with
   feeding the hypernet a format it never sees. The only fair claim is the narrow one — **naive
   concatenation conditioning degrades recall** — and it speaks to neither the architecture's limit
   nor the adapter-as-memory thesis. The real multistep question is tested by iterative single-task
   development, not task-packing.

3. **No evidence Phase-1 traded capacity for peak.** c3 vs warm-start is statistically
   indistinguishable at **every** k (all CIs span 0; the c3=6 vs warm=8 gap at k=8 is 2/24, inside
   the noise, and warm's curve isn't even monotonic). Likewise "c3 beats warm at k=1 (12 vs 9)" is
   **not** significant. Only "c3 (and adapter) > floor" holds.

4. **The handoff's "multi-step" hypothesis is UNTESTED here (not refuted).** I previously read the
   task-packing decay as evidence against "uplift is largest in multi-step." That was a category
   error: the handoff's "multi-step" means the **runner's iterative generation** on one task, which
   this probe never exercised. Task-packing is not multi-step. So this run says nothing about the
   multi-step thesis — it only confirms single-step (k=1) recall is real. The multi-step question
   remains open and needs the runner-based test below.

## Limits (read before citing)

- **n=24, binary pass@1, noisy.** k=8 has only 3 blocks. Treat point estimates descriptively; lean
  on the CIs.
- **The k>1 sweep is off-DESIGN** (concatenated independent tasks) — the hypernet is single-step and
  the runner carries within-run context, not cross-task storage. Treat k>1 as a null instrument, not
  a capacity measurement.
- **The name cue leaks semantic signal** (scale=0 = 5/24 from names alone). The c3−scale0 deltas are
  *above* that floor, so the adapter contribution is real, but absolute recall numbers include the
  name's contribution.

## Reproduce

```
# build: tools/_recall_capacity_probe.py ; driver: tools/_run_capacity_arms.sh
# analyse: tools/_capacity_analysis.py  (reads /tmp/cap/{scale0,warm,c3}.jsonl)
uv run python tools/_recall_capacity_probe.py --ckpt /tmp/phase1/ckpt/c3_t07_lp2_lg1.pt \
    --corpus /workspaces/rune-gpu/benchmarks/mbpp_recall_heldout.jsonl --k-values 1,2,4,8 \
    --out /tmp/cap/c3.jsonl              # --scale0 for the floor; --ckpt <warm.bin> for warm
```

- corpus `mbpp_recall_heldout.jsonl` sha256 `cae274bf1aed…` (n=24); durable source S3
  `s3://elixirtrials-949678234935-us-east-1-artifacts/training-data/github-pairs/` (issue-52 derived).
- c3 checkpoint sha256 `53e24af243a3…`; warm = `doc-to-lora/.../checkpoint-20000/pytorch_model.bin`.
- All three tools are REMOVE-BEFORE-MERGE issue-52 scaffolding.

## Future work (corrected — the multistep test uses the runner, single-step hypernet)

The hypernet stays single-step; do **not** train it on multi-task/packed conditioning. The bet is that
it is an effective **substrate for stepwise context across the runner's iterations on one task**. So:

- **The decisive multistep experiment — on the rune runner, NOT a parallel harness** (standing rule:
  always use rune; change rune if needed). The runner is already prompt-minimal (verified above), so
  the test is: drive `rune run` on tasks that need iteration (repair / code-continuation) under three
  adapter conditions — **scale=0** (`adapter_scaling=0`), **warm Sakana**, **c3 (ours)** — and compare
  final integrate-success / pass@1, plus the success-vs-turn curve. c3/warm > scale=0 ⇒ the adapter
  carries cross-turn context; c3 > warm ⇒ our training improved the substrate. This is the handoff's
  original "state evicted into the adapter, prompt fixed" — which the runner *already does*. Add any
  missing metric/canary logging inside the engine step, deliberately; do not fork the runner.
- **Optimize the hypernet for those iterative generations** (its actual job), not for holding a library
  of independent tasks. The k=1 recall result here is necessary (the channel is open) but not
  sufficient (it does not show the substrate carries *evolving* context across steps).
