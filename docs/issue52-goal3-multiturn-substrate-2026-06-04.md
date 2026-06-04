# Issue #52 — GOAL 3 step 1: multi-turn adapter-as-memory-substrate (v1) — EXPERIMENT CARD

Status: **PRE-REGISTERED (this card written before the full batch)**. Branch
`issue52-bf16-body-contrastive`, PR #55. Date 2026-06-04.

## Question

Does the hypernetwork adapter act as a **memory substrate across the runner's repair turns** —
carrying the prior failing code + error into the next turn so the engine recovers attempt-1
failures — better than (a) no adapter (`scale=0`) and (b) the pre-warmed Sakana doc-to-lora adapter
(`warm`)? The hypernet is single-step by design; the **runner** is multistep. We test whether c3
(our Phase-1 recall checkpoint) is a better stepwise substrate than warm / scale=0.

Win conditions (v1): **c3 (and warm) > scale=0** ⇒ the adapter carries cross-turn context;
**c3 > warm** ⇒ our training improved the substrate. The prompt-FULL "ceiling" arm is **DEFERRED**
(needs an owner-reviewed `prompt_mode: minimal|full` switch; rendering `code.j2`/`code_repair.j2`
would confound adapter memory with prompt stuffing).

## Apparatus — runs on the RUNE RUNNER (standing rule: never a parallel runner)

`rune.bench.runner.run_benchmark` IS the harness: each task is driven through the full
`engine.ainvoke` loop (`code → diagnose → repair`* up to `MAX_RETRIES=4`, continuation sub-loop),
scored against held-out `test_code`, paired by `task_id`. Three arms = three configs; nothing
forked.

| arm | checkpoint | adapter_scaling |
|---|---|---|
| scale0 (floor) | c3 ckpt (irrelevant; B zeroed) | **0.0** |
| warm (Sakana) | `third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin` | 1.0 |
| c3 (ours) | `/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt` (sha256 `53e24af2…`) | 1.0 |

Base model `Qwen/Qwen3-4B-Instruct-2507` bf16, `offload_base=False` (RAM ~15GB; GPU 23GB). Single
source `config.yaml` / `load_rune_config()`.

## Pre-registration (handoff GOAL-3 a–g + advisor blockers) — DO NOT DEVIATE

**(a) Per-step re-encode — NATIVE, confirmed not reimplemented.** `step_node` calls
`render_training_format_trajectory(task, existing_code, feedback)` per action (graph.py:200) and
re-encodes on `accumulated_code` in the continuation sub-loop (graph.py:243). The adapter is
re-derived from the *current* state every turn.

**(b) scale=0 via `adapter_scaling=0` on the SAME path + forward-parity check.**
`scale_lora_b(sd, 0.0)` multiplies every `lora_B` by 0 (adapter.py:27) ⇒ effective LoRA delta
`(xAᵀ)B·s = 0` exactly. **Verified once** by a forward-parity check (base logits == scale=0-adapter
logits, bitwise / max-abs-diff = 0) — logged below in "Parity check".

**(c) Derived-hint channel — named, not oversold.** The minimal prompts are NOT task-free:
`prompt_code` / `prompt_code_repair` / `prompt_plan` inject `{{ project_label }}` =
`task[:200]` (state_to_ctx). MBPP specs are short ⇒ **the full task spec is in the prompt** on every
turn. Additionally repair injects `fix_guidance[:150]`, diagnose injects `error_summary[:300]`.
These are **identical across all three arms**. ⇒ The memory channel under test is NOT the task spec
(in the prompt) but the **prior failing code + error trajectory**, which appears ONLY in the adapter
conditioning (`render_training_format_trajectory(task, existing_code, feedback)`), never in the
repair prompt. This is the central correction to "prompt-minimal = task-only-in-adapter."

**(d) Score repair (+ continuation) turns only; tasks that FORCE the repair loop.**
Slice rule (pre-committed analysis cut): keep tasks where **scale=0 fails attempt-1** (the first
`code` step's output fails the held-out tests). Arm-independent (no adapter), so fair. Single batch
over a common candidate pool; slice applied post-hoc. `decompose`/`plan` are skipped (MBPP trips
`_is_simple_task` ⇒ single `_main` subtask ⇒ clean `code→diagnose→repair`), so repair is the
cross-turn surface, as pre-reg (d) wants.

**(e) Primary metric = SUCCESS-VS-TURN CURVE, not final@N (advisor Blocker 1).**
Final@N on a fails-attempt-1 slice confounds Goal-1 (adapter helping the first code attempt) with
the new repair-memory claim. So:
- Per-arm **fraction-correct-by-turn-k** over the common slice (shape is the proof: scale0 flatter,
  c3 climbs if repair memory works).
- Per-arm **recovery gap = final-success − attempt1-success** = the scalar cross-turn memory signal.
- **Scored POST-HOC against held-out `test_code`**, NOT the engine's internal `feedback.exit_code`
  (that only means "ran without error"; held-out scoring lives in runner.py:162). Per-step
  `generated_code` is persisted via `run_benchmark(sessions_dir=)` → `session.jsonl` `output` field
  (= `rec.output_text` = `extract_partial_code` for code/repair/integrate). Re-score each code/repair
  step's `output` with `strip_self_tests(output) + test_code` in the sandbox.
Secondary: per-step frozen-probe accessibility on the latest body tail vs warm (forgetting canary).

**(f) Recitation watch.** Cross-conditioned at the final step + "does step k output repeat step k−1
verbatim when the diagnosis/spec changed?" Computed post-hoc from `session.jsonl` step outputs.

**(g) Adapter-conditioning tokens vs prompt tokens per step — INSIDE the engine step.**
Deliberate rune change (owner review): a `ModelWrapper.count_tokens` accessor + two guarded
`mlflow.log_metric` calls in the graph.py step block logging `len(tokenizer(trajectory_text))` vs
`len(tokenizer(prompt_text))`. Thesis instrument: prompt ~flat while adapter trajectory grows.

### Blockers locked (advisor)
1. **Headline = curve + recovery gap** (not final@N) — see (e). Re-proves nothing from Goal-1.
2. **Slice DISJOINT from c3's train-40.** Candidate pool = c3-unseen tasks with on-disk test_code:
   heldout-24 ∪ (train160 ∖ train40) = **144 tasks**, all disjoint from the Phase-1 c3 train split.
   (warm/scale0 have zero MBPP exposure; train-on-test risk is c3-only.)
3. **`seed` set explicitly** in run_config for all arms (else `_seed_rng` never fires, runner.py:138;
   arms unpaired). Same seed + same task order ⇒ `seed+i` pairing across arms.

## Power
Pre-register a FIXED slice of ~40–60 tasks failing scale=0 attempt-1, frozen before the 3-arm batch.
Bootstrap CIs (10k, paired by task_id) on per-turn success deltas and the recovery gap. Same
discipline that caught the Goal-1 over-claims (n=24 was underpowered).

## Parity check (b) — PASSED
`tools/_goal3_multiturn_probe.py parity` (base + c3 hypernet, one forward over a
trajectory+code input; base reference via PEFT `disable_adapter()`):
- `max|scale0-adapter − base| = 0.0` (exact) ⇒ `adapter_scaling=0` is a clean no-adapter floor on
  the same engine path. `scale_lora_b(sd, 0.0)` → `lora_B=0` → delta `(xAᵀ)·0·s = 0`.
- `max|scale1-adapter − base| = 18.52` ⇒ the c3 adapter is non-trivially applied (sanity that
  zeroing is meaningful, not a no-op adapter).

## Results
_(to be filled.)_

## Limits (pre-stated)
- Single seed unless replicated; binary pass/fail; n per slice.
- Task spec is in the prompt (project_label) — this is a repair-memory test, not a task-recall test.
- v1 = minimal path only; prompt-FULL ceiling deferred.
- All probe/driver tooling is REMOVE-BEFORE-MERGE; the token-logging hook (g) is the one mergeable
  rune change, pending owner review.
