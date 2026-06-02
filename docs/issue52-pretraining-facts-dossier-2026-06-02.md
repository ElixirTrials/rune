# Issue #52 — Facts dossier ahead of a long-duration training run (2026-06-02)

> **ERRATUM (added 2026-06-02, post-experimentation):** any warm-start recall numbers here sourced from
> `gemma_demo` (notably goal +2.30 / file +1.76 / diff +1.01 / **tail +2.01**) are **GEMMA, not qwen**.
> True qwen warm-start: goal +2.235 / file +1.596 / diff +0.983; code-recall +2.597; **no qwen
> continuation/tail number exists on disk**. Also: the engine runs the 4B base in **bf16** (not 4-bit).
> See `docs/issue52-deliverable4-results-2026-06-02.md`.

Facts only. No interpretation. Every number is from a logged run (MLflow, eval logs, scratchpad,
or state-dict inspection). Sources noted inline. "m−mismatch" = matched-minus-mismatch logprob
margin; "m−zero" = matched-minus-base; "matched−swap" = matched-minus-feedback-swapped-negative.

## 0. System under test
- Base model: `Qwen/Qwen3-4B-Instruct-2507`, loaded 4-bit (nf4, double-quant, bf16 compute), frozen.
- Hypernetwork: ctx-to-lora HyperLoRA perceiver. Warm-start checkpoint `qwen_4b_d2l/checkpoint-20000`
  (842 MB hypernet state). Architecture: r=8, lora_alpha=45.2548, 36 layers,
  target_modules={`down_proj`}.
- Adapter apply contract (validated this branch, anchors 1–3): `effective_scaling = lora_alpha = 45.25`
  (NOT alpha/r), via `combine_lora` + head bias. Engine PEFT-hotswap is logit-identical to the
  functional contract (anchor 3: mean |Δ|=0.04, argmax match).
- Hardware: single 22.03 GiB GPU; ~15 GB CPU RAM. Base+hypernet fit on GPU (no offload).

## 1. Doc2LoRA positive control (PR #53 / experiment branch)
- Unmodified Sakana NIAH reproduction (gemma_demo/checkpoint-80000), rougeL.f1 = **1.0** (n=40).
- Scorecard calibration at known-good recall: needle m−mismatch = **+7.70 nats**, m−zero = +7.13.
- Sakana zero-shot on Rune episodes (m−mismatch): goal **+2.30**, file **+1.76**, diff **+1.01**.
- Continuation/tail m−zero = **+2.01** (Rune #49's tail was **−0.38**).
- Base-family control (qwen_4b_d2l) overall m−mismatch = +1.60 (≈ Gemma +1.69).
- Rune #49 own checkpoint (reference): goal m−mismatch = **+0.0005**, diff m−mismatch = **+0.075**.

## 2. Goal-3 specificity probe (this branch; frozen-10 MBPP, logprob, derangement matched-vs-mismatch)
- Task-in-prompt regime: full-span matched−mismatch = **−0.18** (0/10 positive); both matched and
  mismatch beat zero by ~**+1.2**.
- Task-hidden regime: matched−mismatch = **+1.17** (10/10 positive).
- Span split (task-hidden): signature span m−mismatch = **+3.84**; body span m−mismatch = **+0.14**.
- At contract scale, signature DiD vs the in-prompt function name = **−0.28**.

## 3. Lever A — signature/name enforcement (this branch)
- Adding an explicit exact-name instruction per task: pass rate 7/10 → **9/10**. MLflow run
  `895ccda7fc12473d889a002e2e42fabf`. `_is_simple_task` unchanged on all 10.
- The two flips: `sorted_matrix`→`sort_matrix`, `find_volume`→`find_Volume`.
- Remaining failure mbpp/57 = return-type contract (`'321'` ≠ `321`).
- Baseline per-task code was not saved (so 9-vs-7 is not a controlled per-task delta).

## 4. Lever B — avoid-utility in-context ceiling (this branch)
- On `external_codereview`: critique-in-prompt accepted-over-rejected DiD = mean **+0.17**,
  frac(>0) **0.53** (pre-registered bar 0.6; below it).
- Stratified (diagnostic): base-uncertain subset DiD = **+0.52**; accepted-edit-obvious subset = **−1.38**.
- Plain answer-CE finetune facet result: diff m−zero **+0.72** while diff m−mismatch (specificity)
  **−0.25**; goal specificity flat; only `file` gained specificity.
- Feedback-swap hard negative on diff: m−zero **+2.68**, m−mismatch(generic) **+1.01**,
  m−mismatch(FEEDBACK-SWAP) **+0.174**.

## 5. Mining engine trajectories for failure-bearing episodes (this branch)
- Engine mid-loop runs `strip_self_tests(code)` with no functional test → diagnose→repair fires on
  syntax/import/load-time errors only.
- Instrumented mining run (visible example assert appended as oracle; `_mine_semantic_sessions.py`,
  10 frozen tasks): 7/10 fail→repair→pass chains; 7 "valid". Failure classes across failed steps =
  SyntaxError ×7 + NameError ×5, **AssertionError ×0**.
- Temp-sample avoid-pair probe, frozen-10 (`_temp_sample_avoid_probe.py`, N=16 @ temp0.9,
  adapter@contract): yield 3/10 task pairs; good-rejects = mbpp/16 14, mbpp/20 1, mbpp/56 2.
  In-context ceiling on the 3 pairs: per-task DiD +0.030/+0.207/+0.017, mean +0.0847, frac 1.00.
- Temp-sample avoid-pair probe, 35 non-frozen MBPP validation tasks: yield **5/35** pairs
  (mbpp/67, 84, 100, 103, 434). In-context ceiling n=5: per-pair DiD +0.014/−0.038/+0.199/−0.032/+0.637,
  mean **+0.156**, frac **0.60**. Seed dumped `/tmp/avoid_pairs_val.json`.

## 6. Corpus geometry (external_codereview)
- Train corpus: `/tmp/rune-corpus/external_codereview.train.jsonl`.
- Val corpus: `external_codereview.val.clean.jsonl`, 323 records.
- Episode token lengths (tokenizer Qwen3-4B, first 300 train records), context+answer:
  p50 = 774, p90 = 3268, max = 24963. Context alone: p50 427, p90 1627, max 12514.
  Answer alone: p50 348, p90 1600, max 12449.

## 7. Training infrastructure (this branch)
- Entry: `tools/_distill_entry.py --config configs/issue52_recipe_mvc_4b.yaml [--max-steps N]`,
  run under `tools/run_guarded.sh` (RAM watchdog, kill threshold 13.5 GB).
- Config (`issue52_recipe_mvc_4b.yaml`): warm-start from qwen_4b_d2l; lr 2e-5; grad_accum_steps 8;
  max_steps 300; early_stop_warmup 100; contrastive true, contrastive_weight 1.0, contrastive_margin 1.0;
  val_corpus_path set, val_sample 40, val_steps 100, save_steps 100; load_in_4bit true; use_8bit_optim true.
- Objective: KL (teacher diff agreement) + contrastive hinge `clamp(margin − (lp_m − lp_n_det))` on
  **edit-local** tokens, hard negative = `make_hard_negative(context, other_feedback)` (same Task +
  Current Code, feedback text swapped). Matched piece `−lp_m[active]` and neg piece `+lp_n[active]`
  backward sequentially (memory-bounded; designed for seq=768).
- `effective_scaling = effective_scaling(hypernet) = lora_alpha = 45.25` confirmed (L193/609).
- `val_diff_agreement` selects `checkpoint_best.pt` (matched-vs-base = discipline confound, per advisor).

## 8. Training-run events (this branch, 2026-06-01)
### 8a. Warm-start baseline (before any training), `_feedback_swap_eval.py`, n=60, val, 4-bit
- matched−SWAP = **+0.0185**, frac(>0) **0.48**. matched−zero = **+0.0870**.

### 8b. Smoke attempt 1 — OOM
- max_seq_length 2048 → CUDA OOM at `hypernet_distill.py:343` (`log_softmax(student_logits[:-1].float())`),
  21.94/22.03 GiB in use, +352 MiB requested. Code comment at the contrastive block states the
  memory-bounding "keeps seq=768, peak ~ single-path." Fix applied: max_seq_length → **768**;
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### 8c. Smoke attempt 2 (seq 768) — broken adapter (DOCUMENTED FAILED RUN)
- MLflow `43cf8526ee`. Metrics: margin 3.44→2.07→1.93→1.28→0.96→0.99; scaler_B watched = **1.0** all 6
  points; scaler_A 1.0575; bias_A/bias_B grad_l2 0.06–0.15; kl 16→12; loss ~24; gpu_peak 10.9–12.8 GB;
  diff_agreement (final) 0.020.
- Held-out eval of its checkpoint (n=60): matched−SWAP = **+0.0019** (frac 0.38);
  matched−zero = **−8.81** (uniform ~−7 to −10 across all episodes).
- State-dict diff vs warm-start: all params identical EXCEPT `scaler_B.down_proj`:
  warm-start mean|·| **0.057** / std **0.070** (structured); this run **1.0** / std **0.0** (uniform).
- Cause located: `hypernet_distill.py:186` called `reinit_scaler_b_nonzero(hypernet, 1.0)`
  unconditionally after warm-start load. Function docstring: "Call ONLY when (re)training, never when
  loading a trained checkpoint (its learned scaler_B must be preserved)." Checkpoint preserved at
  `/tmp/smoke_broken_scalerB1.pt`.

### 8d. Code fix
- Added `scaler_b_is_collapsed(hypernet, eps=1e-4)` (all |scaler_B| < eps = zero-init basin).
- Warm-start init now: reinit only if collapsed, else preserve + log mean|·|.
- Regression tests added (`tests/unit/test_scaler_b_init.py`, 4 pass).

### 8e. Smoke attempt 3 (seq 768, fix applied) — FINAL FIXED SMOKE
- MLflow `cbe9da363c`. Metrics: margin 1.007/0.936/1.041/1.025/1.03/0.968 (flat ~1.0); scaler_B watched
  = 0.00226 all points (signed mean of preserved structured tensor); kl 2.76→1.99; loss 2.45–6.80;
  diff_agreement (final) **0.364**; checkpoint artifact logged.
- Saved checkpoint `scaler_B.down_proj`: mean|·| **0.05703** / std **0.06992** (== warm-start).
- Held-out eval (n=60, val): matched−SWAP = **+0.0687**, frac(>0) **0.65**; matched−zero = **+0.5021**.

## 9. Baseline→fixed-smoke deltas (held-out, n=60, same eval)
| metric | warm-start baseline | fixed 60-step smoke |
|---|---|---|
| matched−SWAP | +0.0185 | **+0.0687** |
| frac(matched−SWAP > 0) | 0.48 | **0.65** |
| matched−zero | +0.087 | +0.502 |
- Steps: 60. grad_accum 8. lr 2e-5. Per-step train margin proxy: flat ~1.0.

## 10. Prior root-cause on record (issue #49, from PR #53)
- Stated conclusion: "#49 failed because its training recipe did not train queryable episode memory —
  it optimized an edit/full-revision emission objective that rewards generic, code-driven behavior."
- Ruled out as #49 causes: probe-blindness, capacity, ill-posed facts, base-model family.
- "Training recipe" kept as a bucket (objective, data format, query supervision, scale, batch, init
  still entangled).

---

# Research synthesis (HuggingFace papers, WebSearch, Consensus) — 2026-06-02

Literature relevant to the facts above. Citations are external; mapping-to-our-facts is noted but
the mapping itself is deferred to the advisor/hypothesis section.

## A. The exact method family (Sakana)
- **Doc-to-LoRA: Learning to Instantly Internalize Contexts** (arXiv 2602.15902; HF papers/2602.15902).
  Meta-learned hypernetwork generates LoRA in one forward pass that compresses a context into
  parameters; chunking concatenates per-chunk adapters along the rank dim for long inputs.
  Reported: near-perfect zero-shot **NIAH** at >4× native context, <50 MB vs >12 GB KV cache for 128K,
  internalization <1 s. Validated objective = retrieval/recall (NIAH), context compression.
- **Text-to-LoRA: Instant Transformer Adaption** (arXiv 2506.06105; ICML 2025). Hypernetwork generates
  task adapters from a natural-language *task description*, trained by distilling a library of
  pre-existing per-task LoRAs; zero-shot task adaptation.
- (repo: github.com/SakanaAI/text-to-lora, SakanaAI/doc-to-lora.)

## B. Parametric memory limits of LoRA
- **How LoRA Remembers? A Parametric Memory Law for LLM Finetuning** (arXiv 2605.30260). Loss reduction
  is a **power law in LoRA rank and sequence length**. Token-level **deterministic phase transition**:
  prediction prob p>0.5 is sufficient for verbatim recall under greedy decoding; unresolved
  "bottleneck tokens" can trigger decoding collapse. **MemFT**: reallocate gradient budget from
  already-mastered tokens to below-threshold tokens to raise capacity under constrained rank.
- **Understanding LoRA as Knowledge Memory: An Empirical Analysis** (arXiv 2603.01097).
- **LoRA Learns Less and Forgets Less** (arXiv 2405.09673). Low-rank adapters have limited capacity;
  learn less than full FT but forget less.

## C. Injecting facts into weights vs context
- **Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs** (Ovadia 2023, 262 cit). RAG
  consistently > unsupervised fine-tuning for knowledge injection; LLMs **struggle to learn new facts
  via unsupervised FT**; exposing the model to **many variations/paraphrases of the same fact**
  alleviates this.
- **On the generalization of LMs from in-context learning and finetuning** (Lampinen 2025, 45 cit). FT
  generalizes narrowly (reversal curse); ICL generalizes more flexibly in data-matched settings;
  **adding in-context reasoning traces to the finetuning data improves FT generalization**.
- **Can We Edit Factual Knowledge by In-Context Learning?** (Zheng 2023, 325 cit). In-context editing
  competitive with gradient methods, fewer side effects.
- **Few-shot Learning with Retrieval Augmented LMs / Atlas** (Izacard 2022, 1198 cit). Retrieval reaches
  high QA accuracy with very few examples and far fewer params; index is updatable.

## D. Contrastive learning failure modes
- **Boosting Contrastive SSL with False Negative Cancellation** (Huynh 2020, 215 cit). False negatives
  (negatives that are actually semantically equivalent to the anchor) **discard semantic information
  and slow convergence**; detecting + eliminating/attracting them improves learning.
- **Contrastive sentence representation with adaptive false negative cancellation** (Xu 2023);
  **Seeking False Hard Negatives for Graph CL** (Liu 2024). Same theme: negative quality gates the
  representation; bad negatives cap the achievable margin.

## E. Shortcut / surface-feature learning
- **Exploring and Mitigating Shortcut Learning for Generative LLMs** (Sun 2024). LLMs exploit spurious
  task↔feature↔label correlations; proposes forgetting spurious correlations + learning from in-context.
- **Less Learn Shortcut** (Du 2022). Words highly co-occurring with a label ("biased words") are
  learned first and dominate predictions; down-weighting biased examples mitigates over-reliance.
- **LLMs Can be Lazy Learners** (Tang 2023, 95 cit). LLMs exploit shortcuts in prompts; larger models
  more so.

## F. Alternatives / capacity probes
- **PERK: Long-Context Reasoning as Parameter-Efficient Test-Time Learning** (arXiv 2507.06415).
  Encode context into a low-rank adapter via **test-time gradient updates** (nested meta-learning);
  outperforms prompt-based baselines on long-context reasoning. (An oracle/per-episode-LoRA capacity
  probe analog.)
- **In-Context Meta LoRA Generation** (arXiv 2501.17635, CVAE); **Compressed Context Memory**
  (arXiv 2312.03414, conditional LoRA as memory).

---

# Interpretation, hypotheses, and gating tests (advisor-informed) — 2026-06-02

> The facts and research above are stated without interpretation. This section is the interpretive
> layer and is explicitly opinionated.

## Lead conclusion: the fixed smoke is the #49 pattern in miniature, not a green light
Of the adapter's **+0.50** nat edit-local lift (matched−zero), only **+0.0687 (~14%)** is attributable
to the *specific* feedback (matched−swap); the other ~86% is **generic boosting that fires with the
*wrong* feedback** — the exact #49 signature (m−zero rises via generic boosting while specificity stays
flat) this work was built to detect. "matched−swap moved above baseline" must **not** be read as
"feedback-binding works."

## Rigor gap to close first (T0)
+0.0687 at n=60 (per-episode −0.5…+1.2) is ~1.4 SE from zero; the aggregate move +0.0185→+0.0687 is
likely within noise. Only frac 0.48→0.65 (39/60, ~2.3σ) is suggestive. **Compute the paired
per-episode delta (trained − baseline matched−swap on the same 60 val episodes) and a paired/sign test
before any go/no-go.** Both checkpoints exist.

## Convergent evidence: feedback→edit signal in this corpus is weak
Three independent measurements agree: oracle **in-prompt** ceiling **+0.17** (frac 0.53, below the 0.6
bar); in-**train** contrastive margin **flat ~1.0**; held-out matched−swap **+0.0687**. The Lever-B
stratification explains it: edit-obvious → critique can't move it (−1.38); base-uncertain → feedback
determines the edit (+0.52). Research names the mechanisms: **false negatives** (Huynh — a swapped
feedback that doesn't change the correct edit caps the margin and slows convergence), **single-exposure
FT binding is weak, needs many paraphrases** (Ovadia), **shortcut learning** (Du — the signature
co-occurs with the answer, so the adapter binds the *name*: our signature +3.84 vs body +0.14), and the
**Parametric Memory Law** (loss is power-law in *rank* — r=8 holds low-entropy signatures, not
high-entropy bodies).

## Hypotheses and discriminators
| H | Claim | Test | Decision |
|---|---|---|---|
| **H1** | Corpus feedback→edit mutual information is intrinsically low (data-limited) | T1: stratify to the directive (+0.52) subset; re-measure oracle ceiling + short-train matched−swap | still flat → external_codereview is the wall; long run does **not** run on it |
| **H2** | False negatives cap the gap | label pairs by whether the swapped feedback changes the gold edit; train/eval on true-negatives only | signal rises on filtered → confirmed; long run trains on filtered data |
| **H3** | r=8 cannot hold body/algorithm recall (capacity) | T2: fit an oracle per-episode LoRA on a few body/feedback facts at r=8 vs higher rank | oracle fails at r=8 → raise rank before the long run |
| **H4** | Single-exposure binding is too sparse | augment K paraphrased feedbacks per episode; short-train | signal rises → augmentation is a long-run lever |

## Gating tests before any long-duration run (each ≈ one smoke of GPU)
1. **T0 — paired significance** of the smoke matched−swap delta (cheap; do first).
2. **T1 — directive-feedback stratification (DECISIVE).** Filter train+eval to the base-uncertain /
   feedback-determines-edit subset; short-train; re-measure matched−swap. Jump → false negatives were
   diluting the signal (long run trains on filtered data). Flat → corpus is the wall (do **not** launch
   on external_codereview).
3. **T2 — oracle per-episode LoRA capacity probe** at r=8 vs higher rank.

## Go/no-go thresholds (anchored to calibration, written before the run)
Real recall ≈ **+7.7**, hidden-task specificity ≈ **+1.17**. A long-run "success" of matched−swap ≈ +0.1
is ~1–2% of real binding. Set the success bar in these units, **plus** the retention gate (NIAH/QA
recall preserved) and the generation-stability gate (xgrammar pass@1 not degraded). The unfiltered
external_codereview long run is **not yet justified**; T1 is the single test that most changes that.
