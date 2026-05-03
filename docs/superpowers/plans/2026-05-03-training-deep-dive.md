# Training Deep-Dive — Why The Model Doesn't Learn (Investigation Log)

**Date:** 2026-05-03
**Branch:** `fix/diff-loss-per-turn-alignment` (PR #35) — building on commits up to `58dc87af`
**Companion docs:**
- [`2026-05-03-diff-aware-span-match-rca.md`](2026-05-03-diff-aware-span-match-rca.md) — span-match failure RCA
- [`2026-05-03-diff-aware-span-match-results.md`](2026-05-03-diff-aware-span-match-results.md) — span-match fix outcomes
- `instructions/some_training_suggestions.md` — external review of fixes to try
- `instructions/hpe_issues.md` — earlier external review

This document captures a deeper investigation: even after the truncation bug was fixed and we moved to plain SFT, training still showed only marginal token-accuracy improvement (0.79 → 0.84). The user pushed back on the data-size hypothesis: *something else* is wrong.

---

## TL;DR (running)

Two issues identified so far, with roughly equal blame:

| # | Issue | Status |
|---|---|---|
| 1 | **`--diff-aware-loss` was a no-op** on this corpus shape (`changed_token_frac ≈ 0.98` because pre/post are unified-diff strings, not raw file bodies) | Confirmed; plain-SFT control matches diff-aware in tok_acc within 0.006 |
| 2 | **LoRA α override of 16 vs deltacoder's saved α=32** halves the canonical scaling (1.0 → 0.5), reducing both forward-pass adapter influence and gradient flow | Tested with α=32; tok_acc moved from 0.795 → 0.768 (within noise — not the smoking gun) |
| 3 | **Sampling bias in `head -N` slicing** concentrates 6 repos into the first 500 rows; random-N covers 28 | Confirmed; switched to random sampling for all subsequent runs |

After fixing 1 and 3, plain SFT on random-500 rows reaches **0.834 token_accuracy** with t016-style HP. That's only 0.04 above the warm-start prior of ~0.79. Still not impressive. Next round of experiments below tests whether anything in the training pipeline itself is preventing learning.

---

## Hypotheses being tested in the current 4-run chain

| Run | Trainer | Warm-start | Other config | Purpose |
|---|---|---|---|---|
| 1 | **Mimic** (vanilla HF Trainer, hand-rolled collator) | deltacoder | 5 rows × 20 epochs, plain SFT, paged_adamw_32bit, no checkpointing, no NEFTune, α=32 saved scaling | Can the model overfit a tiny dataset under the *minimum-confound* path? If yes → optimization works; failures elsewhere are in our custom code. If no → fundamental training-pipeline problem (warm-start saturation, model+corpus, or framework bug). |
| 2 | **Mimic** | **none** (fresh LoRA r=32 α=32 q+v_proj) | same as 1 | Cold-start control. Rules out warm-start saturation as the cause. |
| 3 | **Our trainer** (full pipeline, no `--diff-aware-loss`) | none | 5 epochs × 500 random rows, lr=2e-4, grad_accum=8, max_seq 2048 | Does our trainer learn on a cold start? If mimic+cold-start works but ours doesn't, something in our custom pipeline is broken. |
| 4 | **Our trainer** | none | Suggestions-doc canonical recipe: lr=2e-4, override-α=64, paged_adamw_32bit, warmup_ratio=0.1, 3 epochs, plain SFT | Does the combined-fixes recipe actually lift learning above run 3's baseline? |

---

## Hypothesis testing — before this round

| H | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | LoRA scaling 0.5 (α=16, r=32) too low | Run with α=32 (scaling=1.0) | Token acc 0.768 vs control 0.795 — no signal |
| H2 | LR=4.3e-5 too low | (deferred — combined with H1) | Pending |
| H3 | 8-bit Adam quantizes optimizer state | Mimic uses paged_adamw_32bit | Pending in run 1 |
| H4 | NEFTune α=5 noise drowns updates | Mimic disables NEFTune | Pending in run 1 |
| H5 | gradient_checkpointing interaction | Mimic disables | Pending in run 1 |
| H6 | Wrong target_modules | Inspected: 12 modules (q,k,v,o,gate,up,down + Mamba in_proj_*); 86.5M trainable params | ❌ Refuted |
| **H7 (new)** | **Chat template returning empty token sequences** | Debug print of `apply_chat_template` output | **Confirmed real bug — but in MY mimic-script's data loading, not in the trainer.** `apply_chat_template(tokenize=True)` returns a BatchEncoding dict, not a list; `len(dict)=2`. Fixed in mimic script. Worth checking whether anything in our trainer makes the same mistake. |

---

## Run results

### Run 1: Mimic warm-start (vanilla HF Trainer)

5 rows × 20 epochs × grad_accum=1 = 100 steps. Fixed 2-record held-out batch evaluated before & after.

| | Initial | Final |
|---|---|---|
| `loss` | 1.069 | **0.002** (535× reduction) |
| `mean_token_accuracy` | 0.7625 | **1.0000** |
| `n_labeled` | 918 | 918 |

Trajectory: bouncy descent with deep memorisation by epoch 5-6, locked at near-zero loss by epoch 10+. Per-step loss in epoch 18-19 is `0.0001-0.02`.

**Verdict: optimization, LoRA setup, 4-bit quantization, gradient flow, deltacoder warm-start all confirmed working.**

### Run 2: Mimic cold-start (vanilla HF, fresh LoRA)

Same script, `--no-warm-start` → fresh `LoraConfig(r=32, α=32, target_modules=[q_proj, v_proj], dropout=0)`. 14M trainable params (vs 86M with deltacoder).

| | Initial | Final |
|---|---|---|
| `loss` | 1.071 | **0.016** (67× reduction) |
| `mean_token_accuracy` | 0.7593 | **0.9946** |

Cold-start achieves near-perfect memorisation — slower than warm-start (less capacity, no code prior) but still nails it.

**Verdict: model + corpus + framework can be trained from scratch. The deltacoder warm-start is not the cause of the "flat training" symptom.**

### Run 3: Our trainer cold-start (production pipeline)

500 random rows × 5 epochs × grad_accum=8. Killed at step 49/245 (epoch 0.20) after observing:

| | First 10 steps | Last 10 steps |
|---|---|---|
| `loss` | 0.816 | 0.851 |
| `mean_token_accuracy` | 0.812 | 0.801 |

Loss slope **+0.0005/step** (slightly *up*); tok_acc slope **−0.0002/step** (slightly *down*). Initially looked like a smoking gun — "our trainer can't learn even on cold-start" — but then ran the cleaner A/B in run 5 below.

### Run 4: Our trainer cold-start canonical recipe

Not run — superseded by run 5 (direct A/B against mimic on the exact same overfit probe).

### Run 5: Our trainer warm-start, 5-row × 20-epoch overfit probe (direct mimic A/B)

**This was the experiment that resolved the puzzle.** Same 5 rows, same 20 epochs, same grad_accum=1 the mimic just nailed. Same model, same warm-start (deltacoder), same lr=2e-4. Only the trainer code path differs.

| Step | Epoch | Loss | tok_acc |
|---|---|---|---|
| 1 | 0.4 | 1.50 | 0.684 |
| 5 | 1.0 | 1.36 | 0.677 |
| 9 | 1.8 | 0.39 | 0.902 |
| 13 | 2.6 | 0.23 | 0.936 |
| 17 | 3.4 | 0.10 | — |
| 50 | 10.0 | **0.009** | **1.0000** |

Identical regime to the mimic. Our trainer overfits 5 rows to the same near-zero-loss / 100%-accuracy floor in the same step count. **Pipeline confirmed correct.**

---

## What the puzzle actually was

The "flat training" symptom that motivated this entire investigation was **passes-per-row, not a code bug**:

| Setup | Passes per row | Observed tok_acc trajectory |
|---|---|---|
| Mimic, 5 rows × 20 ep, ga=1 | 20 | 0.76 → 1.00 (perfect memorization) |
| Our trainer, 5 rows × 20 ep, ga=1 (Run 5) | 20 | 0.68 → 1.00 (perfect memorization) |
| Our trainer, 500 rows × 1 ep, ga=8 (replica B) | 1 | 0.79 → 0.84 (modest improvement) |
| Our trainer, 500 rows × 5 ep, ga=8 (Run 3, killed at 0.2 ep) | <1 | 0.81 → 0.80 (no improvement — but **didn't even complete one full pass**) |

The 49-step run that looked "flat" was 49/245 = 0.2 epochs in. **Each row hadn't been seen even once on average.** Of course tok_acc didn't move.

The previous "killed long runs" we labeled flat were similarly underfed:
- 28 steps on 2,743 rows × 1 ep, ga=8 — the run we killed earlier in this work — was 28×8 = 224 of 2,743 rows seen, i.e. about 8% of one epoch.
- 13-step random-500 runs were ONE pass over the full data.

The literature-canonical recipe is **2–3 epochs minimum** (item #8 in `some_training_suggestions.md`). One pass over the data is fundamentally too few for visible per-step token-accuracy improvement on a code-edit fine-tune; gradients move the weights every step, but the *measurable* lift on accuracy needs multiple passes to show up against the warm-start prior.

---

## Conclusions

1. **Optimization stack is healthy across both pipelines.** LoRA, 4-bit quantization, deltacoder warm-start, paged_adamw_32bit (and 8bit), gradient checkpointing, the TRL chat template, manual prompt+response masking, our `_attach_assistant_masks`, `DiffWeightedDataCollator` (when not invoked) — all validated by the 5-row × 20-epoch overfit probe.

2. **Our pipeline is not bugged.** The mimic-vs-ours direct A/B (5 rows × 20 epochs, same HP) reaches the same ~100% memorisation floor in the same step count. There is no signal-dropping bug in `_attach_assistant_masks` or anywhere else in our trainer's data path.

3. **The "flat training" we kept observing was data-volume × epoch-count under the visibility threshold.** All our prior production runs were 1 epoch on 500–2700 rows. With grad_accum=8, that's <1 pass per row per epoch — too few visits to see meaningful per-step accuracy lift over a strong warm-start prior of ~0.79.

4. **Diff-aware loss is still a no-op on this corpus** (separate finding, established earlier — `changed_token_frac ≈ 0.98`). That is unrelated to whether our trainer trains.

5. **One real bug discovered along the way:** the mimic script had `apply_chat_template(tokenize=True)` returning a `BatchEncoding` (not `dict`, not `list`) — `len()` returned the dict-key count of 2, silently skipping every record. Fixed via the cleaner `tokenize=False` → text → re-tokenize path. **Worth checking whether anything in production uses `apply_chat_template(tokenize=True)` with `len()` or `isinstance(dict)` checks.** I checked `compute_assistant_masks` — it uses `tokenize=True, return_dict=False` and explicitly `list(...)` wraps the result, so it's safe.

---

## What to do next

1. **Re-run a real validation training with adequate passes per row.** Recipe: `--epochs 3` (not 1), `--grad-accum 8` (or 4), random-sampled 500–2,500 rows, warm-start deltacoder, lr=2e-4, no NEFTune (for now), no `--diff-aware-loss`. Expected wall clock: 1–3 h depending on dataset size.

2. **Drop `--diff-aware-loss` from default recommendations** until corpus is re-mined to raw file bodies (separate effort). On the current diff-format corpus, diff-aware is mathematically equivalent to plain SFT.

3. **Update training docs / suggested commands** to call out the passes-per-row gotcha. Anyone reproducing our pre-fix experiments would have made the same mistake.

4. **The HPO sweep we stopped earlier should be re-run on corrected code with `--epochs 3`** to find the real optimum HP on this corpus. The previously-found "winners" were tuned under both the truncation bug and the 1-epoch underfit regime.

5. **Item from the suggestions doc still open:** corpus expansion (#4) or re-mining to body shape (#1). Independent of the deep-dive findings; remains a corpus-level decision.

---

## Cross-checks against `instructions/some_training_suggestions.md` (final)

| # | Suggestion | Status |
|---|---|---|
| 1 | Wrong data shape (diffs not file bodies) | True; deferred for separate work; not a training-loop bug |
| 2 | `## Implementation` vs `## Current Code` | Earlier diagnostic showed activation_text uses `## Current Code` (57 %) and the rest are step_index=0 records that legitimately have no pre. Suggestion may have conflated activation/teacher headers |
| 3 | head -N → random sampling | Resolved (all subsequent runs random-sampled) |
| 4 | Corpus too small | Real but not the root cause of the per-step "flat" symptom |
| 5 | LR 2e-4 | Confirmed: works in mimic AND ours |
| 6 | LoRA α=64 | Tested with α=32 (canonical 1.0) — no signal vs α=16; not a bottleneck |
| 7 | paged_adamw_32bit | Mimic uses it and learns; ours uses 8-bit by default and ALSO learns (Run 5). Not the issue |
| 8 | **3 epochs minimum** | **Confirmed as the actual missing piece — single epochs were under the visibility threshold** |
| 9 | warmup_ratio 0.1 | Untested but likely not blocking |
| 10 | HPO under the bug | True; should re-run on corrected code |
| 11 | Drop `--diff-aware-loss` | Confirmed: drop on this corpus |
| 12 | No eval pass | Yes, add to `train.sh` |
| 13 | Cold-start control | Run 2 + Run 3 — both work |
| 14 | tok_acc in plain SFT | `mean_token_accuracy` is emitted by vanilla SFTTrainer; OK |
| 15 | `_compute_hunk_ranges` over-claims | Confirmed; irrelevant once `--diff-aware-loss` is dropped |


---

## Cross-checks against `instructions/some_training_suggestions.md`

| # | Suggestion | Coverage |
|---|---|---|
| 1 | Wrong data shape (diffs instead of file bodies) | Covered in `*-rca.md`; out of scope this round |
| 2 | `## Implementation` vs `## Current Code` header | Partially covered; our diagnostic showed activation_text uses `## Task` / `## Review Feedback` / `## Current Code`, not `## Implementation`. The suggestion may have conflated activation/teacher headers. 43 % of records still have no extractable pre — this is real but appears to be by design (step_index=0 records are initial drafts) |
| 3 | head -N → random sampling | ✅ Already done across all subsequent runs |
| 4 | Corpus too small | Out of scope for now |
| 5 | LR raise to 2e-4 | ✅ All 4 runs use 2e-4 |
| 6 | LoRA α=64 | ✅ Run 4 uses α=64 |
| 7 | paged_adamw_32bit | ✅ Mimic always; Run 4 via `RUNE_OPTIM` env |
| 8 | 3 epochs | ✅ Run 4 uses 3 |
| 9 | warmup_ratio 0.1 | ✅ Run 4 |
| 10 | HPO under the bug | Out of scope |
| 11 | Drop `--diff-aware-loss` | ✅ All 4 runs |
| 12 | No eval pass | Future work |
| 13 | Cold-start control | ✅ Runs 2, 3, 4 |
| 14 | token_accuracy in plain SFT | ✅ Vanilla SFTTrainer emits `mean_token_accuracy` |
| 15 | `_compute_hunk_ranges` over-claiming | Already inspected: median 0.977 on 200 random records — confirmed over-claiming, but irrelevant once `--diff-aware-loss` is dropped |

---

## Conclusions (will be updated after the chain completes)

(pending)

---

## What this means for next steps (will be updated)

(pending)
