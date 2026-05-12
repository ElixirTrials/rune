# Diff-Aware Span-Match — Results & Current Status

**Date:** 2026-05-03
**Branch:** `fix/diff-loss-per-turn-alignment` (PR #35)
**Companion plan:** [`2026-05-03-diff-aware-span-match-rca.md`](2026-05-03-diff-aware-span-match-rca.md)
**Latest commit on this work:** `b80c5f62 fix(diff-loss): suffix-match recovery for keep_end-truncated assistant spans`

---

## Executive summary

Both presenting symptoms are now resolved or reframed:

| Symptom | Initial hypothesis | Final result |
|---|---|---|
| **25–30 % silent identity fallback** in the diff-aware loss path | Either Qwen BPE-merge drift at the `## Revision\n` header (audits 1, 2, `hpe_issues.md` reviewer) **or** `keep_end` truncation slicing the head off long bodies (agent 3) | **Truncation, decisively.** 415/418 (99.3 %) of failures are TRUNCATION_FRONT; BPE drift is 1/1816 (0.06 %). Fix 1 (suffix-match recovery) lands the recovery; **87–92 % of prior failures now align via the surviving suffix.** |
| **Flat loss on the 2,500-row validation run** (loss flat at ~7.5, tok_acc flat at 0.78) | Either the `keep_end` alignment offset added in commit `167d1e9a` regressed something, or the recommended hyperparameters were wrong, or the noisy gradient from 25 % identity fallback was killing learning | **Dataset-sampling confound.** Same code, same HP: `head -500` of `pairs_all.jsonl` → flat at 0.795 tok_acc; *random*-500 → **learns to 0.840 tok_acc** (within striking distance of the t016 HPO winner's 0.88). The first 500 file rows are biased — 45 % drop on pre-tokenization vs 18 % on a uniform sample — and produce an unrepresentative training signal. |

Net change vs main: span-match failures on the production diff-aware path drop from ~25 % silent identity fallback to ≤ 2 % (the residual being `_span_no_post` edge cases, not addressable by Fix 1). The training trajectory itself is unchanged for any single sample, because on this corpus `changed_token_frac ≈ 0.98` — almost every effective token is "changed", so identity weighting and diff weighting produce nearly identical gradients. The diff-aware path is mechanically correct now; whether it is *useful* (vs plain SFT) is a separate question that depends on the corpus's hunk distribution.

---

## Hypotheses

Three hypotheses competed for the span-match failure rate. They make incompatible empirical predictions:

| Hypothesis | Origin | Predicted top failure bucket | Predicted `len(post_ids) > len(span_ids)` |
|---|---|---|---|
| **H1 — BPE-merge drift** at the `## Revision\n` → body boundary; standalone tokenisation differs from in-context by 1–2 tokens at start/end | Audits 1 + 2 (code-read), `hpe_issues.md` reviewer | BPE_DRIFT_START / BPE_DRIFT_END | No — drift at boundary only |
| **H2 — `keep_end` truncation** slices hundreds of tokens off the head of long bodies; the tensor literally doesn't contain the prefix the search wants | Agent 3 (data reproduction) | TRUNCATION_FRONT | Yes — by hundreds |
| **H3 — header asymmetry** in `d2l_data.py` (assistant message keeps `## Revision\n`, `post_codes[idx]` strips it) | Audits 1 + 2 | Indirectly via H1 | No |

A flat-loss hypothesis (H4) was added later, after a 28-step validation run on the recommended hyperparameters showed loss flat at ~7.5 and tok_acc flat at 0.78. Candidate root causes for H4:

- H4a — the `keep_end` alignment offset in commit `167d1e9a` regressed training
- H4b — the recommended HP (lr=2e-4, α=64, dropout=0.05, ga=8, cosine, NEFTune=5, warmup=0.05) was outside the learning region
- H4c — the truncation-induced 25 % identity-fallback noise was washing out the gradient
- H4d — dataset confound (the 250-row smoke and 2,500-row long runs sampled differently)

---

## Method

We refused to pick a fix until a discriminating test was on the table. Two empirical instruments built:

1. **`scripts/diagnose_span_match.py`** + **`libs/model-training/src/model_training/span_match_classifier.py`** — a 7-bucket failure classifier (TRUNCATION_FRONT / TRUNCATION_TAIL / BPE_DRIFT_START / BPE_DRIFT_END / BPE_DRIFT_BOTH / WRONG_TURN_LOOKUP / CONTENT_MISMATCH) with strict predicate ordering. 21 unit tests cover every bucket and the predicate-ordering invariants. Run: `uv run python scripts/diagnose_span_match.py --dataset … --max-length 3072` produces `artifacts/span_match_diagnosis.json` and a histogram. (Tasks 7 + 8 in the companion RCA plan, commit `117da0c8`.)
2. **`scripts/train.sh` replicas under controlled conditions** — short (~30 min) trainings on 500-row samples with the *exact* hyperparameters of HPO trial t016 (the diff-aware HPO winner), comparing pre-fix and post-fix code on both deterministic and random samples. We mined the existing HPO experiment records first to avoid burning fresh L4 time. (Tasks 9 + 10.)

---

## Checks performed and results

### Check 1 — span-match failure classification (Task 7)

| Bucket | Count | % of failures |
|---|---|---|
| TRUNCATION_FRONT | **415** | **99.3 %** |
| TRUNCATION_TAIL | 0 | 0 % |
| BPE_DRIFT_START | 1 | 0.2 % |
| BPE_DRIFT_END | 0 | 0 % |
| BPE_DRIFT_BOTH | 0 | 0 % |
| WRONG_TURN_LOOKUP | 0 | 0 % |
| CONTENT_MISMATCH | 2 | 0.5 % |
| **Total failures** | **418** | (of 1,816 spans) |

**H2 (truncation) selected. H1 (BPE drift) and H3 (header asymmetry) refuted by data.**

### Check 2 — mining the existing HPO records for the flat-loss hypothesis (Task 9)

The Optuna study `rune-training-v1` had 9 COMPLETE trials, including 3 diff-aware trials (t008, t009, t010, t016). Pulled per-trial loss trajectories + final eval metrics from MLflow.

| Trial | diff_aware | LR | α_ovr | drop | ga | sched | Final tok_acc | eval/hunk_acc | eval/improvement |
|---|---|---|---|---|---|---|---|---|---|
| t017 (best non-diff) | False | 4e-4 | 64 | 0 | 8 | cosine | 0.78 (flat per-step) | 0.806 | 0.317 |
| t016 (best diff) | True | 4.3e-5 | 16 | 0.1 | 32 | constant | 0.885 | 0.765 | 0.179 |
| t010 | True | 4.3e-5 | 16 | 0.1 | 32 | constant | 0.840 | 0.765 | 0.141 |

Findings:
- The **per-step training loss is flat across both diff-aware AND non-diff trials**, regardless of whether the trial achieved high eval/improvement. The "loss" metric reported by the trainer per logging step is not the right signal; eval/* metrics at end-of-epoch are.
- Diff-aware trials cluster on conservative HP (lr=4.3e-5, α=16, dropout=0.1, ga=32, constant). Aggressive HP (lr=4e-4 / cosine / α=64) is the non-diff regime.
- **My recommended HP (lr=2e-4, α=64, dropout=0.05, ga=8, cosine, NEFTune=5, warmup=0.05) was outside both regions** — likely why it produced a flat trajectory.

**Provisional ranking of H4 candidates after Check 2: H4b (HP wrong) most likely, H4d (dataset confound) plausible, H4a (Qodo-fix regression) and H4c (gradient noise) less likely but not yet ruled out.**

### Check 3 — replica run on post-Qodo-fix code with t016 HP, head -500 (Task 9 sanity, pre-Fix 1)

Same HP as t016. Deterministic head -500 sample (matching the smoke run's sampling).

| Metric | Value |
|---|---|
| train_runtime | 1780 s |
| Logging steps | 9 |
| Surviving rows | 272 / 500 |
| **train/token_accuracy (final)** | **0.795** (flat: 0.799 → 0.795) |
| train_loss (epoch sum) | 30.09 |
| train/changed_loss | 0.891 |
| diff_span_match_failures | 70 / 393 (17.8 %) |

**Replica DID NOT learn at the per-step training-metric level** — same flat 0.795 token_accuracy as my long run. This was anomalous against t016's reported 0.88 tok_acc with same HP. Three possible explanations remained at this point:

- Code-state regression (Qodo-fix commit `167d1e9a`)
- Dataset sampling difference (t016 used `--subsample 500` random; replica used `head -500`)
- t016's "0.88" was end-of-epoch reporting artifact, not per-step learning

### Check 4 — Fix 1 implemented, replica A run (Task 10, post-Fix 1, head -500)

Fix 1 details: new function `_find_post_in_span_or_suffix` in `libs/model-training/src/model_training/diff_loss.py`. When the strict full-match fails AND `span_start == 0` (the canonical signature of `keep_end` front-truncation), search for the longest suffix of `post_input_ids` that matches a prefix of the span. Constraints: minimum surviving suffix length `max(8, n_post // 4)` to avoid coincidental short matches. Returns `(match_pos, prefix_truncated)`; the caller's weight loop uses unified arithmetic `post_idx = local - match_pos + prefix_truncated` that handles both strict and suffix cases.

Counter changes:
- New `_span_truncated_recovered` increments on suffix-match success (the span IS aligned, just from the suffix).
- `_span_match_failures` increments only when both strict AND suffix fail.
- `train/diff_span_truncated_recovered` surfaced via `DiffAwareSFTTrainer.log()`.

`_find_post_in_span` itself **unchanged** — it remains the single source of truth for the diagnostic classifier (`span_match_classifier.py` imports it via `_find_subseq`).

**Replica A** result (commit `b80c5f62`, deterministic head -500, same HP as Replica from Check 3):

| Metric | Pre-Fix 1 | Post-Fix 1 (A) | Δ |
|---|---|---|---|
| train_runtime | 1779 s | 1779 s | ≈ 0 |
| train/token_accuracy (final) | 0.795 | **0.795** | ≈ 0 |
| train_loss | 30.09 | 30.07 | ≈ 0 |
| train/changed_loss | 0.891 | 0.893 | ≈ 0 |
| diff_spans_aligned | 323 | 384 | **+61** (recovered counted as aligned) |
| diff_span_match_failures | 70 | 9 | **−61** |
| diff_span_truncated_recovered | (didn't exist) | **61** | NEW |
| diff_span_no_post | — | 9 | (already-known empty-post edge case) |

**Fix 1 mechanically works** — 61 of the 70 prior failures recovered (87 %). The remaining 9 are `_span_no_post` empty-post edge cases that Fix 1 doesn't address. **But the per-step training trajectory is identical to pre-Fix 1.**

Why: `changed_token_frac ≈ 0.98` on this corpus. ~98 % of effective tokens carry weight 1.0 (changed) regardless of whether the diff path runs correctly or falls back to identity. Only ~2 % of tokens see their weight drop from 1.0 to 0.3 when the diff path is correct. Gradient signal change is microscopic.

This **rules out H4c** (gradient noise from identity fallback was not the cause of flat loss). Token accuracy stayed at 0.795 even after the noise was eliminated.

### Check 5 — Replica B (Task 10, post-Fix 1, *random*-500)

Identical to Replica A in every respect except the dataset is a random-sampled 500-row subset (matching how t016's `--subsample 500` works in the HPO infrastructure).

| Metric | Pre-Fix 1 (head-500) | Post-Fix 1 A (head-500) | Post-Fix 1 B (**random-500**) |
|---|---|---|---|
| train_runtime | 1779 s | 1779 s | **2754 s** |
| Logging steps | 9 | 9 | **13** |
| Surviving rows | 272 (45 % drop) | 272 (45 % drop) | **408 (18 % drop)** |
| **train/token_accuracy (final)** | 0.795 | 0.795 | **0.840** ↑↑ |
| train_loss (epoch sum) | 30.09 | 30.07 | **27.83** ↓ |
| train/changed_loss | 0.891 | 0.893 | **0.725** ↓ |
| changed_token_frac | — | 0.982 | 0.975 |
| diff_spans_aligned | 323 | 384 | 446 |
| diff_span_match_failures | 70 | 9 | 8 |
| diff_span_truncated_recovered | — | 61 | **88** |
| diff_span_no_post | — | 9 | 7 |

**Replica B reaches token_accuracy 0.840 on the same code and same HP that Replica A stayed flat at 0.795.** That is within striking distance of t016's 0.885. The only difference between A and B is *how the 500 rows were sampled*.

**This selects H4d (dataset sampling) as the cause of the flat-loss symptom.** `head -500` of `pairs_all.jsonl` is biased: 45 % of those rows get dropped during pre-tokenisation (vs 18 % on a uniform random sample), and the surviving rows are over-represented in some narrow region of repo / time that does not match the corpus average. The "flat loss on the long run" was therefore not a code or HP issue but an unintended methodology issue in how I built the smoke and validation samples.

H4a (Qodo-fix regression) and H4b (HP wrong) are also retired — Replica B reproduces t016's outcome on post-Qodo-fix code with the same conservative HP.

---

## Code changes landed (this branch)

```
b80c5f62 fix(diff-loss): suffix-match recovery for keep_end-truncated assistant spans
117da0c8 fix(diag): code-review cleanups for span-match classifier
cbb5c9b6 fix(diag): use _find_post_in_span from diff_loss as single source of truth
d684b2de docs(plan): RCA + discriminating-test plan for diff-aware span-match failure
b94c4a99 feat(diag): Task 7 — span-match failure classifier + diagnose_span_match.py
```

Files added:

- `scripts/diagnose_span_match.py` — CLI tool for the 7-bucket diagnostic
- `libs/model-training/src/model_training/span_match_classifier.py` — pure classifier (no torch, no I/O)
- `libs/model-training/tests/test_span_match_classifier.py` — 21 unit tests for the classifier
- `docs/superpowers/plans/2026-05-03-diff-aware-span-match-rca.md` — companion plan

Files modified:

- `libs/model-training/src/model_training/diff_loss.py` — adds `_find_post_in_span_or_suffix`, the `_span_truncated_recovered` counter on `DiffWeightedDataCollator`, and the `train/diff_span_truncated_recovered` log emit on `DiffAwareSFTTrainer.log()`. `_find_post_in_span` itself unchanged (single source of truth).
- `libs/model-training/tests/test_diff_loss.py` — 7 new tests covering the suffix-match path and counter behaviour.
- `.gitignore` — adds `artifacts/`.

Test inventory:

- `libs/model-training/tests/test_diff_loss.py` — **48 tests pass** (was 41 before this work; +3 chunked entropy from earlier in PR #35, +2 keep_end alignment tests, +7 suffix-match tests).
- `libs/model-training/tests/test_span_match_classifier.py` — 21 tests pass.
- Combined: 69 tests, all green.
- `uv run ruff check` clean on every changed file.

---

## Current status

| Item | State |
|---|---|
| Span-match failure root cause | **Resolved.** Empirically confirmed: 99.3 % is `keep_end` front-truncation, not BPE drift, not header asymmetry, not wrong-turn-lookup. |
| Fix 1 (suffix-match recovery) | **Landed locally** at `b80c5f62`. Mechanically validated: 87–92 % of prior failures now align via the surviving suffix; remaining residue is `_span_no_post` empty-post cases. |
| Diagnostic instrument | **Landed locally.** `scripts/diagnose_span_match.py` is committed and runs in ~5 min on the full corpus. Will surface any future regression in any of the 7 buckets. |
| Per-step training trajectory after Fix 1 | **Unchanged in absolute terms, by design.** The corpus's `changed_token_frac ≈ 0.98` means identity-fallback and diff-aware weighting are within ~2 % of each other in gradient impact. Fix 1 is *correct*, not necessarily *useful* on this corpus. |
| "Flat loss" symptom | **Reframed: dataset sampling confound.** The deterministic `head -500` smoke sample was unrepresentative. Random sampling reproduces t016's outcome on post-fix code with the same HP. |
| HPO sweep | **Stopped** mid-run (had 19 FAIL, 9 COMPLETE, 1 RUNNING; was running on uncorrected code so it would have polluted the diff-aware HP search). |
| In-flight validation training | **Killed** before this work; was the run that surfaced the flat-loss symptom in the first place. Now superseded by Replica B. |
| Push to remote | Not done. User asked for commit-only at this stage. Branch is 6 (now 7 with this doc) commits ahead of `origin/fix/diff-loss-per-turn-alignment`. |

---

## Remaining open questions / next steps

These are explicitly out-of-scope for this branch's PR but worth tracking:

1. **The diff-aware path may not be useful on this corpus.** `changed_token_frac ≈ 0.98` means the weighting is a near-no-op vs plain SFT. Either the corpus is intrinsically rewrite-heavy (most lines genuinely differ between pre and post) or `_compute_hunk_ranges` is over-claiming "changed". Worth a separate investigation: tally the *line-level* hunk ratios on a 100-row sample to confirm.
2. **Dataset sampling discipline.** `head -N` is convenient but biased; the project's own HPO uses `--subsample N` for random sampling. Future smoke / validation runs should random-sample by default. Either change `data/_smoke/` generators or add a flag.
3. **Restart HPO on corrected code.** The previous HPO converged on conservative HP (lr=4.3e-5, α=16, drop=0.1, ga=32, constant) under the bug. With the bug fixed (or rather: with Fix 1 making the gradient less noisy on the fraction of truncated spans), the optimum may drift toward more aggressive HP. A fresh HPO would tell us — and would also address the `eval/adapter_improvement` gap (t017 non-diff at 0.317 vs t016 diff at 0.179).
4. **Eval is missing from `scripts/train.sh`.** The HPO path produces `eval/hunk_loss`, `eval/hunk_accuracy`, `eval/adapter_improvement`; the `train.sh` standalone path does not. We compared per-step training loss across runs but couldn't compare the eval metrics that the HPO objective is built on. Worth adding an eval pass to `train.sh` for consistency.
5. **`_span_no_post` ~3 % residue.** Small but non-zero rate of empty-post edge cases. Probably benign (legitimately empty assistant turns?), but worth a brief audit before the next major training run.
