# Diff-Aware Span-Match Failure — RCA, Discriminating Test, Fix Plan

**Date:** 2026-05-03
**Branch:** `fix/diff-loss-per-turn-alignment` (PR #35)
**Status:** two competing root-cause hypotheses; **discriminating test needed before fix**

---

## TL;DR

We have **two distinct symptoms** on PR #35, observed simultaneously on the in-flight 2,500-row validation training:

- **Symptom 1 — silent identity fallback:** ~25 – 30 % of assistant spans fail `_find_post_in_span` and fall back to identity weights. Counter ratio at step 28 = 82 / 274 = 30 %.
- **Symptom 2 — flat loss / no learning:** loss flat at ~7.5 (mean 7.44 first-third, 7.50 last-third; OLS slope -0.036/step ≈ 0); `token_accuracy` flat at 0.78. By contrast the 250-row smoke run with default hyperparameters showed a clear -0.19/step decrease and accuracy 0.795 → 0.814 over 16 steps.

Two parallel investigations into Symptom 1 produced **mutually inconsistent** root causes:

- **Audits 1 + 2 (code read):** Qwen byte-level BPE merge drift at the `## Revision\n` → body boundary. Standalone tokenisation of `post_codes[idx]` differs from in-context tokenisation by ~1 token.
- **Audit 3 (data reproduction):** `keep_end` truncation slices the head off long diff bodies; the tensor literally doesn't contain the prefix the search is looking for.

These are **physically incompatible explanations**:

|              | BPE drift only | Truncation only |
|---|---|---|
| `len(post_ids) > len(span_ids)` | No | Yes |
| `post_ids[1:]` matches inside span | Yes | No (still too long) |
| First failing token | start (or end) | first hundreds |
| Span starts at sequence position 0 | Sometimes | Always |

**Neither investigation's evidence rules out the other** (see "Critique" section below). And neither investigation directly addresses Symptom 2 — flat loss could be downstream of the identity-fallback rate, or it could be an independent hyperparameter / override-alpha / NEFTune issue. The two symptoms can be related or independent.

Until we know the empirical breakdown for Symptom 1 *and* the cause of Symptom 2, any fix is a gamble. This document defines:

1. A **single discriminating test** (`scripts/diagnose_span_match.py`) that classifies every span-match failure into one of seven mutually-exclusive buckets.
2. A **hyperparameter A/B test** that isolates whether Symptom 2 is caused by the new hyperparameters (override-alpha=64, dropout=0.05, NEFTune=5, warmup=0.05) or by the dataset/code state.
3. A **success criterion** for each candidate fix, expressed as the bucket distribution and loss slope we expect post-fix.
4. The **decision rule** for which fix(es) to land based on the empirical breakdown.

No code-under-test changes until both tests run and their results are on the table. The in-flight training has been killed (it had served its purpose: confirming Symptom 2 against the smoke baseline).

---

## Critique of the existing investigations

### What audits 1 + 2 got right
The asymmetry at `d2l_data.py:1069-1071` (`_extract_revision` keeps the header, `_extract_post_revision` strips it) is **real**. If failures are caused by BPE drift, this asymmetry is the cause. The reviewer in `instructions/hpe_issues.md` correctly identified this as the structural mismatch in the data pipeline.

### What audits 1 + 2 missed
They reasoned from a code read without measuring. Specifically:

- Never measured `len(post_ids)` vs `len(span_ids)` on actual failure cases. If post is strictly longer than span, BPE drift cannot be the cause.
- Never quantified the implied prediction: BPE drift at the start should fail at token index ≤ 2 of the search. Truncation should fail because the entire prefix is missing (hundreds of tokens).

### What audit 3 got right
Empirical reproduction on the full corpus, with worked examples showing posts that are 145 / 1498 / 29 tokens longer than their spans. That is **direct evidence of truncation**.

### What audit 3 got wrong
Their classifier — `tail_found = post_ids[-20:] is in span_ids` — is symmetric to BPE-drift-on-the-start-token. A standalone-tokenized post whose first BPE merge differs in-context will:
- Fail the strict full-match (✗)
- Have an identical 20-token tail (✓ matches)
- Have `post_ids[1:]` that matches inside the span (not checked)

The reported "99 % of failures have matching tails" is therefore **not** evidence for truncation specifically — it is also consistent with start-token BPE drift. Agent 3 generalised from 3 worked examples without checking the discriminator on all 477.

### What the reviewer in `hpe_issues.md` missed
The review endorses Fix A (header strip) + Fix C (`[1:-1]` retry) without running any reproduction. They listed three alternative hypotheses (truncation offset error, `lstrip("\n")` on body, `strip_current_code_section` interaction) but treated them as edge cases to verify rather than as primary candidates. They never considered front-truncation by `keep_end` because audits 1 and 2 didn't surface it.

---

## The discriminating test

### Design

For every failed span across the full corpus, classify into exactly one of seven buckets. The buckets are mutually exclusive and collectively exhaustive (every failure ends up in exactly one). The decision tree is evaluated in order:

| # | Bucket | Decision predicate (in order) | What it implies |
|---|---|---|---|
| 1 | **TRUNCATION_FRONT** | `span_start == 0` AND `len(post_ids) > len(span_ids)` AND ∃ k≥1 such that `post_ids[k:] == span_ids[0 : len(post_ids)-k]` (suffix-of-post matches prefix-of-span) | `keep_end` cut the head; data was correct, span was sliced |
| 2 | **TRUNCATION_TAIL** | `span_end == len(input_ids) - 1` AND `len(post_ids) > len(span_ids)` AND ∃ k≥1 such that `post_ids[:-k] == span_ids[-(len(post_ids)-k):]` (prefix-of-post matches tail-of-span) | The tail of the body was cut (would indicate `keep_start` was used somewhere; not expected with our config) |
| 3 | **BPE_DRIFT_START** | `len(post_ids) ≤ len(span_ids)` AND `post_ids[1:]` is a contiguous run inside span AND `post_ids[0]` ≠ corresponding span token | Standalone left-context produced different first BPE merge |
| 4 | **BPE_DRIFT_END** | `len(post_ids) ≤ len(span_ids)` AND `post_ids[:-1]` is a contiguous run inside span | Standalone right-context produced different last BPE merge (the `<|im_end|>` adjacent merge) |
| 5 | **BPE_DRIFT_BOTH** | `len(post_ids) ≤ len(span_ids)` AND `post_ids[1:-1]` is a contiguous run inside span AND #3, #4 don't apply | Both boundaries drifted (rare, multi-byte unicode) |
| 6 | **WRONG_TURN_LOOKUP** | None of the above AND ∃ another `post_codes[j]` (j ≠ assigned turn_idx) whose tokens match the span | Per-turn alignment offset is wrong (Alt-H1) |
| 7 | **CONTENT_MISMATCH** | None of the above | True content drift — `post_codes[idx]` text differs from what's actually in the span |

### Diagnostic outputs

For each failure, log:

- bucket (one of the 7)
- `(conv_idx, span_idx, turn_idx, span_start, span_end, len(post_ids), len(span_ids))`
- `post_ids[:4]`, `span_ids[span_start:span_start+4]` (with decoded text for both)
- `post_ids[-4:]`, `span_ids[span_end-4:span_end]` (decoded)
- For BPE_DRIFT_*: which boundary tokens drifted (id_a → id_b with text)
- For WRONG_TURN_LOOKUP: which `j` matched and how it differs from `turn_idx`
- For CONTENT_MISMATCH: longest common subsequence length / span length ratio

### Aggregate report

```
=== SPAN-MATCH FAILURE BREAKDOWN ===
Total spans:          N
Total failures:       F  (F/N = X.X%)
  TRUNCATION_FRONT:    a (a/F = X%)
  TRUNCATION_TAIL:     b
  BPE_DRIFT_START:     c
  BPE_DRIFT_END:       d
  BPE_DRIFT_BOTH:      e
  WRONG_TURN_LOOKUP:   f
  CONTENT_MISMATCH:    g
```

### Where the test lives

- Script: **`scripts/diagnose_span_match.py`** (committable; promoted from `/tmp/inspect_match_failures.py`).
- One-shot run: `uv run python scripts/diagnose_span_match.py --dataset data/github-pairs/_merged/pairs_all.jsonl --max-length 3072` — writes `artifacts/span_match_diagnosis.json` (full per-failure record) and prints the aggregate report.
- Unit tests for the classifier itself: **`libs/model-training/tests/test_span_match_classifier.py`** — synthetic spans constructed to land in each of the 7 buckets, asserting the classifier gets each right. The classifier MUST be tested in isolation before we trust its corpus-wide output.

---

## What we expect to see (predictions)

| Hypothesis | Predicted top buckets |
|---|---|
| Audits 1+2 are right (BPE drift) | BPE_DRIFT_START dominates (~25 %), TRUNCATION_FRONT minor (a few %) |
| Audit 3 is right (truncation) | TRUNCATION_FRONT dominates (~25 %), BPE_DRIFT_* minor |
| Both are partially right | Both visible at non-trivial rates |
| Some third hypothesis | A bucket nobody predicted dominates |

The empirical breakdown picks the winner. **No fix is implemented before this report exists.**

## Empirical breakdown (Task 7+8 result, commit 117da0c8)

```
=== SPAN-MATCH FAILURE BREAKDOWN ===
Total spans:          1816
Total failures:        418  (23.0%)
  TRUNCATION_FRONT       415  (99.3% of failures)
  TRUNCATION_TAIL          0
  BPE_DRIFT_START          1  (0.2%)
  BPE_DRIFT_END            0
  BPE_DRIFT_BOTH           0
  WRONG_TURN_LOOKUP        0
  CONTENT_MISMATCH         2  (0.5%)
```

Audit 3's truncation hypothesis is **decisively dominant**. The BPE-drift hypothesis from audits 1, 2 and the `hpe_issues.md` reviewer is supported by exactly 1 case in 1816 — empirically below noise. The decision rule (TRUNCATION_FRONT ≥ 50 % → Fix 1) selects **Fix 1 (suffix match in `_find_post_in_span`)** as the primary intervention. **Fix A (header strip) is not warranted** by this evidence; the header divergence between `messages[assistant].content` and `post_codes[idx]` is a real-but-cosmetic data-pipeline asymmetry that affects ≤ 0.2 % of spans. **Fix C (defensive `[1:-1]` retry)** still lands regardless per the plan, as a safety net for any future regression — but by itself it cannot recover the 99.3 % of failures (the retry trims 1–2 tokens, not the 145 / 1498 / 29 that truncation removes).

---

## Decision rule (after the diagnosis runs)

| Empirical pattern | Action |
|---|---|
| TRUNCATION_FRONT ≥ 50 % of failures | Implement **Fix 1** (suffix match in `_find_post_in_span`) — primary |
| BPE_DRIFT_* ≥ 50 % of failures | Implement **Fix A** (strip header in `d2l_data.py:953, 1069-1071`) — primary |
| Both > 20 % | Implement Fix 1 AND Fix A; landing order: A first (corpus-side), then 1 (collator-side); each landing should drop its target bucket independently |
| WRONG_TURN_LOOKUP > 5 % | Re-audit the per-turn alignment offset in `_weights_via_hunk_path` — separate fix, blocking |
| CONTENT_MISMATCH > 5 % | Investigate `_extract_revision` / `_extract_post_revision` for hidden text transformations — separate fix |
| Any bucket dominant but unaddressable by A/1 | Stop and triage; this RCA needs another iteration |

In all cases, **Fix C** (`[1:-1]` retry safety net in `_apply_span_weights`) lands as a defensive measure regardless. Cost is ~6 lines and it produces a separate counter so we can see if drift creeps back in later.

---

## Validation plan: how we prove the fix worked

For each fix that lands, the following gates must pass before the next fix lands or the PR merges:

### Gate 1 — classifier-level proof (synthetic)

`test_span_match_classifier.py` synthesises a span that lands in the targeted bucket *before* the fix. After the fix, re-classify the same synthetic span. The bucket must change (typically to "no-failure / matched").

Concretely for Fix A:
- Pre-fix: build a synthetic record where `messages[assistant].content = "## Revision\n<body>"` and `post_codes = ["<body>"]`. Run through the collator. Assert the failure bucket is `BPE_DRIFT_START`.
- Post-fix: same record. Assert no failure (`_find_post_in_span` succeeds, `_spans_aligned` increments, `_span_match_failures` does not).

For Fix 1:
- Pre-fix: build a synthetic record where the assistant turn has 3000 body tokens but the conversation forces `keep_end` truncation to 1500. Assert failure bucket is `TRUNCATION_FRONT`.
- Post-fix: same record. Assert `_find_post_in_span` returns a negative `match_pos` (signalling "post starts before tensor"), `_apply_span_weights` correctly weights the surviving suffix, `_span_match_failures` does not increment, a new `_span_truncated_recovered` counter increments.

### Gate 2 — corpus-level proof (empirical)

Re-run `scripts/diagnose_span_match.py` on `data/github-pairs/_merged/pairs_all.jsonl` after the fix. Compare the bucket histogram to the pre-fix run. The targeted bucket must drop by ≥ 90 %; total failure rate must drop accordingly. Other buckets must not increase (no regression).

Save both runs as `artifacts/span_match_diagnosis_pre_fixA.json` and `artifacts/span_match_diagnosis_post_fixA.json`. Diff them — anything that moved unexpectedly is a red flag worth investigating before merging.

### Gate 3 — live-run proof (training)

Restart the validation training (same command as the in-flight run, with the new adapter id). Inspect the live MLflow metrics:

- `train/diff_span_match_failures / train/diff_spans_aligned` ratio after step 5 must be ≤ 1 % (target; from the current 25 %).
- New `train/diff_span_truncated_recovered` (added by Fix 1) should track the pre-fix `diff_span_match_failures` rate — i.e. failures that previously were silent identity now show up as recovered.
- Loss / accuracy / entropy trends from the previous run should not regress; if anything, the recovered diff signal should make `train/changed_loss` mildly higher (richer signal) and `train/context_loss` mildly lower (better calibrated).

### Gate 4 — regression test (CI)

Add to `libs/model-training/tests/test_diff_loss.py`:
- `test_keep_end_front_truncation_recovers_via_suffix_match` — only meaningful if Fix 1 lands.
- `test_assistant_content_equals_post_codes_after_extract` — guards Fix A against future regression at the data layer.
- Both must pass on `uv run pytest libs/model-training/tests/test_diff_loss.py -q`.

---

## Files involved (for whichever fix(es) the diagnosis selects)

### If Fix 1 is selected (truncation suffix match)

| File | Why |
|---|---|
| [`libs/model-training/src/model_training/diff_loss.py`](../../../libs/model-training/src/model_training/diff_loss.py) | `_find_post_in_span` (line 262): when strict match fails AND `span_start == 0` (or the equivalent flag), search for the longest suffix of `post_ids` that matches a prefix of the span; return a *negative* `match_pos` to signal "post head was truncated". `_apply_span_weights` (line 438): handle negative `match_pos`, weight the truncated prefix as identity (no per-token diff data is recoverable for it), offset hunk-range character anchors by `post_offsets[k][0]` so weights still land on the right surviving body bytes. New counter: `_span_truncated_recovered`. |
| [`libs/model-training/tests/test_diff_loss.py`](../../../libs/model-training/tests/test_diff_loss.py) | Synthetic front-truncation test asserting the suffix path recovers diff weights on the surviving body. |

### If Fix A is selected (header strip)

| File | Why |
|---|---|
| [`libs/model-training/src/model_training/d2l_data.py`](../../../libs/model-training/src/model_training/d2l_data.py) | Lines 953 (single-turn) and 1069–1071 (multi-turn): `messages[assistant].content` becomes `_extract_post_revision(...)` (header-free) so it equals `post_codes[idx]`. The standalone re-tokenisation in the collator then matches in-context tokenisation exactly. |
| [`libs/model-training/tests/test_d2l_data.py`](../../../libs/model-training/tests/test_d2l_data.py) (or the existing test module for `pairs_to_chat_messages`) | Assert `messages[idx].content == post_codes[turn_idx]` for every turn in the output — guards the asymmetry from creeping back in. |

### Fix C (defensive retry, lands regardless)

| File | Why |
|---|---|
| [`libs/model-training/src/model_training/diff_loss.py`](../../../libs/model-training/src/model_training/diff_loss.py) | `_apply_span_weights` (line 438): after the strict match fails AND any selected primary fix's path also fails, retry with `post_input_ids[1:-1]` (and `[2:-2]` for multi-byte boundary chars). On success, the trimmed boundary tokens get identity weight; rest of body gets diff weights. New counter: `_span_match_recovered_via_trim`. |

---

## Why "strip uninformative tokens like `\n` from the JSONL" still won't help

(retained from prior draft, unchanged by the discrimination question)

A natural-sounding fix is "just remove the newlines from the corpus" — it would not work and would actively damage the model:

| Concern | Effect of stripping newlines from the corpus |
|---|---|
| Python code structure | Indentation + line boundaries define blocks. No newlines = unparseable Python. |
| Production interface | The model is asked to emit normal multi-line code at inference time. Training without newlines would teach it the wrong distribution. |
| Diff hunks | `_compute_hunk_ranges` works on `splitlines(keepends=True)`. No newlines = single-line bodies = no hunks. The whole diff-aware path collapses. |
| Either suspected mechanism | Truncation cause: doesn't matter what we strip; the bytes are missing. BPE-drift cause: removing the body's internal newlines doesn't change the boundary that creates the drift; only a header strip (Fix A) does. |

The right framings remain: **either preserve more of the post in the tensor (Fix 1, suffix match) or remove the boundary that creates BPE drift (Fix A, header strip).** Stripping newlines does neither.

---

## Symptom 2 — flat loss on the long run

### Observation

Live MLflow data from the (now-killed) validation run, all 28 steps:

```
first-third loss mean:  7.443
middle-third loss mean: 7.466
last-third loss mean:   7.499
OLS slope:              -0.036 per step (≈ 0; noise floor)
token_accuracy:         0.78 throughout
entropy:                no monotonic trend
changed_token_frac:     0.99 (was 0.95 in smoke; even more locked-into "all changed")
```

The smoke run on a 250-row subsample with **default** hyperparameters produced a clear -0.19/step slope and 0.795 → 0.814 accuracy over 16 steps. The long run with **recommended** hyperparameters on the full corpus did not.

### What changed between the two runs

| Knob | Smoke (learning) | Long (flat) | Plausibility as flat-loss cause |
|---|---|---|---|
| Dataset | `data/_smoke/pairs_smoke_250.jsonl` (250 rows, head -250 of full file) | `data/github-pairs/_merged/pairs_all.jsonl` (2,743 rows, full) | Possible — the head of the file may be biased toward easier records. |
| Epochs / LR / scheduler | 1 / 2e-4 / cosine | 1 / 2e-4 / cosine | No change. |
| `--override-lora-alpha` | unset (uses deltacoder's saved α) | 64 | **Strong candidate.** If deltacoder's α ≠ 64, this rescales the adapter and may force it into a regime where it has to relearn baseline behavior. |
| `--override-lora-dropout` | unset | 0.05 | Possible. Adds ~5 % noise during training. |
| `--neftune-noise-alpha` | unset | 5 | Possible. NEFTune perturbs the embedding layer; on a model that should be subtle-fine-tuning it can disrupt rather than help. |
| `--warmup-ratio` | unset (default 0.03) | 0.05 | Mild — extends warmup by ~67 %. Unlikely the sole cause. |
| `--encoding-mode` / `--max-seq-length` | multi_turn / 3072 | multi_turn / 3072 | No change. |
| Code state | commit `6e76bae0` (chunked-entropy fix only) | commit `167d1e9a` (chunked-entropy + Qodo fixes incl. keep_end alignment offset, metric-domain change) | **Strong candidate.** The Qodo-fix commit changed real semantics: per-turn alignment now offsets by `n_turns - len(spans)` under `keep_end`. If that offset is wrong (off-by-one, or wrong on records where `len(spans) > n_turns`), the gradient signal is corrupt. |

The two strong candidates (override-alpha and code state) deserve separate isolation. The two mild candidates (dropout, NEFTune) are unlikely to *flatten* learning but could compound.

### The hyperparameter A/B test

Run two short trainings on the same fresh 500-row subsample of `pairs_all.jsonl`, on the current commit (`167d1e9a`):

**Run A — smoke-style HP** (matches the run that learned):
```bash
bash scripts/train.sh \
  --dataset data/_ab/pairs_500.jsonl \
  --adapter-id "ab-smokehp-$(date +%H%M)" \
  --warm-start deltacoder \
  --epochs 1 --grad-accum 8 --max-seq-length 3072 \
  --diff-aware-loss --encoding-mode multi_turn \
  --experiment-name rune-ab-flat-loss
```
(no `--lr`, no `--override-*`, no `--neftune-*`, no `--warmup-ratio` — let trainer defaults apply, identical to smoke)

**Run B — recommended HP** (matches the run that didn't):
```bash
bash scripts/train.sh \
  --dataset data/_ab/pairs_500.jsonl \
  --adapter-id "ab-rechp-$(date +%H%M)" \
  --warm-start deltacoder \
  --epochs 1 --lr 2e-4 \
  --override-lora-alpha 64 --override-lora-dropout 0.05 \
  --warmup-ratio 0.05 --grad-accum 8 \
  --lr-scheduler cosine --diff-aware-loss \
  --neftune-noise-alpha 5 --max-seq-length 3072 \
  --encoding-mode multi_turn \
  --experiment-name rune-ab-flat-loss
```

Each takes ~25 – 30 min on the L4 (500 rows / 8 grad_accum ≈ 60 steps).

### Decision rule for Symptom 2

| Outcome | Implication | Action |
|---|---|---|
| Run A learns (slope ≤ -0.05/step), Run B flat | Hyperparameters cause Symptom 2 | Drop the recommended HP; use defaults. Triage which knob: bisect (alpha-only, dropout-only, NEFTune-only mini-runs) if needed. |
| Both flat | Dataset or code state causes Symptom 2 | Bisect: rerun Run A on commit `6e76bae0` (pre-Qodo-fix) → if it learns, the Qodo-fix commit broke training. If still flat, the full-corpus distribution is the cause (smoke was lucky). |
| Both learn | The first long run was a fluke / batch effect | Re-run the long validation, fixing whatever specific knob changed between the killed run and the new one. |
| Run A flat, Run B learns | Unexpected. Re-audit. | Investigate. |

### Success criterion for Symptom 2

A fix is considered to have addressed Symptom 2 when, on a 500-row run with the chosen hyperparameters, the OLS slope on `loss` over the full run is ≤ **-0.05/step** (between the smoke -0.19 and the killed -0.036). Lower is better. Token accuracy must show a strictly increasing trend on first-third → last-third averages.

---

## Schedule

1. **In-flight training killed** (already done).
2. **Promote `/tmp/inspect_match_failures.py` to `scripts/diagnose_span_match.py`**, extend with the 7-bucket classifier, and write `test_span_match_classifier.py` with synthetic-input tests for each bucket. ~1.5 hours.
3. **Run the span-match diagnosis** on the full corpus → publish `artifacts/span_match_diagnosis_pre_fix.json` and the aggregate breakdown. ~5 min run.
4. **Run the hyperparameter A/B** (Run A and Run B above). ~1 hour wall clock total (sequential on L4) or ~30 min each.
5. **Triage:** combine the bucket histogram and the A/B result. Decide which of (Symptom 1 fix(es), Symptom 2 fix) need to land.
6. **Implement the selected fix(es).** Estimated 1 – 3 hours depending on which fix(es).
7. **Re-run the span-match diagnosis** post-fix (Gate 2). Diff histograms.
8. **Re-run validation training** with the converged config (Gate 3). Watch live counters AND loss slope.
9. Add CI regression tests (Gate 4).
10. Merge PR #35.

---

## Open questions left for after the diagnosis

- **Right-boundary drift (`<|im_end|>` adjacent):** if BPE_DRIFT_END appears in the histogram, the same Fix A logic doesn't address it (Fix A only changes the left context — header). A separate intervention may be needed.
- **Multi-byte unicode boundaries:** if BPE_DRIFT_BOTH appears at non-trivial rate, Fix C's `[1:-1]` is insufficient; need `[2:-2]` retry.
- **Cumulative effect:** if both TRUNCATION_FRONT and BPE_DRIFT_START appear at material rates, landing only one fix will leave the other visible. Plan for two passes.
- **Pre-codes (`_extract_pre_revision`):** `rstrip("\n")` on the body and possible `## Current Code\n` header stripping. Pre-codes feed `_compute_hunk_ranges`, not the in-span match, so this is a *secondary* concern — but if the diagnosis surfaces hunk-boundary anomalies in `train/changed_token_frac` after Fix A or 1 lands, this is the next thing to audit.
- **`strip_current_code_section` interaction (Alt-H3 from the reviewer):** non-first turns have user content elided to `CURRENT_CODE_ELIDED`; `pre_codes` for those turns is from the pre-elision user. If hunk ranges go wrong, this is the suspect.
- **Should the deltacoder warm-start adapter be retrained?** Probably not — the new adapter on top will simply learn the corrected distribution.
