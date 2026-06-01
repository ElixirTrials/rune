# Issue #49 — adapter-as-memory: overnight findings (2026-06-01)

**Status: exploratory negative with a clear next direction.** Short runs (40–600 steps),
one corpus (`external_codereview`), held-out clean val. Not proof; the *data-structure*
argument is what generalizes, the run numbers confirm it.

## TL;DR

We set out to make the HyperLoRA adapter encode a *specific coding trajectory* (so a
matched adapter beats a wrong-context one on the gold edit). After fixing the original
collapse (scaler_B≈0) and correctly wiring a contrastive specificity objective, the
adapter still does **not** carry trajectory memory. Five independent probes converge on
why, and the root cause is upstream of the loss:

1. **No document-style recall.** The adapter does not store the row's code — it makes the
   code body *less* likely than base (copy `matched−zero = −0.31`). Sakana/doc2lora's
   "recover facts from the embedded document" premise does **not** transfer.
2. **It is a generic edit-booster.** It boosts edit-region tokens (`+1.17` over base) at
   the expense of the code body (`−0.31`), net-negative on full-answer.
3. **A faint context-conditioned edit signal exists, but it is code-driven, not
   feedback-driven.** matched−mismatch on edit tokens `+0.075` (grows to `+0.161` at
   scale 1.0); matched−**feedback-swap** only `+0.005`. The adapter weakly knows *which
   edit* from the surrounding code and essentially **nothing** from the review request —
   which is the trajectory fact we actually wanted to bind.

Only claim (3) is the adapter-as-trajectory-memory signal, and it is currently far too
small (and feedback-blind) to be useful.

## Two root causes (co-equal)

### A. Data dilution (training-length-independent; the load-bearing evidence)
Each training row's **answer is the entire revised file re-emitted**, ~89% a verbatim
copy of the `## Current Code` already in the context, with the real change ~10% of
tokens. The **review feedback — the trajectory-specific fact — is only ~5% of the
context (median 23 tokens)**.

| metric (400 train rows) | median |
|---|---|
| answer tokens | 401 |
| context tokens | 472 |
| feedback tokens | 23 (**4.7%** of context) |
| copy fraction (answer verbatim ⊂ pre_code) | **0.89** |
| edit-local token fraction | **0.10** |

Because the student forward is answer-only (the adapter is the sole carrier of context),
we are implicitly asking a rank-r adapter to **regurgitate ~470 tokens of one row's code
verbatim**, then apply a 10% edit driven by 5% of the input. The contrastive
hard-negative (feedback-swap) keeps the code **identical**, so on 89% of tokens matched
and negative adapters must behave the same *by construction* — the trajectory signal can
only live in the ~10% edit. This structurally guarantees the tiny residual we measure.

### B. Conditioning attenuation (mechanistically deeper; measured)
The perceiver→weight mapping flattens context. Extracted features carry the context
(25–31% variation across rows; **5.3%** from feedback alone), but the generated adapter
weights barely move: context-dependent residual is **~1%** of `||W||`, and **feedback-only
moves the weights just 0.4%** (a ~13× attenuation, harder than the ~5× for whole-context).
Cross-row weight deltas concentrate in mid-late layers 21–24; feedback-swap deltas stay
tiny everywhere.

A and B are entangled (feedback may attenuate *because* it is 5% of tokens), but B is
real and measured independently of the loss. Up-scaling the adapter (Sakana-style, tested
0.25→2.0) does **not** recover specificity: matched−mismatch stays at noise at every scale
on the generic checkpoints, and where a small code-residual is exposed (A600), it grows
with scale but **destroys preservation** (0.81→0.45 at scale 1.0). Scaling is a
diagnostic, not a production fix.

## The two lanes (per review methodology)

- **Research bet — adapter-as-trajectory-memory: not met.** matched−mismatched ≈ noise on
  the generic objective; the only non-trivial signal (A600 edit-local) is code-driven, not
  feedback-driven, and collapses preservation when scaled.
- **Product utility — unmeasured; proxy is net-negative.** No pass@1 / generation was run.
  The only proxy is recall logprob, which is **net-negative on full-answer** (matched−zero
  −0.29 → −1.38): the adapter helps the edit region but hurts the code body. Whether the
  generic edit prior helps real pass@1 is **not established**.

## What was built and verified (this is sound)
- Fixed the contrastive hinge to carry gradient through the negative path (was `no_grad`,
  could only lift matched, never suppress the wrong-context adapter), memory-bounded so
  seq stays 768 (`9d03b454`).
- Verified the contrastive term engages: 100% feedback coverage, hard-negatives preserve
  scaffold (median 2.7% token-length delta, no lexical tell), edit-local mask non-empty on
  every row.
- The smoke confirmed the term fires (margin≈1.0) and no OOM — yet matched−swapneg stayed
  ~0, which the probes then explained as cause B, not a loss bug.

## Recommendation (next experiment, with built-in verification)
**Reformulate the distillation target from full-file reproduction to a compact patch /
edit-program (`pre_code → post_code` diff), conditioned on the feedback** — and
**re-run the conditioning probe to confirm the feedback now moves the generated weights.**
Rationale: removes the verbatim-code-cache burden, raises feedback salience from 5% toward
the dominant signal, and reserves adapter capacity for the episode delta. This targets
cause A directly and *tests* whether it relieves cause B (the hypothesis that feedback
attenuates because it is token-sparse). Keep full-code generation as a downstream
integration test, not the core target. Judge by: (i) feedback-swap weight delta rises,
(ii) edit-local matched−swapneg separates, (iii) **without** the preservation collapse seen
under scaling.

If patches do **not** raise feedback→weight movement, the bottleneck is architectural
(perceiver/head capacity, generic-vs-residual factoring, layer placement) and the fix is
representational, not data.

## Artifacts (durable)
- Probes: `tools/diag_conditioning.py`, `tools/diag_weight_sensitivity.py`,
  `tools/diag_recall.py`, `tools/gate_trajectory.py`.
- Gate JSON: `/tmp/rune-ck-trajectory-safe/{gate_smoke_warm_vs_step40,gate_scale_step40,gate_confirm}.json`,
  `recall_a600.log`, `cond_probe_step40.log`.
- Checkpoints (S3): warm-start `checkpoints/hypernet_hpo`, generic run A 200/400/600
  `checkpoints/issue49-final`.
- Full running log: `instructions/scratchpad.md`.
