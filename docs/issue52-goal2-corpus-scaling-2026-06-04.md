# Issue #52 — Goal-2: does scaling the recall corpus (40 → N) keep raising held-out recall?

**Status: COMPLETE (2026-06-04 UTC).** Answer: a modest, sub-threshold YES — point estimates rise on
both metrics and every trained size beats the no-adapter floor, but the per-doubling gain beyond ~80
tasks is within noise at this eval size. Companion: `issue52-goal1-recall-capacity-2026-06-04.md`.

## Design

Pre-registered question (handoff): generalization rose going 10→40 tasks; does *more* disjoint
training keep raising **held-out body accessibility**? Use the recovered train pool.

- **Eval set held FIXED** at the 24-task `mbpp_recall_heldout.jsonl` (sha256 `cae274bf…`) so all sizes
  are comparable.
- **Train sets nested and eval-disjoint:** 40 ⊂ 80 ⊂ 160 (`tools/build_scaling_train_corpora.py`;
  usable pool 206 rows). The existing Phase-1 c3 corpus is exactly the N=40 set.
- **Objective + compute held fixed:** c3 (`body_recall_guarded`, τ=−0.7, λ_p=2, λ_g=1), 48 steps —
  so this isolates *data diversity* at fixed compute (see the compute caveat below).
- **Two metrics on the fixed 24:** (A, pre-registered) absent/body `m_zero` accessibility
  (`_specificity_probe`); (B, secondary) k=1 pass@1 name-cued, spec-absent (`_recall_capacity_probe`)
  vs the scale=0 floor.

## Results

### (A) Accessibility — the pre-registered metric (absent/body `m_zero`, mean ± 95% CI, n=24)

| train size | m_zero | 95% CI | lp_m |
|------------|--------|--------|------|
| warm-start | +0.530 | [+0.370, +0.703] | −1.03 |
| **N=40**   | +0.635 | [+0.432, +0.855] | −0.93 |
| **N=80**   | +0.649 | [+0.445, +0.875] | −0.91 |
| **N=160**  | +0.671 | [+0.476, +0.887] | −0.89 |

Monotonic point-estimate rise (warm < 40 < 80 < 160; +0.141 warm→160), but within-arm CIs overlap
heavily. Paired-by-task deltas beyond 40: **n80−n40 +0.014 [−0.023,+0.051]; n160−n40 +0.036
[−0.0007,+0.070]; n160−n80 +0.022 [−0.008,+0.054]** — every increment spans/just-touches zero. The
n160−n40 trend sits exactly on the significance boundary (suggestive, not confirmed).

### (B) Functional recall — secondary metric (k=1 pass@1 vs scale=0 floor)

| arm | pass@1 | rate | Δ vs scale0 (paired CI) |
|-----|--------|------|-------------------------|
| scale=0 | 5/24 | 0.21 | (floor) |
| warm | 9/24 | 0.38 | +0.167 [0.000, +0.375] |
| **N=40** | 12/24 | 0.50 | +0.292 [+0.083, +0.500] ✓ |
| **N=80** | 15/24 | 0.62 | +0.417 [+0.167, +0.625] ✓ |
| **N=160** | 14/24 | 0.58 | +0.375 [+0.167, +0.583] ✓ |

(✓ = CI excludes 0.) pass@1 jumps 40→80 (+3/24) then plateaus (n160 14/24). Paired beyond-40:
n80−n40 +0.125 [−0.042,+0.292]; n160−n40 +0.083 [−0.083,+0.250]; **n160−n80 −0.042 [−0.125,+0.000]**.
The two metrics *disagree* on the 80→160 step (accessibility +0.022, pass@1 −0.042) — both noise.

### (C) The discriminating comparison — trained vs warm-start (paired, by task)

"Beyond-40" increments are sub-threshold, but the sharper question is whether the *trained* adapter
beats the *warm-start* adapter (i.e., did training on N tasks help at all vs the doc-to-lora prior):

| size | accessibility m_zero − warm | k=1 pass@1 − warm |
|------|------------------------------|-------------------|
| N=40 | +0.105 [+0.033, +0.180] ✓ | +0.125 [−0.042, +0.292] |
| N=80 | +0.119 [+0.049, +0.201] ✓ | **+0.250 [+0.042, +0.458] ✓** |
| N=160 | +0.141 [+0.082, +0.209] ✓ | +0.208 [−0.042, +0.458] |

This is the *more* informative scaling answer: **on accessibility the trained margin over warm is
significant at every size and grows monotonically with N** (+0.105 → +0.141). **On functional pass@1,
N=80 is the first size to reliably beat warm** (CI excludes 0); N=40 does not, and N=160 falls back
just below significance (consistent with its training instability). So: *you need more than 40 tasks
before the trained adapter reliably beats warm-start on functional recall, and ~80 is where it lands.*

## VERDICT

1. **Scaling the corpus keeps nudging held-out recall up — modestly.** On the pre-registered
   accessibility metric the rise is monotonic (0.635 → 0.649 → 0.671); pass@1 corroborates (jump then
   plateau). Every trained size beats the no-adapter floor with CI excluding 0, and N=80/160 beat it
   by more than N=40 / warm.

2. **But gains beyond ~80 tasks are within noise at n=24.** No paired beyond-40 increment is
   individually significant. This is **diminishing returns** relative to the larger 10→40 jump the
   handoff reported. **N=80 ≈ N=160 (tied)** — do not read "monotonic" as "160 wins"; the metrics
   disagree on that step and both deltas are sub-threshold.

   *The sharper, significant story is trained-vs-warm (§C):* accessibility-over-warm is significant
   at every size and grows monotonically (+0.105 → +0.141), and **N=80 is the first size whose
   functional pass@1 reliably beats warm-start.** So "more than 40 tasks, ~80, before training
   reliably beats the prior" is the real scaling answer — not a flat null.

3. **The binding constraint is eval power, not data.** The n160−n40 accessibility CI half-width is
   ≈0.035; resolving a ~+0.04 true effect needs ~4× the eval set, which competes with the training
   pool. A single retrain cannot settle it.

## Why no compute-matched N=160 control was run (free diagnostic)

Fixed 48 steps means larger corpora got fewer passes — N=160 could be under-trained. Resolved for
zero compute by reading training-time slope (early steps 5–20 vs late 30–48 of `distill_metrics`):

| size | loss Δ | lp_matched Δ | reading |
|------|--------|--------------|---------|
| N=40 | −0.75 | +0.01 | flattened (converged) |
| N=80 | −1.53 | +0.54 | still improving — used compute well (and won eval) |
| N=160 | +0.18 | −0.36 | **unstable** — late metrics *degraded*, not "still improving" |

A compute-matched run is justified only when the slope shows *clear continued improvement* (advisor
rule). N=160 does not — it is unstable at fixed compute, not productively under-trained — and it still
**leads** on accessibility (0.671), so it is not badly starved. Spending was therefore declined.

## Limits

- **n=24 eval is underpowered** for the ~+0.03 m_zero / ~+2-per-24 increments at stake.
- **Single seed per size** — the 80-vs-160 wiggle could be training-seed noise.
- **Fixed 48-step compute (confound attenuated, not eliminated)** — at this budget the practical sweet
  spot is ~80 tasks; 160 diverse tasks are harder to fit stably. The slope diagnostic shows N=160
  *unstable* (not merely fewer epochs), but you cannot fully separate "160 tasks need more steps" from
  "160 tasks hurt at this lr/steps" without one matched-epoch ablation (e.g. N=160 vs N=80 at ~2× steps
  on N=80). Untested.
- **Scope — Goal-2 does NOT repair the `diff` retention gap.** Corpus scaling optimizes MBPP *body*
  recall; it does nothing for the edit-local `diff` m−zero (−0.028) on `external_codereview` — a
  different target distribution. Do not let Goal-2 progress imply the diff/edit-local channel is fixed.
- **N=160's accessibility lead (0.671) coexists with unstable training loss** — not a contradiction:
  held-out accessibility can lead even when the training-loss trajectory is noisy. But weight the
  0.671 accordingly.
- **Cross-session reuse control:** the N=40 accessibility (0.635) is reused from the Phase-1 session.
  This is validated, not a soft spot: warm-start reproduced at **exactly 0.530** this session, and
  n40−warm = **+0.105 [+0.033, +0.180]** reproduces the Phase-1 headline (+0.105 [+0.033, +0.182])
  almost exactly — same probe surface, same model/dtype.

## Reproduce

```
uv run python tools/build_scaling_train_corpora.py --sizes 80,160     # nested, eval-disjoint
bash tools/_run_goal2_train.sh   # train c3-objective N=80,160 (48 steps) -> MLflow exp issue52-goal2-scaling
uv run python tools/_fetch_goal2_ckpts.py                              # pull ckpts from MLflow/S3
bash tools/_run_goal2_eval.sh    # spec + cap(k=1) on the fixed 24, arms warm/n80/n160
uv run python tools/_goal2_analysis.py                                 # scaling table + CIs
```

corpus `mbpp_recall_train_{80,160}.jsonl` (eval-disjoint, nested over N=40). All `_`-prefixed tools
are REMOVE-BEFORE-MERGE issue-52 scaffolding. MLflow experiment `issue52-goal2-scaling`.

## Implication for Phase-2 (Goal 3, still deferred)

Corpus scaling is a real but *diminishing* lever; it is not the path to a large recall jump on its
own. Combined with Goal-1 (single-step memory real at k=1), the higher-value Phase-2 levers are
**(a)** the runner-based multistep test — does the single-step hypernet effectively carry *within-run*
stepwise context (repair / code-continuation / development) so it need not live in the prompt, vs
scale=0 — and **(b)** steadier optimization for larger corpora. **NOT** training the hypernet on
multi-task/packed conditioning: the hypernet is single-step by design; the multistep value lives in
the runner's iterative generation, not in storing a library of independent tasks. (Corrects an earlier
mischaracterization in the Goal-1 doc.)
