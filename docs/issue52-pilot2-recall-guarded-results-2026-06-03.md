# Issue #52 — Pilot 2: matched-recall guarded objective — accessibility PASS, no recitation (2026-06-03, us-east)

Follow-up to `docs/issue52-crossover-frozen-probe-results-2026-06-03.md` (pilot 1 =
`body_derangement`, which moved the margin via suppression, not accessibility → not a clean
GO). Pilot 2 redesigns the objective per that doc's recommendation and the AI-engineer
review, and **passes the predeclared accessibility criterion**.

## TL;DR — PASS on ACCESSIBILITY (criteria 1, 2, signature); recitation/utility OPEN

The guarded matched-recall objective raises **matched body accessibility** (the lever pilot 1
could not move): absent/body `lp_matched` −1.026 → −0.736, **Δ +0.290, sign +9/−1 p=0.021,
bootstrap CI [+0.130, +0.448] (excludes zero)**. Signature retained (+3.84 → +5.71). This is
trained-on-test trainability, not generalization.

**Recitation (the user's concern) is RESOLVED for this objective: no drift.** The cross-conditioned
test (adapter on episode *i*, episode *j*'s spec in the prompt) shows the trained adapter follows
the prompt spec **9/10 and recites 0/10 — identical to warm-start**. The +0.290 accessibility gain
came with zero recitation drift. (The absent valid-code rise 0→0.875 does NOT itself test
recitation — see `## Recitation`. Full xgrammar **pass@1 correctness** remains a separate unrun gate.)

## Objective (vs pilot 1)
```
L = L_distill(KL+CE)                         # retention
  + lambda_p * mean relu(target - lp_matched)        # PRIMARY: raise matched toward target (tau=-0.7)
  + lambda_g * mean relu(lp_mismatch - lp_n0)        # GUARD: hold deranged at frozen warm-start lp_n0
```
`lp_n0` precomputed once under the pristine warm-start (deterministic (i+1)%n derangement,
matching the frozen probe). No term rewards `lp_mismatch` falling → no suppression incentive.
Code: `contrastive_mode="body_recall_guarded"` in `src/rune/training/hypernet_distill.py`;
config `configs/issue52_body_recall_crossover_4b.yaml`; 30-step bf16 cross-over on the 10
frozen MBPP bodies. MLflow exp `issue52-body-recall` (44), run `7b82c304`.

## Result vs the predeclared criterion

| criterion (frozen before deltas) | body_derangement | **recall_guarded** | verdict |
|---|---|---|---|
| 1. Δ`lp_matched` absent/body material, CI excludes 0 | +0.075, CI [−0.21,+0.27] | **+0.290, CI [+0.130,+0.448]** | ✅ |
| 2. Δ`lp_mismatch` does not dominate the gain | ~92% (11:1 suppression) | ~52% (~1:1) | ✅ |
| 3. recitation (cross-conditioned) | unmeasured | **no drift: 0/10 recite (= warm-start)** | ✅* |
| + signature retained (absent/sig) | +3.84→+5.95 | +3.84→**+5.71** | ✅ |

*Criterion 3: the recitation-specific concern is resolved (cross-conditioned generation does not
override the prompt — see `## Recitation`). The absent valid-code number (0→0.875) only rules out
gross degeneration. Full xgrammar **pass@1 correctness** is a separate, still-unrun gate;
`ast.parse`/entry_point checks are floors, not correctness.

absent/body decomposition (`lp_zero` constant → Δm−zero = Δmatched):

| | lp_matched | lp_mismatch | lp_zero |
|---|---|---|---|
| warm-start | −1.026 | −1.163 | −1.866 |
| recall_guarded | −0.736 | −1.477 | −1.866 |
| **Δ** | **+0.290** | −0.314 | 0.000 |

Per-episode body m−mismatch +0.137 → +0.742 (sign +10/−0 p=0.002, CI [+0.347,+0.896]);
frac>0 0.60 → 0.90. Snapshot trajectory (frozen-probe surface, steps 0→30):
`snap_absent_matched` −0.498→−0.163 monotone; `snap_absent_m_zero` +0.640→+0.975.

## Why this is a real accessibility win (not pilot 1's suppression)
- `guard_active_frac` stayed **0** — the deranged partner never rose above its warm-start
  baseline, so the guard never fired. The mismatch fell as **emergent specificity** from the
  primary recall objective (optimizing matched-recall incidentally degrades cross-conditioning),
  NOT from an explicit suppression term. The win is driven by matched rising (CI excludes 0).
- The signature span confirms the gradient raises matched without trading away labels.

## Recitation (the user's steer) — OPEN, decisive test pending
Absent-regime valid-Python rose 0.0 → 0.875, but this does NOT test recitation (advisor):
absent prompts ask for nothing specific, so reproducing the stored body is *both* the goal and
valid Python — recitation would score high too. It rules out gross degeneration only. The
present-regime body-lp rise (+0.208) is likewise undiagnostic: adapter-content and prompt-spec
are the SAME MBPP task on this corpus, so they never diverge and there is nothing to override.

**Decisive test (no retrain — uses the existing checkpoint): cross-conditioned present-regime
generation** (`tools/_recitation_probe.py`). Condition the adapter on episode *i*, put episode
*j*'s spec in the prompt; HEALTHY = output defines *j*'s entry_point (followed the prompt),
RECITATION = output defines *i*'s stored entry_point (adapter overrode the prompt).

### Cross-conditioned recitation result — NO recitation induced
| checkpoint | spec-follow (j) | recite (i) | neither |
|---|---|---|---|
| warm-start | 9/10 | **0/10** | 1 (mbpp/19) |
| recall-guarded trained | 9/10 | **0/10** | 1 (mbpp/19) |

The trained adapter follows the in-prompt spec 9/10 and **never recites** the stored body —
identical to warm-start (same lone outlier). The +0.290 accessibility gain came with **zero
recitation drift**. So criterion 3 holds at the spec-following level: raising body recall did
not push the base into recitation/Q&A mode. (Caveat: this is "which function did it write,"
not full **pass@1 correctness** — xgrammar pass@1 is still a separate, unrun gate.)

## Caveats (do not over-claim)
- Trained-on-test **trainability** probe, not generalization; 10-episode MBPP body micro-probe
  = the **continuation** facet (recall-friendly; correct-next ≈ stored body). The
  tried-and-failed/avoid facet still needs a corrected-code target (failure accessible, not
  reproducible).
- mbpp/17 still negative (−0.21); mbpp/19 is the lone matched-delta regression (−0.26) — same
  outlier as pilot 1; watch under corpus expansion.
- Retention (goal/file/diff/code-recall) NOT measured by this probe (deferred — matters before
  the long run). Present-regime generation canary NOT run.

## Next (gated)
1. Add the **present-regime generation canary** (spec-compliance under in-prompt task) to fully
   close the recitation question.
2. **Retention gates** (goal/file/diff/code-recall not regressed) + xgrammar pass@1 in an
   absent-regime eval.
3. **Corpus-quality gate** (real failure-bearing trajectories, in-prompt ceiling, positive
   controls, provenance) — this, not the micro-probe, launches the long run.
4. Tune `tau`/`lambda` and sweep the recall↔spec tradeoff; then short full run → HPO.

## Merge hygiene
All pilot-2 code (`body_recall_guarded` branch, `_recall_*`/`_snap_*` helpers, config, the
body-derangement scaffolding) remains REMOVE-BEFORE-MERGE issue-52 probe code. Promote
deliberately with its own tests/docs only after generalization is shown. (Keep the
`sb0.detach()` + `_artifact_uploaded` size-check bug fixes — mergeable alone.)
