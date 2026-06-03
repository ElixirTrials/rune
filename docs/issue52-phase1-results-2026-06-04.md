# Issue #52 — Phase-1 results (recall objective: trainability → generalization → bench)

**Status: LIVE — Phase-1 HPO running overnight (2026-06-03/04 UTC). This doc + the PR comment
update as configs land.** Durable companions: `issue52-crossover-frozen-probe-results-2026-06-03.md`
(pilot 1), `issue52-pilot2-recall-guarded-results-2026-06-03.md` (pilot 2).

## The arc so far

1. **Pilot 1 (`body_derangement`): QUALIFIED, not GO.** absent/body m−mismatch +0.137→+1.026,
   but 91% deranged-suppression, accessibility (lp_matched) flat (+0.075, CI spans 0).
2. **Scaling audit:** the +0.137 body floor is **real encoding**, not an `alpha/r` apply bug
   (dispositive: sig +3.84 vs body +0.137 in one forward pass).
3. **Pilot 2 (`body_recall_guarded`): accessibility PASS on the trained-on-test 10.**
   Δlp_matched **+0.290**, bootstrap CI **[+0.13,+0.45] excludes 0**; mismatch ~1:1 (not
   dominating); signature retained; guard never fired (emergent specificity). Cross-conditioned
   recitation: **0/10 recite = warm-start** (no drift). Functional pass@1 (real MBPP 3-test
   suites): **present 6→10/10, absent 1→5/10** (scale=0≈0).
4. **Generalization check (the decisive Phase-1 question): the pilot-2 ckpt does NOT generalize.**
   On 24 **held-out** disjoint tasks: absent/body m−zero (accessibility) warm **+0.530** →
   trained **+0.514** (flat); m−mismatch +0.127→+0.165. Signature *did* generalize (+3.56→+4.95).
   ⇒ training on the 10 was **memorization**, not a transferable "encode any body accessibly"
   skill. (This is why held-out data had to come first.)

## Phase-1 experiment (running)

**Question:** does training on a **40-task** held-out-disjoint split (`mbpp_recall_train.jsonl`)
generalize body accessibility to the **24-task** held-out eval (`mbpp_recall_heldout.jsonl`)?

**Pipeline** (`tools/_phase1_orchestrate.py`, unattended, incremental):
- Train-HPO grid (4 configs over τ ∈ {−0.7,−0.5}, λ_p ∈ {1,2}, λ_g ∈ {1,2}), 48 steps each on
  the 40-task split.
- Eval each on held-out 24 (accessibility: Δlp_matched, gated on signature retention).
- Pick best → **pass@1 vs scale=0** on held-out: {scale=0 (base), warm-start adapter,
  best-trained}, present + absent regimes (real MBPP tests).

**Baseline (warm-start, held-out 24):** absent/body m−mismatch +0.127, m−zero +0.530,
lp_matched −1.030, sig +3.564.

### Train-HPO trials (held-out 24 accessibility) — updating
Warm-start held-out baseline: m−zero **+0.530**, lp_matched −1.030, sig +3.56. (Pilot-2 ckpt
trained on the 10 was FLAT on held-out: +0.514.)
| config | τ | λ_p | λ_g | held-out absent/body m−zero | Δ vs warm | lp_matched | sig retained |
|---|---|---|---|---|---|---|---|
| c1 | −0.7 | 1 | 1 | **+0.593** | **+0.063** | −0.967 | +5.16 ✓ |
| c2 | −0.5 | 1 | 1 | **+0.601** | **+0.071** | −0.959 | +5.15 ✓ |
| c3 | −0.7 | 2 | 1 | **+0.635** | **+0.105** | −0.925 | +5.26 ✓ |
| c4 | −0.5 | 2 | 2 | _pending_ | | | |

**Early read:** training on 40 disjoint tasks produces a *small but positive* held-out
generalization (+0.063 m−zero) where training on 10 produced none — directional evidence the
recall objective transfers a little with more data. Modest vs the +0.290 trained-on-test;
significance on n=24 to be assessed.

### Bench pass@1 (held-out 24) — pending best
| arm | present (stability) | absent (capability) |
|---|---|---|
| scale=0 (base) | _pending_ | _pending_ |
| warm-start adapter | _pending_ | _pending_ |
| best-trained | _pending_ | _pending_ |

## Honest read (either outcome is informative)
- **If held-out accessibility moves** (m−zero materially > +0.530, CI excludes 0) **and pass@1
  beats scale=0 in the absent regime** → the recall objective is a *generalizing* lever; Phase 1
  passes; the checkpoint becomes the Phase-2 retention baseline.
- **If held-out stays flat** → the objective *memorizes* regardless of corpus size — a deeper
  limit (more data / different conditioning needed), and Phase 2 RL would be built on sand. Do
  not proceed to Phase 2.

## Method caveats (carried)
n is small (24 held-out); pass@1 is raw greedy + real tests, **not** xgrammar-constrained engine
pass@1; "absent" single-shot is a proxy for the memory regime (the deeper eval is the engine
running multi-step with state evicted to the adapter — fixed prompt, growing adapter memory).
Retention (goal/file/diff/code-recall via `diag_recoverability`) is the other half of gate 1,
still to run. All issue-52 probe code is REMOVE-BEFORE-MERGE scaffolding.
