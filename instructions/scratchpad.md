# Issue #52 scratchpad — body-contrastive cross-over go/no-go (2026-06-03, us-east)

Live working notes: experiments, results, interpretations, plans. **Convention: APPEND new
notes to the BOTTOM, each with a `### [YYYY-MM-DD HH:MM UTC]` timestamp.** Durable writeups:
`docs/issue52-crossover-frozen-probe-results-2026-06-03.md` (pilot 1) and
`docs/issue52-pilot2-recall-guarded-results-2026-06-03.md` (pilot 2). Feedback channel:
`instructions/reflections.md` (AI engineer — I monitor it for changes).

**## Current verdict (latest):** Pilot 2 `body_recall_guarded` = accessibility PASS + recitation
clean. Authoritative state + plan = the **latest timestamped block at the BOTTOM** of this file.

## Task
Resume the pending go/no-go from `instructions/handoff.md` §8: does a 30-step body-span
contrastive (derangement) fine-tune of the warm-start hypernet *move* the absent/body
m−mismatch (the code-body recall floor, +0.137) when re-scored by the FROZEN E1 probe?

## What I did (chronological)
1. **Orientation.** Confirmed warm-start ckpt present (sha `6438b46c…` ✓), GPU idle (L4 23GB),
   gpu env synced (torch 2.12+cu130, transformers 5.9, peft 0.19.1). `/workspaces/rune-gpu`
   is a symlink → `/workspaces/content` (the A.2 path fix already in place).
2. **Missing probe deps.** Probe `tools/_specificity_probe.py` needs `scoring_core` +
   `benchmarks/mbpp_phase0_iter.json`. Initially thought both lost. → `scoring_core` was at
   `tools/scoring_core.py` (committed cef27c7; I'd looked in the wrong subdir). Only
   `mbpp_phase0_iter.json` genuinely missing (not in any commit, gitignored, lost on recycle).
3. **Reconstruction + GATE A.** Rebuilt `mbpp_phase0_iter.json` from the committed corpus
   (`configs/issue52_mbpp_body_crossover.jsonl`); description recovered by stripping the
   render scaffold; **byte-exact round-trip** `render_training_format_trajectory(desc)==context`
   for all 10. (`tools/_reconstruct_mbpp_phase0.py`.) Why it matters: surface drift would hit
   the *trained arm only* → false NULL.
4. **Parity gate (A.5).** Re-ran the 30-step probe under guards, bf16. Reproduced `c401f0c0`
   to 4 decimals; trained `checkpoint_step30.pt` sha matched historical. MLflow→S3 works here
   (checkpoints upload + auto-delete local; I fetch step30 to /tmp transiently).
5. **GATE B.** Fresh warm-start frozen probe → absent/body **+0.1370** (hist +0.137),
   absent/sig **+3.8365** (hist +3.8–4.09). Reconstruction is faithful.
6. **Trained probe + decision.**

## Results (frozen probe, absent regime — the decisive one)
| span | warm m−mismatch | trained m−mismatch | warm m−zero | trained m−zero |
|---|---|---|---|---|
| body | **+0.137** | **+1.026** | +0.840 | +0.915 |
| sig  | +3.837 | +5.948 | +5.245 | +6.121 |

Body m−mismatch: +10/−0, sign p=0.002, bootstrap CI [+0.558,+1.299].

## THE decomposition (lp_zero identical across runs ⇒ Δm−zero = Δlp_matched, exact)
absent/body: lp_matched −1.026→−0.951 (**+0.075**), lp_mismatch(deranged) −1.163→−1.977
(**−0.814**), lp_zero −1.866 (0). → the +0.889 m−mismatch gain is **91% deranged-suppression,
8% matched-rise**.

absent/sig: lp_matched −2.284→−1.409 (**+0.875**), lp_mismatch −6.121→−7.356 (−1.236). →
matched genuinely rises for the signature (proves the gradient CAN raise matched).

Accessibility stat-test (lp_matched / m−zero per-episode delta): mean +0.075, sign **+9/−1
p=0.021**, bootstrap CI **[−0.212,+0.274] spans zero** (dragged by mbpp/19 −1.075). →
directionally broad, magnitude negligible.

## Interpretation / verdict
**QUALIFIED — gradient-reachable but NOT a clean GO.** The contrastive body objective opens
the specificity margin massively + reproducibly, but ~entirely by **suppressing the wrong
episode's body** (discriminability), not by **recalling the right one** better (accessibility,
which is flat). This is the exact "margin movement, not matched rising" the handoff flagged
as not-the-desired-signal. Not a NULL (metric moved hugely); not the generic-boost FAIL
(mismatch fell, didn't rise with matched); a third under-specified outcome.

## Plan (next, gated — iterate the pilot, NOT the long run)
- Redesign objective: reward **matched rising while mismatch is held** (joint accessibility +
  specificity). NB pitfall (advisor): "optimize m−zero" alone walks into the CE generic-boost
  confound (base fixed ⇒ raises matched AND mismatch). Target the joint condition, not m−zero.
- Add **xgrammar pass@1** as a first-class pilot gate: specificity-via-suppression risks
  *destructive interference* on non-matched content. NOT yet measured.
- Retention gates (goal/file/diff/code-recall) + generation-stability NOT evaluated by this
  probe — deferred, name them as such.
- Only a matched-body accessibility move → corpus-quality gate → short full run → HPO.

## AI-engineer feedback (reflections.md 2026-06-03) — received + incorporated
- Verdict endorsed; framed as **objective misspecification** (not hypernet/probe fault).
- **Reconciles the advisor's m−zero warning:** m−zero CAN be primary IF the derangement
  hinge is kept as a **guard** (penalize when lp_mismatch rises or gain comes with Δmatched≤0).
  Unguarded m−zero = generic-boost trap; guarded = safe. → updated doc recommendation.
- Train-loop (matched↑ mismatch↓) vs frozen-probe (91% suppression) GAP: causes = hinge on
  easy negs / readout-token diffs / 30-step loss-surface overfit. Action: next pilot logs
  BOTH in-loop readout AND a cheap frozen-probe snapshot every N steps. (added to doc)
- Mechanism: suppression = gradient path of least resistance (deranged = strong stable neg;
  raising matched competes w/ warm-start binding, fewer high-surprisal toks than sig). (added)
- mbpp/19 (−1.075 accessibility) flagged: if one episode drives suppression-heavy win, corpus
  expansion matters as much as loss shape. (added)
- pass@1 = first-class pilot gate (destructive-interference risk); retention deferred OK only
  while not promoting. (added, named explicitly)

## Scaling-bug audit (2026-06-03, user challenge: "is +0.137 a rune scaling artifact?")
VERDICT: NO. The +0.137 body floor is real, not the 8×-too-weak `alpha/r` bug.
- Runtime: `effective_scaling = lora_alpha = 45.2548` un-divided (old bug = alpha/r = 5.66).
  Probe uses this. use_bias=True (head bias in combine_lora), r=8, 36 layers.
- Code identity: rune `_lora_delta` == ctx_to_lora native `lora_forward` (same einsum
  `(x@Aᵀ)@B`, same `* scaling`). `_parity_engine_vs_functional.py` (scratch) proved
  engine PEFT apply == functional apply on this exact qwen ckpt.
- DISPOSITIVE: sig m−mismatch +3.84 vs body +0.137 are scored in the SAME forward pass,
  SAME scaling, SAME adapter. A scaling bug is a global delta multiplier → cannot make
  sig specific while body is flat. The asymmetry = what the hypernet ENCODES, not apply.
- The "+7 high mismatch" the user recalls = Gemma Doc2LoRA NIAH calibration (pure Sakana,
  doc facts, ~100% recall), a DIFFERENT base+ckpt+fact-type. Not the qwen body number.
ARCHITECTURE (user reaffirmed): prompt must NOT grow per iteration → absent regime IS the
design, so absent/body +0.137 is THE operative number. Retract my earlier "is engine
in-prompt?" conditional. Raising absent/body accessibility is the core mechanism, not optional.

## NEXT PILOT DESIGN — matched-recall guarded objective (2026-06-03, option b)

Problem the redesign fixes: `body_derangement` opened the margin by suppressing the deranged
partner (−0.814) while matched body held (+0.075). We want matched-body lp to RISE
(accessibility) with mismatch HELD (specificity), no suppression reward, no generic boost.

### Loss (per body token i in the matched episode's body span)
lp_m = log p(body_i | matched adapter); lp_n = log p(body_i | deranged-partner adapter);
lp_n0 = deranged body lp under the WARM-START hypernet, precomputed once at step 0 (frozen);
τ = matched target lp.

```
L = L_distill(KL+CE on matched answer)        # retention: keep sig/goal/file/diff bindings
  + λ_p · mean_i relu(τ − lp_m_i)             # PRIMARY: raise matched toward τ (hard tokens only)
  + λ_g · mean_i relu(lp_n_i − lp_n0_i)       # GUARD: hold deranged ≤ warm-start (anti generic-boost)
```

Why this hits the criterion:
- No term rewards lp_n FALLING → suppression has zero incentive (relu silent below baseline).
- Primary acts only while lp_m < τ → focuses gradient on low-recall body tokens, stops at τ.
- Guard acts only while lp_n > baseline → catches generic boost (matched+mismatch rise together).
- Specificity margin becomes EMERGENT (τ − lp_n0), not the optimized quantity. ⇒ a win can
  only come from matched rising with deranged held = exactly the predeclared success criterion.

Defaults (first guess; the 30-step cross-over shows if matched moves): τ=−0.5 (between
warm-start −1.0 and oracle −0.22), λ_p=1.0, λ_g=1.0, distill weight unchanged.

### Implementation (minimal, reuse existing structure — NO greenfield gradient path)
1. `DistillConfig`: `contrastive_mode="body_recall_guarded"` (keep `body_derangement` for A/B);
   `matched_target_lp=-0.5`, `guard_weight=1.0`, `primary_weight=1.0`.
2. Precompute `lp_n0` per episode at step 0 (warm-start deranged body lp): reuse
   `_generate_lora_dict`+`_student_logits`+`_gold_logprobs` over `_deranged_partner_context`
   + `_body_span_mask`. Freeze in a dict keyed by task_id.
3. New branch in the contrastive block: L_primary (grad through matched), L_guard (grad through
   neg), backward sequentially (memory-bounded, like the current neg piece). scaler_B guard intact.
4. Periodic frozen-probe snapshot every `frozen_probe_steps` on K=5–10 episodes (absent/body
   Δlp_matched, Δlp_mismatch) → MLflow, so we tune against the deciding metric (engineer ask).
5. `configs/issue52_body_recall_crossover_4b.yaml`; unit tests (relu directions, baseline frozen,
   guard=0 when deranged≤baseline, primary=0 when matched≥τ).
6. Validate on the SAME 10-episode cross-over first (cheapest trainability probe, directly
   comparable). Only a matched-rise → corpus-quality gate → short full run.

Alternative considered (engineer's Option B): `max(0,margin−(lp_m−lp_n)) + λ·(−lp_m)`. Rejected
as primary because the hinge still rewards suppression; our guarded form removes that reward.

### RECITATION RISK (user steer 2026-06-03) — recall ≠ generation utility
Sakana recall worked because train==inference was Q&A (internalize→answer). Ours is NOT: the
base must GENERATE CODE, not recite the trajectory. Maximizing verbatim body lp can drive the
base into recitation/Q&A mode that overrides the task (scratch adapter-as-memory-report:
spec-divergence at adapter scaling ≥0.49). Recall ↑ can come WITH pass@1 ↓.

Implications baked into the design:
- **Target the DESIRED GENERATION, not verbatim recall.** Loss target = the correct next
  action conditioned on the adapter (absent). Facet-dependent:
  - continuation: correct next ≈ the body ⇒ recall and generation coincide (so the 10-MBPP
    body cross-over is a valid FIRST probe, but recall-friendly by construction — don't
    over-generalize a win here to the retry facet).
  - tried-and-failed/avoid: correct next ≠ the stored failure ⇒ adapter makes the failure
    ACCESSIBLE not REPRODUCIBLE; target = corrected code. Needs a generation-target objective,
    NOT verbatim recall. Design when real failure-bearing trajectories exist.
- **τ MODEST (accessibility floor, not recitation ceiling).** relu(τ−lp_m) already stops at τ;
  keep τ conservative (start −0.7, not the oracle −0.22) and treat the recall↔spec-compliance
  tradeoff as a SWEEP, not a fixed point.
- **Generation is CO-PRIMARY, not just a gate.** Every snapshot scores generation quality
  (valid/correct code) in BOTH regimes: absent (does the adapter help produce correct code?)
  AND present (does raising absent-recall CORRUPT present-regime spec-following? = the
  recitation-dominance canary). A recall win that drops generation is a FAIL.

## IMPLEMENTED (2026-06-03) — body_recall_guarded objective + detailed MLflow logging
- `DistillConfig`: `contrastive_mode="body_recall_guarded"`, `matched_target_lp=-0.7`,
  `primary_weight`, `guard_weight`, `snapshot_steps`, `snapshot_episodes`.
- Loss helpers: `_recall_terms` (pure relu directions, unit-tested), `_recall_guarded_term`
  (primary grad-through-matched + readout; returns guard_pending), post-backward GUARD piece
  (grad-through-deranged, relu vs frozen lp_n0). `_precompute_recall_baselines` freezes lp_n0
  under warm-start at step 0 (deterministic (i+1)%n derangement, matches probe).
- Per-step MLflow (via contrastive_metrics): lp_matched/mismatch/zero, lp_n0_baseline,
  primary_loss, guard_loss, primary_active_frac, guard_active_frac, recall_m_mismatch.
- `_recall_snapshot` (steps 0/10/20/30): FROZEN-PROBE surface body lp matched/mismatch/zero in
  absent+present regimes (snap_absent_m_mismatch, snap_absent_m_zero, snap_present_*) +
  valid-code generation rate (snap_gen_valid_absent). Bridges the in-loop-vs-probe gap.
- Tests: 4 new unit tests (relu directions, config, empty metrics); 325 unit pass; ruff+mypy clean.
- config: `configs/issue52_body_recall_crossover_4b.yaml`. Running 30-step pilot now.
- ALL REMOVE-BEFORE-MERGE (issue-52 probe scaffolding) unless the pilot validates the lever.

## PILOT 2 RESULT (run 1, MLflow exp 44 run cd63226d) — ENCOURAGING (gen canary pending)
Frozen-probe-surface snapshot trajectory (absent/body), steps 0->30:
- snap_absent_matched  -0.498 -> -0.281 -> -0.180 -> -0.163  (Δ +0.335 ACCESSIBILITY ROSE)
- snap_absent_mismatch -0.660 -> -0.989  (Δ -0.329)
- snap_absent_zero     -1.138 constant (base, correct)
- snap_absent_m_zero   +0.640 -> +0.975  (Δ +0.335 = Δmatched, the clean accessibility gain)

vs body_derangement: matched Δ+0.075 / mismatch Δ-0.814 (11:1 suppression). Guarded:
matched Δ+0.335 / mismatch Δ-0.329 (~1:1). Redesign moved the RIGHT lever — matched
accessibility rose ~4.5x, suppression dropped from dominant to balanced. guard_active_frac=0
throughout (deranged never rose above baseline; guard never needed to fire — primary recall
term did the work). loss 5.4->0.055.

CAVEATS (do NOT call a win yet):
- GEN CANARY FAILED on a bug (empty-answer float upcast -> embedding rejects float ids).
  Fixed (_snap_full dtype=long); re-running. The recitation safety check = user's key concern
  = exactly what was unmeasured.
- present-regime matched ALSO rose (+0.208) -> possible recitation creep; gen canary adjudicates.
- snapshot = 8-episode MEANS; need per-episode frozen-probe CI + signature retention (run
  tools/_specificity_probe on checkpoint_step30.pt + _crossover_decision).
- trained-on-test trainability, not generalization.

NEXT: gen-fixed re-run -> frozen probe + decision on new ckpt (per-episode CI, sig retention,
recitation via gen valid-code absent+present). THEN assess vs full predeclared criterion.

## PILOT 2 (run 2, gen fixed, MLflow run 7b82c304) — ACCESSIBILITY PASS, recitation OPEN
Frozen probe on trained ckpt vs warm-start (authoritative per-episode):
- absent/body lp_matched -1.026 -> -0.736 (Δ +0.290), lp_mismatch -1.163 -> -1.477 (Δ -0.314).
- ACCESSIBILITY: Δlp_matched +0.290, sign +9/-1 p=0.021, bootstrap CI [+0.130,+0.448] EXCLUDES 0.
  => PASSES criterion 1 (body_derangement failed it: +0.075, CI spanned 0).
- m-mismatch +0.137->+0.742; signature retained +3.84->+5.71. guard_active_frac=0 (emergent
  specificity, not explicit suppression). gen valid-code absent 0->0.875.

ADVISOR CORRECTION (important): criterion 3 (recitation) NOT met by absent valid-code — in the
absent regime reproducing the body IS the goal AND valid python, so recitation scores high too.
Recitation only shows when adapter-content & prompt-intent DIVERGE (never on same-task corpus).
ast.parse != xgrammar pass@1 (syntactic floor, not correctness). snapshot lp_matched (-0.163)
vs frozen-probe lp_matched (-0.736) disagree ~0.57 (different surfaces) — verdict rests on
frozen probe; don't mix surfaces in reporting.
=> Verdict: PASS on ACCESSIBILITY (crit 1,2,sig); recitation/utility OPEN.

DECISIVE recitation test (tools/_recitation_probe.py, no retrain): adapter=i + prompt-spec=j,
does gen follow j (healthy) or recite i's stored body? Running warm-start vs trained now.

## ============ CONSOLIDATED STATE (2026-06-03, end of session) ============

### FINDINGS (what we established, in order)
1. **Go/no-go (pilot 1, body_derangement) = QUALIFIED, not clean GO.** Frozen probe: absent/body
   m-mismatch +0.137 -> +1.026, but decomposition = 91% deranged-SUPPRESSION, 8% matched-rise;
   accessibility (lp_matched) flat (+0.075, CI spans 0). Margin moved, accessibility didn't.
   Triply confirmed (me + advisor + AI engineer). Doc: docs/issue52-crossover-frozen-probe-
   results-2026-06-03.md. Parity bit-exact (sha d296a4e2 == historical); GATE A/B passed.
2. **Scaling-bug audit: the +0.137 is REAL, not an alpha/r artifact.** effective_scaling=45.25
   (un-divided), rune _lora_delta == ctx_to_lora native lora_forward. Dispositive: sig(+3.84) vs
   body(+0.137) in ONE forward => encoding asymmetry, not apply bug. The "+7 high mismatch" =
   Gemma NIAH (different base/ckpt/facts), not qwen body.
3. **Architecture (user): prompt does NOT grow => absent regime IS operative.** absent/body is THE
   metric; present-regime numbers irrelevant to the product. Raising absent accessibility = core.
4. **PILOT 2 (body_recall_guarded) = ACCESSIBILITY PASS.** Δlp_matched +0.290, sign +9/-1 p=0.021,
   bootstrap CI [+0.130,+0.448] EXCLUDES 0 (crit 1 PASS; pilot 1 failed it). mismatch -0.314
   (~1:1, not dominating, crit 2 PASS). signature retained +3.84->+5.71. guard never fired
   (emergent specificity). Doc: docs/issue52-pilot2-recall-guarded-results-2026-06-03.md.
   MLflow exp 44 run 7b82c304.
5. **Recitation (user steer) — RESOLVED: no drift.** Cross-conditioned gen (adapter=i,
   prompt-spec=j): WARM-START 9/10 follow, 0/10 recite. TRAINED recall ckpt **identical: 9/10
   follow, 0/10 recite** (same mbpp/19 neither outlier). The +0.290 accessibility gain induced
   ZERO recitation. Absent valid-code 0->0.875 does NOT itself test recitation (advisor). Full
   xgrammar pass@1 CORRECTNESS still a separate unrun gate (entry_point check is a floor).

### INTERPRETATION
- The guarded matched-recall objective is the RIGHT lever: it raises body ACCESSIBILITY (what the
  product needs) where the contrastive-hinge (pilot 1) only opened a margin by suppression.
- The win is modest in magnitude (lp_matched -1.03 -> -0.74; oracle is -0.22, real NIAH ~+7.7) and
  trained-on-test (trainability, not generalization) on the recall-FRIENDLY continuation facet.
- Recitation is the live risk the accessibility number can't see; the cross-conditioned test is the
  one that bites. [SUPERSEDED by 19:30 UTC] Warm-start clean AND trained 0/10 recite => crit 3 (recitation) RESOLVED, no drift.
- ast.parse valid != xgrammar pass@1 (syntactic floor, not correctness).

### PLAN (gated, in order)  [SUPERSEDED — see the latest timestamped block at the BOTTOM for the current plan]
1. [DONE — 0/10 recite, no drift] Trained recitation result read; crit 3 resolved.
2. Retention gates (goal/file/diff/code-recall not regressed) + real xgrammar pass@1 in absent eval.
3. Sweep tau (-0.7 start) / lambda; recall<->spec tradeoff curve.
4. CORPUS-QUALITY GATE (real failure-bearing trajectories, in-prompt ceiling, positive controls,
   provenance) — this, NOT the micro-probe, launches the long run.
5. Tried-and-failed/avoid facet: target = CORRECTED code (failure accessible, not reproducible) —
   different objective from verbatim recall; design when real trajectories exist.
6. Short full run -> HPO over engine params for pass@1. Not before.

### ARTIFACTS / merge hygiene
- Code: body_recall_guarded branch + _recall_*/_snap_* helpers in hypernet_distill.py;
  configs/issue52_body_recall_crossover_4b.yaml; tools/_recitation_probe.py,
  tools/_crossover_decision.py, tools/_reconstruct_mbpp_phase0.py. 4 unit tests added (329 pass).
  ruff (tools/_recitation_probe.py has E501s to clean) + mypy clean on src.
- ALL issue-52 probe code = REMOVE-BEFORE-MERGE unless generalization shown. Keep sb0.detach() +
  _artifact_uploaded size-check bug fixes (mergeable alone).
- MLflow exps: 43 (issue52-body-crossover/pilot1), 44 (issue52-body-recall/pilot2). S3 works here.

## Open / housekeeping
- Stale MLflow run `a40838af` (exp 43) still RUNNING (orphan from a prior attempt) — clean up.
- Determinism: sha matched historical, but I do NOT assert bitwise-determinism-across-tf5.9
  as a finding (surprising; verdict doesn't need it). Metric parity + GATE B carry it.

---

### [2026-06-03 19:30 UTC] SESSION-END SUMMARY + NEXT STEPS
**Findings**
- Pilot 1 (body_derangement): QUALIFIED, not GO. m-mismatch +0.137→+1.026 but 91%
  deranged-SUPPRESSION; accessibility flat (Δlp_matched +0.075, CI spans 0). (+ verified +0.137
  is real encoding, NOT a scaling/alpha-r bug.)
- Pilot 2 (body_recall_guarded): ACCESSIBILITY PASS + no recitation. Objective = primary
  relu(τ−lp_matched) + guard relu(lp_mismatch−lp_n0_warmstart). Result: Δlp_matched +0.290,
  sign +9/−1 p=0.021, bootstrap CI [+0.130,+0.448] EXCLUDES 0; mismatch ~1:1 (not dominating);
  signature retained +3.84→+5.71; guard never fired (emergent specificity). Cross-conditioned
  recitation test: trained 0/10 recite = warm-start → recall did NOT induce recitation.
  MLflow exp 44 run 7b82c304.
- Status: guarded matched-recall is the validated lever. Trainability only (trained-on-test, 10
  MBPP bodies, continuation facet). NOT yet: generalization, pass@1 correctness, retention, corpus.

**Next steps (gated, in order — do NOT skip to the long run)**
1. Real gates on pilot-2 ckpt: xgrammar pass@1 CORRECTNESS (absent regime) + retention
   (goal/file/diff/code-recall not regressed vs warm-start). entry_point/ast checks so far = floors.
2. Sweep τ (−0.7 start) and λ_p/λ_g; map the recall↔spec tradeoff.
3. Present-regime recitation canary clean (0/10) — extend to >1 partner/ep if tightening wanted.
4. CORPUS-QUALITY GATE = what launches the long run (NOT the micro-probe): real engine
   trajectories + failure-bearing episodes, in-prompt CEILING, positive controls, provenance,
   causal failure→next-action alignment.
5. Tried-and-failed/avoid facet needs a DIFFERENT objective: target = CORRECTED next code
   (failure ACCESSIBLE not REPRODUCIBLE), not verbatim recall. Design when real trajectories exist.
6. Short full run → THEN HPO for pass@1.

Merge hygiene: all issue-52 probe code = REMOVE-BEFORE-MERGE; keep sb0.detach() +
_artifact_uploaded size-check fixes (mergeable alone).

---

### [2026-06-03 19:31 UTC] AI-engineer reflection 3 (reflections.md) — incorporated
Endorses pilot 2 (accessibility PASS, recitation resolved) + the gated next-steps. New
operational items folded in:
- **τ/λ sweep diagnostic:** watch `guard_active_frac` — if it RISES, the guard is firing
  (generic-boost or deranged drift). If matched PLATEAUS below τ while mismatch keeps falling,
  we're sliding back to pilot-1 (suppression) geometry → back off λ_g / raise λ_p or τ.
- **Reporting hygiene (reaffirm):** headline numbers from ONE instrument (`_specificity_probe`);
  snapshots are training telemetry only (they read ~0.57 optimistic vs the frozen probe).
- Confirmed: do NOT launch long run; need corpus gate + pass@1 + retention on the pilot-2 ckpt
  (or its τ-sweep successor). No change to plan — these are guardrails for step 1–2.

---

### [2026-06-03 19:32 UTC] AI-engineer reflection 4 — incorporated (plan refinements)
Endorses the arc; concrete refinements folded into the gated plan:
1. **ORDER: gate 1 BEFORE τ-sweep.** Run retention + pass@1 on the FIXED pilot-2 ckpt 7b82c304
   FIRST. τ-sweep is for the recall↔spec frontier, NOT for unlocking the long run (classic
   "probe win, bench lose" risk). [supersedes the earlier sweep-then-gate ordering]
2. **pass@1 gate def:** warm-start pass@1 = FLOOR; trained must be ≥ floor within noise, per-task
   table, using the rune bench / xgrammar harness (NOT probe logprobs — accessibility ⊥ pass@1).
3. **MLflow sweep column:** log Δlp_matched / |Δlp_mismatch| (fraction of m-mismatch gain from
   matched vs mismatch) on a 5-episode frozen snapshot every N steps → detect pilot-1 geometry
   EARLY (don't wait for step 30).
4. **Corpus-quality gate additions:** HELD-OUT episodes (not in the 10), NEGATIVE CONTROL (shuffle
   trajectory-id vs context), in-prompt CEILING = tokens-to-limit per episode (so "absent" ≠ a
   truncation artifact). Without holdout the gate only certifies memorization of the pilot set.
5. **mbpp/19** (the shared 1/10 "neither") is the SAME accessibility outlier from pilot 1, not
   noise — keep per-episode recitation+accessibility dashboards; do NOT aggregate it away.
6. **Tried-and-failed facet:** name it as a distinct contrastive_mode facet tag early (avoid
   copy-paste reuse of body_recall_guarded); target = logprob/RL on CORRECTED continuation tokens
   with the failure span MASKED from the generation target (not failure verbatim).
7. **Wording:** prefer "validated TRAINING OBJECTIVE for continuation-body accessibility under
   absent conditioning" over bare "validated lever" in external docs (avoid product-validation read).

---

### [2026-06-03 19:42 UTC] RESEARCH: two-stage training (recall → outcome RL) — recommendation
User idea: stage 1 = recall (what we did), stage 2 = signal is pass@1 / continuation success /
repair-debug success. Researched (consensus, HF papers, web, context7/TRL). Verdict: SOUND and
mainstream, with 3 system-specific caveats that change HOW, not WHETHER.

LITERATURE (grounding):
- Two-stage SFT/representation→RLVR is the DOMINANT validated recipe for code. ExecVerify
  (two-stage: exec-reasoning→code-gen, +5.9% pass@1, 7B≈32B); DRIVE (two-stage GRPO comp-code);
  Agentic RL for code repair (SFT→GRPO = 7–20% abs gains); Agent-RLVR (SWE pass@1).
- PROCESS > OUTCOME for code: PRLCoder (line-level process-RL beats outcome-only); RePair
  (process APR, <20B ≈ commercial); PRM survey. → user's "continuation/repair success" (process)
  alongside pass@1 (outcome) is well-supported and likely NECESSARY for density.
- CATASTROPHIC FORGETTING real at 1–7B, worsens with scale (Luo 2023). Mitigate: KL-to-stage-1,
  replay recall objective, or COOPERATIVE SFT+RL (Beyond Two-Stage, 2509.06948).
- Sparse binary reward = vanishing gradients (ConfClip). Hypernet+RL has precedent (Keynan 2021;
  zero-shot-transfer-RL 2022) but in control/meta-RL, not LLM-adapter-gen.
- TRL GRPOTrainer supports custom verifiable reward fns; but our policy = the HYPERNET emitting
  the adapter (not the base/PEFT), so reuse GRPO loss in a CUSTOM loop (our distill loop already
  does custom functional-LoRA backprop) — TRL not drop-in.

3 CAVEATS specific to us:
1. Credit assignment: pass@1 reward must reach the HYPERNET through non-differentiable
   gen+exec → policy gradient (GRPO), hypernet-as-policy. Novel; feasible via our existing
   custom-backprop loop. Standard RLVR optimizes the base; we optimize the memory-generator.
2. Reward density: pure pass@1 on tiny/real corpora is too sparse (vanishing grads). BLEND
   outcome (pass@1) + process (per-step continuation/repair success, compile/test deltas).
3. Forgetting: stage-2 RL can erase the stage-1 +0.290 accessibility. KL-anchor to stage-1
   hypernet OR replay recall OR cooperative joint.

WHY two-stage is RIGHT FOR US (the strong argument): stage 1 optimizes a PROXY (logprob
accessibility — necessary-not-sufficient, the exact gap we keep hitting); stage 2 optimizes the
ACTUAL objective (task success). Two-stage directly resolves the proxy-misalignment. Stage 1 =
warm-start so sparse-reward RL isn't sample-starved.

RECOMMENDATION (gated): pursue two-stage, but (a) stage-2 reward = blended outcome+process,
(b) GRPO group-relative in a custom hypernet-as-policy loop, (c) retention via KL-to-stage-1 /
recall-replay (or cooperative), (d) ONLY after stage-1 accessibility + pass@1 floor (running) +
a REAL failure-bearing corpus exist. Consider cooperative SFT+RL as a variant vs strict sequential.
Do NOT start stage 2 before the corpus-quality gate.

---

### [2026-06-03 20:35 UTC] GATE-1 pass@1 SMOKE (real MBPP 3-test suites, end-to-end) — strong positive
tools/_pass1_probe.py, warm-start vs trained recall ckpt (7b82c304), per-episode:
- PRESENT (spec in prompt, STABILITY gate): warm 6/10 -> trained 10/10. Recall training did NOT
  break in-context generation; improved it.
- ABSENT (from memory, no spec, CAPABILITY = operative regime): warm 1/10 -> trained 5/10.
  scale=0 absent ≈ 0 (no adapter, no spec = no info). So adapter provides the memory; training
  5x'd it. The +0.290 logprob accessibility TRANSLATED to +4 functional pass@1.
CAVEATS: trained-on-test (memorization, not generalization — held-out set is next); n=10
descriptive; raw greedy gen + real tests, NOT xgrammar-constrained engine pass@1. Absent PASS
sets differ slightly per-episode (warm passed mbpp/12; trained passes 11,14,17,18,56).
=> Gate-1 pass@1 half: PASS (improved both regimes). Retention half (goal/file/diff/code-recall
via diag_recoverability) still to run. This is the "smoke before HPO" and it's positive.

AI-engineer reflection (19:42) incorporated: forgetting -> KL-anchor to stage-1 hypernet +
recall-replay, prefer COOPERATIVE joint over strict sequential, log accessibility/signature
canaries every N RL steps. Rewards -> execution-GATED process + outcome, avoid naive pass-rate
surrogates. Credit assignment genuinely novel (OP-LoRA is supervised); K=4 rollouts, group-rel
advantage on adapter-gen weights. Stage 1 only certifies "memory channel open," not correct gen.

USER DIRECTION (this turn): (1) memory-exercising eval is required AND the governing principle =
NEVER fill the prompt with what belongs in the adapter (whole point = use adapter to spare the
context window). (2) BUILD HELD-OUT SET FIRST (before train HPO). Proceeding to build held-out
MBPP recall corpus (disjoint from the 10) + negative control, then held-out accessibility probe.

---

### [2026-06-03 20:45 UTC] HELD-OUT generalization of pilot-2 ckpt (24 disjoint tasks) — DOES NOT GENERALIZE
_specificity_probe --corpus mbpp_recall_heldout.jsonl, warm-start vs pilot-2 (trained on the 10):
- absent/body m-mismatch: warm +0.127 -> trained +0.165 (Δ +0.038, ~flat)
- absent/body m-zero (ACCESSIBILITY): warm +0.530 -> trained +0.514 (Δ -0.016, FLAT)
- absent/sig m-mismatch: warm +3.56 -> trained +4.95 (signature DID generalize +1.39)
=> The +0.290 accessibility gain was MEMORIZATION of the 10, NOT a transferable skill. On
held-out tasks the trained hypernet ≈ warm-start on BODY accessibility (signature generalizes,
body does not — consistent with the whole project thesis). User's "build held-out first" was
exactly right.

OVERNIGHT GOAL (sharpened): does training on the 40-task TRAIN split generalize body
accessibility to the 24 held-out? That is the real Phase-1 question. Orchestrator: train-HPO on
mbpp_recall_train (40) -> eval held-out 24 accessibility (gated sig+recitation) -> best -> pass@1
bench {scale=0, warm-start, best} on held-out -> retention. Honest outcome either way:
generalizes => real lever; flat => objective memorizes regardless of corpus (deeper limit).
Crash-safety: commit e98172a pushed; orchestrator writes incremental results + I commit per config.

---

### [2026-06-03 21:37 UTC] PHASE-1 COMPLETE — PASSES (generalizing). PR #55 finalized.
Train-HPO (40-task split) -> held-out 24 eval. Best c3 (tau-0.7 lp2 lg1, MLflow run fe72f9ddd69c):
- held-out accessibility Δlp_matched +0.105, CI [+0.033,+0.182] EXCLUDES 0 (17/24). GENERALIZES
  (trained-on-10 was flat). lambda_p is the lever; guard (lambda_g=2) hurts.
- functional pass@1 held-out (real MBPP 3-test), ABSENT/from-memory: scale0=0/24, warm=3/24,
  c3=8/24. PRESENT (stability): 19/24=base (no regression).
=> recall objective is a GENERALIZING lever (not memorization). c3 = Phase-2 retention baseline.
LIMITS: 8/24 partial; n=24 binary; raw-greedy not xgrammar; single-shot absent = harshest proxy.
REMAINING gate-1: retention scorecard (goal/file/diff/code-recall). Deeper: engine multi-step eval.
NEXT: retention -> engine eval -> scale corpus -> Phase-2 cooperative RL (KL-anchored, canaries).
Commits pushed through Phase-1 COMPLETE; PR #55 comment + docs/issue52-phase1-results-2026-06-04.md.

---

### [2026-06-04 07:56 UTC] Dataset logging wired + external_codereview corpus RECOVERED
Context: gate-1 retention was blocked on `external_codereview.val.clean.jsonl` (only in /tmp,
never logged). Investigated recoverability: NOT on disk, no restic, S3 ListBucket denied on the
eu-west-2 MLflow artifact bucket, and `run.inputs.dataset_inputs == []` on every prior run
(dataset tracking was never used). It was NOT lost — raw source survived on S3.

RECOVERY (deterministic, reproducible):
- Source: `s3://elixirtrials-949678234935-us-east-1-artifacts/training-data/github-pairs/external_codereview.unrolled.jsonl`
  (7,670 rows, sha256 4931fe03). chain: `split_corpus.py` (family-keyed sha256 split, val5/test5)
  -> `corpus_split_qc.py` (TF-IDF near-dup clean, cos>=0.9 dropped; sklearn 1.9.0).
- Result: val.clean 323 rows sha256 7e3692df; test.clean 343 rows sha256 744715a6;
  train 6930 / val 370 / test 370. Pushed durable to `.../github-pairs/splits/`.

DATASET LOGGING (mergeable product code, commit 25dcbd2):
- `src/rune/tracking.py::log_dataset(uri,*,name,context)` — thin wrapper over MLflow
  `MetaDataset(resolve_dataset_source(uri))` + `log_input`; metadata-only (logs by reference, safe
  for ~90MB), MLflow's built-in digest (no hand-rolled sha256). Tested (sqlite round-trip).
- Wired: train (corpus_dir->training), bench + bench-hpo (tasks_file->test).
- Backfill: MLflow exp `corpus-registry` run `register-external_codereview-2026-06-04` (ea4f3c43),
  4 inputs logged by S3 URI. Gate-1 retention UNBLOCKED on the data side.

INTERPRETATION/HYPOTHESES (carried for next session):
- Single-turn (absent/from-memory) pass@1 UNDERSTATES the thesis (8/24 is a floor). The design's
  real payoff is MULTI-TURN: prior work lives in the adapter, prompt stays flat. Must be measured
  vs scale=0 control. This is the next decisive eval, before scaling the corpus.
- Convention going forward: no run trains/evals on an unlogged dataset; pass the durable S3 URI
  (not /tmp) to `log_dataset`. diag_*/gate_* `--val` should default to the S3 URI.
NEXT: run diag_recoverability on the restored val.clean (only blocker now = base-model/4-bit
default fix) -> retention scorecard -> multi-step engine eval (prompt fixed, adapter memory) vs
scale=0 -> scale corpus 40->N -> Phase-2 cooperative RL (KL-anchored to c3, accessibility/sig canaries).

---

### [2026-06-04 08:15 UTC] GATE-1 RETENTION — QUALIFIED PASS (cross-domain, 3/4 targets). Closes gate-1.
WHAT I RAN: `tools/run_guarded.sh tools/diag_recoverability.py` (defaults fixed first — see below).
  ckpt c3 `/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt` (sha256 53e24af2…, MLflow run fe72f9ddd69c);
  base **Qwen/Qwen3-4B-Instruct-2507 in bf16** (the recipe that produced +0.105 — orchestrator
  `load_in_4bit: false`, `_specificity_probe` with no `--load-4bit`); val = recovered
  external_codereview.val.clean (n=24, sha256 7e3692df…, durable
  s3://elixirtrials-949678234935-us-east-1-artifacts/training-data/github-pairs/splits/external_codereview.val.clean.jsonl;
  local /tmp copy bit-identical). scaling=45.25 (effective_scaling==lora_alpha). log
  `/tmp/phase1/logs/recoverability_c3.log`.
BLOCKER FIXED: `diag_recoverability.py` hardcoded a 4-bit nf4 load of `Qwen/Qwen3.5-9B` (wrong
  model AND wrong dtype → adapter conditioned on mismatched activations). Changed defaults to
  c3 ckpt + Qwen3-4B-Instruct-2507 bf16, made 4-bit opt-in (`--load-4bit`). This was the only
  remaining gate-1 blocker.
RESULTS (target | n | m-mismatch | m-zero; bet needs BOTH>0):
  goal  | 24 | +0.539 | +0.297  ✓ both  (recovers the review request, episode-specifically)
  diff  | 24 | +0.058 | -0.028  ✗ m-zero flat (exact edit-local tokens not lifted over no-adapter)
  tail  | 24 | +0.108 | +0.224  ✓ both  (semi-Markov: last-N lines that DRIVE THE NEXT STEP)
  avoid | 14 | +0.155 | +0.862  ✓ both  (logp(accepted)-logp(rejected) at edit hunk: avoids failed)
INTERPRETATION: c3's episodic memory channel is OPEN and episode-specific cross-domain. goal/tail/
  avoid all clear both bars on external_codereview — a DIFFERENT domain than c3's MBPP-recall
  training, so this is generalization of the recoverability skill, not in-domain retention. The
  two operationally important targets pass: `tail` (the state that drives the next engine step)
  and `avoid` (don't-repeat-the-rejected-approach). `diff` flat on m-zero = the verbatim edit is
  high-entropy and not lifted over base; it remains weakly episode-specific (m-mismatch +0.058).
LIMITS (honest): point estimates, NO CI/bootstrap — this tool reports sign+magnitude only, weaker
  than the Phase-1 "CI excludes 0" bar; per-target n varies (avoid n=14, rows silently skipped
  when no feedback / <7 lines / no replace-hunk). Cross-domain by construction (sets the
  pre-Phase-2 recoverability BASELINE the forgetting canaries compare against), not in-domain.
HYPOTHESIS/EXPECTATION: Phase-2 RL must not erode goal/tail/avoid m-zero below these values; if a
  canary drops, forgetting. diff being flat predicts edit-token recall is the weakest channel —
  candidate for the corpus-scale / objective work, not a Phase-2 blocker.
NEXT: gate-1 CLOSED (pass@1 half done Phase-1; retention half = this). Decisive eval = multi-step
  engine vs scale=0 (prompt fixed, adapter memory growing). Also in flight: DRY config.yaml
  refactor (all tools -> Qwen3-4B-Instruct-2507 instruct via single config; user directive).

---

### [2026-06-04 08:40 UTC] DRY config refactor — single source of truth for base model (mergeable)
WHAT I DID (user directive: "everything uses the instruct model so the pre-warmed Sakana adapter
  is compatible; one config.yaml with all settings"):
- `config.yaml` (repo root): all PipelineConfig settings; model_id=Qwen/Qwen3-4B-Instruct-2507.
- `rune.config`: added `DEFAULT_MODEL_ID` const (=instruct), `load_rune_config()` resolver
  (defaults <- config.yaml/RUNE_CONFIG <- RUNE_* env, later wins), `RUNE_BASE_MODEL` env override,
  refactored `from_env` via shared `_env_overrides()`. Default model_id flipped 9B->4B-instruct.
- Swept all hardcoded `Qwen/...` literals -> `load_rune_config().model_id` across 19 tools +
  `d2l_train.py` (-> DEFAULT_MODEL_ID). Only remaining literal = DEFAULT_MODEL_ID in config.py.
  Fixes a latent bug: diag_*/gate_* tools defaulted to Qwen3.5-9B (wrong model for issue-52 ckpts).
- Docs reconciled: CLAUDE.md base-model line + RAM figure (18GB->~8GB bf16 for 4B).
- Tests: updated test_config/test_d2l_train defaults; added TestLoadRuneConfig (file read + env-
  over-file precedence). VERIFY: ruff src+tests clean, mypy src clean, 329 unit pass.
NOTE: tools are still REMOVE-BEFORE-MERGE scaffolding; config.py/config.yaml/tests are mergeable.
NEXT (unchanged): multi-step engine eval vs scale=0 (the decisive one) -> scale corpus -> Phase-2.
Not committed (awaiting user go).

### [2026-06-04 08:48 UTC] config refactor — wired the PRODUCT entrypoints (advisor-caught gap)
The 08:40 sweep wired tools but `rune run`/`rune bench` still built bare `PipelineConfig()` when no
`--config` was passed — they never read repo-root config.yaml (values only coincidentally agreed).
Fixed `src/rune/cli.py`: default path now `load_rune_config()` (config.yaml + RUNE_* env), so
editing config.yaml actually drives the product. ruff/mypy clean, 329 unit pass, CLI import OK,
`_specificity_probe --help` OK (reordered imports resolve). Open item for user: training
hyperparameters (D2LTrainConfig) are a SEPARATE config surface — fold into config.yaml or keep two?

### [2026-06-04 09:00 UTC] config refactor — UNIFIED to one surface (training reads config.yaml)
Per user: one config.yaml for everything. Added nested `training:` section (D2LTrainConfig fields);
`load_config`/`load_rune_config` ignore it; new `rune.training.d2l_train.load_train_config(path=None)`
reads it and INHERITS model_id from top-level (single source; precedence RUNE_BASE_MODEL >
training.model_id > top-level model_id > DEFAULT_MODEL_ID). `rune train --config` takes the
config.yaml path, defaults to repo-root. Orchestrator/_distill_entry flat-YAML path untouched
(separate scaffolding). VERIFY: ruff src+tests clean, mypy clean, 334 unit pass (5 new),
rune train --help OK. Mergeable. Still not committed (awaiting user go).

### [2026-06-04 09:10 UTC] config — env-override symmetry fix (advisor-caught)
`rune run/bench --config X` previously used `load_config(X)` which skipped RUNE_* env overrides
(only the no-arg path applied them) — so RUNE_BASE_MODEL silently no-op'd with an explicit config,
contradicting the documented universal precedence. Fixed: `load_rune_config(path=None)` now takes an
optional path and applies env overrides on either branch; both CLI commands call
`load_rune_config(config)`. Verified: `RUNE_BASE_MODEL=Org/Z ... load_rune_config('config.yaml')`
-> Org/Z. ruff/mypy clean, 335 unit pass (+1). Handoff headline corrected to "qualified pass".
