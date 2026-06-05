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

### [2026-06-04 08:48 UTC] SESSION START — goals extracted from handoff_20260604T082918Z; reflections monitor armed
WHAT I RAN: read handoff + reflections (full) + tail of this scratchpad; checked git/GPU/RAM/ckpts.
STATE: branch issue52-bf16-body-contrastive, PR #55. GPU free (0/23GB), RAM 10GB avail. c3 ckpt +
c1/c2/c4 present at /tmp/phase1/ckpt/. Recall corpora present (train=40, heldout=24). Phase-1 PASS,
gate-1 retention QUALIFIED PASS (per handoff §where-we-are).
GOALS (priority order, from handoff):
  1. Multi-step engine eval vs scale=0 — THE DECISIVE EXPERIMENT. Engine runs N steps, state
     EVICTED INTO THE ADAPTER, prompt held FIXED, adapter memory growing; compare vs scale=0 (no
     adapter) on same tasks. Governing principle: never fill the prompt with what belongs in the
     adapter. A result without the scale=0 control is not a result.
  2. Scale recall corpus 40 -> N (use recovered `train` split); test if more disjoint tasks keep
     raising held-out body accessibility.
  3. Phase-2 cooperative outcome-RL (distill + recall-replay + GRPO, KL-anchored to c3, canaries
     every N). DO NOT start before goals 1-2 give a real failure-bearing signal.
REFLECTIONS CONSTRAINTS (pre-registration for goal 1, from reflections 2026-06-04 block):
  - Fixed prompt budget per step (log tokens-to-limit per episode).
  - scale=0 AND warm-start on the SAME task DAG layer (not only c3).
  - Primary metric = integrate success / pass@1 at final step; secondary = per-step accessibility
    canary (same frozen probe as Phase-1).
  - Failure mode to watch: recitation dominance (step k adapter encodes step k-1 body verbatim);
    final-step cross-conditioned recitation alone is insufficient.
  - "Single-turn understates the thesis" is a HYPOTHESIS until the protocol returns numbers.
MONITOR: armed inotify+MD5 watch on instructions/reflections.md (this session, task bixy8zff5).
PLAN: (a) understand how the engine conditions the adapter on trajectory + whether an
"adapter-as-memory, fixed-prompt" mode exists or must be built; (b) advisor review of the decisive-
experiment design BEFORE building; (c) build/run the multi-step eval with scale=0 control; (d) then
goal 2 corpus scaling. NEXT: inspect engine adapter-conditioning path; call advisor.

### [2026-06-04 09:05 UTC] DESIGN FORK (advisor-reviewed) — decisive eval = recall-CAPACITY probe, NOT engine rebuild
ENGINE FINDING (why not the LangGraph engine): the current engine puts prior state INTO THE PROMPT
(`code_trajectory`, `existing_code`, `repair_history`, `code_outputs` all render into the action
templates), while the adapter is conditioned only on a NARROW LOCAL `(task, current_code, feedback)`
via render_training_format_trajectory — NOT a growing memory. So a scale=0 control inside the engine
is DIRTY (model still reads prior state from the prompt with the adapter off), and the handoff's
"state evicted into the adapter, prompt held FIXED, adapter memory growing" is not how the engine
works today. Rebuilding the engine to a prompt-minimal/growing-memory mode is a multi-day build that
confounds thesis-failure with new-code-failure. REJECTED.
DECISION: the decisive experiment lives on the `_pass1_probe` ABSENT harness (spec-absent prompt,
--scale0 native floor) extended to MULTI-TASK ACCUMULATION. The 8/24-vs-0 single-turn result already
came from this harness; "absent pass@1 is a floor not the ceiling" = a continuation of this metric
family, not a different apparatus.
LINCHPIN (untested assumption the whole thesis rests on): can ONE adapter hold MULTIPLE tasks and
still recall each? Warm start is doc-to-lora (one doc -> one adapter); multi-task conditioning may
not survive. THIS is the failure-bearing signal the handoff wants before Phase-2. 1-2h probe, not a
build. If recall collapses at k=2, THAT is the finding — report, don't engineer around it.
PROBE DESIGN (`tools/_recall_capacity_probe.py`, REMOVE-BEFORE-MERGE):
  - Condition ONE adapter on a trajectory concatenating k task descriptions (k=1,2,4,8); query each
    of the k tasks NAME-CUED, spec absent: "Write the Python function `{entry_point}` you studied".
  - Per-k pass@1 vs REAL MBPP 3-test suites (heldout 24 tasks: k=1->24 blocks, 2->12, 4->6, 8->3).
  - Record per-within-block POSITION (recency effect) + which `def NAME(` the model emits
    (cross-task interference = wrong-function retrieval, the reflections recitation failure mode).
  - Prompt token length logged per query: name-only -> ~FLAT regardless of k, while adapter
    conditioning grows. Demonstrating flat-prompt + growing-memory IS the thesis.
ARMS (pre-registered, per reflections): (A) c3 ckpt; (B) warm-start; (C) scale=0 with the SAME
name-cued prompt = the critical control. If scale=0 solves it from the name alone, the name leaks
the spec and the adapter isn't doing the work; adapter >> scale=0 is the win condition.
NEXT: build the probe; 1-block smoke per k; full run under run_guarded.sh; log dataset URI/digest.

### [2026-06-04 09:30 UTC] capacity probe BUILT + smoke-validated; full 3-arm run launched
WHAT I RAN: `tools/_recall_capacity_probe.py` smoke on 4 heldout tasks, c3, k=1,2.
INSTRUMENT VALIDATION (the design works):
  - prompt_tokens FLAT ~28-30 across all k; study_tokens grows with k (add_lists block 63->227 for
    k=1->2). Flat-prompt + growing-adapter-memory is correctly instrumented = the thesis surface.
  - emitted `def NAME(` ALWAYS == queried name (interference=0): name cue disambiguates perfectly;
    failures are WRONG-BODY-for-right-name, i.e. a clean memory-capacity failure, not confusion.
SMOKE SIGNAL (n=4, noisy): c3 k=1 pass@1=2/4 (0.50); k=2 pass@1=0/4 (0.00). merge_sorted_list
  PASSED at k=1 then FAILED at k=2 — direct capacity degradation. Matches advisor's predicted
  failure mode ("if recall collapses at k=2, that is the finding"). Need full n to confirm.
DATASET (mandatory logging): corpus benchmarks/mbpp_recall_heldout.jsonl sha256 cae274bf1aed n=24;
  durable S3 source for splits per handoff: s3://elixirtrials-...-us-east-1-artifacts/training-data/
  github-pairs/ (recall corpora are issue-52 derived). c3 ckpt sha256 53e24af243a3 (matches handoff).
FULL RUN: `tools/_run_capacity_arms.sh` -> 3 arms (scale0, warm, c3) x k={1,2,4,8} on all 24 heldout,
  under run_guarded.sh watchdog. Out: /tmp/cap/{scale0,warm,c3}.jsonl. Monitor armed on /tmp/cap.
HYPOTHESIS: if c3 >> scale0 at k=1 but both decay toward scale0 as k grows, the adapter memory is
  real but LOW-CAPACITY (1-2 items) — that bounds Phase-2 and the "engine N-step" ambition. If c3
  holds flat across k, capacity is high and the multi-step thesis is supported. NEXT: read full
  results, compute per-k decay curve + per-position recency, interpret, write verdict block.

### [2026-06-04 09:15 UTC] DECISIVE EXPERIMENT RESULT — adapter memory is REAL but CAPACITY-BOUND; Phase-1 traded capacity for peak
WHAT I RAN: tools/_recall_capacity_probe.py, 3 arms x k={1,2,4,8} on all 24 held-out MBPP-recall
tasks (heldout sha256 cae274bf1aed). Name-cued, spec ABSENT. scale=0 = no-adapter floor (the
control the handoff demands). c3 sha256 53e24af243a3; warm = doc-to-lora checkpoint-20000.
RESULTS — pass@1 /24 (rate):
  k    scale0      warm        c3
  1    5 (.21)     9 (.38)    12 (.50)
  2    5 (.21)     7 (.29)    10 (.42)
  4    5 (.21)     6 (.25)     9 (.38)
  8    5 (.21)     8 (.33)     6 (.25)
PAIRED c3-scale0 delta (bootstrap 10k, paired by task_id at fixed k):
  k=1 +0.292 CI[+0.083,+0.500]  <- CI EXCLUDES 0 (real adapter contribution beyond the name)
  k=2 +0.208 CI[+0.000,+0.417]  (CI touches 0)
  k=4 +0.167 CI[+0.000,+0.375]  (CI touches 0)
  k=8 +0.042 CI[-0.125,+0.250]  (CI SPANS 0 — indistinguishable from floor)
CONTROLS/INSTRUMENT: interference=0 at every k/arm (name cue always retrieves the RIGHT function
name; failures are wrong-BODY, a clean capacity failure). prompt_tokens 27-34 (FLAT) while
study_tokens 36-498 (grows with k) — flat-prompt + growing-memory thesis surface confirmed. scale0
flat at 5/24 across all k (k-invariant floor, as it must be: study material never reaches the model).
INTERPRETATION:
  1. The adapter-as-memory channel is REAL at low load: c3 +0.292 over the no-adapter floor, CI
     excludes 0. The handoff's scale=0 control is satisfied AND passed at k=1.
  2. CAPACITY is the binding constraint. c3's advantage DECAYS MONOTONICALLY (12->10->9->6) and is
     statistically gone by k=8. The channel holds ~1-2 items reliably, frays by 4, ~floor by 8.
  3. KEY TRADE-OFF: warm is FLATTER than c3 (warm k=8=8 > c3 k=8=6). Phase-1 training bought higher
     PEAK single-item recall (12 vs 9) at the COST of multi-item capacity — the single-task
     accessibility objective ERODED capacity. Actionable for Phase-2.
  4. This CONTRADICTS the handoff's optimistic goal-1 framing ("uplift largest in multi-step; 8/24
     is a floor not a ceiling"). Uplift is LARGEST at k=1 and SHRINKS with accumulation. For this
     ckpt + this conditioning, single-item recall is near the CEILING of the memory channel.
CAVEATS (honest limits):
  - n=24, binary pass@1, noisy; k=8 has only 3 blocks. warm non-monotonicity (6 then 8) is within
    noise. c3's 12,10,9,6 monotone decay is the more credible trend but each point is ~+/-0.1.
  - **CONDITIONING-FORMAT CONFOUND (important):** multi-task study was NAIVE — k descriptions
    concatenated into the single `task=` slot of render_training_format_trajectory, a format the
    hypernet NEVER saw in training (single-task only). The decay may partly reflect format mismatch,
    NOT a hard architectural capacity limit. This BOUNDS this conditioning approach, not the ceiling.
  - name cue leaks semantic signal (scale0=5/24 from names alone); deltas are above that floor so
    real, but absolute numbers include the name's contribution.
HYPOTHESIS for goal 2 (reframed): training on MORE disjoint tasks (40->N) and/or on MULTI-TASK
conditioning may FLATTEN the decay curve (raise capacity), which matters more than raising the k=1
peak. Capacity, not peak accessibility, is the Phase-2 lever.
NEXT: advisor review of interpretation + the format-confound; then goal 2 design (capacity-targeted).
Promote verdict to docs/issue52-*.md.

### [2026-06-04 09:25 UTC] CORRECTION (advisor-caught over-claim) — only the k=1 c3-vs-floor result survives
The 09:15 headline ("CAPACITY-BOUND; Phase-1 traded capacity for peak") asserted more than the data
support. Computed the missing c3-warm paired CIs; they kill two claims:
c3-WARM paired delta (bootstrap, by task_id):
  k=1 +0.125 CI[-0.042,+0.292]   k=2 +0.125 CI[-0.042,+0.292]
  k=4 +0.125 CI[+0.000,+0.250]   k=8 -0.083 CI[-0.250,+0.083]   -> EVERY k SPANS ZERO.
WARM-scale0: k=1 +0.167 CI[+0.000,+0.375] (touches 0); k>=2 mostly touch/span 0.
WHAT SURVIVES (the only CI-clean claim): c3 - scale0 at k=1 = +0.292 CI[+0.083,+0.500], EXCLUDES 0.
  => The adapter-as-memory channel is REAL at k=1 (single item), in-distribution conditioning.
WHAT DIES / IS DEMOTED:
  1. "Phase-1 traded capacity for peak" — DEAD. c3 vs warm indistinguishable at every k (all CIs
     span 0). The c3=6 vs warm=8 at k=8 gap (2/24) is within noise; warm's curve isn't monotonic.
  2. "Capacity ceiling / k=8 collapse" — NOT a finding. CONFOUNDED: k>1 fed the hypernet a
     concatenated-description format it NEVER saw in training (single-task only). "Recall degrades
     as k grows" conflates capacity with OFF-DISTRIBUTION-conditioning generalization. The fair claim
     is narrow: "naive concatenation conditioning degrades recall" — a property of MY conditioning,
     not the channel. k>1 is UNTESTED for true capacity.
  3. Even "c3 beats warm at k=1 (12 vs 9)" — NOT significant (CI spans 0). Only c3 > floor holds.
CORRECTED VERDICT (3 lines): (a) memory channel real at k=1, CI-backed (c3 +0.292 over no-adapter
floor); (b) k>1 capacity is an OPEN QUESTION — multi-item conditioning was off-distribution, not a
measured limit; (c) no evidence Phase-1 traded capacity for peak.
GOAL 2 (advisor steer — keep as pre-registered, do NOT redirect): does 40->N disjoint TRAINING tasks
keep raising k=1 held-out accessibility (the thing that rose 10->40 and that we just measured
cleanly)? The "train on multi-task conditioning to raise capacity" idea is a genuine NEW hypothesis
from this run but needs an in-distribution multi-item design (hypernet trained on multi-task
trajectories first) — LOG AS FUTURE WORK, don't fold into Goal 2.
NEXT: write durable doc with only the surviving claim + open questions; then Goal 2 corpus scaling.

### [2026-06-04 09:35 UTC] GOAL 1 closed -> docs/issue52-goal1-recall-capacity-2026-06-04.md ; GOAL 2 plan
Promoted the surviving claim + open questions to the durable doc. Goal 1 deliverable: the decisive
scale=0 control PASSES at k=1 (c3 +0.292 CI excl 0); k>1 capacity left OPEN (off-distribution
conditioning confound). 3 probe tools are REMOVE-BEFORE-MERGE.
GOAL 2 (pre-registered): does scaling disjoint TRAINING tasks 40->N keep raising k=1 held-out
accessibility (Δlp_matched via _specificity_probe) AND k=1 pass@1 (capacity probe k=1)?
DESIGN: hold the EVAL set FIXED at the current 24 heldout (rows[40:64]) for comparability; grow only
train. Builder emits train = rows[:40] + rows[64:64+extra], disjoint from heldout. Sizes: N=40
(existing c3), 80, 160 (cap by MBPP availability; sanitized test=257, ~200 usable after
bodyless/crossover filter). Train each with the c3 objective (body_recall_guarded, τ=-0.7, λ_p=2,
λ_g=1), 48 steps, same as Phase-1. Eval all on the fixed 24: (a) _specificity_probe m-zero/lp_matched
(the 10->40 metric); (b) capacity probe k=1 pass@1 vs scale0 floor.
NEXT: build scaling corpora (hold heldout fixed); verify disjoint; train N=80,160; eval.

### [2026-06-04 09:30 UTC] goal-2 training in flight; checkpoint-retrieval gotcha
Built nested+disjoint scaling corpora (40 subset 80 subset 160; eval fixed 24; pool=206 usable).
Trained c3-objective (τ=-0.7,λ_p2,λ_g1) 48 steps on N=80 (done rc=0) and N=160 (running, step ~20).
GOTCHA: run_hypernet_distillation uploads ckpts to MLflow(S3) and DELETES the local staging .pt
(disk guard). So local checkpoint_dir holds only distill_metrics.jsonl. Must fetch from MLflow
(tools/_fetch_goal2_ckpts.py, maps run->corpus size via corpus_path param, downloads
checkpoint_step48.pt). First fetch hung on the ACTIVE n160 run (artifact absent) — hardened to skip
status!=FINISHED. Fetch AFTER n160 completes, then eval (spec m_zero + cap k=1) on fixed 24.
n40 reuse: /tmp/phase1/heldout_c3_t07_lp2_lg1.jsonl absent/body m_zero=0.6351; cap k=1 pass@1=12/24.

### [2026-06-04 09:50 UTC] GOAL 2 RESULT — corpus 40->N: point estimates rise on BOTH metrics, but per-doubling gains NOT significant at n=24
WHAT I RAN: trained c3-objective (τ=-0.7,λ_p2,λ_g1, 48 steps, FIXED compute) on nested disjoint
train sets N=40(existing)/80/160 (40⊂80⊂160, all disjoint from the FIXED 24 heldout). Eval on the
fixed 24: (A) absent/body m_zero accessibility via _specificity_probe; (B) k=1 pass@1 (name-cued,
spec absent) via capacity probe vs the scale=0 floor (5/24). Ckpts fetched from MLflow (842MB each).
RESULTS:
(A) accessibility m_zero (mean [95% CI], n=24):
    warm +0.530 [.370,.703] | n40 +0.635 [.432,.855] | n80 +0.649 [.445,.875] | n160 +0.671 [.476,.887]
    -> MONOTONIC point rise warm<n40<n80<n160 (+0.141 warm->n160), but within-arm CIs heavily overlap.
(B) k=1 pass@1 (delta vs scale0 floor, paired CI):
    warm 9/24 +0.167[.000,.375] | n40 12/24 +0.292[.083,.500]✓ | n80 15/24 +0.417[.167,.625]✓ |
    n160 14/24 +0.375[.167,.583]✓   (✓=CI excl 0). pass@1 climbs 12->15->14 (jump then plateau).
PAIRED tests (the right significance test, by task) — scaling BEYOND 40:
    m_zero:  n80-n40 +0.014[-.023,+.051]  n160-n40 +0.036[-.0007,+.070]  n160-n80 +0.022[-.008,+.054]
    pass@1:  n80-n40 +0.125[-.042,+.292]  n160-n40 +0.083[-.083,+.250]   n160-n80 -0.042[-.125,+.000]
    -> EVERY beyond-40 increment SPANS/TOUCHES 0. NOT individually significant at n=24. The n160-n40
       accessibility trend is right at the boundary (lower -0.0007) — suggestive, not confirmed.
VERDICT (honest): scaling 40->N keeps NUDGING held-out recall UP on both metrics (accessibility
monotone; pass@1 jump-then-plateau), and ALL trained sizes beat the no-adapter floor with CI excl 0,
with n80/n160 beating it by MORE than n40/warm. BUT the per-doubling gain 40->160 is WITHIN NOISE at
n=24 — diminishing returns vs the larger 10->40 jump the handoff reported.
CONFOUND (important, must flag): FIXED 48 training steps means larger corpora got FEWER passes over
their data — N=160 may be UNDER-TRAINED for its size. "Diminishing returns" could be a COMPUTE limit,
not data saturation. Disentangling needs N=160 with more steps (matched epochs), OR a larger eval set
(n=24 underpowered for ~+0.03 m_zero / ~+2/24 increments).
NEXT: advisor on framing + whether to run the compute-matched N=160 control; then durable doc.

### [2026-06-04 10:00 UTC] GOAL 2 confound resolved (free slope diagnostic) -> DECLARE DONE, no extra spend
Advisor steer: read training-time slope before spending on a compute-matched N=160 run. Smoothed
early(5-20) vs late(30-48) means of distill_metrics:
  n40 : loss -0.750, lp_matched +0.009 (FLAT), diff_agree +0.082  -> converged/flattened.
  n80 : loss -1.534, lp_matched +0.536 (still improving strongly), diff_agree +0.079 -> used compute
        well; also won eval (pass@1 15/24). Sweet spot at 48-step fixed compute.
  n160: loss +0.176 (UP), lp_matched -0.358 (DOWN), diff_agree +0.064 -> NOT cleanly under-training-
        with-slope; late metrics DEGRADED = UNSTABLE at fixed compute (160 diverse tasks, 48 steps,
        lr 2e-5). But n160 still LEADS accessibility (0.671) so it is NOT badly starved.
DECISION: a compute-matched run is justified only if the slope says "still clearly improving"
(advisor rule). n160 does NOT — it is unstable, not productively-improving. And the binding
constraint is EVAL POWER (n=24), which one retrain cannot fix (half-width ~0.035 on m_zero). So:
DECLARE GOAL 2 DONE. No extra training spend.
GOAL-2 VERDICT (final, honest): scaling disjoint train 40->N nudges held-out recall UP on the
PRE-REGISTERED metric (accessibility m_zero monotone 0.635->0.649->0.671) and on the secondary metric
(pass@1 12->15->14, jump-then-plateau); ALL trained sizes beat the no-adapter floor (CI excl 0), n80/
n160 by more than n40/warm. BUT per-doubling gains 40->160 are WITHIN NOISE at n=24 (paired CIs span/
touch 0) — diminishing returns vs the larger 10->40 jump. n80 ~= n160 (TIED; metrics disagree on
80->160: accessibility +0.022, pass@1 -0.042, both noise). At fixed 48-step compute the practical
sweet spot is ~80 tasks; 160 is harder to fit stably. Single-seed (seed variance unmodeled).
NEXT: write durable doc (lead with accessibility; state n80~=n160 tie; flag single-seed + fixed-
compute + n=24 power). Goal 3 stays deferred. Then final verification + commit decision.

### [2026-06-04 10:05 UTC] SESSION COMPLETE — goals 1&2 done+documented; verification clean; goal 3 deferred
DELIVERABLES (durable, in /workspaces/content on branch issue52-bf16-body-contrastive / PR #55):
  - docs/issue52-goal1-recall-capacity-2026-06-04.md (decisive scale=0 eval; memory real at k=1)
  - docs/issue52-goal2-corpus-scaling-2026-06-04.md (40->N: modest, diminishing; sweet spot ~80)
  - tools/_recall_capacity_probe.py + _capacity_analysis.py + _goal2_analysis.py +
    build_scaling_train_corpora.py + _fetch_goal2_ckpts.py + drivers (all REMOVE-BEFORE-MERGE)
  - benchmarks/mbpp_recall_train_{80,160}.jsonl (nested, eval-disjoint)
  - Next-session handoff: instructions/handoff_20260604T100000Z.md (Phase-2-ready, reshaped by findings)
HEADLINE FINDINGS:
  G1: adapter-as-memory REAL at k=1 (c3 +0.292 over no-adapter floor, CI excl 0). Multi-item capacity
      OPEN (k>1 conditioning was off-distribution; c3 vs warm indistinguishable at all k). Engine is
      the wrong vehicle for the decisive test (prompt carries state -> dirty scale=0).
  G2: corpus 40->N nudges held-out recall up (accessibility monotone; pass@1 jump-then-plateau), all
      sizes beat floor, but per-doubling gains within noise at n=24. n80~=n160. N=160 unstable at
      fixed 48-step compute. Diminishing returns vs the 10->40 jump.
VERIFY: ruff src clean, mypy src clean (37 files), 335 unit tests pass. No src/ changes (only tools/
  benchmarks/ docs/ scaffolding). NOT committed (awaiting user go per operating rules).
GOAL 3 (Phase-2 RL): DEFERRED. Now unblocked but a high-variance multi-session build. Highest-value
  next step (per findings): train hypernet on MULTI-TASK conditioning, then re-run capacity probe to
  test the true multi-item ceiling IN-DISTRIBUTION — more informative than more single-task RL data.
REFLECTIONS MONITOR: armed all session (task bixy8zff5); no AI-engineer guidance fired.

### [2026-06-04 10:10 UTC] advisor-final tightening — trained-vs-warm SHARPENS goal 2; interference claim verified
Advisor caught: I tested beyond-40 increments + vs-floor, but never trained-vs-WARM (the right "did
training help" baseline). Computed (paired, by task):
  accessibility m_zero - warm: n40 +0.105[+0.033,+0.180]✓  n80 +0.119[+0.049,+0.201]✓
                               n160 +0.141[+0.082,+0.209]✓  -> SIGNIFICANT at every size, GROWS with N.
  pass@1 - warm: n40 +0.125[-0.042,+0.292]  n80 +0.250[+0.042,+0.458]✓  n160 +0.208[-0.042,+0.458]
                 -> N=80 is the FIRST size to reliably beat warm; n40 doesn't, n160 falls back below
                    sig (its instability). Sharper goal-2 answer: "need >40 (~80) tasks before the
                    trained adapter reliably beats warm-start on functional recall." NOT a flat null.
  CROSS-SESSION CONTROL VALIDATED: warm reproduced EXACTLY 0.530 this session; n40-warm +0.105
  reproduces the Phase-1 handoff headline (+0.105 [+0.033,+0.182]) -> reused n40 number is trustworthy.
Interference claim verified (emitted_def != entry_point over all rows): adapter arms (warm/c3/n80/
  n160) 0/96 wrong-or-missing name; scale0 12/96 had NO valid def. So "name cue retrieves the right
  name, failures are wrong-BODY" holds for the ADAPTER arms (tightened the goal-1 doc accordingly).
Both docs updated: goal-2 +section (C) trained-vs-warm + verdict + limits (n160 instability reconcile,
cross-session control); goal-1 interference sentence tightened. Goal-1 k=1 verdict UNCHANGED.

### [2026-06-04 10:20 UTC] autonomous tick — PR #55 CI note (no action taken; needs user decision)
PR #55 OPEN, MERGEABLE; only red check = `lint-and-type-check` (FAILURE). Diagnosed: CI runs
`uv run ruff check .` (WHOLE repo) -> 32 E501 line-too-long, ALL in tools/ at commit 718a7ff
(pre-existing; zero src/tests failures). This is the handoff-acknowledged "tools/ pre-existing lint
debt = REMOVE-BEFORE-MERGE scaffolding." This session's new tools add ~49 more E501 (current tree=83).
NOT actioned: work uncommitted (push unauthorized), debt pre-existing+accepted, and the real fix is a
POLICY decision the user owns — (a) exclude tools/ from ruff (per-file-ignore/extend-exclude), (b) wrap
lines, or (c) leave it (tools removed before merge). Local-only fixes give no CI benefit + churn the
review. Goals 1&2 remain done/verified; goal 3 deferred. Reflections monitor still armed.

### [2026-06-04 10:35 UTC] CORRECTION (project owner) — hypernet is SINGLE-STEP; multistep lives in the RUNNER
User correction to my goal-1 framing: the hypernet is NEVER trained on multistep/multi-task
conditioning — it is ALWAYS single-step. The RUNE RUNNER is multistep, and the hypernet must be
OPTIMAL as the SUBSTRATE that carries stepwise context across the runner's iterations on ONE task
(long-running repair, code-continuation generation, development). So "adapter memory growing" =
one coherent run's evolving state carried across steps, NOT k independent tasks packed into one
adapter.
WHAT THIS FIXES:
  - My goal-1 capacity probe (pack k disjoint MBPP tasks, recall-by-name) tested an OFF-DESIGN
    scenario (I'd flagged it off-distribution; it's worse than that — it models something the system
    doesn't do). The k>1 decay measures nothing real about the architecture or the thesis.
  - WITHDRAWN: "train the hypernet on multi-task conditioning to raise capacity." Wrong-headed —
    the hypernet stays single-step by design.
  - SURVIVES: the k=1 result (single-step recall beats scale=0, in-distribution).
CORRECTED decisive multistep experiment (the real next step): run ONE task through N runner
iterations (repair/continuation) with the accumulating trajectory carried by the ADAPTER (prompt
held minimal, NOT re-fed code/error history), vs scale=0 (no adapter, same minimal prompt); primary
= integrate-success/pass@1 at final step. This is the handoff's original "state evicted into the
adapter, prompt fixed." Tooling start: diag_continuation_probe.py / cont_probe.py / gate_trajectory.py.
ARTIFACTS UPDATED: docs/issue52-goal1-* (correction box + verdict #2/#4 + limits + future-work),
docs/issue52-goal2-* (Phase-2 implication), instructions/handoff_20260604T100000Z.md (Phase-2 steer),
memory issue52-goals12-findings. NOTE: my earlier "engine is the wrong vehicle" was about the CURRENT
engine putting state in the PROMPT; the corrected test is precisely to run the runner in a
prompt-minimal/adapter-carries-context mode — i.e. fix that, don't avoid the runner.

### [2026-06-04 10:40 UTC] reflections fired — incorporated AI-engineer pushbacks (4) on the session block
The AI engineer endorsed: diag_recoverability fix, goal-1 over-claim autopsy, owner correction
(single-step hypernet / runner-multistep), goal-2 trained-vs-warm discriminant, config unification.
PUSHBACKS incorporated (all durable-artifact wording fixes):
  P1 "gate-1 CLOSED oversells": it is NOT closed. issue#52 scorecard needs BOTH m-mismatch AND m-zero
     >0 per target; `diff` m-zero=-0.028 (edit-local memory still flat). Retention was on an --n 24
     PILOT subsample of 323-row val.clean, no bootstrap; avoid n=14 underpowered. -> handoff "Where we
     are" reworded to "Phase-2 retention baseline established (goal/tail/avoid); diff edit-local flat;
     full-val retention pending." (NOT "QUALIFIED PASS/closed".)
  P2 "handoff drift": the obsolete "MULTI-ITEM capacity never tested in-distribution" line removed;
     replaced with "the open question is the RUNNER's within-run multistep (adapter carries trajectory
     across repair/continuation, prompt minimal), NOT k-task packing (off-design)."
  P3 goal-2: added (a) fixed-48-step confound is ATTENUATED not eliminated — needs a matched-epoch
     ablation (N=160 vs N=80 at ~2x steps) to separate "needs more steps" from "hurts at this lr"; and
     (b) SCOPE note: Goal-2 optimizes MBPP body recall and does NOT repair the diff/edit-local gap on
     external_codereview. (single-seed + pass@1 15->14 disagreement already in Limits.)
  P4 pre-registration for the runner-based multistep test folded into handoff next-step (a-e): per-step
     re-encode on current state (training contract, not frozen step-0); scale=0 each step w/ identical
     minimal prompt (no repair_history in prompt); primary=success@stepN, secondary=frozen-probe
     accessibility on latest body tail vs warm (forgetting canary); recitation = final cross-cond PLUS
     step-k-vs-(k-1) verbatim when spec changed; log adapter-cond tokens vs prompt tokens per step.
CI policy (reaffirmed): for merge either ruff `extend-exclude=["tools"]` OR fix E501 in tools kept
  through review; 83 E501 block CI regardless of REMOVE-BEFORE-MERGE intent. User's call.
ARTIFACTS: handoff_20260604T100000Z.md (Where-we-are + next-step pre-reg), docs/issue52-goal2-*
  (Limits: matched-epoch + diff-scope). goal-1 doc already corrected at 10:35. No experiments re-run.

### [2026-06-04 10:45 UTC] NEXT EXPERIMENT (design) — multi-turn adapter-as-memory-substrate test
GOAL (user): show the adapter enables MULTI-TURN reasoning as a memory substrate — better than
scale=0 AND better than the warm pre-warmed Sakana checkpoint. Hypernet stays SINGLE-STEP; the RUNNER
is multistep and re-encodes the adapter each turn from the evolving trajectory.
OPERATIONAL THESIS: hold the PROMPT MINIMAL each turn (no prior code/errors in it); then the ONLY
channel that can carry cross-turn context is the adapter. Multi-turn progress under a minimal prompt
IS the memory substrate working. Builds on the VALIDATED single-step signal in
diag_continuation_probe.py (real > zero > contradictory across a token boundary) + gate_trajectory.py
(matched/mismatch/zero edit-local scoring). New = extend to a MULTI-TURN loop.

DESIGN — 2x2 (prompt-mode x adapter), isolates the two win conditions:
                 scale=0(no adapter)   warm Sakana        c3 (ours)
  prompt-MINIMAL      floor             arm B              ARM A (the claim)
  prompt-FULL         --                --                 ceiling reference
  - A > scale=0 (minimal): adapter is CARRYING cross-turn context (prompt has no history). [>scale0]
  - A > warm  (minimal):  OUR training improved the substrate, not just any adapter.       [>Sakana]
  - A ~= prompt-FULL ceiling: KILLER result — adapter recovers context-window performance at near-zero
    prompt cost ("never fill the prompt with what belongs in the adapter").
DOMAIN: multi-turn REPAIR first (clearest memory-of-prior-attempt, real pass/fail) — tasks that fail
  attempt-1 and need iterative fixing; then code-CONTINUATION (extend diag_continuation_probe across
  multiple boundaries).
PROTOCOL INVARIANTS (pre-registered): per-step hypernet RE-ENCODE on the CURRENT state via the training
  contract (render_training_format_trajectory) — NOT a frozen step-0 adapter; identical minimal prompt
  template every turn (NO repair_history/code_trajectory in prompt); identical sandbox seeds per step
  across arms; same task set, paired.

METRICS TO MONITOR:
  PRIMARY (does the substrate enable multi-turn progress?)
   1. Final task success @ turn N — pass@1 on real tests at last turn (headline; A vs scale0 vs warm).
   2. Success-vs-turn CURVE — fraction passing by turn k. SHAPE is the proof: scale0 flat (no memory),
      adapter arms climb. Flat-scale0 + rising-c3 = cleanest "memory accumulates" visual.
   3. Strict-improvement rate — frac turns fixing >=1 prev-failing test without breaking others.
  SECONDARY (mechanism + guards)
   4. Per-step accessibility canary (frozen probe) — gold-logprob of the LATEST code/body tail under
      the current adapter vs warm = forgetting guard (does turn-k context erode turn-(k-1)?).
   5. Edit-local matched-mismatch-zero per turn (reuse gate_trajectory scoring) — carried signal is
      SPECIFIC to this run's trajectory, not generic lift.
  INSTRUMENT (the thesis surface — sells "memory substrate")
   6. Adapter-conditioning tokens vs PROMPT tokens per turn — prompt ~flat while adapter trajectory
      grows = "context lives in the adapter, sparing the window" (multi-turn analogue of study_tokens
      vs prompt_tokens from the capacity probe).
  FAILURE MODES (pre-registered kills)
   7. Recitation — does turn k repeat turn (k-1) VERBATIM when feedback/spec changed? memory must
      ADAPT not parrot (cross-cond at final turn + step-to-step verbatim check).
   8. Error compounding — per-step hot-swap can compound mistakes; a DECLINING adapter curve catches it.
POWER: pre-register a FIXED slice of ~40-60 tasks that genuinely fail base on attempt-1 (from the train
  pool, paired across arms), bootstrap CIs on per-turn success deltas (same discipline that caught the
  goal-1 over-claims; n=24 was underpowered).

BUILD PLAN: prompt-minimal multi-turn runner harness (add a minimal-prompt mode that routes state ONLY
  into adapter conditioning — current engine puts history in the prompt) + 3-arm driver + metrics
  1/2/6 first (headline). Nucleus to extend: diag_continuation_probe.py; scoring: gate_trajectory.py.
STATUS: design only, awaiting user go to scaffold. NOT started. Goal 3 (Phase-2 RL) stays after this
  shows a real multi-turn signal. Conclusions so far unchanged: single-step recall real at k=1
  (c3 +0.292 vs floor, CI excl 0); corpus scaling diminishing (sweet spot ~80); k>1 task-packing
  off-design; diff edit-local retention still flat (full-val pending).

### [2026-06-04 11:00 UTC] CORRECTION + STANDING RULE — runner is ALREADY prompt-minimal; ALWAYS use rune, never a parallel runner
TWO things settled this turn:
1) VERIFIED (Jinja2 templates) — my earlier claim "the engine puts prior state in the PROMPT -> scale=0
   dirty -> engine is the wrong vehicle" is FALSE. Evidence:
   - Action fields: field2=trajectory_template (rich: existing_code/repair_history/code_outputs/
     accumulated_code), field3=prompt_template (the GENERATION prompt). The GENERATION prompt uses the
     prompt_* family, which is MINIMAL and explicitly defers to the adapter:
       prompt_code "Follow the architecture plan IN YOUR CONTEXT"; prompt_plan "...spec IN YOUR
       CONTEXT"; prompt_integrate "Combine all implementations FROM YOUR CONTEXT"; prompt_code_continue
       = task_description[:200]; prompt_code_repair = subtask_name + fix_guidance[:150] (diagnosis dir);
       prompt_diagnose = error_summary[:300] (only raw-error one; diagnose doesn't execute code).
   - trajectory_template is NEVER rendered by the engine (defined in state.py only; zero render calls).
     Live adapter conditioning = render_training_format_trajectory(task, current_code, feedback) in
     graph.py. So CODE/feedback history -> the ADAPTER; the prompt stays minimal.
   => The runner is ALREADY prompt-minimal / adapter-as-memory by design. scale=0 on it is essentially
      CLEAN (minor hints: fix_guidance[:150] repair, error_summary[:300] diagnose). My "need to build a
      prompt-minimal harness" was WRONG.
2) STANDING RULE (owner): ALWAYS use the rune runner/engine for experiments; NEVER build a parallel
   runner / reimplement the generation path in a one-off probe. If rune lacks something, CHANGE RUNE
   (carefully, with owner review) — don't fork. My capacity probe + proposed harness violated this;
   retracted. (Only the goal-1 k=1 number survives, and it's in-distribution single-step.)
REVISED NEXT EXPERIMENT (multi-turn memory substrate) — runs ON RUNE:
   - Drive `rune run` on tasks needing iteration (repair / code-continuation) under 3 adapter
     conditions: scale=0 (adapter_scaling=0), warm Sakana, c3 (ours). Same tasks, paired.
   - Win conditions: c3/warm > scale=0 (adapter carries cross-turn context); c3 > warm (our training
     improved the substrate); c3 ~= a prompt-FULL reference (adapter recovers context-window perf at
     near-zero prompt cost).
   - Metrics unchanged (success@N, success-vs-turn curve, strict-improvement, per-step accessibility
     canary, edit-local matched/mismatch/zero, adapter-cond-tokens vs prompt-tokens, recitation,
     error-compounding). Pre-registration items a-e now mapped onto the ENGINE path (per-step re-encode
     already native; scale=0 via adapter_scaling=0; logging hooks added INSIDE the engine step, not a
     wrapper).
   - Changes to rune likely needed (small, owner-reviewed): per-step token-count logging; metric/canary
     hooks; a clean experiment switch for the 3 adapter arms. NO new runner.
ARTIFACTS UPDATED: docs/issue52-goal1-* (retracted-premise box + Future-work -> rune), handoff
  (goal-1 summary retraction + next-step "run on rune, never parallel"). Scratchpad 10:45 build-plan
  superseded by this block (use rune; don't build a harness).

### [2026-06-04 11:05 UTC] reflections fired again — incorporated P1–P5 on the multistep design
AI engineer endorsed: 10:40 incorporation, 11:00 template audit (independently confirmed code.j2/
code_repair.j2 exist but NOT wired in policy.py), 10:45 pre-registration (adopt as-is).
PUSHBACKS incorporated (handoff updated):
  P1 Handoff goal-1 summary still said "k>1 capacity is an OPEN QUESTION, not a measured limit" —
     CONFLICTS with owner correction. Replaced: "k>1 task-packing is OFF-DESIGN — withdrawn, not
     measured. Do NOT rerun the capacity probe."
  P2 prompt-FULL "ceiling" arm is NOT rune-as-shipped — it needs the orphaned code.j2/code_repair.j2
     (history in prompt) = a deliberate SECOND code path. DEFERRED from v1. v1 = 3 arms (scale0/warm/
     c3) on the MINIMAL path only. Prompt-FULL only after an owner-reviewed `prompt_mode: minimal|full`
     switch that swaps prompt_* <-> rich templates WITHOUT changing adapter conditioning. Claiming
     c3-minimal~=prompt-FULL before that confounds adapter memory with prompt stuffing.
  P3 scale=0 semantics: adapter_scaling=0 scales lora_B only (scale_lora_b) — VERIFY once via forward-
     parity check that effective LoRA delta is EXACTLY zero, log in experiment card. Also repair/diagnose
     prompts inject fix_guidance[:150]/error_summary[:300] = a DERIVED-HINT CHANNEL (not raw trajectory,
     not empty); acceptable only because identical across arms — name it, don't oversell "prompt
     minimal" as "task-only."
  P4 score REPAIR (+continuation) turns ONLY; decompose/plan don't exercise cross-turn code memory. Use
     tasks that FORCE the repair loop (fail attempt-1 in sandbox) so integrate@N reflects adapter memory
     not planning quality.
  P5 hygiene: handoff "no AI-engineer guidance fired" was false — fixed to "fired twice (~10:14,~10:26);
     both incorporated." (This scratchpad's 10:20 block is append-only history; superseded by 10:40+.)
ARTIFACTS: handoff goal-1 summary (P1), GOAL-3 next-step pre-registration a–g (P2/P3/P4), scratchpad-
  convention line (P5). NO experiments re-run. Next action per engineer: add engine logging hooks +
  run the 3-arm minimal `rune run` study; defer prompt-FULL until the prompt_mode switch is designed.

### [2026-06-04 12:30 UTC] SESSION START — GOAL 3 step 1 (v1 multi-turn substrate) pre-registered; advisor-reviewed
READ: PRODUCT.md, CLAUDE.md, handoff_20260604T100000Z.md, full scratchpad, full reflections (P1-P5).
ENV: GPU 0/23GB free; RAM 15GB (offload_base=False mandatory); c3 ckpt present /tmp/phase1/ckpt/
  c3_t07_lp2_lg1.pt; warm = third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/
  pytorch_model.bin; reflections monitor armed (bg bn1fd1guc, baseline md5 64d55ac2).
APPARATUS (verified by reading src): run_benchmark IS the v1 harness — drives full engine.ainvoke
  (code->diagnose->repair* up to MAX_RETRIES=4 + continuation), scores held-out test_code, paired by
  task_id. 3 arms = 3 configs (checkpoint + adapter_scaling). NOTHING forked (standing rule honored).
PRE-REG CONFIRMATIONS:
  (a) per-step re-encode NATIVE (graph.py:200 render_training_format_trajectory each action; :243 in
      continuation). Confirm-not-reimplement satisfied.
  (b) scale=0 = adapter_scaling=0 -> scale_lora_b zeroes lora_B (adapter.py:27) -> delta exactly 0.
      Forward-parity check pending (will log max-abs logit diff in card).
  (c) DERIVED-HINT FINDING (important, corrects 11:00 paraphrase): prompt_code/prompt_code_repair/
      prompt_plan inject project_label = task[:200]. MBPP specs short => FULL TASK SPEC IS IN THE
      PROMPT every turn, all arms. So memory channel under test = prior failing CODE+ERROR carried in
      the adapter (NOT in the repair prompt), NOT the task spec. Repair prompt has task+fix_guidance
      [:150] only; prior code lives solely in adapter conditioning.
  (d) score repair turns only; slice = tasks failing scale=0 attempt-1. MBPP trips _is_simple_task ->
      single _main subtask -> clean code/diagnose/repair (planning-free), exactly pre-reg (d).
  (g) token-logging hook = ModelWrapper.count_tokens + 2 guarded mlflow.log_metric in graph.py:328
      block. Deliberate rune change, owner review, not committed.
ADVISOR BLOCKERS LOCKED:
  B1 headline = SUCCESS-VS-TURN CURVE + recovery gap (final-attempt1), NOT final@N (final@N confounds
     Goal-1 attempt-1 help with repair-memory). Score each turn POST-HOC vs held-out test_code
     (engine feedback.exit_code only means "ran"); persist via run_benchmark(sessions_dir=) ->
     session.jsonl output field (= per-step extracted code).
  B2 slice DISJOINT from c3 train-40. Candidate pool = heldout-24 + (train160\train40) = 144 c3-unseen
     tasks WITH on-disk test_code. Verified disjoint.
  B3 set seed explicitly (config.seed default None => _seed_rng never fires => arms unpaired).
ARTIFACT: docs/issue52-goal3-multiturn-substrate-2026-06-04.md (full pre-registration card).
NEXT: (1) forward-parity check + 3-arm smoke on 2-3 tasks (sizes wall-clock, shakes warm load);
  (2) candidate-pool scale=0 run -> freeze 40-60 attempt-1-fail slice; (3) 3-arm batch; (4) post-hoc
  curve + bootstrap CIs + canaries. Nothing committed (awaiting go).

### [2026-06-04 12:55 UTC] PARITY (b) PASSED; token hook (g) landed; driver built; smoke running
TOKEN HOOK (g) — the one deliberate rune change (owner-review, NOT committed):
  - src/rune/model/wrapper.py: ModelWrapper.count_tokens(text) (add_special_tokens=False).
  - src/rune/engine/graph.py: 2 guarded mlflow.log_metric (adapter_cond_tokens, prompt_tokens) in
    the existing step MLflow block. Guarded by mlflow.active_run() so CPU CI unaffected.
  - VERIFY: ruff+mypy clean on both; 335 unit tests pass.
FORWARD-PARITY (b) PASSED: tools/_goal3_multiturn_probe.py parity ->
  max|scale0-adapter - base| = 0.0 EXACT (disable_adapter ref) => adapter_scaling=0 is a clean
  no-adapter floor on the same path. max|scale1 - base| = 18.52 => c3 adapter non-trivially applied.
  Logged in card "Parity check (b) — PASSED".
DRIVER: tools/_goal3_multiturn_probe.py (REMOVE-BEFORE-MERGE) — parity/run/score/analyze. `run`
  drives run_benchmark(sessions_dir=) on the rune engine (no forked generation). `score` re-scores
  each code/repair step's session.jsonl output vs held-out test_code (held-out scoring, B1) + counts
  trajectory vs prompt tokens post-hoc (g, belt-and-suspenders with the engine hook). `analyze`
  applies the scale0-attempt-1-fail slice (B2-disjoint pool) + paired bootstrap CIs on final-success
  delta and RECOVERY GAP delta (B1 isolator). ruff clean.
CANDIDATE POOL: benchmarks/goal3_candidate_pool.json — 144 c3-disjoint tasks (heldout-24 +
  train160\train40), sha256 e9e34f66, all with held-out 3-assert test_code.
NEXT: 3-arm smoke (3 tasks) running -> size wall-clock + verify warm load -> then full batch.

### [2026-06-04 13:25 UTC] SMOKE WALL-CLOCK ANOMALY — suspect continuation explosion on scale0
Smoke (3 arms x 3 tasks) still on scale0 FIRST task after >15min, GPU active (68%), no session.jsonl
yet. Advisor flagged: don't size from a cold-start-contaminated first task; BUT this is past cold
start. Suspected cause = the continuation sub-loop (graph.py): scale0 (base, no memory) emits output
that never validate_syntax-passes -> up to cont_budget=5 rounds x 2048 tok PER code/repair action,
across the full code->diagnose->repair* loop (~5 code/repair actions) = ~tens of 2048-tok gens/task.
That is REAL runner behavior but makes the FAILING arm very expensive (cost coupled to outcome).
NOTE: driver lacks logging.basicConfig so engine INFO ("continuation round N") is suppressed -> blind
to it live; will confirm from first session.jsonl output lengths when it lands (waiter b0lki29w2).
NOT KILLED (advisor: let it finish for the warm .bin load-validation; bounded by budget/cont_budget).
DECISION PENDING (needs first-task data, not yet a scope change): if a single failing task is ~15-20
min, even a 60-task all-3-arm batch is >>12h. Levers (advisor): cut POOL size (slice >=40), NEVER
budget; background under watchdog across session boundary is the intended multi-session path. If
continuation is the culprit, a defensible faithful option is documenting it, not disabling it.

### [2026-06-04 14:30 UTC] ROOT-CAUSED the slow run — 3 engine miswirings FIXED (owner-steered, deep trace)
Owner steer: "overdecomposing or not recognizing EOS"; "all extraction through pydantic, revisit json
outputs, don't give up"; "log everything to MLflow"; "deep tracethrough to catch miswirings preemptively".
DID a full pipeline trace (generate->extract->parse->policy) + settled H_A vs H_B with a live raw dump.
SETTLED H_A (decisive, no-GPU-needed reasoning + 1 GPU raw dump): result.text = `{"code":"```py\n..."}`
  = VALID JSON, grammar+pydantic WORKING. Fences live INSIDE the .code value. So structured output is
  fine; the user's JSON path is sound (don't give up on it). NOT H_B (no silent pydantic bypass).
THREE MISWIRINGS (docs/issue52-goal3-pipeline-traceThrough-2026-06-04.md):
  M1 [DOMINANT] markdown fence inside JSON code value -> sandbox SyntaxError line 1 -> logically-correct
     code fails spuriously (mbpp/106 `return tpl+tuple(lst)` is CORRECT, would pass de-fenced) ->
     pollutes the attempt-1-fail slice + triggers needless repair. FIX: _strip_code_fences() sanitizes
     the pydantic-parsed .code value in extract_code_from_raw; non-pydantic fallbacks now LOG LOUDLY
     (owner: extraction through pydantic). Tests added.
  M2 targeted diagnose + hallucinated subtask_name ("write_function" for "_main") -> diagnosis on phantom
     key -> select_action never routes _main to repair -> diagnose x9 livelock until budget (the slowness).
     FIX: targeted-diagnose fallback attaches guidance to the actual target + reopens it. Test added.
  M3 driver MLflow optional + smoke ran without it -> nothing to see. FIX: `run` now ALWAYS
     configure_mlflow+tracked_run+log_dataset+params(arm/ckpt/scaling/seed/pool sha)+pass@1; engine step
     auto-logs per-turn trajectory/prompt/output + (g) token metrics. Server up localhost:5000.
RULED CLEAN: xgrammar (works), continuation sub-loop (doesn't fire; outputs short -> my earlier
  "continuation explosion" theory was WRONG), EOS (recognized), thinking phase (real cost not a bug).
VERIFY: 338 unit tests pass (+3), ruff+mypy clean on changed src. NOT committed.
CONSEQUENCE: M1 (not M2) was the dominant cost. Post-fix attempt-1 passes skip the loop; per-task
  should drop ~15-20min -> minutes. MUST re-measure steady-state before sizing batch (reduce pool not
  budget; slice >=40). NEXT: verification smoke (3 arms, few tasks, WITH mlflow) to confirm no livelock
  + visible runs + new wall-clock.

### [2026-06-04 15:45 UTC] Code extraction refactored to MAINTAINED LIBRARIES (owner: no fragile regex); verify results
OWNER STEER: use known packages (json_repair, parse_llm_code, instructor), look at outlines; "do your
research, don't reinvent or use fragile non-generalizable regex."
RESEARCH (WebFetch/WebSearch + EMPIRICAL tests on our real captured strings, CPU `uv run --with`):
  - json-repair (PyPI, MIT, v0.60.1, ACTIVE): robust JSON parse — extracts JSON from prose, REPAIRS
    truncated JSON, returns '' for non-JSON. ADOPTED for the JSON layer.
  - parse_llm_code (MIT, 3 stars, last release 2024): extract_first_code works on fenced/comment but
    returns None on CLEAN (no-fence) AND truncated code -> would DROP correct solutions. REJECTED
    (empirically fragile for our cases; ironic given the "no fragile" ask).
  - llm-output-parser: only parse_json/parse_xml, NOT a code-block extractor. WRONG TOOL.
  - instructor: API/Ollama/vLLM clients only, NO raw-HF support, no standalone parser -> DOESN'T FIT
    our local xgrammar+HF path. REJECTED (documented).
  - outlines: generation-time constraint for transformers (peer of xgrammar). Like xgrammar it
    constrains JSON STRUCTURE but CANNOT stop fences inside a free `code` STRING field -> wouldn't fix
    the bug; we already have working xgrammar. NOT ADOPTED (documented why).
  - markdown-it-py (CommonMark ref parser, used by rich/mkdocs/jupyter, ALREADY installed via rich):
    EMPIRICALLY handles ALL cases — fenced, comment-prefixed, TRUNCATED (unclosed fence->EOF), and
    CLEAN (passthrough); fence-only match avoids mis-extracting indented Python bodies. ADOPTED.
DECISION (research-backed, beats blind adherence to named pkgs): extract_code_from_raw pipeline =
  json-repair (robust JSON) -> pydantic model_validate (THE CONTRACT, owner ask) -> markdown-it-py
  fence extraction. Deleted the fragile regex (_strip_code_fences) AND the custom src/rune/engine/
  json_repair.py + test (superseded by the json-repair PACKAGE — "don't reinvent the wheel"). Deps:
  uv add json-repair markdown-it-py. VERIFY: 329 unit tests pass, ruff+mypy clean (src+tests).
VERIFY SMOKE (regex-fix version, 3 arms x5; representative of post-M1/M2 engine):
  - scale0 pass@1 = 0.80 (4/5) <- FENCE FIX CONFIRMED: correct code now passes attempt-1 (mbpp/106 =
    1 `code` action, NO repair loop). Pre-fix it livelocked + failed all. M1 was indeed the dominant bug.
  - wall-clock: passing task ~90s (1 action, thinking 1024 dominates), failing task ~10min (full repair
    loop). scale0 avg 229s/task. => batch sizing: passing tasks cheap, failing tasks costly.
  - warm/c3 crashed rc=1 @2s = NameError 're' (imported parse.py MID-REFACTOR before I restored import
    re). ARTIFACT of edit timing, NOT a bug; fixed (tests green). Re-running all 3 arms now (verify2).
NEXT: confirm verify2 (lib code, all 3 arms, mlflow) -> then size + launch full batch on fixed engine.

### [2026-06-04 16:10 UTC] VALIDATION — ran REAL captured failure modes through the new lib pipeline
Owner: "run failure modes through these new tools to see if it solves the problem before committing."
PULLED 5 REAL raw code/repair result.text from MLflow (verify scale0 run) -> through the REAL new
extract_code_from_raw (json-repair -> pydantic -> markdown-it-py):
  - raw is valid JSON: 5/5 (H_A holds at scale: xgrammar+pydantic fine).
  - raw code value had ``` fence: ~1/5 (failure mode rate for scale0).
  - NEW extract removed fence: 5/5 (markdown-it). FENCE PROBLEM = SOLVED.
  - ast.parse after extract: 3/5. The 2 non-parsing are GENUINE MODEL-CONTENT failures, NOT
    extraction bugs:
      * tuple_to_int (HAD fence): fence is a COMPLETE block (raw 994B, valid JSON, closing ``` present),
        removed cleanly; still SyntaxErrors because the MODEL'S code is broken (defines tuple_to_int but
        asserts call string_to_tuple_list; unbalanced `)]` `)`). OLD kept fence -> SyntaxError too (worse).
      * step_4 repair: model output PROSE ("The function checks if...") not code; no fence; returned
        verbatim. Can't extract code from prose. Genuine model failure.
  - The engine RECOVERS these via the repair loop (tuple_to_int step_2 repair -> clean parseable code).
PROOF the fence fix is real end-to-end: verify scale0 pass@1 = 0.80 (4/5); mbpp/106 passes attempt-1
  (1 code action). OLD regression on the one fenced complete case: OLD .code verbatim -> SyntaxError;
  NEW -> fence removed (residual error is the model's own broken code, not ours).
VERDICT: the new tools SOLVE the fence failure mode (the actual bug). Residual non-parsing = genuine
  model errors (broken code / prose), orthogonal to extraction, handled by the repair loop. Changes are
  safe to keep. Gathering more samples from verify2 (warm/c3 adapter arms) for a larger sample.

### [2026-06-04 16:45 UTC] DIAGNOSED why the runner generates poorly (owner: prompts/adapter/langgraph)
Posted findings to PR #55 (issue-comment 4622149905). Then diagnosed attempt-1 generation quality from
15 real task-runs (verify+verify2). docs/issue52-goal3-runner-quality-diagnosis-2026-06-04.md.
ATTEMPT-1: 8 PASS / 5 SyntaxError / 2 NameError. THREE root causes:
  1. PROMPT over-generation (dominant): prompt_code.j2 "Write tests FIRST, then implement" -> instruct
     model writes whole pytest MODULES / extra functions / comment walls -> exceeds max_tokens 2048 ->
     continuation extends -> still truncates -> SyntaxError. mbpp/113 scale0: output began
     "# test_check_integer.py\nimport pytest..." grew to 13,835 chars -> unterminated string. median
     out_len PASS=394 vs FAIL=908 (max 13835). 4/15 define >1 func; 4/15 emit self-tests (stripped=waste).
  2. ADAPTER over-conditioning (the "adapter templates" issue): adapter applied at effective scaling
     45.25 (un-divided alpha; right for logprob PROBE, too strong for free GEN). warm degenerated
     attempt-1 to literally `{"code":"error"}` (5 chars) on mbpp/108+115 -> NameError; repair recovered.
     Matches prior "spec-divergence at scaling>=0.49". HYPOTHESIS (thesis-relevant): c3 (trained on the
     trajectory format) degenerates LESS than warm (doc-to-lora, never saw it) — c3/mbpp106 attempt-1 =
     69 chars concise+correct vs scale0 362 / warm 394. n=1, c3 arm still running.
  3. Missing entry_point in prompt -> model picks wrong fn name -> NameError. BenchTask.entry_point
     exists but is NOT threaded into engine (make_initial_state drops it; ctx has none).
  Secondary: project_label = task[:200] truncates long docstring examples (mbpp/108 cut mid-list).
NOT the problem: LANGGRAPH wiring (post-fix repair loop advances+terminates+recovers: verify2 scale0
  5/5, warm 5/5 FINAL); xgrammar/pydantic (valid JSON); adapter conditioning FORMAT (well-formed) — it's
  conditioning STRENGTH, not format.
RECO (owner-review; change runner+experiment): (1) prompt -> "only the function {entry_point}, no tests/
  extras/prose"; (2) thread entry_point into engine+prompt; (3) raise project_label cap; (4) lower
  GENERATION-time adapter scaling vs the 45.25 probe value OR treat c3<warm degeneration as a result.
NEXT: await verify2 c3 (degeneration-rate c3 vs warm = thesis signal); then owner decision on prompt/
  scaling fixes before the full batch. Nothing committed.

### [2026-06-04 17:20 UTC] Implemented owner-approved runner fixes 1-3; HPO (fix 4) infra confirmed
OWNER decisions: (1) prompt — remove "Write tests FIRST" but DON'T force "no tests" either (neutral);
(2)+(3) DO IT (thread entry_point, raise project_label cap); (4) adapter scaling -> bench HPO, signal
when ready.
IMPLEMENTED (mergeable engine changes, owner-approved, NOT committed):
  - prompt_code.j2 / prompt_code_repair.j2: dropped "Write tests FIRST, then implement to pass them";
    added conditional `{% if entry_point %}Implement the function `{{entry_point}}`.{% endif %}`
    (neutral on tests).
  - entry_point threaded: RunState gains entry_point; make_initial_state(task,budget,entry_point="");
    state_to_ctx sets ctx["entry_point"]=state.get(...); run_benchmark passes task.entry_point. rune run
    free-form -> "" (omitted in prompt). 
  - _PROJECT_LABEL_CAP 200 -> 1200 (was truncating MBPP docstring asserts mid-example).
  - Tests: added test_templates entry_point cases (render + neutrality + omitted-when-empty); updated
    _ctx() superset. VERIFY: 331 unit pass, ruff+mypy clean.
VERIFY2 FINAL (pre-prompt-fix, fixed-engine): scale0 5/5, warm 5/5, c3 4/5. c3 ATTEMPT-1 is the
  CLEANEST substrate: passed 108 (warm degenerated->"error") and 113 (scale0 over-gen 13.8k->fail) with
  concise code; one hard degeneration (115, out_len 7) repair couldn't fix. => thesis-relevant: c3
  (trained on the trajectory format) > warm/scale0 at attempt-1, but adapter scaling 1.0 still
  occasionally degenerates -> that's exactly fix 4 (HPO the magic scaling).
FIX 4 INFRA: src/rune/bench/hpo.py ALREADY tunes adapter_scaling + temperature + presence_penalty +
  max_phase_iterations + cont_multiplier; ranges from config.yaml hpo: (currently {} -> must populate).
  objective requires ALL 5 ranges defined. checkpoint for HPO must point to c3 (bench has no --checkpoint
  -> use a dedicated hpo config with checkpoint_path=c3, or env). Plan to signal: populate hpo ranges
  (adapter_scaling focus ~[0.1,1.5]), tasks=goal3_pool_simple (entry_point now threaded), n_trials,
  cost est. Validating prompt fix now (promptfix smoke scale0+c3) before signaling HPO-ready.

### [2026-06-04 17:45 UTC] Owner plan: no-HPO validation FIRST, then sped-up HPO
EARLY no-HPO signal (new prompt, scale0): max_out_len 13835 -> 312 (over-generation ELIMINATED; no
more pytest-module rambling). Full promptfix smoke (scale0+c3 @ production settings) still running.
HPO SPEEDUPS applied to configs/goal3_scaling_hpo.yaml (owner-suggested): max_tokens 2048->768,
  thinking_budget 1024->512, max_phase_iterations(retry budget) 10->6, cont_budget 5->3, n_trials 12->6.
  Est cost now ~2h (was ~8h). Search: adapter_scaling[0.1,1.5] + presence_penalty[0,1.5]; rest pinned.
  Tasks benchmarks/goal3_hpo_tasks.json (30, entry_point threaded). checkpoint=c3.
PLAN: (1) finish no-HPO validation (promptfix) -> report good effect (attempt-1 up, faster, no
  over-gen); (2) on owner go, launch sped-up HPO to find the magic adapter scaling.

### [2026-06-04 18:30 UTC] SYSTEMATIC DEBUG — single-word degeneration; Phase-1 root cause hypothesis
BUG: model emits degenerate single-word "code" e.g. {"code":"success"|"python"|"INTERNAL_ERROR"|
"unexpected_token"}. Confirmed GENUINE model output (raw result.text), not extraction. Hits scale0
(NO adapter) AND c3 -> NOT purely adapter. Intermittent (sampling).
PHASE-1 EVIDENCE (cheap, decisive — tokenizer/template, no GPU):
  - Qwen3-4B-Instruct-2507 = NON-THINKING instruct variant: chat template adds NO <think> block;
    assistant prompt is just `<|im_start|>assistant\n`. Real turn-end = <|im_end|> (151645).
  - BUT inference.py thinking phase sets eos_token_id=</think> (151668) ONLY. So the model's natural
    <|im_end|> does NOT stop generation -> the "thinking" phase rambles PAST the model's turn end for
    up to thinking_budget=1024 tokens (code + <|im_end|> + garbage/new-turn), then a FAKE </think>\n is
    appended, then the STRUCTURED phase generates {"code":...} from that corrupted prefix -> degenerates
    to a high-freq status word.
HYPOTHESIS (single, testable): the forced </think>-terminated thinking phase is wrong for this
  non-thinking model; it corrupts the context and the structured phase degenerates. Predicts:
  thinking=True -> degenerate, thinking=False -> clean; INDEPENDENT of adapter and schema.
TEST (owner hint): tools/_degen_probe.py — OUTSIDE the runner, 2x2x2 ablation adapter(base/c3) x
  schema(CodeResult/none) x thinking(on/off), N=4 on a degeneration-prone prompt; logs degen rate +
  thinking-phase tail. Running. If thinking=False is clean across the board -> fix = don't force the
  thinking phase for this model (thinking_budget=0 / instruct-aware).

### [2026-06-04 19:30 UTC] ROOT CAUSE CONFIRMED — thinking phase (primary) + presence_penalty (secondary suspect)
FAITHFUL reproduction (exact degenerate prompts mbpp/108/113/115, engine settings, schema=True, N=6x3):
  adapter=base thinking=ON  -> DEGEN 6/18 (33%)
  adapter=base thinking=OFF -> DEGEN 2/18 (11%)
  adapter=c3   thinking=ON  -> DEGEN 5/18 (28%)
  adapter=c3   thinking=OFF -> DEGEN 0/18 (0%)
SMOKING-GUN mechanism (thinking-phase text on degenerate runs):
  base: '...return sorted(combined)\n```<|im_end|>\n<|im_end|>\n<|im_start|> user\nYou are' — model
        FINISHED code, emitted <|im_end|>, then HALLUCINATED A NEW USER TURN. 
  c3:   '***END OF RESPONSE***<|im_end|>\n***END OF RESPONSE***<|im_end|>...' repeated.
  => CONFIRMED: inference.py thinking phase sets eos=</think> (151668) only, NOT the model's real
     turn-end <|im_end|> (151645). The non-thinking Qwen3-Instruct-2507 never emits </think>, so the
     "thinking" phase runs PAST the turn end (rambles / starts a new turn) up to thinking_budget=1024;
     fake </think> appended; structured phase generates from the corrupted prefix -> single-word collapse.
FIX (primary): thinking phase is WRONG for this non-thinking model -> thinking_budget=0 (the engine's
  non-thinking path) takes c3 0/18, base 3x lower. CONFIRMED by ablation (Phase-3 minimal test flips it).
RESIDUAL/secondary (owner suspect VALIDATED): base thinking=OFF still 2/18. presence_penalty=1.5 applies
  ONLY in the structured phase, subtracting 1.5 from every already-emitted CODE token (indentation,
  return, :, newlines) -> aggressive against normal repetitive code -> plausible push to short collapse.
  Running presence ablation (thinking OFF, presence 1.5 vs 0, base+c3) to pin it.
NEXT: presence result -> then FIX via model-profile generalization in config.yaml (thinking flag,
  presence_penalty, dtype, attn, + adapter checkpoint + warmstart checkpoint — owner ask). Then re-run.

### [2026-06-04 19:55 UTC] THOUGHTS & CONCLUSIONS — GOAL-3 runner debugging arc (consolidated)

WHAT THIS SESSION ACTUALLY BECAME: started as "run the v1 multi-turn experiment", became a deep
debugging of WHY the rune runner generates badly. That was the right pivot — running the experiment on
a broken runner would have produced meaningless numbers. The runner is now understood end-to-end.

THE BUGS, IN CAUSAL ORDER (each found by tracing real model output, not guessing):

1. ENGINE MISWIRINGS (fixed, tested, uncommitted):
   a. Markdown fence INSIDE the JSON code value -> sandbox SyntaxError on line 1 -> correct code fails
      spuriously. Fixed: json-repair -> pydantic -> markdown-it-py (maintained libs, no fragile regex;
      deleted custom json_repair.py). Validated on real captured outputs: fence removed 5/5.
   b. Targeted diagnose + hallucinated subtask_name -> diagnosis on phantom key -> diagnose livelock
      (10 wasted gens/task). Fixed: attach guidance to the real target.
   c. MLflow not logging the runs. Fixed: driver always logs.
   These three made the runner SLOW + the experiment INVALID. With them fixed the repair loop works
   and recovers most failures (final pass@1 5/5 on smokes).

2. GENERATION-QUALITY ROOT CAUSES (the real story, owner-steered):
   PRIMARY (CONFIRMED): the forced "thinking phase" is wrong for Qwen3-4B-Instruct-2507 (a NON-thinking
     model). inference.py sets the thinking-phase eos to </think> (151668) ONLY; the model's real
     turn-end <|im_end|> (151645) is NOT a stop, and a non-thinking model never emits </think>. So the
     "thinking" runs PAST the turn end — caught red-handed rambling into a hallucinated new user turn
     ('...<|im_end|>\n<|im_start|> user\nYou are...') — then a fake </think> is appended and the
     structured phase generates from that corrupted prefix, collapsing to a single word
     ({"code":"success"|"python"|"INTERNAL_ERROR"}). Faithful ablation (real prompts, schema=True, N=18):
     thinking ON 33%/28% degen (base/c3) -> thinking OFF 11%/0%. Turning the thinking phase off is the
     fix; it took c3 to 0/18.
   SECONDARY (owner suspect, ablation pending): presence_penalty=1.5 applies only in the structured
     phase, subtracting 1.5 from every already-emitted CODE token (indentation/return/:/newlines) —
     aggressive against normal repetitive code; likely the residual base 2/18 with thinking off.
   PROMPT (fixed): "Write tests FIRST" made the model emit whole pytest modules -> 13.8k chars ->
     truncation. Removed (neutral on tests); added entry_point naming; raised project_label cap 200->1200.

KEY INSIGHT (the lesson): EVERY one of these is a MODEL-SPECIFIC behavior hardcoded into a supposedly
  general runner — </think> token, the thinking phase itself, presence_penalty=1.5 (a Qwen card rec),
  dtype, attn_implementation, the adapter scaling contract. The runner pretends to be model-agnostic but
  silently assumes one model's quirks. That is the generalization debt, and this bug is a symptom.

CONCLUSION / DECISION: fix the bug AS the generalization (owner directive). config.yaml gets a per-model
  profile selected WITH the model:
    model: {id, thinking: bool, presence_penalty, dtype, attn_implementation, [stop/think tokens]}
    adapter: {checkpoint, warmstart_checkpoint, scaling}
  inference.py/ModelWrapper read the profile instead of hardcoding. thinking:false -> skip the broken
  thinking phase (the fix). Selecting a model now forces selecting its generation contract, so a
  thinking-phase / eos / penalty mismatch CANNOT silently recur on a new model.

OPEN / NEXT:
  - confirm presence_penalty ablation (pending) -> decide its default in the profile.
  - implement the model/adapter profile in config.yaml + wire inference.py/wrapper to read it; keep
    backward-compatible defaults; tests.
  - re-run the exact degenerate prompts through the FIXED engine -> verify degeneration ~0.
  - THEN: no-HPO validation of all fixes (good-effect baseline) -> sped-up HPO for adapter_scaling +
    presence_penalty magic numbers.
  - Nothing committed all session (owner rule). Mergeable engine fixes: extraction pipeline, diagnose
    target, token hook, entry_point threading, prompt, project_label cap, + the upcoming model profile.

META (process): systematic-debugging discipline paid off — my FIRST degeneration probe used the wrong
  prompt and "worked" (0/4), which would have falsely exonerated the thinking phase. Reproducing on the
  EXACT failing prompts before concluding is what turned a guess into a confirmed cause. Owner's two
  pushes (test outside the runner w/ and w/o adapter & schema; check presence_penalty) were both right.

### [2026-06-04 20:05 UTC] PRESENCE ABLATION RESULT — secondary cause CONFIRMED; degeneration fully characterized
presence_penalty ablation (thinking OFF, N=6x3 real prompts, schema=True):
  base presence=1.5 -> DEGEN 2/18
  base presence=0.0 -> DEGEN 0/18   <- presence=0 eliminates the residual
  c3   presence=1.5 -> DEGEN 0/18
  c3   presence=0.0 -> DEGEN 0/18
=> SECONDARY CAUSE CONFIRMED (owner hunch): presence_penalty=1.5 (flat OpenAI-style, subtracts 1.5 from
   every already-emitted token, INCLUDING legitimately-repeated code tokens) drives ~11% of base
   failures. presence=0 fixes them.
FULL CHARACTERIZATION of the single-word/incomplete degeneration:
   thinking OFF + presence_penalty 0  ==> 0/18 BOTH arms. Degeneration eliminated.
   (thinking phase = primary, the dominant + adapter-independent driver; presence_penalty = secondary,
    base-mostly. c3+thinking-off was already clean.)
NOTE: the Qwen "presence_penalty=1.5" card rec is for chat repetition, NOT code; applying a flat
   presence penalty to code (which must repeat indentation/keywords) is harmful. Belongs in the model
   profile as a tunable; HPO can refine, but 0 (or small) is the right default for code gen.
DECISION: fix = model/adapter profile in config.yaml. Defaults for THIS model: thinking:false,
   presence_penalty:0.0 (was 1.5). Plus dtype/attn + adapter checkpoint + warmstart checkpoint.

### [2026-06-04 20:20 UTC] REFLECTIONS FIRED — 7 AI-engineer pushbacks incorporated
Engineer ENDORSES (strong): batch->debug pivot; project_label-in-prompt correction; B1 pre-reg; parity;
M1/M2 + maintained extraction; thinking-phase root cause + MODEL PROFILE generalization ("not a one-off
thinking_budget=0 hack buried in a yaml leaf" — my profile approach is right).
PUSHBACKS (all adopted):
  P1 ALL pre-fix smoke numbers are STALE (verify/verify2/promptfix ran through different engine states):
     c3 4/5, scale0 0.80, warm 5/5 = NOT thesis evidence. RULE: after profile+thinking_budget=0+presence
     default, rerun ONE 3-arm smoke on the FROZEN B2 slice, THEN batch. No thesis numbers from earlier smokes.
  P2 REGIME mismatch: Goal-3 (spec in prompt via project_label) != Phase-1/capacity (spec absent). c3>warm
     at attempt-1 in Goal-3 = "adapter helps iterative REPAIR given spec-in-prompt", NOT absent-spec memory.
     Keep evidence panels separate in PR/issue.
  P3 gen scaling != probe scaling: don't HPO gen_scaling while reporting logprob probes at 45.25 unlabeled.
     Pre-register probe_scaling (frozen 45.25 for _specificity_probe/recoverability) vs gen_scaling
     (HPO [0.1,1.5] for rune run/bench).
  P4 HPO ONLY after: profile+tests; degenerate-rerun ~0 with thinking:false; no-HPO 3-arm on frozen slice.
     (matches my order.)
  P5 presence default: ablation landed -> presence_penalty 0.0 default for instruct profile. DONE.
  P6 report by OUTCOME STRATUM (pass-at-attempt-1 vs needed-repair), not single pass@1 headline. B2 slice
     40-60 failing x3 arms x~10min = multi-session under run_guarded.
  P7 COMMIT SURFACE RISK: large mergeable delta, ZERO commits -> loss on recycle. On owner go, split:
     (1) extraction/diagnose (2) goal-3 observability (3) prompt/entry_point (4) model profile. <- flag owner.
ACTION NOW: finish model profile (config.py done; wrapper dtype/attn; config.yaml; tests) -> degenerate
  rerun through ENGINE (verify ~0) -> then frozen-slice 3-arm smoke. Mark all earlier smoke verdicts stale.

### [2026-06-04 21:10 UTC] COMMITTED + PUSHED (granular); GATE = fixed runner beats pre-fix
COMMITS (PR #55, pushed 718a7ff..07dd6ae): 5 focused engine commits + 1 remove-before-merge durability:
  930cbfc extraction (json-repair+pydantic+markdown-it-py)
  7927ebf diagnose livelock fix
  002a913 token-accounting hook
  9592c8a entry_point + drop forced-tests prompt + full spec
  a1b4a78 per-model generation profile (thinking_budget=0, presence=0.0, dtype/attn) = the FIX
  07dd6ae goal-3 tooling/docs/journal [remove-before-merge]
GATE (post-fix vs pre-fix promptfix; runner-quality, small-n, NOT thesis evidence per P1):
  scale0: attempt1 2/5->6/6, degen 3/5->0/6, steps 3->1
  c3    : attempt1 0/4->5/6, degen 2/4->0/6, final 3/4->5/6, steps 3->1
  => fixed runner succeeds MORE + far more efficiently (1 step vs 3, degeneration eliminated). The exact
     degenerate tasks (108/113/115) now pass attempt-1 in 1 step.
CAVEATS: task counts differ (promptfix interrupted 5/4 vs postfix 6); small-n smokes. Per P1, the THESIS
  gate is the clean no-HPO 3-arm run on a FROZEN 40-60 slice, reported by outcome stratum.
NEXT (AI-engineer stricter gate): freeze B2 slice -> ONE clean 3-arm smoke (profile fix) -> full batch ->
  B1 curves + paired CIs. THEN HPO (ranges updated: presence [0,0.3], gen_scaling [0.1,1.5] != probe 45.25,
  thinking off, temp [0,0.4]). All earlier smoke verdicts marked STALE.

### [2026-06-04 21:40 UTC] OWNER REDIRECT — spec-in-adapter / name-in-prompt (decisive runner test)
KEY REALIZATION (owner): post-fix, scale0 ~= c3 because the SPEC IS IN THE PROMPT (project_label) on
easy MBPP -> base solves single-turn -> adapter redundant. We engineered away the repair loop too.
OWNER IDEA (implement): make the adapter LOAD-BEARING on the runner ->
  - prompt_mode="reference": PROMPT references the mission by NAME only ("implement `merge_sorted_list`,
    spec + DoD in your context"); the SPEC (+ signature, Phase-1 style) lives ONLY in the adapter
    conditioning (render_training_format_trajectory task slot). scale0 then has the name only (no spec
    anywhere) -> FLOOR; c3/warm succeed iff the adapter carries the spec. The GAP = adapter-as-memory ON
    THE RUNNER (runner-native version of the capacity-probe k=1 +0.292). 
  - embed signature ("def fn(...):") in the adapter context (Phase-1 conditioned on task+partial code).
  - MULTI-TURN: use a harder real problem (SWE-bench / internet) where attempt-1 genuinely fails so the
    repair loop runs (MBPP is too easy / single-turn after the fixes).
PLAN: add prompt_mode (config + step_node override of project_label) [deliberate engine feature, owner-
  requested]; build a reference pool w/ signature in description; run 3-arm (scale0/warm/c3) reference;
  compare scale0(name-only floor) vs c3(spec-in-adapter). Then SWE-bench for multi-turn.

### [2026-06-04 22:10 UTC] BUILT prompt_mode reference_a/b (spec-in-adapter); experiment launched
ENGINE FEATURE (owner-requested, uncommitted): config.prompt_mode {full|reference_a|reference_b}.
  reference_*: prompt refers to mission by NAME; spec lives ONLY in adapter conditioning; empty
  sections omitted; prompt uses the EXACT section names in the adapter.
  - reference_a: adapter = `## Task\n{spec}` (faithful to c3 training surface).
  - reference_b: adapter = `## Mission / ## Specification / ## Definition of Done / ## Current Code`
    (signature stub via _derive_signature, arity from doctest). Off-distribution for c3 (no retrain).
  graph.py: _split_spec, _derive_signature, render_reference_adapter + step_node branch (code/repair).
  templates prompt_reference_a.j2 / _b.j2. driver --prompt-mode. VERIFY: mypy+ruff clean, 331 tests.
EXPERIMENT (issue52-goal3-specinadapter): A+B x {scale0(name-only FLOOR), warm, c3} x 8 tasks.
  HYPOTHESIS: scale0 solves only name-leaking tasks (floor); c3/warm > scale0 => adapter CARRIES THE
  SPEC on the runner (runner-native capacity-probe k=1). A (on-distribution) expected > B (off-dist for
  current c3). If c3 ~= scale0 even here, the trained recall doesn't transfer to runner generation.

### [2026-06-04 22:55 UTC] SPEC-IN-ADAPTER RESULT — adapter carries the spec ON THE RUNNER (positive)
prompt_mode reference (prompt = mission NAME only; spec lives ONLY in adapter), 8 MBPP tasks:
                  reference_a   reference_b
  scale0 (name)     2/8           2/8
  warm              2/8           2/8
  c3                4/8           5/8
=> scale0=warm=2/8 FLOOR (only name-leaking tasks: tuple_to_int, mbpp/118; warm adapter adds NOTHING).
   c3 DOUBLES the floor with the spec ONLY in the adapter (per-task: c3 also solved check_integer,
   empty_dit, merge_sorted_list, mbpp/120 — tasks the name can't carry). => the c3 adapter genuinely
   CARRIES THE SPEC on the real rune runner (runner-native confirmation of Phase-1 +0.292 / capacity k=1).
   c3's TRAINING is what matters (warm doesn't carry it).
SURPRISE: reference_b (Mission/Spec/DoD + def signature stub) 5/8 > reference_a (plain ## Task) 4/8,
   EVEN THOUGH B is OFF-DISTRIBUTION for c3 (never trained on it). The structured framing + signature
   anchor HELPED generation -> retraining c3 on the B format is a promising lever.
REGIME NOTE (keep separate, AI-eng P2): this is the SPEC-ABSENT-FROM-PROMPT regime (= Phase-1/capacity),
   now on the runner. Distinct from the multi-step repair-memory test (spec in prompt, adapter carries
   prior code across turns) running now.
DECISION (owner): record findings (PR+scratchpad), then REVERT all prompt_mode/reference A/B code; keep
   the branch ONE path (full mode), no modes/fallbacks. Findings persist; experimental code does not.

### [2026-06-04 23:40 UTC] Variant C REFINED (owner) + reference_c run launched
OWNER: mission name should appear in the adapter; generic arg names won't link; strengthen prompt<->context link.
REFINED reference_c (closer to dist + precise + strong link):
  ADAPTER = `## Mission\n{entry_point}` + `## Task\n{spec}` + `## Current Code\n{REAL signature}`.
    Mission name explicit (links to prompt's "mission `X`"); real arg names from the recall reference
    (num1,num2,num3 / test_list,test_tup / ...) via threaded BenchTask.signature (mirrors entry_point).
  PROMPT references its Mission/Task/Current Code + names the mission. Output the complete function.
THREADING: BenchTask.signature -> make_initial_state -> RunState.signature -> ctx -> render_reference_adapter.
  ref pool benchmarks/goal3_ref_pool.json (8 MBPP, real sigs); multistep tasks given real sigs too.
  VERIFY: 331 tests, ruff+mypy clean.
RUN (issue52-goal3-refc): 1-turn reference_c (8 MBPP, scale0 floor vs c3) THEN multi-turn reference_c
  (4 hard tasks, scale0 vs c3, budget 10). Compares to scale=0 for single + multi-turn benefit.
  Prior: refa c3 4/8, refb c3 5/8, scale0/warm floor 2/8. C isolates: name-anchor+real-sig vs B's restructure.

### [2026-06-05 00:30 UTC] MULTI-TURN reference_c CONFOUNDED (famous names) + reference_b1 built/launched
TRAINING TEMPLATE (ground truth, byte-exact = render_training_format_trajectory(spec,"","")):
  `## Task\n{spec docstring+doctest}\n\n## Current Code\n[EMPTY]\n\n## Review Feedback\n[EMPTY]`. Target=body.
  => NONE of A/B/C match: A dropped the empty headers; B restructured; C added Mission+filled CurrentCode.
1-TURN reference_c: scale0 2/8, c3 3/8 (A=4,B=5,C=3 c3 — all within noise at n=8; robust = c3 > 2/8 floor).
MULTI-TURN reference_c (4 hard tasks): scale0 3/4 > c3 1/4 -> CONFOUNDED. EVIDENCE: scale0 (NAME-only
  prompt, no adapter) produced REAL impls FROM THE NAME — decode_string=stack decode, int_to_roman=roman
  map, merge_intervals=real merge (famous LeetCode, memorized by name). The c3 adapter (scale 45.25)
  PERTURBS the memorized solutions -> worse. ONLY calculate (ambiguous name) shows the adapter helping:
  scale0 wrong (x+y, didn't know it's expr-parsing); c3 right signature/approach (spec from adapter). =>
  the "hard" tasks leak by name -> invalid adapter test. FIX = OPAQUE names.
reference_b1 (owner spec): reference_b content on OUR training headers ->
  ## Task: "Mission: {ep}\n{prose}"; ## Current Code: "{real sig}\n    # complete the implementation";
  ## Review Feedback: "To be done: {assert}". Mission name in BOTH prompt and adapter (strongest linker).
  prompt references Task/Current Code/Review Feedback. mypy/ruff clean, 331 tests.
LAUNCHED refb1: 1-turn reference_b1 (8 MBPP, slot into A/B/C) + multi-turn reference_b1 on OPAQUE tasks
  (solve_p1..p4 — name can't leak, spec only in adapter -> VALID c3-vs-scale0 test). bwexze7j2.

### [2026-06-05 01:15 UTC] RESULTS + INTERPRETATION + PLANS (spec-in-adapter A/B/C/b1; over-perturbation)

RESULTS — 1-turn reference variants, c3 pass@1 (8 MBPP, spec ONLY in adapter, name in prompt):
  floor (scale0/warm, name-only): 2/8
  reference_a (plain ## Task, empty sections DROPPED):            4/8
  reference_b (Mission/Spec/DoD restructure + generic sig):       5/8
  reference_c (## Mission + ## Task + real sig in Current Code):  3/8
  reference_b1 (Mission-in-Task + real sig+comment in CurrentCode + "to-do:assert" in Feedback): 1/8 (!!)
RESULTS — multi-turn (hard tasks):
  reference_c on FAMOUS-named tasks: scale0 3/4 > c3 1/4 (CONFOUNDED — see below).
  reference_b1 on OPAQUE tasks (solve_p1..p4): scale0 2/4, c3 PENDING. [NB scale0 2/4 on opaque is
    suspicious — check whether the opaque rename truly removed leak, or base stumbles into solutions.]

INTERPRETATION (key findings):
1. TRAINING-SURFACE SENSITIVITY is real and dominant. The byte-exact distill surface =
   `## Task\n{spec}` + EMPTY `## Current Code` + EMPTY `## Review Feedback` (target=body). b1 FILLS the
   two sections that were empty in training -> c3 1/8, BELOW the 2/8 no-adapter floor. So filling the
   empty training slots DEGRADES c3 below baseline. reference_a (closest, just dropped the empty headers)
   = 4/8. => c3's recall is fragile to off-distribution conditioning; the enrichments (signature, DoD,
   restructure) HURT more than help once they push off the trained surface. (AI-eng P2 + my 00:30 note.)
   UNTESTED + likely-best = the TRAINING-EXACT variant: `## Task\n{spec}` + EMPTY Current Code + EMPTY
   Review Feedback (headers KEPT) + name-only prompt. None of A/B/C/b1 is this.
2. n=8 is DESCRIPTIVE ONLY (AI-eng P1): A4/B5/C3/b1·1 are within noise; paired CI spans zero. Robust
   claim = c3 > 2/8 floor in the on-distribution variants (A/B/C) -> adapter IS load-bearing on the
   runner for spec-absent-from-prompt (= on-engine Phase-1/capacity story; NOT repair-memory Goal-3).
   Do NOT promote B>A or pick a winner without replication on the held-out 24 with bootstrap.
3. OVER-PERTURBATION (owner hypothesis) — SUPPORTED, two mechanisms:
   (a) MAGNITUDE: adapter applied at effective scaling 45.25 (un-divided Sakana alpha) is strong; on
       tasks the base already solves (famous names, spec-in-prompt) it perturbs the base's own solution
       -> c3 <= scale0. Scaling (gen_scaling multiplier <1) is the magnitude knob -> HPO.
   (b) DIRECTION: c3 = body_recall_guarded -> trained to REPRODUCE the reference body. That direction
       conflicts with the base's OWN (also-correct) solution, so even at lower magnitude it nudges away
       from the base's answer toward a specific reference impl. Scaling reduces but doesn't remove this.
   AI-eng P3: the famous-name multi-turn c3<scale0 is name-leak AND 45.25-perturbation entangled;
   disentangle by an opaque-task arm c3@lower-scaling vs c3@45.25 (hold gen_scaling fixed across arms).

DECISION on "is there something BESIDES optimizing scaling?": YES.
  - FORMAT (free, structural, do FIRST): use the TRAINING-EXACT surface (empty Current Code/Review
    Feedback kept) + name prompt. The b1<floor result says staying on-distribution matters MORE than
    enrichments. Add as reference_d ("training_exact") and re-run A vs d.
  - OBJECTIVE (Phase-2, deeper): the DIRECTION conflict won't yield to scaling. Retrain the adapter on
    an OUTCOME/correctness signal (pass@1), not reference-body reproduction, + KL-anchor to the base so
    it SUPPLIES the spec without overwriting the base's own correct code. This is the planned cooperative
    outcome-RL; the over-perturbation finding is its motivation.
  - SCALING (HPO): the magnitude lever; run after format. Search gen_scaling [0.1,1.5] distinct from the
    FROZEN probe_scaling 45.25 (AI-eng P3).

PLANS (ordered):
  1. Add reference_d = TRAINING-EXACT (## Task + empty Current Code + empty Review Feedback, name prompt);
     re-run 1-turn A vs d on the 8 MBPP. Expect d >= A (most on-distribution).
  2. Replicate the best 1-2 variants on the held-out 24 with PAIRED BOOTSTRAP CIs (AI-eng P1) before any
     durable doc claim.
  3. Disentangle name-leak vs scaling on multi-turn: opaque tasks, arms c3@45.25 vs c3@gen_scaling (AI-eng P3).
  4. THEN scaling HPO (gen_scaling) on the chosen surface.
  5. If format+scaling leave a perturbation cost -> Phase-2 outcome-RL + KL-anchor (the structural fix).
HYGIENE (AI-eng P4/P5): persist these numbers (docs+scratchpad done); after recording, REVERT the
  experimental prompt_mode unless promoting ONE chosen surface (likely training-exact/A) as a reviewed
  product feature. Keep regimes separate: post-fix gate (spec-in-prompt) = runner FIXED; spec-in-adapter
  = adapter LOAD-BEARING; do not merge into one headline.
REFLECTIONS: 5 AI-engineer pushbacks (P1 n=8 CI; P2 lead with b1/training-surface; P3 gen-scaling
  confound; P4 revert prompt_mode; P5 keep gate vs adapter-memory separate) — all incorporated above.

### [2026-06-05 01:25 UTC] b1 FINAL — WORST variant; over-perturbation confirmed via format deviation
b1 c3: 1-turn 1/8 (BELOW the 2/8 no-adapter floor); multi-turn opaque 0/4. b1 is the WORST in everything.
=> CONFIRMS training-surface sensitivity: b1 FILLS the ## Current Code + ## Review Feedback sections that
   were EMPTY in distillation -> pushes c3 so far off-distribution it degrades BELOW baseline. "Don't
   perturb so strongly you lose what the model trained on" (owner) = literally what b1 did via FORMAT,
   not scaling. Ranking c3 1-turn: b(5) > a(4) > c(3) > floor(2) > b1(1). On-distribution-ness tracks it:
   a/c keep plain ## Task; b restructures but its content is spec-faithful; b1 fills empty slots = worst.
CAVEAT: scale0 opaque multi-turn 2/4 is SUSPICIOUS (attempt-1 outputs were stdin-reading stubs w/ no
   spec; shouldn't pass solve_pN(args) tests) -> likely a scoring/repair artifact; do NOT trust scale0
   opaque floor without checking. c3 0/4 is the clean signal (off-dist b1 on hard tasks -> total fail).
NEXT (unchanged plan, sharper): reference_d = TRAINING-EXACT surface first; replicate on held-out 24 w/
   bootstrap; then gen_scaling HPO; Phase-2 outcome-RL+KL if perturbation persists. REVERT prompt_mode
   experiment after recording (keep at most ONE chosen surface as a reviewed feature).

### [2026-06-05 01:50 UTC] LAUNCHED flavor x scaling HPO on bench (metric = held-out pass@1)
DECISIONS (owner): metric = pass@1 (held-out validation); search = prompt/adapter FLAVOR (categorical)
  x adapter_scaling (continuous). Rest pinned to clean profile (thinking 0, presence 0, temp 0.3,
  max_tokens 768, budget 4).
FLAVORS (spectrum training-faithful -> precise/prompt-coupled):
  training_exact (## Task + EMPTY Current Code + EMPTY Review Feedback — byte-exact distill surface;
    the missing most-faithful one) | reference_a (plain ## Task) | reference_c (## Mission+## Task+real
    sig) | reference_b (Mission/Spec/DoD+sig) | reference_b1 (all filled + to-do + directive prompt).
IMPL: added training_exact (graph.py render_reference_adapter + prompt_training_exact.j2 + gate);
  extended src/rune/bench/hpo.py objective to suggest_categorical("prompt_mode") when configured
  (opt-in, no default change). config configs/goal3_flavor_hpo.yaml. 24-task pool
  benchmarks/goal3_ref_pool_24.json = heldout-24 (c3-disjoint) + real sigs. VERIFY 331 tests, ruff/mypy clean.
RUN: `rune bench --hpo --config configs/goal3_flavor_hpo.yaml --tasks-file goal3_ref_pool_24.json --fresh`.
  c3 checkpoint; 24 trials; tuning/val 0.70/0.30; MLflow rune-bench-hpo. ~2-3h. Optuna TPE over
  (flavor x scale) -> best flavor+scale by held-out pass@1. Addresses n=8 noise (24-task held-out) +
  over-perturbation (scale is searched) + format question (flavor is searched).
HYPOTHESES going in: training_exact/reference_a (on-distribution) win on flavor; optimal scale < 1.0
  (over-perturbation). If best pass@1 only ~= name-floor, the recall doesn't transfer -> RL needed.
THEN (owner): proceed to RL training (Phase-2 outcome-RL + KL-anchor) informed by HPO — the structural
  fix for the DIRECTION-conflict (c3 reproduces reference body) that scaling can't resolve.

### [2026-06-05 02:25 UTC] HPO MLflow logging FIXED + restarted; multi-turn status
LOGGING FIX (src/rune/bench/hpo.py): MLflowCallback metric_name="tuning_pass_at_1" (was generic "value")
  + a progress callback logging best_tuning_pass_at_1 / trial_pass_at_1 / trials_completed as a metric
  SERIES on the PARENT run (so the optimization curve is traceable without opening every nested trial).
  Thread-safe via MlflowClient direct calls. Test updated; 6 hpo tests pass; ruff/mypy clean.
  RESTARTED the flavor x scaling HPO with proper logging (b06c2ar6n); old run's signal (low scale wins,
  reference_c/a lead) already captured. Interim (pre-restart, 9 trials): best reference_c@0.92 -> 0.41
  tuning; HIGH scale (1.2-1.47) clustered at ~0.18 = OVER-PERTURBATION confirmed in HPO.
MULTI-TURN status (owner Q): the HPO is 1-TURN MBPP (spec-in-adapter pass@1). Multi-turn results so far
  are POOR + CONFOUNDED: reference_c on famous-named hard tasks scale0 3/4 > c3 1/4; reference_b1 on
  OPAQUE hard tasks scale0 2/4, c3 0/4 (scale0 2/4 itself suspect). c3 worse than scale0 driven by
  (a) famous-name leak, (b) over-perturbation across REPAIR turns (adapter fires every turn -> worse than
  1-turn). PLAN: eval the 1-turn HPO winner (flavor, LOW scale) on the OPAQUE multi-turn set (cheap) —
  low scale should perturb less per turn. Multi-turn is where the reproduction-objective adapter hurts
  MOST -> strongest motivation for Phase-2 outcome-RL + KL-anchor. A dedicated multi-turn HPO is possible
  but expensive (hard tasks x repair x trials); do the winner-eval first.

### [2026-06-05 03:10 UTC] GRAPH REVISION (owner) — model-driven decompose + adapter carries the episode
INVESTIGATION (summary): policy.py select_action ALREADY does decompose->plan->code->diagnose->repair
  (per failing subtask)->integrate(only if >1). Three real problems found:
  P1 `_is_simple_task` REGEX gate in step_node pre-empts decompose: matches phrases ("write a function")
     + <200 words -> fabricates a synthetic _main subtask, skips decompose. So MBPP never decomposes; the
     hard "Implement calculate..." tasks DO -> their plan step runs full-mode and LEAKS the spec into the
     prompt (the multi-turn confound). The MODEL never decides.
  P2 adapter conditioning carried only (task, LATEST code, LATEST feedback) — render_training_format_
     trajectory ignored ctx.code_trajectory/repair_history. "What's been tried" was NOT in the adapter.
  P3 ROOT: c3 distilled with EMPTY ## Current Code / ## Review Feedback -> never trained to USE non-empty
     code/feedback; at repair the episode is recorded but the adapter can't leverage it. RL must fix.
REVISIONS (graph.py, 333 tests, ruff/mypy clean):
  R1 DELETED _is_simple_task + the synthetic-subtask gate + dead _SIMPLE_* consts; decompose ALWAYS runs,
     the MODEL decides 1 vs N. Strengthened prompt_decompose_concise.j2 with an explicit decision rule
     ("ONE subtask for a self-contained function; multiple only for separable components").
  R2 render_training_format_trajectory now carries the EPISODE: appends "## Previous Attempts" (prior
     failing code -> error, capped) when history exists; attempt-1 stays BYTE-IDENTICAL to the distill
     surface (test-locked). step_node passes ctx.code_trajectory[:-1] (attempts before the current one).
  R3 (next) = this enriched surface is the RL TRAINING surface: train c3/successor on episodes with
     non-empty code/feedback/Previous-Attempts so the adapter LEARNS to use what's been tried (fixes P3).
NB the running flavor HPO imported the OLD code (gate present, minimal conditioning) -> unaffected; it
  still answers the 1-turn flavor x scaling question on single-subtask MBPP. Post-revision, MBPP will
  decompose (model-decided) — slightly more steps; that's intended.

### [2026-06-04 20:52 UTC] Goal-3: codegen over-escape root cause + freeform fix (shipped)
- **Encoding question answered (no).** episode + embed probes (toy + 3 hard tasks w/ headroom): adapter raises lp(FIX) a lot (not pure recitation), but task-conditioning collapses the fix-vs-failure margin even with NO failure in context (task_only control) — so embedding/prompt changes can't be the lever. Encoding is second-order. docs/issue52-goal3-encoding-probe-2026-06-04.md.
- **Real bottleneck found via live engine run (c3, 4 hard tasks, pass@1 1/4).** Diagnose emitted phantom "syntax/unclosed-quote" guidance → misleading adapter conditioning → repair livelock. Root cause (MLflow raw outputs): model over-escapes `\n`→`\\n` inside JSON `{"code":...}` string → decodes to literal backslash-n → code collapses to ONE line → phantom line-1 SyntaxError. 2/4 hard fails were pure serialization, not logic. Both `\n` and `\\n` are valid JSON ⇒ no grammar can reject it.
- **Fix (advisor cut): abandon JSON-wrapping for code.** code/repair/integrate output_schema=None (freeform ```python fence), de-fence with markdown-it (extract_code_block), validate with ast/compile. Removed extract_code_from_raw + CodeResult/IntegrateResult + json.dumps continuation re-wrap. Probe: freeform 3/3 compile vs JSON stochastically collapsing. 330 unit tests + mypy + ruff green.
- **De-confound re-run:** 0 literal-`\n`, 0 phantom line-1 diag; int_to_roman FAIL→PASS; first-pass code pass@1 1/4→2/4. CAVEAT: steps=3 everywhere ⇒ repair loop never engaged (clean code → no diagnose trigger). "Repair" baseline still UNMEASURED; RL question needs a repair-triggering slice.
- Commits pushed (issue52-bf16-body-contrastive): bench MLflow HPO; engine freeform codegen + graph; remove-before-merge probes/docs. Follow-up pre-merge: re-run stratified single-turn gate (freeform regression coverage is n=1).

### [2026-06-04 21:35 UTC] Public-example oracle shipped; repair now engages
- Root cause of "no repair": engine in-loop check = run_in_sandbox(strip_self_tests(code)) = bare def → exit 0 → code_passed; held-out tests only at scoring. decode_string crashes on PUBLIC case yet engine marked passed (crash in a test call never executed in-loop). diagnose→repair fired ONLY on module-load crashes ⇒ adapter-as-repair-memory never exercised on correctness.
- Fix (rune/engine/oracle.py): extract_public_checks (stdlib doctest → equality asserts calling entry_point, WITH actual-vs-expected msg), defines_function (AST), build_probe → graph wires into sandbox + logs oracle_fired. No held-out leak (public example already in prompt; pass@1 gates on full held-out).
- Validated (4 hard tasks): decode_string NOW repairs (real IndexError traceback → accurate diagnosis "ensure stack not empty before pop", vs prior phantom "unclosed quote"); merge_intervals/int_to_roman no spurious repair; calculate passes its lone public example but fails held-out 100/10/2 → motivates model-judge. 341 unit tests + mypy + ruff. Commit f08ef97 (production, not RBM).
- NEXT: model-judge as SEPARATE arm (advisor: judge injects model content into the repair episode the adapter carries — keep out of deterministic baseline). Then re-test multi-turn thesis; slice = tasks failing public example on attempt 1.

### [2026-06-04 21:50 UTC] OVERNIGHT PLAN (autonomous; user asleep)
Goal: rerun earlier Goal-3 experiments on the CORRECTED runner (freeform codegen + public-example oracle + in-loop model-judge), reach conclusions, and complete an HPO run. Record + commit + push + PR-comment regularly for recoverability.

Corrected runner now = (a) code emitted freeform (no JSON over-escape), (b) public-example oracle gives in-loop correctness signal so diagnose→repair engages on logic errors, (c) model-judge (always on) flips code→failing when it names a concrete failing input the public example misses.

Steps:
1. Validate model-judge on 4 hard tasks: does it flip `calculate` (100/10/2 int-div) WITHOUT false-positiving correct tasks (int_to_roman/merge_intervals)? [in progress]
2. scale0 baseline (no-adapter floor) on goal3_ref_pool_24 (24 MBPP held-out, all have doctest+entry_point) with corrected runner → pass@1 floor.
3. Launch flavor×adapter_scaling HPO (configs/goal3_flavor_hpo.yaml: prompt_mode ∈ {training_exact, reference_a/b/b1/c} × adapter_scaling[0.1,1.5], c3, 24 trials, 0.70 tuning split, metric=held-out pass@1) on the corrected runner. Resumable (sqlite, load_if_exists).
4. Analyze HPO: best flavor+scaling, validation pass@1 vs scale0 floor = adapter value on corrected runner.
5. Conclusions on the adapter-as-memory thesis (single-turn pass@1; repair now engages).
Decisions: judge kept ON during HPO (user: "corrected runner"). If HPO too slow to finish overnight, reduce n_trials but keep ≥12; sqlite DB makes it resumable regardless. All runs logged to MLflow + sessions on disk.

### [2026-06-04 22:40 UTC] Judge validated → FALSE-POSITIVES; HPO will run oracle-only
Re-ran judge validation (4 hard tasks) after fixing the integrate metric-name crash (oracle_fired/"" trailing slash → MLflow reject; would have crashed every HPO trial; fixed commit 492693d).
Result pass@1 2/4: int_to_roman PASS, merge_intervals PASS, calculate FAIL, decode_string FAIL.
Behaviours:
- decode_string: oracle flips (IndexError on public 3[a]2[bc]) → 4 repair rounds → still FAIL (genuinely hard).
- calculate: JUDGE flips (held-out 100/10/2 gap the public example misses) → 4 repair rounds → still FAIL. Judge DID engage on a real gap (good) but couldn't fix in 4.
- int_to_roman: JUDGE FALSE-POSITIVE — flagged correct code "wrong on input 4"; its own `reason` chain-of-thought concluded the code is CORRECT ("✅ No failing input found"), but the parsed verdict was correct=false. Root cause: JudgeResult emitted `correct` BEFORE `reason` ⇒ verdict-before-reasoning. Spurious repair ran but happened to keep it correct (no regression this time, but risk + ~35min/4tasks slow).
Decisions:
- Judge root-cause fix: reorder JudgeResult → (reason, failing_input, correct) so verdict follows analysis; prompt asks brief-reason-first. Commit 9d16a66. STILL UNVALIDATED as in-loop signal.
- OVERNIGHT HPO runs on the deterministic public-example ORACLE with model_judge=OFF (configs/goal3_flavor_hpo.yaml model_judge:false; probe _build_cfg model_judge=False). Rationale: judge false-positives correct code → corrupts pass@1 (HPO metric) + 4x slower; advisor recommended judge as a SEPARATE arm not baked into baseline. This follows "once that [judge] works, test the rest" — judge does NOT work yet, oracle does.
Sequence now: scale0 baseline (oracle, full mode, adapter off) on ref_pool_24 → floor + HPO timing; then flavor×scaling HPO (c3, oracle, --fresh) on ref_pool_24 → best validation pass@1 vs floor.

### [2026-06-04 22:58 UTC] scale0 floor + HPO launched
- scale0 baseline (corrected runner, oracle, judge OFF, prompt_mode=full/spec-in-prompt, adapter OFF) on goal3_ref_pool_24: **pass@1 0.792 (19/24)** in 23 min (~58s/task). High floor — base model solves most simple MBPP unaided; limited headroom. This is the no-adapter reference.
- Launched flavor×adapter_scaling HPO: configs/goal3_flavor_hpo.yaml (model_judge:false), goal3_ref_pool_24, c3 ckpt, **n_trials=16** (reduced from 24 to guarantee overnight completion), --fresh (old buggy-runner study discarded). Tunes prompt_mode∈{training_exact,reference_a/b/b1/c} (spec-in-adapter) × adapter_scaling[0.1,1.5]; 0.70 tuning split → 17 tuning / 7 held-out val; metric=tuning pass@1, reported val_pass@1. Resumable (sqlite optuna_bench_hpo.db). Launcher tools/_run_goal3_hpo.py writes /tmp/goal3/hpo_result.json.
- Conclusion frame: best val_pass@1 (spec-in-adapter) vs 0.792 floor (spec-in-prompt) = whether the adapter carries the spec as well as the prompt (adapter-as-memory). NOTE high floor compresses the signal.

### [2026-06-05 03:40 UTC] HPO COMPLETE (16 trials, 4.3h)
flavor×adapter_scaling HPO (c3, corrected runner, oracle, judge OFF) on goal3_ref_pool_24:
- BEST: prompt_mode=**reference_a** @ adapter_scaling **0.627** (max_phase_iter 4, temp 0.3, presence 0).
- tuning pass@1 = **0.588** (10/17); held-out **validation pass@1 = 0.571 (4/7)**.
- best held since trial 2; other flavors (reference_c/b/b1, training_exact) and high scaling did not beat it → spec-in-adapter plateaus ~0.57–0.59.
- vs scale0 spec-in-prompt floor 0.792 (on 24). Running apples-to-apples scale0 on the SAME 7 held-out (val7: mbpp/115,133,118,119,106,113,135), then c3 prompt_mode=full on val7 (adapter as ADDED memory on top of prompt) for a 3-way: spec-in-adapter (HPO) vs spec-in-prompt-no-adapter (scale0) vs spec-in-prompt+adapter (c3-full).
PRELIM CONCLUSION: on this easy pool, moving the spec OUT of the prompt INTO the adapter (reference modes) costs accuracy — the hypernet doesn't carry the full task spec as reliably as the prompt. Consistent with earlier single-turn findings. Repair now genuinely engages (oracle) but the pool is too easy/high-floor to show multi-turn repair value; need a harder, repair-triggering slice for the repair-memory thesis.

### [2026-06-05 04:55 UTC] CORRECTION (owner caught it): scale0-full had spec in prompt → invalid floor
- prompt_mode=full: code prompt's `Project:` = task[:1200] = FULL spec+doctest. So scale0-full (0.792) is a spec-in-PROMPT ceiling, NOT a no-context floor. The adapter is irrelevant when the prompt has the answer.
- reference_a mode: prompt = mission name only, spec ONLY in adapter (proper memory test).
- TRUE memory comparison (both reference_a, minimal prompt): scale0 (adapter OFF, no spec) = **0.333 (8/24)** vs c3 (adapter ON, spec in adapter) = **0.583 (14/24)** ⇒ **adapter lifts +6/24 (+0.25)**. THE ADAPTER CARRIES THE SPEC AS MEMORY — thesis works single-turn. (Still < 0.792 prompt ceiling ⇒ lossy channel.)
- Reverses earlier (flawed) "spec-in-adapter worse" headline. c3-full-on-top-of-prompt remains ~neutral (18 vs 19/24) — adapter's value is as spec CARRIER, not extra boost on a complete prompt.
- NEXT (owner ask): HPO on HARDER/LONGER tasks for multi-turn gains. Only hard tasks on disk = 8 multistep (4 unique algos + 4 opaque). Need to curate ~24 harder pool (MBPP attempt-1-fail slice) or expand multistep. Floor will be lower on hard tasks → memory gain should be larger + repair engages.

### [2026-06-05 05:15 UTC] Per-flavor best + harder-task plan (owner-directed)
- Verified ALL reference flavor prompts contain ONLY the mission name (entry_point) — no spec leak (unlike full mode). reference_c/b1 say "spec/Specification" only as prose pointing to the adapter context. So flavor HPO = clean spec-in-adapter, minimal prompt. ✓ (owner check)
- Per-flavor best (clean 16-trial HPO, tuning pass@1): reference_a / reference_c / reference_b1 TIED at 0.588; reference_b 0.471; training_exact (exact c3 training fmt) WORST 0.294. → best templates = reference_a/b1/c (precise/structured beat training-faithful).
- HARDER-TASK PLAN (owner: curate attempt-1-fail slice; HPO where ADAPTER contains context; floor is just reference):
  1. multistep memory test (8 hard tasks, reference_a): scale0 floor [running] vs c3 [next] → memory gain on hard tasks.
  2. Discovery: reference_a c3, max-iters=1 (attempt-1 only), candidate_pool 144 → tasks failing attempt-1 = hard slice (repair engages).
  3. Hard slice = candidate-pool attempt-1 fails + 8 multistep.
  4. HPO: prompt_mode∈{reference_a,reference_b1,reference_c} × adapter_scaling, c3 (spec in adapter), on hard slice; scale0 reference floor for reference. Metric pass@1 + per-turn curve (multi-turn gain).

### [2026-06-05 05:25 UTC] Published-benchmark calibration (owner ask: MBPP for this model)
Qwen3-4B-Instruct-2507 model card reports NO MBPP/HumanEval (Qwen3 drops saturated benchmarks). Reported coding: MultiPL-E 76.8, LiveCodeBench v6 35.1, Aider-Polyglot 12.9.
- Our scale0-full (spec-in-prompt) 0.792 on easy MBPP subset ≈ native MultiPL-E 76.8 → runner is SANE, not broken; high floor is the model being genuinely good at easy MBPP (saturated).
- LiveCodeBench 35.1 (harder/contest) shows ~2x headroom → confirms harder-task direction. For item-4 multi-benchmark, LiveCodeBench/MultiPL-E/EvalPlus are the comparables (need harder task sets).
PLAN (owner-consolidated): (1) full HPO ✓ done; (2) curate hard slice [next]; (3) multi-turn experimentation on hard; (4) then RL / more training / multi-benchmark run.

### [2026-06-05 06:00 UTC] Hard-task memory NULL result + LiveCodeBench pipeline built
- HARD-TASK MEMORY (8 multistep, reference_a, minimal prompt): scale0 floor 0.25 (2/8) vs c3 @0.627 ALSO 0.25 (2/8). On genuinely hard tasks the adapter did NOT lift pass@1 (vs +0.25 on easy pool). ⇒ hypernet carries SIMPLE specs but not complex algorithmic specs. (n=8, caveat.) c3 passed decode_string+merge_intervals.
- LIVECODEBENCH v6 (owner: bench on LCB, use existing packages, minimal impl):
  - benchmarking_tools.md confirms LCB for contamination-resistant evals (medium-high glue).
  - Approach: rune GENERATES, official LCB harness GRADES (zero grading code from us). Isolated /tmp/lcbenv (numpy+tqdm; datasets NOT needed — build eval sample manually).
  - GRADER VALIDATED: codegen_metrics on hand-written sols → stdin correct 1.0/wrong 0.0; functional correct 1.0/wrong 0.0. ✓ (test properly)
  - test6.jsonl = v6 delta (175 problems); Feb-Apr 2025 window = 131 (Qwen 35.1 slice); 63/175 functional, rest stdin.
  - tools/_lcb_run.py (rune env: LCB→BenchTask→engine→generations json) + tools/_lcb_grade.py (lcbenv: official codegen_metrics). PROBING generation on 4 problems now before any full run (owner: probe first).

### [2026-06-05 06:40 UTC] LCB pipeline VALIDATED + generation failure on hard problems
- LCB grader (official codegen_metrics, isolated lcbenv) end-to-end: probe 4 problems → pass@1 0.25 (1/4). Grader trustworthy. ✓ pipeline READY.
- BUT rune generates EMPTY code on 3/4 hard problems (abc391_d/e/f). abc391_a (easy) → valid stdin script, passes.
- rune-internal pass@1 (0.75) is BOGUS: stdin problems get empty test_code in my converter → engine sandbox runs empty code → exit 0 → false pass. Only LCB grade real.
- CAUSE (same class as codegen JSON bug): plan/decompose/diagnose still emit JSON structured output → on hard problems the long plan JSON TRUNCATES → PlanResult validation fails → "re-planning" loop → budget exhausted before code → EMPTY. The freeform fix was code-only; plan/decompose/diagnose remain fragile on long outputs.
- Ties to hard-task diagnosis: the agentic engine (decompose→plan→...) breaks on hard tasks (over-decompose, re-plan loops, oracle-stops-early, easy-tuned scaling over-perturbs). Hard tasks need: robust plan (json_repair/freeform/higher max_tokens), stronger in-loop signal (all public examples), scaling re-tune, decompose guard.

### [2026-06-05 07:00 UTC] SESSION SUMMARY — everything tried (consolidated, owner-requested)
**A. Runner correctness fixes (committed, validated):**
1. Freeform codegen — code was JSON-wrapped; model over-escaped \n→\\n → 1-line phantom SyntaxError poisoning repair. Fixed: emit freeform ```python, de-fence (markdown-it) + ast. Probe: freeform 3/3 compile vs JSON stochastic collapse.
2. Public-example oracle (rune/engine/oracle.py) — engine in-loop check was bare-def→exit0 (no logic signal). Now runs spec doctest in-loop. decode_string now repairs with real traceback→accurate diagnosis.
3. In-loop model-judge — FALSE-POSITIVES correct code (flagged correct int_to_roman; verdict-before-reasoning). Reorder fix (reason→verdict) committed but UNVALIDATED + slow. KEPT OFF.
4. Fixed integrate MLflow metric crash (oracle_fired/"" trailing slash).

**B. HPO (16 trials, corrected runner, judge OFF, easy ref_pool_24):** best reference_a @ scaling 0.627, tuning 0.588, val 0.571. Per-flavor: reference_a/b1/c tied 0.588, training_exact WORST 0.294.

**C. Memory results (the core thesis):**
- WRONG first: scale0-full 0.792 vs spec-in-adapter 0.583 — INVALID (full mode leaks full spec into prompt via project_label[:1200]).
- CORRECTED (both reference_a, minimal prompt): scale0 (adapter OFF, no spec) 0.333 vs c3 (spec in adapter) 0.583 = **adapter +0.25 (+6/24). Thesis works single-turn (easy).** Lossy channel (<0.79 prompt ceiling).
- c3-full-on-top-of-prompt: ~neutral (18 vs 19/24; n=7 looked +, reversed n=24).
- HARD (8 multistep, reference_a): scale0 0.25 == c3 0.25 (2/8). NO gain on hard.

**D. Benchmark calibration:** Qwen3-4B-Instruct-2507 card: NO MBPP/HumanEval; MultiPL-E 76.8, LiveCodeBench v6 35.1, Aider 12.9. Our scale0-full 0.79≈MultiPL-E → runner sane; LCB 35.1 = harder, 2x headroom.

**E. Hard-task failure diagnosis (docs/issue52-goal3-hardtask-failure-2026-06-05.md):**
- Adapter carries full spec (no truncation) — capacity NOT the issue.
- Oracle stops at SINGLE public example: c3 reaches public attempt-1 5/8 (vs scale0 1/8) but held-out only 2/8 — ships wrong code. Public≪held-out on hard.
- Scaling 0.627 (easy-tuned) OVER-PERTURBS: trades tasks (gains merge_intervals, BREAKS int_to_roman which scale0 solved 7/7).
- Plan step LEAKS full spec into prompt (project_label) in both arms — muddies floor.
- Over-decomposition: calculate→2 subtasks, never completes/defines entry_point.

**F. LiveCodeBench v6 (owner: bench it, minimal impl):**
- Pipeline READY: rune GENERATES → official LCB codegen_metrics GRADES (isolated /tmp/lcbenv, no rune changes). Grader VALIDATED (stdin+functional correct 1.0/wrong 0.0). tools/_lcb_run.py + _lcb_grade.py.
- Probe 4 problems: LCB pass@1 0.25 (1/4) BUT 3/4 EMPTY code. rune-internal 0.75 BOGUS (stdin empty test_code→false pass).
- Empty cause: plan/decompose/diagnose STILL JSON → long hard outputs TRUNCATE → PlanResult validation fail → re-plan loop → budget exhausted → empty. (Freeform fix was code-only.)

**G. NOW BRAINSTORMING (owner hypothesis):** focused per-step prompt + EPISODIC adapter carrying the RIGHT context per step (local sub-goal state for code/repair; ALL subtasks' results for integration). Overall goal in adapter, NOT full spec every prompt. Each subtask = full dev cycle around its sub-goal. (Adapter MUTATION vs hotswap = later.) Checking this hypothesis first.

### [2026-06-05 07:15 UTC] BRAINSTORM design decisions (episodic conditioning) + roadmap
Owner hypothesis converged via Q&A:
- EPISODIC ADAPTER carries the RIGHT context per step (not full spec every prompt):
  - code/repair: condensed overall goal + CURRENT sub-goal (desc + its check) + current code + error (local episode).
  - integration: overall goal + ALL subtasks' results/code + integration error.
- PROMPT: thin, focused on the current step's sub-goal ("implement <sub-goal>"/"repair"). No full spec leak.
- PER-SUBTASK SIGNAL = BOTH: (1) executable acceptance check emitted by decompose (example I/O/assert for the sub-goal) AND (3) fixed reason-first model-judge vs sub-goal desc. Either fails → repair (adapter carries failure episode). Each subtask = full dev cycle.
- DECOMPOSE = model-decides, BOUNDED+VERIFIED: cap ≤3; default 1 for single-function; each subtask maps to a named piece of the entry_point; verify integration DEFINES entry_point. Single-function → 1 subtask, check=public example.
- Adapter MUTATION vs hotswap: LATER (after this hypothesis).
ROADMAP (owner, after episodic): 1) robust plan/decompose/diagnose (json_repair/freeform/higher max_tokens) — fixes empty-code/re-plan truncation loop (COUPLED to episodic decompose which emits more structured output → more truncation risk). 2) stronger in-loop signal = feed ALL public examples (LCB has several). 3) re-tune adapter_scaling for hard (sweep <0.627). 4) decompose guard via AGENT INSTRUCTION not fragile regex.
METHOD (owner): smoke/probe test each change + report; record findings to scratchpad.

### [2026-06-05 07:25 UTC] CONSOLIDATED RESULTS TABLE (all pass@1, for the record)
All corrected runner (freeform codegen + public-example oracle), model_judge OFF, c3 ckpt, seed 0.

EASY POOL (goal3_ref_pool_24, 24 simple MBPP):
| config | spec location | adapter | pass@1 |
| scale0 full | prompt | off | 0.792 (19/24) |  <- spec-in-PROMPT ceiling (NOT a floor; full mode leaks spec)
| reference_a scale0 | nowhere (mission only) | off | 0.333 (8/24) |  <- TRUE no-context floor
| reference_a c3 @0.627 (HPO best) | adapter | on | 0.583 (14/24) |  <- ADAPTER MEMORY +0.25 over floor
| c3 full @1.0 | prompt+adapter | on | 0.750 (18/24) |  <- adapter on top of prompt: ~neutral (vs 19)

HELD-OUT 7 (val7 = mbpp/115,133,118,119,106,113,135):
| scale0 full | 0.714 (5/7) | ; reference_a c3 (HPO val) 0.571 (4/7) ; c3 full @1.0 0.857 (6/7, REVERSED on 24 — n=7 mirage)

HPO (16 trials): best reference_a@0.627 tuning 0.588/val 0.571. Per-flavor best tuning: reference_a/b1/c=0.588 (tied), reference_b=0.471, training_exact=0.294 (worst).

HARD (8 multistep, reference_a, minimal-ish prompt):
| scale0 0.25 (2/8: decode_string,int_to_roman) ; c3 @0.627 0.25 (2/8: decode_string,merge_intervals)
| c3 reaches PUBLIC example attempt-1 5/8 vs scale0 1/8 (adapter helps reach oracle) but held-out 2/8 both.
| c3 BROKE int_to_roman (scale0 7/7 held-out → c3 fails 1st). Over-perturbation @0.627.

LIVECODEBENCH v6 (test6.jsonl, Feb-Apr2025=131 problems, 63 functional/112 stdin):
| Grader (official codegen_metrics, isolated lcbenv) VALIDATED: stdin/functional correct 1.0, wrong 0.0.
| Probe 4 problems scale0 full: LCB pass@1 0.25 (1/4); 3/4 EMPTY code (plan JSON truncation re-plan loop). rune-internal 0.75 BOGUS (stdin empty test_code).

PUBLISHED BASELINE (Qwen3-4B-Instruct-2507 card): MultiPL-E 76.8, LiveCodeBench v6 35.1, Aider 12.9 (NO MBPP/HumanEval). Our scale0-full 0.79≈MultiPL-E → runner sane.

BRAINSTORM APPROVED: episodic conditioning + bounded-verified decompose + per-subtask (executable check AND fixed judge) + fold robustness #1 in. Validate: unit/smoke probes THEN tiny live slice. Next: write spec.

### [2026-06-05 07:35 UTC] All experiments recorded; STARTING IMPLEMENTATION (episodic design)
Spec: docs/superpowers/specs/2026-06-05-episodic-conditioning-design.md (committed e8b2bd1).
Confirmed all results recorded (session summary 07:00 + results table 07:25). Implementation phases (TDD, unit/smoke probes first per owner):
- Phase A: robust structured parsing (json_repair for decompose/plan/diagnose + graceful degrade to N=1) + decompose schema (overall_goal, acceptance_check, builds) + bound ≤3. [#1 folded in; fixes empty-code/re-plan loop]
- Phase B: render_episode_adapter (context-per-step: local for code/repair, all-subtasks for integrate) + thin focused prompts (no spec leak).
- Phase C: per-subtask signal — executable acceptance_check (oracle) AND fixed reason-first judge; integration runs ALL public examples.
- Phase D: wire graph/policy; AST-verify entry_point at integration (no regex).
- Phase E: unit/smoke probes report → tiny live slice (2 single-fn + 1 multi-component).

### [2026-06-05 08:10 UTC] Episodic design IMPLEMENTED (Phases A-D committed); E (live slice) running
- Phase A (95a9928): robust _loads_structured (json_repair) for decompose/plan/diagnose; schema overall_goal/acceptance_check/builds; cap <=3; degrade to N=1.
- Phase B (0e7e98d,4804f24): render_episode_adapter (context-per-step) + prompt_mode='episodic' + thin prompt_episodic_* templates.
- Phase C (7526a13): build_subtask_probe — per-subtask acceptance_check is the in-loop signal.
- Phase D (add334d): integration AST-verify entry_point defined; decompose prompt instructs new schema + <=3 bound (agent instruction).
- 360 unit tests + mypy + ruff clean throughout.
- Phase E: live smoke slice (2 single-fn + calculate, episodic c3 @0.627) running. Checking: non-empty code (no re-plan loop), N=1 for single-fn, calculate decomposes <=3 with acceptance_checks, focused conditioning.

### [2026-06-05 08:25 UTC] Episodic Phase E live-slice VALIDATION ✓ (mechanism works)
Smoke (3 tasks, episodic c3 @0.627, 8 min):
- mbpp/106 single-fn: N=1, 1 acceptance_check, repair cycle ran, code non-empty → FAIL (check-repair didn't fix).
- mbpp/108 single-fn: N=1, clean 3 steps → PASS.
- multistep/calculate: N=3 (capped), 3 acceptance_checks (one per subtask), per-subtask dev cycles (plan/code/diagnose/repair each), code non-empty → FAIL.
VALIDATED:
- ✓ NO empty code / re-plan loops (robust parsing Phase A works — the LCB 3/4-empty cause is fixed).
- ✓ Bounded decompose: single-fn→N=1, calculate→N=3 capped.
- ✓ Each subtask emits + uses its own acceptance_check (model-authored: e.g. calculate's evaluate subtask check = assert calculate("2+3*4")==14).
- ✓ THIN prompt ("Implement `evaluate_arithmetic_expression`... goal+check in your context") + FOCUSED episodic adapter (## Task = current sub-goal + Acceptance, NOT full spec). The conditioning inversion works.
- ✓ Per-subtask dev cycles with repair engaging on the per-subtask check.
OBSERVATIONS (tuning, not mechanism): calculate decomposed to 3 but FAILED — budget=8 likely too low for 3 per-subtask cycles + integrate (no integrate step reached). Multi-subtask needs more budget.
NEXT (roadmap): budget tuning for multi-subtask; then larger comparison episodic vs reference_a vs scale0 on hard slice to see if episodic IMPROVES held-out pass@1.
