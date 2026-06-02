# Issue #52 — PREDECLARED experiment spec (T0, E1, E2) — FROZEN 2026-06-02

> **FROZEN BEFORE any trained-checkpoint delta** (leakage rule). Scoring rules, masks,
> negatives, row-selection, truncation, and go/no-go thresholds below are fixed prior to
> running any GPU arm. No threshold may be revised after seeing a trained-checkpoint number.
> Produced by the CPU-only scout+synthesis workflow (run wf_b7030969-6d7); reviewed and
> frozen by the main loop. Companion: docs/issue52-deliverable4-handoff-2026-06-02.md.

> ⚠️ **ATTRIBUTION CORRECTION (load-bearing).** The deliverable-4 handoff's headline
> warm-start numbers — goal +2.30 / file +1.76 / diff +1.01 / tail/continuation +2.01 —
> are **GEMMA (gemma_demo)**, NOT qwen_4b_d2l. True qwen warm-start (/tmp/d2l_qwen_ep.log):
> goal **+2.235** / file **+1.596** / diff **+0.983**, overall m−mismatch +1.604; qwen
> code-recall +2.597 (/tmp/d2l_qwen_code.log). **There is NO qwen continuation/tail number
> on disk** — any body/tail ceiling claim must come from a fresh run and may never be sourced
> from the gemma +2.01. The calibration ladder below uses the qwen rungs.

---

PREDECLARED EXPERIMENT SPEC — Issue #52 (T0, E1, E2)

STATUS: FROZEN BEFORE any trained-checkpoint delta. Leakage rule: scoring rules, masks, negatives, row-selection, truncation, and go/no-go thresholds below are fixed prior to running any GPU arm. No threshold may be revised after seeing a trained-checkpoint number.

ATTRIBUTION CORRECTION (binds the whole spec): deliverable4's headline warm-start numbers goal +2.30 / file +1.76 / diff +1.01 and tail/continuation +2.01 are GEMMA (gemma_demo), NOT qwen_4b_d2l. They MUST NOT be used as qwen warm-start. The true qwen warm-start episode-recall numbers are goal +2.235 / file +1.596 / diff +0.983 (/tmp/d2l_qwen_ep.log). There is NO qwen continuation/tail number on disk; any body/tail ceiling claim must come from a fresh run and may never be sourced from the gemma +2.01.

---

## Existing-outputs triage (do FIRST)

On-disk continuation / ceiling / recall numbers (source path : value):

QWEN warm-start (qwen_4b_d2l/checkpoint-20000), trustworthy for this base:
- /tmp/d2l_qwen_ep.log — episode recall (matched − mismatch), n=12 each: goal +2.235 (m-zero +3.756, frac 1.00); file +1.596 (m-zero +3.949, frac 0.75); diff +0.983 (m-zero +2.681, frac 0.83); OVERALL m-mismatch +1.604.
- /tmp/d2l_qwen_code.log — Sakana free-form code recall, n=8: mean m-mismatch +2.597, m-zero +8.354, gen_accuracy 0.75, frac(>0) 0.88.
- /tmp/fbswap_baseline.log — feedback-swap, ckpt qwen_4b_d2l/checkpoint-20000: matched-SWAP +0.0185, frac 0.48, matched-zero +0.0870, n=60. (small matched-zero ⇒ likely the 2048-default path.)

Trained recipe-4b (checkpoints/issue52-recipe-4b/checkpoint.pt), uncontrolled provenance:
- /tmp/fbswap_smoke60b.log — matched-SWAP +0.0687, frac 0.65, matched-zero +0.5021, n=60.
- /tmp/fbswap_smoke60.log — SAME ckpt: matched-SWAP +0.0019, frac 0.38, matched-zero −8.8097 (degenerate/collapsed scaler_B run). The +0.0019↔+0.0687 swing on one checkpoint proves an uncontrolled knob (max_seq_length OR commit-1553026f scaler_B fix). Outer logs do not record argv ⇒ max_seq_length / scaler_B-fix state unrecoverable.

GEMMA (DO NOT attribute to qwen):
- /tmp/d2l_runeep.log — goal +2.302 / file +1.759 / diff +1.012 (the numbers deliverable4 mislabels).
- /tmp/d2l_cont.log — continuation, 2 synthetic CASES: mean m-mismatch +1.975, m-zero +2.010 (the only continuation/tail number on disk; GEMMA, synthetic, low-n).
- /tmp/d2l_coderecall.log — Sakana reference code recall, n=8: +7.118 (m-zero +9.567).

Ceiling-probe artifacts (NOT the episode ceiling):
- /tmp/avoid_ceiling.log — base-only in-context ceiling for the DIFFERENT 'avoid' task: gate_DiD −0.0229, frac 0.47, VERDICT FAIL/WEAK. Not reusable as goal/file/diff/body ceiling.
- /tmp/temp_sample_avoid.log — base-only ceiling on good MBPP pairs (hidden-failure critique), n=3: mean DiD +0.0847, frac 1.00, VERDICT PASS (well-posedness control only).

VERDICT: Existing dumps do NOT answer E1/E2's ceiling question. There is NO in-context CEILING arm (doc/code text in the prompt, no adapter, same scored span) for goal/file/diff/body in any harness (rune_episode_recall.py, rune_continuation.py, _specificity_probe.py). The only prefix-in-prompt apparatus (_avoid_ceiling_probe.py) is base-only, a different task, and FAILED — explicitly not reusable. Therefore:
- ANSWERED on disk (no re-run): matched / mismatch / zero arms for qwen episode recall, code recall, and feedback-swap warm-start (+0.0185).
- STILL NEEDS A FRESH GPU RUN: (1) the ceiling arm (~10-line code add: doc-prefix-in-prompt, no adapter, same span), (2) the T0 trained-vs-warm-start paired delta under one fixed regime, (3) the E1 oracle arm (no oracle code exists today), (4) the E2 counterfactual arm. None of the historical recipe-4b feedback-swap numbers may be used as the trained delta (uncontrolled knob).

---

## T0 — paired significance (cheap rigor closure, NOT the decision)

Harness: tools/_feedback_swap_eval.py. Success metric = mean(matched − swap) on the edit-local mask; matched−zero is discipline only.

Code changes (exact):

(b) Second-checkpoint arm (do FIRST; it dictates structure). Add `--ckpt2`. Load base+tok ONCE; load hyp1 and hyp2 via load_hypernetwork (non-mutating for scaler_B, hypernetwork.py:319). Build `recs` ONCE (lines 95–108) so both arms see byte-identical context/answer/pre_code/feedback and an identical `pool`. Refactor the per-episode body (lines 129–146) into `score_episode(hyp, eff, li, ans_ids, em, neg_fb) -> (lp_m, lp_n, lp_z)`, called per hyp. `eff` (effective_scaling = that ckpt's lora_alpha, NOT alpha/r; adapter_contract.py:54) and `li` (layer_indices) are recomputed per hyp (preserve line 89 logic) so each adapter applies at its own trained scale.

(a) Per-episode JSONL dump. Add `--out PATH`. One JSON line per episode:
`{"row_idx": i, "task_id": rec.get("task_id"), "n_ans_tok": len(ans_ids), "n_edit_tok": int(sum(em[1:])), "neg_idx": (i+1)%len(pool), "ctx_hash": <sha1 of (context|answer|pre_code) bytes>, "arm1": {lp_m,lp_n,lp_z,matched_swap,matched_zero}, "arm2": {...}, "eligible": bool, "skip_reason": "ans<2"|"no_edit"|"nan_arm1"|"nan_arm2"|null}`. `row_idx` = enumerate index over `recs` so it joins across runs.

(c) Truncation-alignment + byte-identical-row assertions (mechanisms):
1. ONE `--max-seq-length` value for BOTH arms, passed identically to `_prepare_ids` (the scored span) AND `_generate_lora_dict`→`extract_activations_with_model` (the adapter conditioning). max_seq_length governs BOTH; it is not a neutral knob. Use a single fixed value (recommend 768, the training regime; do not reuse the historical 2048 path). Assert a single scalar is threaded to both call sites in both arms.
2. Checkpoint-INDEPENDENT eligible set precomputed once from (tok, max_seq_length, pre_code): `len(ans_ids) >= 2 AND sum(em[1:]) > 0`. Assert identical regardless of checkpoint.
3. Per-row byte hash: compute `sha1((context|answer|pre_code).encode())` once from `recs`; write to dump; `assert` arm1 and arm2 read the same `recs` object (trivially true since built once — makes the contract explicit and auditable in the dump).
4. NaN pairing (checkpoint-DEPENDENT skip, line 143): intersect the non-NaN episode sets of the two arms; drop an episode from BOTH means if EITHER arm is NaN. `assert n_arm1 == n_arm2`; log the dropped row_idx list. Both reported means share one denominator.

EXPLICITLY FORBIDDEN: comparing historical +0.0185 (baseline.log) against +0.0687 (smoke60b.log). Different invocations; argv (max_seq_length) and scaler_B-fix state unrecoverable; conditioning regime differs, not just the scored span. Re-measure BOTH arms fresh, in ONE process, under ONE fixed max_seq_length, before any T0 claim.

Stats (heavy-tailed; not t-test alone):
- Paired bootstrap CI on per-episode `arm2.matched_swap − arm1.matched_swap` (10k resamples, 95% percentile CI), computed from the JSONL dump over the shared-denominator episode set.
- Sign test on the per-episode paired difference (binomial, H0 p=0.5).
- Row-level scatter: arm1 vs arm2 matched_swap per episode (identify whether a delta is broad or driven by a few heavy-tailed rows).

Go/no-go (in calibration-ladder units; see Cross-cutting; do NOT anchor to NIAH +7.7):
- WIN: paired bootstrap 95% CI on (trained − warm-start) excludes 0 AND sign test p < 0.05 AND the trained mean clears the next ladder rung above warm-start chance (≥ body +0.14; see ladder). A bare "+0.0687 > 0" is NOT a win.
- NULL/NO-GO: CI includes 0 OR sign test n.s. T0 closes the significance question; it is not the product decision — that is E1/E2.
Positive control for T0: code-recall qwen +2.597 (/tmp/d2l_qwen_code.log) proves the substrate CAN bind code facts, so a T0 null is weak-signal, not a broken harness.

---

## E1 — capacity vs representation (oracle vs hypernet @ matched rank; lead discriminator)

Harness base: tools/_specificity_probe.py (10 frozen MBPP tasks, reference solutions, matched / mismatch-derangement / zero, present vs hidden regimes).

FROZEN scoring rule — BODY mask only, never signature, never edit-local:
- Mask family = `span_bounds` (tools/_specificity_probe.py:133–150). Signature span = answer-token range [lo, hi) of the `def <entry_point>(` line; BODY = [hi, len). E1 scores the BODY span [hi, len) only. The discriminator is body +0.14 vs signature +3.84 — scoring the wrong span destroys the experiment.
- Scoring-validity assertion (FROZEN): the `(0,0)` missing-marker fallback (line 144) silently makes BODY = the FULL answer including the signature, collapsing the discriminator. Predeclare: HARDEN line 144 to RAISE if `j < 0` (marker not found). Every E1 episode must assert the marker is found, or the episode is excluded with an explicit reason — never scored under the (0,0) fallback.
- Metric: `scoring_core.mean_gold_logprob` (next-token, log_softmax over float32 logits, t-1 convention, float64 accumulation). Do not vectorize to gather().sum() — the identical-math guarantee is the control. MAX_ANS_TOK=96 cap retained; confirm body fits.

Oracle = UPPER BOUND on the SAME substrate (not proof the hypernet objective is wrong). Same hidden-code facts, same MBPP tasks, same BODY masks, same mismatch-derangement negatives, same prompts (hidden/absent-template regime: render via render_training_format_trajectory with current_code+feedback empty, _specificity_probe.py:232/281) for oracle and hypernet.

Oracle parity contract (FROZEN; mismatch makes the "upper bound" 8× off or over-capacity):
- `peft.get_peft_model(base, LoraConfig(r=8, target_modules=['down_proj'], lora_alpha=8*45.2548, lora_dropout=0.0))`. target_modules MUST be `['down_proj']` (PEFT defaults to attn q/v — silently over-capacity). lora_alpha MUST be `r*45.2548` because PEFT applies alpha/r while the hypernet/functional path applies lora_alpha UN-DIVIDED (adapter_contract.py:54, wrapper.py:21–44). down_proj × 36 layers = the hypernet's layer_indices (36 = Qwen3-4B layer count) ⇒ identical substrate.
- Train oracle CE on the episode answer span only, answer-preserving truncation via `_prepare_ids` at max_seq_length=768 (supervise only ans positions). gradient_checkpointing may be True for the oracle (PEFT-native forward), but MUST stay False for any hypernet/functional-LoRA arm.
- Score the oracle through the IDENTICAL path and IDENTICAL BODY mask as the hypernet. Hypernet matched-rank arm reuses _specificity_probe.py unchanged (it already loads any --ckpt and uses effective_scaling+assemble_adapter+_student_logits).

Cross-over control: a tiny hypernet fine-tune on the EXACT facts the oracle succeeds on. REUSE (no new training code): `tools/run_guarded.sh <log> tools/_distill_entry.py --config <tiny.yaml> --max-steps <few>`. tiny.yaml = clone configs/issue52_recipe_mvc_4b.yaml; corpus_path → a JSONL of just those facts; `grad_accum_steps <= n_rows` (default 8 ⇒ with <8 rows the optimizer NEVER steps); keep contrastive on; early_stop_warmup=100. Verify diff_token_frac > 0 (skip_zero_diff=True drops rows where teacher==base top-1). DO NOT reinit scaler_B (warm-start preservation is automatic; an unconditional reinit_scaler_b_nonzero inflates B ~17× and destroys the adapter — see the −8.8 collapse run). Then re-run _specificity_probe.py --ckpt <new>.pt on the BODY span.

Decision table:
- oracle good @ r8 + hypernet bad @ r8 → REPRESENTATION wall. Next lever: cross-over fine-tune / hypernet objective change. If cross-over moves the hypernet to oracle territory, it is a trainability gap, not a capacity wall.
- oracle bad @ r8, good @ higher rank → CAPACITY. Next lever: raise r / add chunks.
- both bad @ high rank → DATA / ARCHITECTURE (facts not learnable on down_proj substrate, or supervision wrong).

Minimal GPU plan: serialized single-GPU. (1) train oracle r8 down_proj (1 smoke-unit). (2) score oracle + hypernet on BODY span, one process (0.5 unit; hypernet matched/mismatch/zero on disk for episode recall but NOT for the BODY span, so this run is required). (3) optional oracle r16/r32 if r8 oracle is bad (1 unit each). (4) cross-over tiny fine-tune + re-score (0.5 unit) only if branch 1 fires. Implementation path: mirror tools/diag_pre_corpus_gate.py:149–162 PEFT setup; 4-bit base via the hypernet_distill BitsAndBytesConfig block (lines 137–152).

---

## E2 — directionality (minimal-edit counterfactuals; scored on action-consequences)

Harness base: third_party/doc-to-lora/rune_episode_recall.py (+ add ceiling arm). The mismatch arm currently = the NEXT episode's REAL doc-adapter, ctxs[(i+1)%len] (line 94) — a structured negative, not random. State this in frozen rules.

Counterfactual construction (FROZEN):
- Minimally edited: change ONLY the causal arrow / next-action implication. Preserve tokens, local code, file/goal scaffold. The matched doc and the counterfactual doc must be near-identical bag-of-events, differing only in directionality of the consequence.
- Same-bag-of-events control: a sibling doc with the same events but no directional flip (re-ordered/neutral) — to absorb lexical-overlap effects.
- FORBIDDEN: bare time-reversal or were↔heading text swaps (lexical artifacts the model can exploit without modeling direction).
- E2 mismatch arm = the constructed counterfactual doc (replace ctxs[(i+1)%len] at line 94 with the counterfactual ctx); keep matched = own doc; keep zero = base.

Scoring (FROZEN): score the NEXT-STEP ACTION/code tokens (the consequence the direction determines), NOT "what happened first?" recall. Same mean_gold_logprob math, same MAX_ANS_TOK=48 cap (note: several diff cells hit len=48 and are truncation-weakened — flag truncated episodes in the dump and report them separately). Compare matched vs counterfactual AND matched vs the same-bag-of-events control.

CEILING arm (~10-line add, FRESH GPU required — no qwen episode/body/tail ceiling exists on disk; never source it from gemma +2.01): add `logits_ceiling(model, full_with_doc_prefix)` — build the prompt with the doc text prepended, model.base_model, NO adapter, score the SAME answer span; add a `per_target['*_ceil']` column. Mirror in rune_continuation.py if the tail facet is needed. The action-binding score is reported as a fraction of (ceiling − zero), so a low absolute logprob is read against what the in-prompt ceiling itself achieves.

Adapter contract: native lora_alpha=45.2548, r=8, down_proj only — no scaling knob; counterfactual and matched docs both applied at native scale (model.patch_lora_forward).

Positive control (NOT on disk — construction requirement): curate ONE episode where flipping the causal direction provably changes the correct next action (e.g., "test added then code written" vs "code written then test added" ⇒ different correct next edit). The ceiling arm on this episode must show matched/ceiling ≫ counterfactual; if it does not, the harness, not the adapter, is broken.

---

## Cross-cutting predeclared gates (write before any run)

Calibration ladder (anchored to on-disk QWEN rungs; NIAH +7.7 is a trivially-passable ceiling and is NOT used):
- Rung 0 (feedback chance / near-null): ≈ +0.018 (warm-start feedback-swap).
- Rung 1 (body recall, current floor): +0.14.
- Rung 2 (diff recall): +0.98.
- Rung 3 (file recall): +1.60.
- Rung 4 (goal recall): +2.24.
- Rung 5 (signature recall, strong binding): +3.84.
Thresholds: A "WIN" must clear at least one rung above the relevant baseline AND pass its significance test (T0) / decision branch (E1) / ceiling-fraction (E2). Movement that stays within Rung 0–1 noise is NULL regardless of sign.

Retention gate (run on every trained/fine-tuned checkpoint; must be preserved vs warm-start, no regression beyond CI):
- Episode recall (goal/file/diff) matched−mismatch must not drop below warm-start CI (qwen +2.235 / +1.596 / +0.983).
- Code recall (qwen +2.597) must not regress.
- matched−zero discipline must not collapse (the −8.8 signature is a known scaler_B-collapse failure mode — gate on matched−zero > 0).

Generation-stability gate: xgrammar-constrained pass@1 (rune bench) must not degrade vs the warm-start baseline. A representation/capacity gain that breaks structured generation is not a win.

Positive control per experiment (so a null distinguishes weak-signal from broken-harness):
- T0 / E1: code-recall qwen +2.597 — substrate can bind code facts.
- Ceiling well-posedness: temp_sample_avoid PASS (+0.0847, frac 1.00).
- E2: the curated direction-flips-action episode (construction requirement above).

---

## Implementation order + GPU budget

1. [CPU] Edit tools/_feedback_swap_eval.py: add --ckpt2, --out, refactor to score_episode, single max_seq_length threaded to both call sites in both arms, eligible-set precompute, ctx_hash, NaN pairing + n_arm1==n_arm2 assert. (pure code)
2. [CPU] Edit tools/_specificity_probe.py: harden span_bounds line 144 to RAISE on missing marker; confirm BODY span [hi,len) is the scored span. (pure code) — covers E1 mask freeze.
3. [CPU] Write the predeclared spec durably (this document) and freeze. (the orchestrator/main loop writes it, not the subagent)
4. [CPU] Add ceiling arm (~10 lines) to rune_episode_recall.py (and rune_continuation.py if tail needed). (pure code) — covers E2/E1 ceiling.
5. [CPU] Author E1 oracle trainer/scorer under tools/ (PEFT r8 down_proj, lora_alpha=8*45.2548), mirroring diag_pre_corpus_gate.py. (pure code)
6. [CPU] Construct E2 counterfactual docs + same-bag control + curated positive-control episode. (pure code/data)
7. [GPU, serialized] T0: one process, both arms (warm-start + recipe-4b), fixed max_seq_length=768, --out dump. ~1 smoke-unit. [Not covered by disk — historical numbers uncomparable.]
8. [GPU, serialized] E1: train oracle r8 (~1 unit) → score oracle + hypernet on BODY span (~0.5 unit). [Body-span hypernet arm NOT on disk.] Conditional: oracle r16/r32 (~1 unit each); cross-over fine-tune + re-score (~0.5 unit).
9. [GPU, serialized] E2: episode-recall with ceiling + counterfactual + control + positive-control episode (~0.5–1 unit). [Ceiling/counterfactual NOT on disk.]

Total GPU: ~3 smoke-units baseline, +1–3 conditional on E1 branches. Everything in steps 1–6 is pure CPU code work. Reusable from disk WITHOUT a run: qwen episode-recall matched/mismatch/zero, qwen code-recall, warm-start feedback-swap +0.0185 (as the warm-start arm reference only — the trained delta must still be re-measured fresh in step 7). NOT covered by any disk artifact: every ceiling arm, the controlled T0 trained delta, the oracle arm, the E2 counterfactual.

---

## E1 CROSS-OVER CONTROL — PREDECLARED (frozen 2026-06-02, before the run)

After E1 r8 verdict (representation wall: oracle holds body @ r8, hypernet binds signature +4.09 but
body only +0.14 episode-specifically). The cross-over asks: is that body-representation gap REACHABLE
by gradient (trainability/objective) or not (architecture/conditioning attenuation)?

DESIGN (advisor-hardened):
- Fine-tune the HYPERNET (not a per-episode LoRA) for a few steps on the EXACT 10 frozen MBPP absent-
  regime episodes, so its GENERATED adapter better binds the BODY.
- LOSS = CONTRASTIVE on the BODY span: hinge pushing matched-body-logprob ABOVE the derangement-
  partner-conditioned body logprob. NOT plain CE (CE raises lp_m and lp_x together = generic boost =
  the confound; would show fake m-zero movement with flat m-mismatch). The derangement negative is IN
  the training loss, not just eval.
- Reuse hypernet_distill's generation->apply->backprop core + scaler_B preservation (do NOT reinit
  scaler_B). Swap 3 knobs: span edit-local->BODY; negative feedback-swap->derangement partner; corpus
  codereview->10 MBPP absent episodes.
- Re-score with _specificity_probe.py (4-bit, absent, body span) on the fine-tuned ckpt.

FRAMING: this measures TRAINABILITY (can gradient move the representation at all), trained-on-test by
design (mirrors the oracle). A gain is NOT product-generalization evidence.

THRESHOLD (bar = the hypernet's OWN signature binding +4.09, NOT oracle +21.7 which is unreachable for
an amortized generator):
- body m-mismatch +0.14 -> >= +1.0  => MOVED. Objective/trainability gap -> FINE-TUNE is the lever.
- body m-mismatch stays ~+0.14 (< +0.5) => DID NOT MOVE. Architecture/conditioning attenuation back on
  the table (rank/chunks won't help either; the adapter-generation path can't route body detail).
- retention check: signature m-mismatch must not collapse (don't trade sig binding for body).
