# Issue #49 D2L Retrain — Working Scratchpad

Running log of progress, results, fixes, and decisions. Newest entries at the bottom of each section.

- **Branch:** `feat/issue49-d2l-retrain`
- **Spec:** `docs/superpowers/specs/2026-05-30-issue49-d2l-self-distillation-retrain-design.md`
- **Plan:** `docs/superpowers/plans/2026-05-30-issue49-d2l-self-distillation-retrain.md`

## Core decisions (locked)
- **Mechanism:** modified Sakana **D2L privileged-context self-distillation**. Teacher = frozen base + trajectory in-context (`disable_adapter()`); student = base + generated adapter, trajectory removed. Top-K KL over the answer span, **masked to diff tokens** (base≠teacher). **No per-bin oracle stage** (leaner; teacher is the frozen base).
- **Three anti-degeneracy fixes:** (1) reinit `scaler_B` non-zero (escape the zero collapse basin), (2) disable the L1 sink (`gen_lora_l1_reg_coef=0`), (3) diff-token masking + track `diff_agreement` (not `top1_agreement`).
- **Root cause confirmed at source** (`ctx_to_lora/modeling/hypernet.py`): `scaler_B` zero-init, `scaler_A` ones-init, `B = B_raw·scaler_B` → gradient trap.
- **Corpus (already mined):** `s3://elixirtrials-949678234935-eu-west-2-artifacts/training-data/github-pairs/` — `all_unrolled.jsonl` (8,517 rows, 91% diffs) and `external_codereview.unrolled.jsonl` (7,670 rows, 100% diffs). Schema: `activation_text`/`teacher_text`/`pre_code`/`post_code`/`quality_score`/`metadata`. No re-mining.
- **DoD:** gate-validated checkpoint + 2–3 task tiny bench. No full MBPP/HumanEval campaign, no HPO, no engine-side StepRecord re-mine.

## Progress log

### 2026-05-30 — planning
- Brainstormed → spec → plan committed (`ca3ba7f8`, `a36eb650`).

### 2026-05-30 — CPU implementation (workflow + recovery)
- First workflow (`wvwg1zl9x`) ran Tasks 1–10 to green commits, then **failed on a StructuredOutput technicality** mid-Task-11 (work was durable in git regardless). Instance also shut off mid-run.
- **Committed (Tasks 1–10):**
  - `ff6abc03` Task 1 — RAM watchdog `tools/run_guarded.sh`
  - `87eb17d3` Task 2 — `collapse_metrics.py` (diff_agreement, optimizer coverage)
  - `ca19af27` Task 3 — strict-load key audit (#49 §D)
  - `3479d330` Task 4 — `reinit_scaler_b_nonzero` (#49 §A)
  - `ff9dc952` Task 5 — top-K KL + diff-mask loss primitives
  - `74b38eab` Task 6 — diff-masked distill step loss + gradient unit test
  - `3e385243` Task 7 — extract activations under `disable_adapter()` (#49 §D)
  - `f8a2addc` Task 8 — combine_lora + head-bias rank contract (#49 §D)
  - `932a7873` Task 9 — content-based retrieval/contrast gate (#49 §B)
  - `c39e26bc` Task 10 — removed dead oracle + plain-SFT paths; Stage-2 → hypernet_distill
- **Task 11 recovery:** loop was complete in working tree but uncommitted with 5 ruff + 1 mypy errors (interrupted before cleanup). Fixed: `subprocess.run(check=False)`, line lengths, loop-var shadow (`line`→`stripped`), transformers `.to()` stub overload (split chained call on `Any` var). **263 unit tests pass, ruff clean, mypy clean.** Committed `3c83a476` (Task 11).
- **Task 13** (workflow `w53wfcmgt`) — `render_training_format_trajectory` in `graph.py` (wired into both `generate_adapter` call sites; human-facing prompt templates unchanged) + `--json-out` on the 3 probes. Committed `b185b667`. Note: probe `code_continue` trajectory carried no unique signal beyond `accumulated_code` (now → `## Current Code`); the "continue" instruction lives in the prompt side, not the trajectory → nothing dropped.
- **CPU side COMPLETE & GREEN:** Tasks 1–11, 13 all committed. **264 unit tests pass, ruff clean, mypy clean (34 files).** Working tree clean except unrelated PR48 doc.

## Open / next
- [ ] **Task 12 — Stage 0 synthetic NIAH discriminator** (GPU, decisive). Need to write `tools/diag_synthetic_overfit.py` first. Run under `tools/run_guarded.sh`; gate = `real_hit > zero_hit` AND `real_hit > contradictory_hit`. Branch: collapses → mechanical bug; recalls → loop sound.
- [ ] **Task 14 — real-corpus train + gate run + tiny bench** (GPU).

### 2026-05-30 — CRITICAL BUG found in agent-written D2L loop (pre-GPU review)
Reviewing `hypernet_distill.py` before the Stage-0 run surfaced a blocking bug in `_student_logits`:
- It does `base_model.load_state_dict(weights, strict=False)` where `base_model` is a **plain** `AutoModelForCausalLM` (not PEFT-wrapped) and `weights` are PEFT-keyed lora tensors from `_to_peft_state_dict`.
- (1) PEFT keys match **nothing** on a non-PEFT model → adapter never applied → student ≡ base.
- (2) `load_state_dict` copies under `no_grad` → **gradient never flows to the hypernet**.
- Net effect: every record would hit `loss.requires_grad == False` → `skipped += 1` → 0 training. Stage-0 would falsely report "collapse" for a harness reason, not the science.
**Correct mechanism (confirmed in `ctx_to_lora`):** functional patching. `lora_layer.lora_forward` patches each target `Linear.forward` to add `B@A@x·scaling` using the grad-carrying generated `lora_dict` straight from `hypernet.generate_weights(...)` (NOT PEFT-converted); `apply_lora_to_layers(model, layer_indices, lora_dict, n_qs=tensor([1]))` injects per-context A/B; restore `forward_orig` after. V1 used the same via `apply_functional_lora`. The package's modulated model does `patch_lora_forward()` → `model(**inputs, return_generated_lora=True)` → differentiable student forward.
**Fix plan:** rewrite the student forward in our loop to: extract activations (under `disable_adapter`) → `hypernet.generate_weights` (grad) → functional-patch base Linears → forward answer-only → restore. Keep teacher-in-context + diff-mask. Then run Stage-0.

### 2026-05-30 — Stage-0 harness + two more bugs fixed (GPU run in progress)
- Advisor review of the fix added: (a) a cheaper **adapter-application contract** (1 fwd+bwd: student≠base AND `scaler_B.grad`≠0) as the go/no-go *before* the 20-step run; (b) use **module-level** `lora_layer` fns (bare perceiver lacks the wrapper's patch methods); (c) feed **raw grad-carrying `lora_dict`** from `generate_weights`; (d) **positional** layer indexing (package's `apply_lora_to_layers` uses absolute → wrong for non-contiguous layers); (e) explicit `train_scaling` (don't inherit lora_alpha — repo history of 8× over-scaling).
- **Fix 1 committed `fed4566b`:** rewrote `_student_logits` to apply the adapter functionally (`_functional_lora` ctx + `lora_forward`) on the grad-carrying lora_dict; added `_generate_lora_dict` (no no_grad). Was: `load_state_dict` of PEFT keys into a plain base → no-op + broke autograd → would have falsely shown collapse.
- **`tools/diag_synthetic_overfit.py`** written + committed `446068a7`: Phase0 contract → Phase1 overfit (needles `73921/frobnicate/48207/qux`) → Phase2 real/zero/contra recall. Gate = real>zero AND real>contra.
- **Fix 2 committed `174ca793`:** first GPU run crashed `AttributeError: 'LoraConfig' has no attribute 'layers'`. Correct source is `hypernet.config.layer_indices` (per `wrapper.py:40`). Same bug was latent in `hypernet_distill.py:109`. Fixed both.
- GPU env OK: 22.5GB GPU free, flash-attn 2.8.3, checkpoint cached locally (`~/.cache/rune/checkpoints/8e815654733a4579.pt`, 2.5GB — no S3 pull). 264 unit tests still green.
- **Stage-0 run relaunched** under `tools/run_guarded.sh` → `/tmp/synth.log`, json `/tmp/rune-issue49-synth.json`. *Status: running (base model loads in ~1-2 min, then contract → train → recall).*
- **Monitor `boad9rshr`** armed on `instructions/reflections.md` (persistent) per user request — re-read + weigh against approach on each change.

### 2026-05-31 — Reviewer reflections incorporated (instructions/reflections.md)
Four points; all refine (don't overturn) the approach. Actions:
- **(1) Preservation region:** diff-mask narrows the objective; a broad-perturbation adapter could raise `diff_agreement` while wrecking non-diff tokens. → log answer-span KL/NLL on **masked-out (agreement) tokens** + add no-change/preservation examples to the gate; consider a small preservation term in the loss. **Status: TODO before "loop sound".**
- **(2) Stronger Stage-0 read:** string-hit recall is confounded (decoding variance, prefix leakage, format, disruptive contra). → add **logprob/forced-choice needle probe**, **shuffled/irrelevant control**, multiple seeds; downgrade interpretation to "gradient path CAN memorize a toy record," not loop validation. **Status: TODO; re-run harness.**
- **(3) Causal collapse evidence:** non-zero reinit fixing training ≠ proof zero-init was the dominant cause. → **init ablation** (scaler_B=0 vs 1, same batch, first-step grad norms for head/scaler_B). **Status: TODO.**
- **(4) Teacher-quality ceiling (most strategic):** D2L caps student at frozen base+context competence; weak teacher on real edits → adapter compresses weak behavior. → **teacher-quality audit** on real rows (base+context vs `post_code`/edit-local) BEFORE blaming the hypernet for Stage-1 failure. Confirms oracle is a real pass@1 lever, deferred not dismissed. **Status: TODO in Stage 1.**
- **(5) Train/inference parity (CRITICAL):** the functional-patch *training* path and the PEFT-export + hot-swap *inference* path must apply the SAME tensors to the SAME layers at the SAME effective scaling. Positional-vs-absolute layer indexing divergence could let Stage-0 pass while inference is wrong → invalidates the tiny bench, reproduces "trains fine / inert at inference." → add an **equivalence check**: one generated `lora_dict` → functional-path logits vs PEFT-export+hot-swap logits on the same input at matched scaling; assert close BEFORE trusting Stage-0 for the full loop. Also pins the §D scaling. **Status: TODO (required gate).**
Decision: let the in-flight Stage-0 run finish for the **contract** (decisive plumbing check) + a *preliminary* recall signal read narrowly; then build **harness v2** with (1) preservation KL/NLL on agreement tokens, (2) logprob/forced-choice needle + shuffled control + seeds, (3) init ablation, (5) train/inference parity; (4) teacher-quality audit lands in Stage 1. Re-run before any "loop sound" claim. Fold (1)-(5) into spec/plan gate definitions.

### 2026-05-31 — Stage-0 run #1 result: CONTRACT PASS, RECALL FAIL (informative)
JSON `/tmp/rune-issue49-synth.json`; log `/tmp/synth.log`.
- **Contract PASS:** `adapter_applies=true` (Δlogit absmax **22.1**), `loss_requires_grad=true`, `scaler_b_grad_abs_sum=66`, `grad_flows=true`. → plumbing fixes (functional apply + scaler_B reinit + L1 off) VALIDATED; NOT the mechanical-collapse branch.
- **Training learns:** loss 10.4→2.4, diff_agreement 0→~0.2, `head/grad_l2≈0.78` (head learning), `scaler_B` held at 1.0 (its grad tiny ~0.01 — fine, B=B_raw and head carries the signal).
- **Recall GATE FAIL:** real=zero=contra=0.0. **real and contra outputs byte-identical** (`99999999`,`uxuxux`,`77702720`) and degenerate-repetitive; differ from zero only as garbage. = issue #49 "active but content-independent" signature, now non-inert.
- **Read (narrow, per reflection 2):** moved inert→active, not yet active→retrieving. Two confounds: (a) **over-scaling** — Δabsmax 22 at scaling=2.0 → runaway/degenerate decoding masks content; (b) free-form greedy recall confounded.
- **Open question:** weak content signal suppressed by scaling/decoding, OR hypernet ignores context (adapters identical across trajectories)? → harness v2 must measure: **recall scaling sweep** (0.25/0.5/1.0/2.0), **logprob needle probe**, **trajectory-sensitivity cosine** (real vs contra generated-adapter weight cosine — if ~1.0, context encoder isn't conditioning = deeper than scaler_B), plus more steps (200+). Then (1) preservation, (3) init ablation, (5) parity.

### 2026-05-31 — Measurement-validity cautions (reflections.md) → consolidated harness v2
Reviewer hardened the measurement design. Current `diag_traj_sensitivity.py` is a COARSE first look (read caveated). Build ONE rigorous measurement harness incorporating all accumulated cautions, run once:
- **Forced-choice matched-foil margin (decisive content test):** hold all text fixed except the needle value; under context-A(73921) check P(73921) > P(11111), under context-B(11111) check P(11111) > P(73921). Margins vs MULTIPLE distractors; foils matched for token length/frequency. (Beats real-vs-contra logprob and cosine.)
- **Cosine only paired** with per-layer cosine + **generated-weight L2 norms** (guard tiny-denominator noise) + functional logit deltas.
- **Pareto per scale** (no "best scale"): preservation KL, repetition/entropy, correct-vs-foil margin, real-vs-contra separation.
- **Preservation tail metrics:** worst-token KL, frac positions > KL threshold, repetition — not just mean.
- **≥1 generative/edit-level check** before declaring retrieval useful.
- **Stronger contract (harness v2):** not `student≠base`+grad≠0 (Δ=22 shows it's too weak); require a BOUNDED, DIRECTIONAL move toward teacher-preferred tokens at diff positions.
Plan: stop chasing each reflection with a new GPU run; consolidate into harness v2, run once, then decide context-encoder-conditions? vs scale/training issue.

### 2026-05-31 — Sensitivity probe (coarse, UNTRAINED) result
`/tmp/rune-issue49-sensitivity.json`. reinit scaler_B=1, no training.
- **weight_cosine ≈ 0.99999** for ALL context pairs (A_73921|B_11111 rel_L2 0.0039; A|C_zorblax 0.0054; B|C 0.0060). `CONTEXT_SENSITIVE: False`.
- needle real−contra logprob = noise across scaling (0.005,-0.003,+0.20,-0.11). degeneration rises 0.21→0.92 around scaling 0.5–1.0, drops at 2.0.
- **Caveats (reviewer):** untrained; A vs B differ by ONE token (cosine≈1 partly expected); BUT C_zorblax is a wholly different context and still 0.99999 — the concerning datapoint; global cosine can hide high-leverage late-layer diffs.
- **Fork:** H1 = hypernet can't condition on context here (aggregator/wiring washes out context — deeper than scaler_B) vs H2 = it can but needs real training (60 steps far too few) to learn content-carrying adapters for our domain. → harness v2 measures forced-choice matched-foil margin + per-layer cosine + weight norms, BEFORE and AFTER a longer (≥300-step) train, to disambiguate. Consulting advisor on the branch.

### 2026-05-31 — Decisive H1/H2 experiment launched (`tools/diag_forced_choice.py`, ad3468e5)
Advisor: stop perfecting the instrument; take the ONE reading that branches the project. Forced-choice matched-foil margin is the only metric immune to the generic-perturbation confound. Also: starting from the COLLAPSED checkpoint (conditioning pathway never got gradient) → cosine≈1 is EXPECTED, doesn't prove H1; only the POST-SATURATION margin decides.
Experiment: train synthetic A(73921)+B(11111) to saturation (400 steps), scaler_b_init=0.1 (1.0 too hot — degeneration 0.58 @ scaling 0.25), then:
- **margin_A = lp(73921|ctxA) − lp(11111|ctxA)**, **margin_B = lp(11111|ctxB) − lp(73921|ctxB)**, across scaling sweep; H2 if both >0 where degeneration<0.5.
- centered-delta vs neutral context (reviewer: raw cosine measures the shared prior; the context delta W(ctx)−W(neutral) is what matters) — rel_delta + centered cosine.
- feature-cosine A vs C localizer (only matters if H1: features differ but weights cosine~1 → aggregator washes out context, suspect eager-attn perceiver patch; features ~identical → activation-extraction bug).
**Branch:** both margins>0 → H2 (conditioning works, 60 steps was too few) → proceed to real corpus. both≈0 after saturation → H1 (pathway broken) → debug, don't train more.
Refinements (reflections.md, forced-choice): (1) "immune to perturbation" too strong → read positive train-pair margins as "pathway CAN FIT the 2-fact task," not readiness; (2) A/B-only conflates conditioning with memorizing 2 associations → **added held-out unseen value (55555) test** for a reusable binding mechanism; (3) don't fully defer PEFT parity → minimal parity smoke before long real runs. Folded (1)+(2) into the script (commit a438d86d): now reports `train_pair_fits` AND `heldout_generalizes`. Superseded/killed the first run; relaunched complete version.
*Status: running (~8-12 min). Watch bqc3vad6t. JSON /tmp/rune-issue49-forced-choice.json.*
Verdict map: heldout_generalizes → strong H2 (reusable binding) → proceed to corpus (after minimal PEFT parity smoke). train_pair_fits only → capacity but possible memorization → more controls before corpus. neither → H1, debug pathway via feature_cosine + centered-delta localizers.
Deferred to real-corpus stage (per advisor scope discipline): Pareto fronts, tail-KL, multi-distractor batteries, generative/edit checks, minimal-pair context battery, train/inference PEFT-parity (#5), teacher-quality audit (#4).

### 2026-05-31 — ★ DECISIVE RESULT: H2 CONFIRMED (conditioning pathway works) ★
`/tmp/rune-issue49-forced-choice.json` (commit a438d86d). Trained synthetic A(73921)/B(11111) 400 steps, scaler_b_init=0.1, train_scaling=1.0.
- **VERDICT: train_pair_fits=True, heldout_generalizes=True.**
- train loss 4.28→0.01 by step 50 (saturated).
- **forced-choice sweep:** scaling **0.5** → margin_A=+0.36, margin_B=+0.67, **heldout(55555)=+0.14**, degen 0.42 (clean). scaling 1.0 → margins +0.57/+0.92, heldout≈0. scaling 2.0 → margins large but heldout=-0.61 + degen 0.71 (overfit/over-scale).
- **centered_delta_cosine A vs B = -0.39** while raw_cosine=0.91 → reviewer point #1 vindicated: context deltas (vs neutral) point OPPOSITE directions; raw cosine hid strong conditioning. rel_delta_A=0.22, rel_delta_B=0.30.
- **feature_cosine A vs C = 0.53** → activation extraction produces distinct features (not the bug).
**Conclusions:**
1. The hypernet CONDITIONS on trajectory and learns a REUSABLE content-binding mechanism (held-out unseen value generalizes). The D2L approach is VIABLE.
2. Run #1's "cosine~1 / real==contra" = collapsed starting checkpoint + 60-step undertraining + over-scaling (2.0), NOT a broken pathway.
3. **Scaling sweet spot ≈ 0.5** (NOT old 7.84 — issue warning confirmed). Over-scaling overfits train pair while killing generalization + degenerating.
4. Interpretation downgraded per reviewer: this proves capacity+generalization on a synthetic content-binding task — strong go signal, not real-corpus readiness proof.
**Next:** (a) minimal **PEFT train/inference parity smoke** (reviewer #3) before corpus; (b) then Stage-1 real-corpus D2L train at low scaling (~0.5) + teacher-quality audit (#4); update spec/plan scaling + interpretation. CHECKPOINT WITH USER before the larger corpus-training GPU investment.

### 2026-05-31 — H2-result cautions (reflections.md) → hedged claim + stronger pre-corpus gate
- (1) heldout margin +0.14 is SMALL, ONE value, ONE format → "go/no-go signal," NOT proven robust generalization. Strengthen: several held-out values (matched 5-digit tokenization), seed variance, neutral-distractor survival, BEFORE headlining "reusable binding."
- (2) centered-delta now the trusted diagnostic; raw cosine = non-decisive. Pre-register centered-delta + failure modes for Stage 1; don't revert to raw cosine when noisy.
- (3) scaling 0.5 = local operating point, not a real-corpus law → starting prior + small bounded sweep at Stage 1.
**Claim downgraded to:** conditioning pathway is NOT broken + clean go-signal. Pre-corpus gate (consolidated, one GPU run): multi-value held-out + neutral distractor + (≥2 seeds) + **PEFT train/inference parity smoke**. Then Stage-1 corpus at low scaling. CHECKPOINTING WITH USER on the corpus investment.

### 2026-05-31 — Pre-corpus robustness gate launched (user option 1; `tools/diag_pre_corpus_gate.py`, 223b1e87)
One bounded run (~15-20 min) settling the reviewer's robustness cautions before the corpus investment:
- **multi-value held-out** forced-choice margins (4 unseen 5-digit values per config) vs **multiple distractors** (worst-case margin).
- **variance:** 2 configs with DIFFERENT training pairs + seeds (73921/11111 seed0; 42042/86753 seed1).
- **PEFT train/inference parity smoke:** functional-path logits vs PEFT export+hotswap on the SAME lora_dict at matched scaling (alpha=r, scale_lora_b×S) → max abs logit diff + cosine; pass if diff<0.5. (Layers are contiguous 0–31 so positional≡absolute here; parity validates transpose/scaling.)
- Gate pass = (every config ≥75% held-out values positive at some non-degenerate scaling) AND parity pass.
- Monitor re-armed (b0uqygvt2) after idle shutoff.
*Status: running. Watch + JSON /tmp/rune-issue49-pre-corpus.json.*

### 2026-05-31 — Reflections on gate launch (2 points)
- (1) **PEFT parity smoke only exercised contiguous 0-31** → doesn't test positional-vs-absolute. RESOLVED durably via CPU unit test (577ec832): both `_to_peft_state_dict` and `_functional_lora` use enumerate(layer_indices) (positional slot → absolute layer), verified on non-contiguous [0,5,10]. So parity holds for non-contiguous by construction; no need to lock to contiguous.
- (2) **Gate is synthetic numeric-token binding** → passing justifies real-corpus INVESTMENT, not that the adapter binds edit-relevant semantic facts. → Stage-1 real-corpus audit must check the signal on CODE EDITS (does a code-review-trajectory adapter make the model prefer the corrected token / produce post_code), not numeric needle recall. Reinforces teacher-quality audit (#4). **Stage-1 success criterion = edit-relevant, not needle recall.** Logged for Stage 1.

### 2026-05-31 — Real-corpus path: verified structure + fixed blocking bugs (reviewer §data-path)
Peeked S3 `external_codereview.unrolled.jsonl` (3 rows): keys task_id/activation_text/teacher_text/pre_code/post_code/quality_score/metadata. **teacher_text STARTSWITH activation_text** → context=activation_text, answer=teacher_text[len(activation_text):] (the `## Revision …` block). quality_score present (0.4, 0.28). activation_text ~1-2k chars.
- **#1 mapper (BLOCKING) FIXED (837779e7):** loop read `record["context"/"answer"]` but corpus has activation_text/teacher_text → KeyError. Added `_map_record` (+ synthetic passthrough + prefix-strip + fallback) and `_corpus_stats` (logs raw/mapped/skipped/empty). Unit-tested.
- **#3 truncation (BLOCKING) FIXED (837779e7):** `_teacher_base_logits` did ctx+ans then `[:max_length]` → long context pushed supervised span onto CONTEXT tokens. Added `_prepare_ids` (keep FULL answer, front-truncate context keep-end; answer-head if answer alone > max). teacher/student now share `ans_ids`. Unit-tested (incl answer>max). Updated 3 tool call sites.
- **#6 spec overclaim FIXED (2d2298dc):** loop trains REAL context only; "worse under contradiction" is EMERGENT/eval (confirmed by forced-choice positive-only training), NOT trained. Negative-context training = optional enhancement, not precondition.
- 271 unit tests pass; ruff+mypy clean.
**Stage-1 prerequisites still open (reviewer):**
- #2 diff-token fraction: real D2L gradient stat = per-row teacher≠base top-1 fraction after tokenization/truncation + count of zero-disagreement rows (NOT "100% diff coverage"). Build into a corpus-stats/teacher-audit GPU pass.
- #4 quality/metadata: loop ignores quality_score/metadata/pass_at_1 — inspect distribution from a sample; decide filtering.
- #5 **teacher-quality audit BEFORE training** (stratified sample: base+context vs post_code / edit-local). Weak teacher → faithfully-learned weak behavior → fails pass@1.
- Reviewer #2(gate-launch): Stage-1 success criterion must be EDIT-RELEVANT (prefer corrected/post_code tokens), not numeric needle recall.
These (#2/#4/#5 + edit-relevant audit) → one GPU "corpus-readiness + teacher-quality" script before the real D2L train.

### 2026-05-31 — Reflections on real-corpus fixes (3 more)
- (prefix brittleness) startswith verified on 3-row peek only; brittle to whitespace/template drift/re-rendered context. **FIXED (d8d8031c):** `_corpus_stats` now reports exact_prefix vs fallback rate, answer char-len distribution (min/median/p90/max), and sampled fallback task_ids. Run it corpus-wide before trusting training; high fallback rate ⇒ revisit mapper.
- (quality_score not a nicety) scores ~0.28-0.4 may encode confidence/usefulness; unweighted training may emphasize noisy edits. → corpus-readiness pass: inspect quality distribution + correlation with teacher/base disagreement, answer length, edit size before deciding to drop/weight. **[Stage-1 prereq]**
- (teacher audit stratification) include LARGE/multi-location post_code changes; token-level preference for small local edits can look good while missing global review intent. → stratify teacher lift by edit size/source, report separately. **[Stage-1 prereq]**

### 2026-05-31 — ⚠ Pre-corpus gate result: PARITY PASS, HELD-OUT NOT ROBUST (gate_pass=False)
`/tmp/rune-issue49-pre-corpus.json`.
- **PEFT parity PASS:** max_abs_logit_diff=0.14, cosine=0.99995. Train functional path ≡ PEFT export+hotswap. Train→inference equivalence holds. (Real positive.)
- **Held-out NOT robust:** 
  - cfg0 (73921/11111): train-pair margins +0.36/+0.67 (0.5), +0.57/+0.92 (1.0). Held-out 4 values frac_positive 0.25 (0.5) / 0.5 (1.0), mostly NEGATIVE.
  - cfg1 (42042/86753): train-pair MIXED — +0.91 / **−0.24** (one trained value fails!) at 0.5; held-out frac_positive 0.25.
- **Revised read:** earlier forced-choice held-out +0.14 (single value 55555) was NOT representative — across 4 values it's noise ~0. Pathway CONDITIONS + FITS trained pairs + parity holds, but does NOT generalize to unseen values from a **2-record** overfit → closer to MEMORIZING trained associations than reusable content-binding. Reviewer caution #2 vindicated.
- **Two readings:** (a) approach won't generalize (don't invest in corpus) vs (b) 2 records CAN'T teach a general value-binding rule (pure memorization expected when overfitting 2 examples); generalization needs corpus DIVERSITY → the real 7670-row corpus is exactly what would induce it. cfg1's trained-value fail (−0.24) is a yellow flag (instability / value-tokenization sensitivity).
- **Proposed disambiguator (cheap, synthetic):** train on MANY value pairs (e.g. 20-50 distinct MAGIC_OFFSET values), THEN test held-out. Robust held-out positivity with diversity ⇒ generalization emerges with diversity ⇒ strong go for corpus. Still fails ⇒ memorization-only, deeper concern. Better than jumping to corpus. → consulting advisor.

### 2026-05-31 — Advisor redirect: teacher-quality audit is THE next step (not diversity)
- Held-out generalization is UNMEASURABLE from 2-record overfit (memorizing 2 mappings = global optimum). gate_pass=False is NOT a verdict on the approach. Real takeaways: **parity PASS** + fits trained pairs. cfg1 −0.24 = tokenization noise, ignore.
- Synthetic numeric track = proxy ceiling. Real-corpus training IS the diversity experiment on the right distribution. Don't run 20-50 value synthetic diversity.
- **Discriminating insight:** synthetic task was RIGGED teacher≫base (unguessable needle → guaranteed diff-token gradient). Real corpus has NO guarantee; original collapse = teacher≈base (~84%) → no gradient. **Dominant risk + cheap KILL CRITERION:** on real code-review rows, does base+activation_text beat base-alone on the answer span?
- **NEXT = teacher-quality audit (checkpoint-AGNOSTIC: teacher=base+context, no hypernet):** sample real rows stratified by edit size; teacher-forced NLL/top-1 on ## Revision answer span (whole-span first; edit-local via pre/post diff later); compare base+context vs base-alone. **Decisive = diff-token fraction** (frac answer tokens where base+context right & base-alone wrong). ~0 → STOP/rethink (oracle, reframe). Healthy → green light for corpus (held-out tested on real rows there).
- **Advisor Q (before corpus, not before audit):** is `hypernet_hpo/checkpoint.pt` aggregator Sakana-pretrained or overwritten by Rune's collapsed training? If overwritten/damaged, weak binding on corpus ≠ method failure. Investigate before concluding from corpus run; consider fresh/Sakana warm-start for corpus.

### 2026-05-31 — Teacher-quality audit launched (`tools/diag_teacher_quality.py`, 25423eb1)
Advisor+reviewer converged: cheapest highest-value uncertainty = does real data contain a learnable signal. Audit (checkpoint-agnostic, teacher=base+context, no hypernet):
- diff-token fraction WHOLE-SPAN + EDIT-LOCAL (difflib pre_code vs answer tokens — reviewer #3 boilerplate concern), NLL lift, stratified by edit size, quality_score correlation.
- **Leakage check (reviewer):** count rows where context contains "## Revision" (must be ~0; else base+context lift is leakage artifact). By construction context=activation_text excludes the revision.
- Corpus downloaded local: /tmp/rune-corpus/external_codereview.unrolled.jsonl (7670 rows).
- Verdict green if edit-local diff-token frac ≥0.10 AND nll_improvement>0.
*Status: running. Watch bkm5k3zgh. JSON /tmp/rune-issue49-teacher-quality.json.*
**Reflections handled:** leakage (added check), whole-vs-edit-local (both reported), checkpoint provenance (logged as separate factor — investigate before CORPUS run, not before audit; audit is checkpoint-agnostic).
**Still pending before corpus run:** checkpoint provenance (Sakana-pretrained aggregator vs Rune-collapsed-trained — CPU inspect after audit); quality_score filtering decision.

### 2026-05-31 — ★ TEACHER-QUALITY AUDIT: GREEN (kill-criterion passed) ★
`/tmp/rune-issue49-teacher-quality.json`, n=120.
- **leakage_rows=0** (context never contains ## Revision) → lift is real, not artifact.
- teacher_acc 0.88 vs base_acc 0.73; teacher_nll 0.65 vs base_nll 1.28 (**nll_improvement +0.63**, halves loss).
- **diff_token_frac edit-local 0.143** (whole 0.19) — distillable signal on edit-bearing tokens. small=0.17, large=0.12 (both strata real; large edits addressed).
- quality_vs_diffedit_corr=-0.13 (weak NEG) → quality_score not a good proxy for distillable signal; don't aggressively high-quality-filter.
- **VERDICT GREEN** (edit-local ≥0.10 AND nll_improvement>0). Real corpus does NOT reproduce teacher≈base weak-gradient collapse condition → there IS gradient to distill. This validates the TEACHER has signal (not yet that hypernet compresses it — that's the corpus run).
- Aside: forced-choice already showed this checkpoint's aggregator CONDITIONS (centered-delta -0.39, fits pairs) after scaler_B reinit → aggregator functional, mitigates "damaged init" worry. Still doing a quick weight-health inspection before corpus.
**→ Green light for corpus training.** Last pre-corpus checks: checkpoint health/provenance, then launch Stage-1 D2L at low scaling (~0.5), edit-relevant eval. CHECKPOINT WITH USER on the big run.

### 2026-05-31 — Checkpoint health + corpus-run pre-launch requirements
- **Checkpoint health (CPU inspect):** aggregator sum_l2=820 absmax=3.64, head 696, NO nan, NO all-zero — HEALTHY (not damaged). scaler_B collapsed (norm 0.096, reinit fixes), bias_B=0, bias_A norm 1.13 (matches issue). 428M params. + forced-choice showed it conditions ⇒ "damaged init" confound ALLAYED. OK to use this checkpoint + scaler_B reinit for corpus.
- **Reviewer pre-launch requirements (corpus run):**
  1. Keep interpretation split: audit=data/teacher signal; training=compression. Don't back-infer audit was wrong if training fails.
  2. 0.143 clears 0.10 but small margin / n=120 → **log teacher/base diff-token fraction ONLINE over actual training rows** (is the sample representative?).
  3. **Define EARLY-STOP criteria before launch:** nonzero skipped-safe training, bounded degeneration, student→teacher on diff tokens, **no collapse of agreement-token preservation** (reviewer #1). Else loss falls for wrong reason.
- **→ Pre-launch TODO (code, before the big run):** add to `hypernet_distill` training loop: online diff_token_fraction, preservation metric (agreement-token NLL/KL: student vs teacher on NON-diff tokens), early-stop guard. Then launch Stage-1 corpus at low scaling (~0.5), edit-relevant eval. **CHECKPOINTING WITH USER on the big run.**

### 2026-05-31 — Guardrails + MLflow wired; corpus SMOKE launched (user: "add guardrails then launch")
- **Pre-launch guardrails committed (ab95f2ef):** online `diff_token_frac` (vs audit's 0.14), `preservation_agreement` (student↔teacher on NON-diff/agreement region — catches broad-perturbation damage), `should_early_stop` (abort after warmup if skip_frac>0.5 OR diff_agreement<0.02 OR preservation<0.5). DistillConfig fields: early_stop_warmup=150, min_diff_agreement=0.02, min_preservation=0.5, max_skip_frac=0.5. 279 unit tests pass.
- **MLflow wired (user: check in during training):** `_try_mlflow` → localhost:5000 (HTTP 200, mlflow 3.12), logs params + per-step metrics + early_stop tag; JSONL fallback. Never fatal.
- **Launcher `tools/run_corpus_distill.py` (4c99b02e).** Corpus smoke launched: 200 steps, scaler_b_init=0.1, train_scaling=0.5, lr=2e-4, max_seq_len=1024, exp=issue49-d2l-smoke, out=/tmp/rune-ck-smoke. Watch bz8tyzmot. **Smoke health checklist:** corpus_stats sane (low fallback), diff_token_frac ≈0.14-0.19, preservation high (≥0.8), no early-stop, MLflow logging, loss falling.

### 2026-05-31 — HPO proposal (user: consider HPO; v1 optuna + researched params/ranges)
v1 `bench/hpo.py` tuned ENGINE/inference params (adapter_scaling, temperature, cont_multiplier, presence_penalty, max_phase_iterations). For TRAINING the hypernet, different knobs:
| param | range | rationale |
|---|---|---|
| learning_rate | 5e-5 – 5e-4 (log) | distillation typical 1e-4–2e-4; current 2e-4 |
| scaler_b_init | 0.05 – 0.3 | synthetic: 0.1 good, 1.0 too hot (degeneration) |
| train_scaling | 0.25 – 1.5 | synthetic sweet spot ~0.5; >2 overfits+degenerates |
| topk | 25 – 100 | D2L top-K KL support |
| l1_reg_coef | 0 – 1e-5 | the collapse sink; keep ~0, small upper just in case |
| grad_clip | 0.5 – 2.0 | stability |
| num_epochs | 1 – 3 | corpus size 7670 |
| warmup_ratio | 0.0 – 0.1 | gentle start |
**Objective (NOT loss — falls for wrong reasons):** maximize held-out **edit-local diff_agreement** on a held-out split of REAL rows, with a hard penalty if preservation<0.5 or degeneration high (Pareto/constrained). Reuse Optuna + MLflow (optuna-integration[mlflow] is a dep).
**Gating:** HPO is BLOCKED until a baseline corpus run trains end-to-end without early-stop AND the eval metric (edit-relevant held-out) is wired — else we'd optimize a constant/broken objective (the v1 HPO mistake the issue flagged). So: baseline run first → wire eval → THEN HPO (~20-30 trials, short max_steps per trial). Recorded; not run yet.

### 2026-05-31 — Reflections on smoke/HPO plan (3)
- (1) **preservation ≥0.5 too loose** — broad perturbation could pass 0.5 while wrecking easy syntax tokens. → watch raw preservation DISTRIBUTION on smoke; RAISE min_preservation (→~0.8) once scale understood. Treat 0.5 as a hard floor, expect ~0.85+ when healthy.
- (2) **max_seq_len consistency RESOLVED:** teacher-quality audit ran at max_length=1024 (its default, not overridden) and smoke uses 1024 → diff_token_frac IS comparable to audit 0.14-0.19. For FULL run consider 2048 (preserves more context ⇒ ≥ signal; audit already passed at 1024). Interpret online diff_token_frac at the run's truncation length.
- (3) **HPO held-out split:** freeze train / val (HPO objective) / UNTOUCHED final-test split before tuning; 7670 rows + many knobs → short trials can overfit val. Added to HPO plan.

### 2026-05-31 — Length generalization (user Q): chunking + combine_lora (built into ctx_to_lora)
- Perceiver aggregator is length-AGNOSTIC by design: fixed learned latents cross-attend variable-length context (`aggregator.py::forward(ctx_features[bs,seq_len,dim], attn_mask, position_ids)`); adapter size independent of L. Our TRUNCATION caps length, not the architecture.
- D2L long-context (NIAH 5× base ctx window) = **chunking + combine_lora**, both shipped:
  - `data/processing.py::split_too_long_ctx` (max_ctx_chunk_len); **n_chunks SAMPLED during training** ("sampled (train)/derived (eval)") → trains across a length distribution.
  - `lora_merger.py::combine_lora(loras, n_chunks)` concatenates per-chunk rank-r deltas along rank axis → merged adapter rank = K·r (rank_per_group = n_chunks*base_rank).
  - Length scales by ADDING CHUNKS not bigger forwards: peak mem = 1 chunk; total = K× per-chunk (chunks sequential via `lora_forward_packed`/use_sequence_packing).
- **To generalize our loop:** chunk context instead of truncate; sample n_chunks/example (D2L recipe); per-chunk position_ids; grad-checkpoint + flash-attn to push per-chunk size. **PEFT export/hotswap must allocate variable rank K·r** — extends the rank contract (`merge_head_bias_rank`/parity test) already built.
- **Caveats:** rank K·r grows with length → larger adapter, inference cost, practical ceiling. Training-recipe + export change, NOT a flag. **NOT blocking:** code-review rows ~700-1300 tok fit one 1-2k window. → schedule as POST-BASELINE enhancement (chunked/n_chunks-sampled training); update spec/plan with a "length generalization" section then.
- **Reviewer caveats for the chunked regime (when built):** (a) chunking changes rank/merge/inference-cost AND the generated-weight distribution → it needs ITS OWN parity + preservation tests; don't assume one-window success transfers. (b) **Confound:** sampling n_chunks expands rank, so gains may come from RANK EXPANSION not better memory use → compare fixed-total-rank vs chunk-merged-rank (or report cost/quality trade-off explicitly). (c) baseline must prove adapter-as-memory on one-window real edits BEFORE adding this capacity-expanding mechanism (attribution).

### 2026-05-31 — ⚠ Corpus smoke OOM'd (training-memory wall) → memory fixes
- MLflow run created OK (exp 44) — monitoring works. OOM on FIRST student forward (`_student_logits`, decoder MLP), 22GB full.
- **Cause:** frozen 9B bf16 base ~18GB + trainable hypernet (~0.9GB) + AdamW states (fp32 2×428M ≈ 3.4GB) + grads (~0.9GB) + autograd activations over ~600-token real answers. Synthetic fit only because answers were ~10 tokens. CLAUDE.md "base+hypernet fit 23GB" was INFERENCE (no Adam/grad/activations).
- **Fixes (standard, low-risk, keep bf16 base so functional patch unchanged):**
  1. **gradient checkpointing** on base (recompute activations in backward). NOTE: HF checkpoints only when model.training=True → set base_model.train() but keep params requires_grad=False (Qwen has no dropout). use_reentrant=False, use_cache=False.
  2. **8-bit Adam** (bitsandbytes Adam8bit) for hypernet → Adam states 3.4→~0.85GB.
  3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (anti-fragmentation; error suggested it).
  4. reduce max_seq_length to ~768 for smoke.
  - Escalation if still OOM: 4-bit NF4 base (QLoRA) — but functional lora_forward calls torch.nn.Linear.forward(self,x) which breaks on bnb Linear4bit → would need custom forward using forward_orig. Try non-quantized path first.
- Implement as DistillConfig flags (gradient_checkpointing=True, use_8bit_optim=True), re-run smoke.

### 2026-05-31 — Memory fixes applied + smoke #2 (reflections on OOM fix folded in)
Committed (c0cb4886, 12a4d133): gradient_checkpointing (base train()+frozen, use_reentrant=False) + 8-bit Adam + expandable_segments. Reflections folded:
- (1) train()-mode dropout risk → **`_base_is_deterministic` self-check** (two forwards must match); auto-disables checkpointing→eval() if non-deterministic.
- (2) seq-len signal: keeping **1024** (matches audit) now that fixes free memory; only drop to 768 if still OOM (would then log truncation rate separately).
- (3) 8-bit Adam fragility → **watch scaler_A/scaler_B/head VALUE trajectories** (not just grads) to confirm quantized optimizer updates collapse-critical params.
- (4) recorded: train-time memory budget ≠ inference-fit. (Consider updating PRODUCT/CLAUDE.md later.)
- Smoke #2: 150 steps, seq 1024, exp issue49-d2l-smoke2, watch bcp1xs07u. Escalation if still OOM: 4-bit NF4 base (needs custom functional forward, since lora_forward calls torch.nn.Linear.forward — incompatible with bnb Linear4bit).

### 2026-05-31 — Smoke #2: no OOM but grad-checkpointing INCOMPATIBLE with functional patch → pivot to 4-bit base
- Memory fixes worked (reached backward, no OOM). But **CheckpointError: "different number of tensors saved forward vs recomputation (49 vs 43)"** — gradient checkpointing recomputes the layer forward, and the monkeypatched lora_forward closing over external grad-requiring A/B breaks checkpoint's tensor accounting. Known incompatibility.
- **Pivot (robust, standard QLoRA):** 4-bit NF4 base → frees ~13GB → NO checkpointing needed. Requires rewriting `_functional_lora` to compute base_out via the layer's **forward_orig** (not torch.nn.Linear.forward, which breaks on bnb Linear4bit) + manual einsum delta. Custom forward works for any base layer dtype.
- DistillConfig: add load_in_4bit=True; gradient_checkpointing default→False (incompatible). base stays eval().
- **Note: train/inference base-precision mismatch** — train teacher=4-bit, engine inference base=bf16. LoRA deltas usually transfer, but flag: evaluate gates on the SAME precision used to train, decide inference precision later. Teacher audit was bf16; 4-bit teacher slightly degraded but teacher-vs-base COMPARISON stays consistent (both 4-bit).
- Memory budget 4-bit: base ~5.5GB + hypernet 0.9 + 8bit-adam 0.85 + grads 0.9 + activations(seq1024,no-ckpt) ~5-8GB ≈ 13-16GB ⟶ fits 22GB.

### 2026-05-31 — 4-bit base implemented; smoke #3 (reflections folded)
Committed (5565a6de, cac27aa5): 4-bit NF4 base (QLoRA, load_in_4bit=True, double-quant, bf16 compute); functional LoRA rewritten to add delta on layer's **forward_orig** (bnb Linear4bit compatible); `_lora_delta` extracted + **equivalence-contract unit test** (reviewer #3). gradient_checkpointing default off (incompatible). 282+ tests pass.
Reflections folded:
- #1 (re-audit teacher in 4-bit): smoke #3's **online diff_token_frac in 4-bit IS the check** — compare to bf16 audit 0.14. If ≈0.14 → signal survives quantization; if ≪ → 4-bit degrades teacher → reconsider. (Avoids a separate audit run.)
- #2 (dual-precision eval): Stage 3/4 gates must eval BOTH 4-bit (train-matched) and bf16 (engine-target), labeled. **[pending Stage 3/4]**
- #3 (custom forward equivalence): `_lora_delta` unit-tested (math); 4-bit integration exercised by smoke.
- Smoke #3: 4-bit, seq 1024, 150 steps, exp issue49-d2l-smoke3, watch bch0prg9t.

### 2026-05-31 — ★ Smoke #3 HEALTHY (4-bit, real corpus) → FULL BASELINE LAUNCHED ★
**MLflow exp `issue49-d2l-smoke3` (id 46), run d3e02eeb6c48, FINISHED.** End-to-end on real corpus, 4-bit, no OOM/error/early-stop. Trajectory (step→): loss 4.30→~2-3.5 (noisy, batch=1), **diff_agreement 0.056→0.16→0.485→0.234** (RISING — adapter learns to match teacher on diff tokens), **preservation 0.997→~0.90** (agreement region intact), **diff_token_frac ~0.19-0.31 in 4-bit ≈ bf16 audit 0.14-0.19** (signal survives quantization — reviewer #1 ✓), scaler_B grad 0.35 (moving, non-collapse), head grad 0.87 (learning), skipped=0 (mapper healthy on real rows).
- Raised min_preservation 0.5→0.7 (718f2be7; observed ~0.9 steady-state).
- **FULL BASELINE LAUNCHED:** MLflow **exp `issue49-d2l-baseline` (id 47), run cba3c6a38e4b4116a1b2c76bddab86e2, RUNNING.** 4-bit, seq 1024, 1 epoch (~7670 steps, ~2-4h), out=/tmp/rune-ck-baseline. URL http://localhost:5000/#/experiments/47/runs/cba3c6a38e4b4116a1b2c76bddab86e2. Watch b2hmxssy0 (fires on complete/early-stop/OOM).
- **Convention (user):** always report MLflow run+experiment IDs.
- **After baseline:** Stage 3 §D inference correctness + scaling re-measure (dual-precision 4-bit/bf16 eval — reviewer #2), Stage 4 retrieval/contrast gates + tiny bench (edit-relevant). Then HPO (gated on baseline + eval wired).

### 2026-05-31 — Baseline interpretation guardrails (reflections)
- (1) batch-1 noisy: report ROLLING MEDIAN/windowed avg of diff_agreement, not single points. 0.485→0.234 = "signal exists," not stable level.
- (2) preservation ~0.9 can DRIFT DOWN as adapter fits diff tokens → watch the trend over the run, not just the 0.7 early-stop floor.
- (3) **training completion ≠ success.** Must run post-training **dual-precision (4-bit train-matched + bf16 engine-target) edit-relevant held-out gates** before declaring success. Baseline is a TRAINING run only.
- Action after baseline: BUILD edit-relevant held-out eval (frozen split) + dual-precision gate; only then judge.

### 2026-05-31 — Baseline mid-run snapshot (step 610/7670, RUNNING) — GOOD DIRECTION
exp issue49-d2l-baseline (id47) run cba3c6a38e4b4116a1b2c76bddab86e2. Windowed (first30→last30 median): diff_agreement 0.167→0.203 (rising, max 0.66), preservation 0.883→0.892 (stable, min 0.732 >floor, no drift), loss 3.19→2.74 (falling), diff_token_frac ~0.24-0.27 (stable data prop), scaler_B/grad 0.30→0.38 (active), head/grad ~0.83, skipped=0. Healthy: learning + preservation intact + no collapse/degeneration/early-stop. diff_agreement rise modest so far (8% in) — watch for plateau. NOT a verdict — needs post-train dual-precision edit-relevant gates.
**TODO while baseline runs:** build edit-relevant held-out eval (frozen split, 4-bit + bf16) for Stage-4 gate.

### 2026-05-31 — Held-out leakage fix: family split + RELAUNCH (reviewer)
- Whole-corpus baseline (exp 47) was UNJUDGEABLE (no held-out → post-hoc eval contaminated). STOPPED it.
- `tools/split_corpus.py` (37cb379e): deterministic hash of **metadata.source_task_id** (PR family) → train/val/test. **train 6930 (2420 fam) / val 370 (149) / test 370 (124), NO family leak.** Files: /tmp/rune-corpus/external_codereview.{train,val,test}.jsonl.
- **RELAUNCHED baseline on TRAIN split:** exp **issue49-d2l-baseline-split (id 48), run b09bbc30e15e4e18b581ee6975b6229d, RUNNING.** url http://localhost:5000/#/experiments/48/runs/b09bbc30e15e4e18b581ee6975b6229d. out=/tmp/rune-ck-baseline. Watch bts98fe4l. → checkpoint now JUDGEABLE on held-out val/test families.
- **Next while training:** build edit-relevant held-out gate that evaluates the trained checkpoint on val/test (families never trained) in BOTH 4-bit + bf16. THIS is the success verdict (training completion is not).

### 2026-05-31 — INVESTIGATION: loss non-monotonicity / drops (user-flagged)
Pulled full history exp issue49-d2l-baseline-split (id48) run b09bbc30e15e (56 pts, step 550).
**Data:**
- loss windowed medians (8): [3.69, 3.36, 3.21, **0.0**, 1.98, 2.2, 3.1, 3.67] — non-monotonic, a 0.0 window, rising tail.
- loss min=0.0 max=5.03 mean=2.61 stdev=1.24. **zero-loss points: 6/56.** negative: 0.
- **corr(loss, diff_token_frac) = 0.877** ← dominant finding.
- diff_agreement windowed: [0.084,0.171,0.196,0.0,0.203,0.232,0.148,0.136] (rose 0.08→0.23 then dipped in hard tail).
- preservation windowed: [0.938,...,0.867] — gentle DOWNWARD drift ~0.07 (still >0.7 floor).

**Diagnosis (NOT training divergence):**
1. **Batch-1 row-difficulty variance is the dominant cause** — loss corr 0.88 with per-row diff_token_frac. Loss ≈ "how many diff tokens this single row has," not learning trajectory. Spikes = hard rows; dips = easy rows.
2. **Zero-diff rows → loss exactly 0** (6/56): rows where teacher==base on the whole answer span hit `distill_step_loss` mask.sum()==0 → returns 0. Creates the 0.0 window / sharp drops. Artifact of the masked objective, not failure.
3. **No shuffling + corpus grouped by repo/PR** → consecutive steps are correlated difficulty BLOCKS, so windowed loss/diff_agreement track WHICH rows are in the window, not time. The rising loss tail + diff_agreement dip = later windows are harder rows, not forgetting.
4. batch=1, no grad-accum → high-variance single-example updates.

**Conclusion:** train-loss is a POOR progress monitor here (confounded by row difficulty + ordering). Real learning signal (diff_agreement) did rise 0.08→0.23 early. Preservation drift (0.94→0.87) is mild, partly hard-tail rows; watch it.

**Fixes (recommended, to make a meaningful run):**
- **Shuffle corpus** (deterministic) → decorrelate consecutive steps; windowed metrics reflect time not repo-blocks. (Easy, high value.)
- **Gradient accumulation** (eff. batch 8–16) → average over difficulty → smoother loss + better gradient. (High value.)
- **Periodic VAL eval** on the held-out val split (fixed yardstick, difficulty/ordering-independent) → the honest progress curve, not train-loss. Doubles as the gate metric.
- (minor) skip zero-diff rows (no signal) → removes 0-loss artifact + wasted steps.
→ Restart baseline with shuffle + grad-accum + periodic val eval; train-loss alone is not interpretable as progress.

### 2026-05-31 — DeepChecks consideration (user Q) — DECISION: data/split QC only
Verified current DeepChecks NLP (property-drift, embeddings-drift domain classifier, data-integrity + train-test-validation suites; maintained).
- **USE for:** one-time corpus + SPLIT validation — embeddings/property drift train-vs-val/test (independent check that held-out is representative + catches NEAR-DUPLICATE leakage a key-based family split can't) + data-integrity (dupes/empty/outliers). Complements `_corpus_stats` + family split + teacher audit.
- **DON'T use for:** training dynamics (MLflow's job; DeepChecks is point-in-time, wouldn't have helped today's loss non-monotonicity); model-quality gates (assumes classification/regression — no primitives for adapter-collapse/retrieval/forced-choice/edit-local/preservation; our bespoke gates are the right tool).
- **Plan:** optional `tools/deepchecks_corpus.py` run ad-hoc on train/val/test (CPU); NOT in the training loop; extra dev-dep + embeddings. Complementary, not a replacement. Awaiting user go before adding the dep.

### PENDING TRAINING ACTION (from loss investigation)
Restart baseline with: shuffle (deterministic) + gradient accumulation (eff batch 8-16) + periodic VAL eval (held-out, the honest progress curve) [+ optional skip zero-diff rows]. Current run's train-loss is confounded (corr 0.88 with row difficulty + unshuffled repo blocks) → not interpretable as progress.

### 2026-05-31 — Both done: training fixes + corpus/split QC; baseline v2 running
**Training fixes (1f4d063d):** shuffle + grad-accum(8) + skip-zero-diff + periodic held-out VAL eval (val_diff_agreement/val_preservation) — the ordering-independent progress curve. **Baseline v2 RUNNING:** exp **issue49-d2l-baseline-v2 (id49), run fff67d9060e24e62aa907cd62a7a63dd**. url .../experiments/49/runs/fff67d9060e24e62aa907cd62a7a63dd. train split + val split, grad-accum8, val_steps100. Watch b7xw3j8o7.
**DeepChecks (user asked):** HARD-INCOMPATIBLE with our sklearn 1.8 (deepchecks calls removed `max_error` scorer → import fails). Downgrading sklearn risks the training env. Decision: deliver equivalent DATA-QC via lightweight `tools/corpus_split_qc.py` (sklearn TF-IDF; deepchecks_corpus.py kept as ref). **TODO: `uv remove deepchecks` AFTER v2 training** (avoid venv churn mid-run; it survived the add). committed 2a0900a2.
**QC FINDING (real):** family split alone leaves **near-dup leakage** — val **12.7%** (47/370), test **7.3%** (27/370) held-out contexts cosine≥0.9 to train (same-repo boilerplate). Integrity clean (0 dup/empty). Properties comparable across splits (ctx_len median ~2180, answer ~1861; long tail to 120k chars truncated by answer-preserving). → wrote **clean held-out: val.clean 323, test.clean 343** (near-dups removed). **Stage-4 gate MUST eval on the .clean splits** for honest generalization.

### 2026-05-31 — Baseline v2 OOM'd on long rows (shuffle surfaced them) → relaunch at seq768
- exp49 run fff67d90 OOM at ~opt-step 30 (240 records), 21.85GB used, tried 460MB w/ 181MB free. val never ran (OOM before step100). Only 3 logged pts — no useful result.
- **Cause:** SHUFFLE now hits the corpus's max-length (1024-tok) rows that smoke#3's unshuffled first-150 missed. Peak at 1024 tok dominated by answer-span LOGITS (1024×151936×4B ≈0.6GB EACH × teacher/base/student ≈1.9GB) + 36-layer activations. 4-bit base + grad-accum didn't grow mem (grads in-place); it's per-row activation/logit peak. expandable_segments was set but it's a genuine capacity edge, not just fragmentation.
- **Fix:** lower max_seq_length 1024→**768** (cuts logits+activations ~25%; answer median ~600 tok so minimal signal loss; log truncation). Relaunch v2. If still OOM → 512 and/or reduce teacher/base full-logit retention to top-k+argmax (~1.2GB saving, code refactor).

### 2026-05-31 — CORRECTION (user catch): OOM was largely the METRIC path, not just forward activations
I was OVERCONFIDENT attributing OOM to forward activations. User flagged loss/metric cost (unchunked CE/softmax over large vocab). Inspecting `topk_kl_loss`: it called **`student_logits.float()` TWICE over full vocab (151936)** — ~0.5GB per [N,V] fp32 tensor, and since student_logits carries grad, BOTH retained in the autograd graph → ~0.9-1.4GB avoidable. Teacher `.float()` adds transient ~0.5GB. This was plausibly the marginal OOM cause, NOT (only) 36-layer activations.
- **Fix (e5d93228):** logsumexp reduces over vocab without a persistent fp32 copy; upcast only the gathered [N,K] slices. (Backward through logsumexp still needs the bf16 [N,V] softmax — inherent — but the 2× fp32 full-vocab copies are gone.)
- **Empiricism:** added `gpu_peak_gb` + `ans_len` to logged metrics → stop guessing; the peak will show the real ceiling and whether it tracks ans_len.
- **Restart v2c at 1024** (audit-matched; the ~1GB saving should clear what 768 worked around): exp **issue49-d2l-baseline-v2c**, out /tmp/rune-ck-baseline-v2c, val=val.clean. Watch will confirm + report gpu_peak_gb.
- Lesson: measure memory, don't assert. Future: if still tight, chunked-CE/cut-cross-entropy over the answer positions decouples logit memory from N (the deeper fix the user is pointing at).
- **AWAITING EMPIRICAL gpu_peak_gb** (v2c exp issue49-d2l-baseline-v2c, RUNNING; first log at opt-step10=80 micro-steps after ~2.5min load). TO RECORD when it lands: min/max/median gpu_peak_gb (vs 22.03GiB), and whether 1024 now CLEARS the OOM. Watch b3w18in00.

### 2026-05-31 — Measurement-rigor corrections (reviewer) — folded in (45b8de3c)
- (1) **gpu_peak_gb was a WINDOW high-water mark** (reset only at log steps / 80 micro-steps) → saturates on worst row → corr(peak, ans_len) meaningless. FIXED: reset_peak per micro-step so logged peak ↔ that row's ans_len. **CAVEAT: v2c is running the OLD window instrumentation** → its gpu_peak is only a ceiling/"did 1024 clear OOM" signal; do NOT compute corr(peak,ans_len) from v2c. Per-row corr available only on the NEXT run.
- (2) **Don't upgrade "plausible" → "validated" from a successful rerun** (ordering/split/allocator differ). ISOLATION: `tools/bench_loss_mem.py` measures OLD vs NEW topk_kl_loss peak on the SAME [1024,151936] grad tensor → direct saving, no training confounds. **RUN after v2c frees the GPU** (can't run concurrently). TO RECORD: OLD peak, NEW peak, SAVING GB.
- Language discipline: v2c clearing OOM = SUPPORTS loss-fix; bench_loss_mem = ISOLATES it.

### 2026-05-31 — ★ EMPIRICAL memory verdict (measured, not asserted) ★
- **Loss-fix isolation (`tools/bench_loss_mem.py`, [1024,151936]):** OLD peak **3.42GB** → NEW **1.87GB** → **SAVING 1.556GB** (no training confounds). The metric-path `.float()` over full vocab WAS a real, substantial cost — reviewer/user hypothesis VALIDATED.
- **BUT v2c (1024, loss-fixed) STILL OOM'd** (~step30, tried 460MB w/181MB free). Empirical window gpu_peak 20.5–22.7GB at ans_len 345–792 → the **forward-activation baseline (~20GB) DOMINATES**; the 1.56GB loss saving is necessary but insufficient at 1024.
- **Reconciliation (honest):** BOTH contributors. (a) metric path 1.56GB (the catch — valid), (b) student-forward retained 36-layer activations for the adapter-delta backward at 1024 tok = the larger ~20GB baseline. My original "forward activations" was directionally right; my correction "metric path" also right; neither alone = the full picture. Lesson reaffirmed: MEASURE.
- units note: gpu_peak_gb is /1e9 GB; capacity 22.03 GiB = 23.65 GB. OOM rows peak >23.6GB.
- **ACTION: relaunch at seq 768 + loss-fix + per-row instrumentation.** 768 scales both forward (~0.75×) and loss (~1.4GB) → fits with margin; v2b ran at 768 (old loss) before. Deeper option if needed: gradient-checkpointing-compatible forward (incompatible with monkeypatch today) or activation offload.

### 2026-05-31 — Metric constancy explained + seq-768 justified empirically (user Qs)
v3 (768) data, 15 log pts:
- **Why "constant":** (a) `scaler_B/grad_l2`, `head/grad_l2`, `scaler_A/grad_l2` = (none) → **grad-accum logging BUG**: rec built AFTER optimizer.zero_grad → grads cleared. FIXED (72538e73): capture grad norms before zero_grad. (b) `scaler_B/mean`=0.1001, `scaler_B/l2`=1.6016, `scaler_A/l2`=15.93 ~flat = DESIRED anti-collapse behavior: reinit scaler_B→0.1 only needs to STAY non-zero so B_raw gets gradient; the HEAD learns B_raw (`head/l2` 724.419→724.591 moves at 4th sig-fig; **diff_agreement 0.055→0.525 rises**). Sub-precision drift, not a stall. Real learning metrics all vary (uniq 15/15).
- **Seq 768 justified:** per-row peak now correct; **corr(gpu_peak_gb, ans_len)=0.974** (peak IS length-driven). Linear fit: peak ≈ **10.8GB + 0.0128GB/token**. N=1024 → ~23.9GB > 23.65GB cap → OOM on long rows EVEN with the 1.56GB metric fix (matches v2c). N=768 → max 20.36GB, ~3.3GB headroom (safe). ~896 ≈22.5GB feasible w/ thin margin. So **metric fix necessary but NOT sufficient for 1024; 768 needed.**
- **v3b RUNNING** (768 + loss-fix + per-row peak + grad-log fix): exp **issue49-d2l-baseline-v3b**, out /tmp/rune-ck-baseline-v3b, val=clean. Watch bticzyq5o. This run should COMPLETE + give full diagnostics (confirm scaler_B/head grad flow now logged).

### 2026-05-31 — Length resilience analysis (user Q): is 768 a permanent ceiling? NO.
Confirmed in code: `_student_logits` is ANSWER-ONLY + the GRAD path; single `max_seq_length` conflates context-encoding (extract_activations, no_grad, cheap) AND answer span (student backward, grad, expensive). corr(gpu_peak, ans_len)=0.974 → **768 is an ANSWER-SPAN backward-memory limit, not a context limit.**
THREE lengths:
1. **Inference GENERATION length: FULLY unconstrained** — adapter = fixed-size LoRA weights (no seq dim); base generates any length.
2. **Inference CONTEXT length: not hard-constrained** — Perceiver = fixed latents × variable context; inference is far cheaper than training (no backward/teacher/student) so longer contexts fit on same HW. Limit = distribution shift (OOD beyond trained ctx length): runs, may degrade.
3. **768 = TRAINING answer-span memory bound only; does NOT constrain inference.**
**Levers to make it HW-resilient (not stuck at 768):**
- **(cheap, high-value) DECOUPLE context vs answer length in training:** `max_context_length`≈2048-4096 (no_grad, cheap → hypernet SEES long contexts → resilient) + `max_answer_length`≈768 (bounds backward mem). Same memory, removes the 768 conditioning ceiling. Small code change (split the one knob; extract_activations + teacher use ctx cap, student uses ans cap). **Recommend doing soon.**
- **(deferred) chunk + combine_lora (D2L):** arbitrary length via in-distribution chunks → rank K·r adapter; sample n_chunks in training. Needs own parity/preservation tests + rank-expansion-confound control.
Bottom line: generation length free; checkpoint RUNS on longer contexts (OOD risk); decouple ctx/ans now + chunking later for true HW-scaled resilience.

### 2026-05-31 — Length-resilience CAVEATS (reflections) → requirements on decoupling
I OVER-CLAIMED "context extraction is cheap/free." Corrections (now requirements if we implement decoupling):
- (1) **context extraction NOT free** — base forward over full ctx + `output_hidden_states` materializes features [1,n_layers,seq,dim] (~0.3MB/token → ~1.2GB @4k) + forward working set; scales with ctx length. → MUST log peak-mem + wall-clock SEPARATELY for ctx-extraction vs answer-backward before raising ctx; don't assume same memory.
- (2) **truncation-policy contract:** diff mask = teacher(ctx+ans) vs base(ans); changing ctx length changes the teacher → diff_agreement/preservation/diff-token NOT comparable across configs unless the comparator truncation is pinned + identical everywhere a comparison is claimed. Make truncation explicit.
- (3) **claim precision:** say "not constrained by generated ADAPTER SIZE" (solid), NOT "free." Positional/runtime/OOD remain → require a long-context parity/preservation SMOKE before calling it HW-resilient.
Decoupling still worth doing but ships WITH: per-path mem/wall instrumentation + explicit identical truncation contract + long-context smoke. Awaiting user go.

### 2026-05-31 — ★ v3b: held-out generalization CONFIRMED + HPO decision ★
exp issue49-d2l-baseline-v3b (id53). 768 holds (peak≤20.4GB). **Grad flows to collapse-critical params** (scaler_B/grad 0.05-0.41, head/grad 0.23-0.57 — logging fix works). **HELD-OUT (clean val) step100: val_diff_agreement=0.177, val_preservation=0.926.** → loop generalizes to real code-review trajectories on UNSEEN families. Hypothesis supported; original collapse fixed.
- Rate 19.7s/opt-step → full epoch ~4h (too long for now). v3b undertrained (100/866 = 12%, default hyperparams).
- **DECISION: HPO overnight.** Justified: never tuned hyperparams; one config; 0.177 may be improvable. (Final-only would need confidence config is near-optimal — we lack it.) Best HPO trial → extend to final train.
- **Smoke gate (`tools/diag_corpus_gate.py`, 99fc506c):** real/zero/contra on clean held-out, dual-precision (4bit+bf16) — content-specificity confirmation. Run on v3b step-200 checkpoint (saves ~step200, ~30min) BEFORE committing overnight: confirms real>zero (adapter helps) AND real>contra (content-specific), not generic.
- **HPO harness** (`tools/hpo_train.py`, building): Optuna sqlite (resumable), maximize held-out val_diff_agreement (penalize if val_preservation<0.7), short trials (max_steps~80), search lr/scaler_b_init/train_scaling/topk/grad_accum. ~20-24 trials overnight (~30min/trial incl base reload).

### 2026-05-31 — PLAN: what I'm doing now & why (time-boxed; user wants overnight launch in ~1.5h)
GOAL: converge within the hour, then launch overnight HPO (or final training if HPO unneeded). v3b already shows held-out generalization (val_diff_agreement 0.177) → loop works.
STEPS (in order):
1. **Build `tools/hpo_train.py`** (NOW, CPU) — Optuna study (sqlite, resumable, MLflow per trial), objective = held-out val_diff_agreement on clean val (penalize preservation<0.7), short trials (~80 steps), search lr/scaler_b_init/train_scaling/topk/grad_accum. WHY: never tuned hyperparams; this is the overnight job that finds a good config (then a final train extends the best trial). Built before launch so it's ready.
2. **Wait for v3b → step 200** (~30min) for a saved checkpoint + 2nd val point. WHY: need a real checkpoint to run the confirming smoke gate; 2-pt val trend.
3. **Run `diag_corpus_gate.py` on v3b@200** — real/zero/contra on clean held-out, dual-precision. WHY: val_diff_agreement 0.177 alone doesn't prove CONTENT-specificity; the gate's real>contra check does. This is the hypothesis smoke test the user asked for. Confirms before spending the overnight.
4. **Decide + launch overnight:** gate passes (real>zero, real>contra) → launch HPO overnight. If gate fails (real≈contra) → diagnose, don't burn overnight on a generic-perturbation adapter.
Fallback if v3b@200 too slow: use val curve (100,200) as the evidence + gate on whatever checkpoint exists; HPO can also serve as the long run.

### 2026-05-31 — v3b ckpt@200 saved; held-out val RISING 0.177→0.201; running smoke gate
v3b val_diff_agreement: step100=0.177, step200=0.201 (RISING on clean held-out). Stopped v3b (have evidence), snapshotted ckpt → /tmp/rune-ck-v3b-step200.pt. Running `diag_corpus_gate.py` (real/zero/contra, dual-precision 4bit+bf16, n=40, scaling0.5) on it. Watch b3b4cyey4. GATE_PASS = real>zero AND real>contra both precisions → confirms content-specificity → launch overnight HPO. Then `tools/hpo_train.py` (validated objective-read) under watchdog.

### 2026-05-31 — HPO RESEARCH SYNTHESIS: hyperparameters + metrics (collapse/overfit-aware)
Sources: doc2lora (arXiv:2602.15902 / SakanaAI/doc-to-lora / pub.sakana.ai) + ctx_to_lora/configs.py (authoritative for our codebase) + KD tuning lit (temperature/alpha, dynamic schedules) + LoRA tuning guides (Unsloth, Raschka, LoRA-dropout/Bayesian-HPO papers).
**Doc2LoRA confirmations:** teacher=full-ctx / student=base+gen-adapter (= our setup); **KL ≫ next-token CE** (F1 0.819 vs 0.763 SQuAD) → our top-K KL choice validated. ctx_to_lora defaults: **lr 4e-5, weight_decay 0.01, warmup 100 steps, lora_dropout 0, gen_lora_l1_reg_coef 0 (sink OFF — keep 0), NEFTune α 5.0**, KL at implicit T=1.

**(A) HYPERPARAMETERS TO OPTIMIZE — tiered:**
- T1 (dominant): `learning_rate` 3e-5–3e-4 log (D2L 4e-5; our 2e-4 is HIGH — high LR overfits short runs, tune FIRST); `train_scaling` 0.25–1.5 (≈LoRA α/r, sweet spot ~0.5; r:α≈2 optimal); `scaler_b_init` 0.05–0.3 (collapse-basin escape, not too hot).
- T2 (regularization/generalization — ADD, D2L uses, we currently omit): `weight_decay` 0.0–0.1 (D2L 0.01); `warmup_ratio` 0.0–0.1 (D2L 100 steps).
- T3 (objective/batch): `topk` 25–100; `grad_accum_steps` 4–16; `temperature` 1.0–3.0 (KD extension — NOT a D2L knob, exploratory).
- Optional: NEFTune α (D2L 5.0, embed-noise regularizer); lora_dropout (D2L 0, unreliable short-run → skip).
- **FIXED, do NOT tune:** rank / target_modules / layer_indices (architecture, set by checkpoint); **`gen_lora_l1_reg_coef`=0 (the collapse sink — never sweep up)**.

**(B) OBJECTIVE metric:** maximize **held-out `val_diff_agreement` on the CLEAN val split** (generalization). NOT train-loss (confounded, corr 0.88 w/ row difficulty). NOT `top1_agreement` (can't detect collapse — base≈teacher ~84%).

**(C) GUARD metrics (so HPO can't "win" via collapse / triviality / overfit):**
- `preservation ≥ ~0.8` (penalize below) — broad-perturbation guard (don't wreck the agreement region).
- **trajectory-sensitivity**: centered-delta cosine of generated weights across DIFFERENT contexts < ~0.9 — adapter must CONDITION on context, not emit a constant adapter (anti-collapse/anti-triviality).
- **contradiction control**: val `real_diff_agreement > contra` — content-specific, not format-overfit (the gate metric).
- `scaler_B absmax` tripwire + ΔW norm + per-component grad norms — cheap collapse pre-filters (log, NEVER promote).
- **train–val diff_agreement GAP** — large ⇒ overfitting; monitor/penalize.
- **val-peak early-stop**: val_diff_agreement rising→plateau→decline ⇒ overfit → keep best-step ckpt.

**(D) Composite objective (recommended):**
`obj = val_diff_agreement − λp·max(0, 0.8−val_preservation) − λc·max(0, val_contra−val_real) − λs·[traj_cosine>0.9]`
(current `hpo_train.py` only penalizes pres<0.7 → SHOULD add contradiction + trajectory-sensitivity guards.)

**(E) Collapse-prevention vs Overfit-prevention (explicit):**
- ANTI-COLLAPSE: scaler_b_init>0 (tune, never→0); l1=0; diff_agreement-not-top1; trajectory-sensitivity guard; scaler_B/grad tripwires.
- ANTI-OVERFIT: held-out val objective; weight_decay (ADD, D2L 0.01); warmup (ADD); val-peak early-stop; train-val gap; NEFTune (optional); contradiction control (not format-overfit).

**Implementation gap vs current `hpo_train.py`** (tunes lr/scaler_b_init/train_scaling/topk/grad_accum; penalizes pres<0.7): ADD `weight_decay` (cheap — `_build_optimizer` arg, D2L-backed) + contradiction guard in val eval (moderate) for the overnight; queue `warmup_ratio`/`temperature`/NEFTune + trajectory-sensitivity guard + val-peak early-stop as next. Prioritize weight_decay + contradiction guard before launch if time; else launch current HPO and note the deltas.

### 2026-05-31 — ★ SMOKE GATE result (v3b@200): adapter HELPS but specificity INCONCLUSIVE → FINAL not HPO ★
`/tmp/gate.log` (4bit pass ok; bf16 pass OOM'd — bf16 inference at 768 ctx >22GB). 4bit_train_matched n=39:
- **real=0.201, zero=0.000, contra=0.2005, preservation=0.908.**
- ✅ real ≫ zero → adapter non-inert + helps; preservation healthy.
- ⚠️ real ≈ contra (margin 0.0005) → content-specificity NOT demonstrated. CAVEATS: (a) weak contradiction control (a different REAL row isn't semantically opposed — both induce generic "code-review mode"; reviewer flagged this), (b) aggregate diff_agreement dilutes the minority edit-specific tokens, (c) undertrained (step200, 12%). Synthetic forced-choice DID show specificity (matched foils, edit-local) → capacity exists. So INCONCLUSIVE, not refutation.
**DECISION: overnight = FINAL TRAINING (longer), NOT HPO.** Why: HPO maximizing val_diff_agreement is premature when that objective can't yet separate specific vs generic (real≈contra); HPO would optimize a possibly-generic signal. Final training (a) strong ckpt regardless, (b) lets specificity emerge under the diff-masked objective, (c) gives a real ckpt to re-gate with a STRONGER specificity test. (User delegated HPO-vs-final to findings; findings ⇒ final.)
**Overnight final run prep:** multi-epoch (≈3) on train split, val on clean val, + research-backed anti-overfit: ADD weight_decay (D2L 0.01) + BEST-VAL checkpoint saving (keep peak-val ckpt, not the final/overfit one). Then re-gate the result with: bf16-memory fix (smaller n / 4bit-primary), EDIT-LOCAL forced-choice specificity (not aggregate), and a HARDER contradiction control (same row, edit/feedback removed).
**Post-final TODO:** strengthen gate (edit-local + harder contra + bf16 mem); THEN HPO once objective is specificity-aware (add contradiction/trajectory-sensitivity guards per research synthesis).

### 2026-05-31 — ★ OVERNIGHT FINAL TRAINING LAUNCHED ★
exp **issue49-d2l-final (id54), run 3abe16db7e83416fb3013f5dc39acd0e**, out /tmp/rune-ck-final. url http://localhost:5000/#/experiments/54/runs/3abe16db7e83416fb3013f5dc39acd0e. Watch byu1x3k2j.
Config: 3 epochs train split (~2600 opt steps ≈14h), val=clean every 100, grad-accum8, seq768, **lr 1.5e-4** (↓ from 2e-4 toward D2L 4e-5 for long-run overfit control), **weight_decay 0.01** (D2L), **best-val checkpoint saving** (checkpoint_best.pt at peak val_diff_agreement). 4-bit base, scaler_b_init 0.1, train_scaling 0.5, topk 50.
**Why final not HPO:** gate showed adapter helps (real≫zero) + preserves, but specificity inconclusive (real≈contra, weak control) → HPO objective not yet specificity-trustworthy → train longer first (lets specificity emerge + strong ckpt), re-gate with stronger test, THEN HPO if needed.
**MORNING TODO (when watch byu1x3k2j fires):**
1. Read MLflow exp54: val_diff_agreement curve (PEAK + best_val_step tag), train trend, scaler_B/head grad, preservation drift, no early-stop. Use **checkpoint_best.pt** (not final-step).
2. Re-gate with STRONGER specificity: fix bf16 OOM (n=20 or 4bit-primary), EDIT-LOCAL forced-choice (not aggregate diff_agreement), HARDER contradiction (same row minus edit/feedback). Does real>contra emerge after full training?
3. If specificity confirmed → done (validated checkpoint) or HPO to optimize (with specificity-aware objective: add contradiction + trajectory-sensitivity guards per research synthesis). If still real≈contra after full training → deeper investigation (objective may need explicit content term / negative-context training).
Current HPO harness (tools/hpo_train.py) ready but objective needs the specificity guards before use.

### 2026-05-31 — Sharper specificity test (user idea) → DECIDES overnight plan
Reflections reinforced: (1) don't claim "content-specific generalization" yet — safe claim = "unseen-family teacher-matching under the metric improved"; (2) **final run optimizes the SAME objective that failed real-vs-contra → may amplify GENERIC review-mode, not specificity** → run stronger gate on intermediate/best-val ckpts, keep periodic ckpts; (3) checkpoint_best.pt (by val_diff_agreement) ≠ best-specificity → choose by COMPOSITE post-hoc (diff_agr, preservation, real-vs-contra edit-local margin, train-val gap); (4) bf16 OOM → report 4-bit-train-matched until memory-safe bf16 gate.
**Stopped the overnight final run** to test specificity FIRST (user: elucidate before the night run). Built `tools/diag_specificity.py`: edit-local (difflib pre_code vs revision) MATCHED vs MISMATCHED vs ZERO logprob — decoding-variance-free, header cancels in matched−mismatched margin. Running on v3b@200 (4-bit, n=30). Watch bu875dwqu.
**DECISION LOGIC:**
- matched > mismatched (margin>0, frac wins>0.5) → real specificity signal (even if small, v3b undertrained) → overnight = FINAL training to AMPLIFY (keep periodic ckpts + composite selection + re-gate intermediates).
- matched ≈ mismatched (margin~0) → objective rewards GENERIC, more training won't fix → overnight should change OBJECTIVE: enable **negative-context training** (D2L add_negative_prompt) or add a contrastive term, NOT more-of-same.
Also TODO: completion test (user idea: diff in adapter lets model finish edit it otherwise can't) — add if logprob margin is borderline.

### 2026-05-31 — Specificity-test methodology refinements (reflections) + REFINED decision tree
- (1) report adapter LIFT over zero, not just raw. My design controls target difficulty (matched & mismatched score the SAME revision_i; only adapter source differs → matched−mismatched IS the source effect, zero cancels). Still report matched−zero & mismatched−zero lifts from JSON.
- (2) lexical-overlap confound: mismatched (row i+1 ctx) NOT length/type-matched, and revision_i may share tokens with context_i → positive result = DIRECTIONAL signal → confirm with COMPLETION test, don't declare definitive.
- (3) near-zero at v3b@200 ≠ "objective structurally generic" (it's UNDERTRAINED). Distinguish "inconclusive (undertrained)" vs "objective generic."
**REFINED decision tree (apply to result):**
- clear matched>mismatched (margin>0, frac>0.6, matched−zero>mismatched−zero) → directional specificity → overnight = FINAL to amplify; confirm with completion test on best-val/later ckpt.
- borderline/near-zero margin → DO NOT conclude objective-broken. Run completion test + a SHORT extra training slice (or test a later ckpt); re-test specificity. Only if STILL ~0 → change objective (negative-context/contrastive). Avoid both "blind 14h final" and "premature objective change."
- matched<mismatched (negative) → red flag; investigate before any overnight.

### 2026-05-31 — ★★ SPECIFICITY RESULT: GENERIC edit-booster, NOT trajectory-specific (v3b@200) ★★
`/tmp/rune-issue49-specificity.json` (edit-local, n=30, 4-bit, scaling0.5):
- matched=-2.0527, mismatched=-2.0523, zero=-3.499. **margin matched−mismatched=-0.0004 (frac wins 0.50).** matched−zero=+1.45, mismatched−zero=+1.45.
- **Both user tests answered:** (1) "diff lets model predict edit it otherwise can't" → YES, big (+1.45 nats ≈4.3× prob over zero). (2) matched vs mismatched → ZERO specificity (wrong-trajectory adapter helps the edit equally). Clean (edit-local, target-controlled, decoding-variance-free) — NOT a weak-control artifact.
- **Conclusion:** the corpus-trained adapter is a GENERIC code-edit booster, not adapter-as-memory of the SPECIFIC trajectory. The diff-masked KL objective has a generic optimum (review-mode adapter) satisfiable WITHOUT trajectory specificity. Matches the issue's core worry.
- **Caveat (reviewer):** v3b@200 undertrained → can't yet PROVE "objective structurally generic" vs "specificity not emerged." Resolve via checkpoint TRAJECTORY.
**OVERNIGHT DECISION: final training (3 epochs) with FREQUENT KEPT checkpoints (numbered, save_steps 300) → morning: specificity-gate the trajectory.** If margin grows with training → emerges (undertrained). If flat across all ckpts → objective generic → implement NEGATIVE-CONTEXT/contrastive training (D2L add_negative_prompt; spec already flagged this as the fix if contradiction-worsening too weak — now empirically confirmed too weak). This overnight DISTINGUISHES the two (more scientific than assuming) + yields a useful generic ckpt regardless.
- Alternative offered to user: skip straight to negative-context training overnight (assumes objective-generic). Chose trajectory-test (don't assume; reviewer caution).

### 2026-05-31 — Morning-gating + negative-context design refinements (reflections)
- **Morning specificity gate: SWEEP scalings (0.25/0.5/1.0)** per checkpoint — single scale (0.5) can hide a small context-specific delta (generic dominates) or saturate both adapters. (Enhance diag_specificity.py to accept --scalings.)
- **Judge by SPECIFICITY TRAJECTORY, not val_diff_agreement.** Clean signature of generic-strengthening = **matched−mismatched FLAT while matched−zero RISES** across checkpoints. Plot both across /tmp/rune-ck-final/checkpoint_step*.pt.
- **Negative-context (if specificity stays flat): use HARD negatives, NOT random rows** (random wrong rows behave like generic edit prompts → re-learns "code-review mode"). Hard negatives: same task/source family + file/edit-type, OR **same row with edit-bearing feedback removed/contradicted** (sharpest — isolates trajectory binding). This is the key design fix for the contrastive objective.
**Overnight final RUNNING:** exp issue49-d2l-final, out /tmp/rune-ck-final, numbered ckpts every 200 + best-val, val=clean every 150, 3 epochs, lr1.5e-4, wd0.01. Watch byu1x3k2j (set earlier) — NOTE: I relaunched the run, so re-set watch on /tmp/corpus_final.log.

### 2026-05-31 — Checkpoint durability + disk-fill mitigation
- **Were NOT logged as MLflow artifacts** (run showed artifacts:NONE; only torch.save to /tmp). FIXED code (00f36773): `_save_checkpoint(..., mlflow_handle)` → mlflow.log_artifact("checkpoints/") (S3-backed) on best/numbered/final. Future runs durable.
- In-flight run (old code): synced existing 3 ckpts to S3 + **sync+prune sidecar** (setsid, READ-only pgrep guard, NEVER kills): syncs /tmp/rune-ck-final→s3://.../checkpoints/issue49-final/ every 10min, then prunes local to **best + 2 newest numbered** (rest durable in S3). Local capped ~2.4G; root free 45G (overlay disk, NOT tmpfs → no RAM-OOM). Verified working.
- ⚠️ HAZARD LEARNED: never pkill/kill on patterns matching training cmdline (`run_corpus_distill`, `rune-ck-final`) — nearly killed training. Use [p]ython read-only pgrep; never kill by broad -f match.

### 2026-05-31 — ADAPTER-AS-MEMORY IMPROVEMENT BLUEPRINT (reflections) — the contrastive-fix recipe (if trajectory gate confirms generic)
1. **Contrastive loss term:** matched adapter must beat zero AND hard-negatives on the SAME target edit → matched KL/CE + margin penalty when score(neg_adapter,target) ≳ score(matched,target). On EDIT-LOCAL tokens.
2. **Separate generic prior from trajectory residual:** estimate generic adapter (empty/avg/irrelevant ctx); penalize tiny/non-discriminative W(ctx)−W(generic). Eval: matched−zero rises from generic; memory needs matched−mismatched AND centered-delta up.
3. **Data curriculum:** upweight rows w/ NECESSARY facts (high teacher lift, high matched-vs-hardneg separation, identifiers/literals/API in trajectory, LOW base/zero solvability); downweight boilerplate/generic-review rows. (Many-contexts→same-obvious-revision rows = booster-good, memory-bad.)
4. **Hard negatives CONSTRUCTED (not random):** same row w/ feedback removed; same row w/ key identifier/literal CONTRADICTED; same repo/family different requested change; same pre_code w/ shuffled review comments. Make "use THIS trajectory" the shortest path to lower loss.
5. **Paired batches:** each minibatch = matched + hard-neg contexts for the SAME target edit → direct contrast in one optimizer step (else gradient noise keeps rewarding generic).
6. **Specificity gate INSIDE train/val + SELECTION:** log matched−zero, matched−mismatched, preservation, train-val gap. **Do NOT let HPO optimize val_diff_agreement alone — it selects the best GENERIC adapter.** matched−mismatched must be FIRST-CLASS (objective or selection), even if expensive/less frequent. (Supersedes earlier HPO-objective note: contradiction/specificity is primary, not just a guard.)
7. **If contrastive STILL fails → inspect hypernet INPUT exposes discriminative facts:** trajectory-text ablations (remove feedback sentence/identifier/literal → do generated weights + edit-local logprobs move?). Barely move ⇒ upstream conditioning/representation problem, not the loss.

### 2026-05-31 — PARALLEL PLAN (user): build contrastive system while training runs
GPU busy with final run (~14h). Build the "correct system" (contrastive/specificity) on CPU NOW; monitor triggers analysis on training completion; then run the new system; replace training approach at end of experimentation.
WORKSTREAMS (CPU, parallel to GPU training):
1. **Contrastive training system** (`src/rune/training/contrastive.py` + loop flag): pure TDD helpers — `strip_review_feedback(activation_text)` (hard-neg context = feedback removed) + `contrastive_margin_loss(lp_matched, lp_neg, margin)` (matched must beat feedback-stripped neg on edit-local gold by margin). Wire into loop behind `contrastive=True`: per row, matched + hard-neg adapters, total = diff-masked KL + λ·margin. Specificity-aware val metric (matched−mismatched) + selection.
2. **Trajectory-gate script** (`tools/gate_trajectory.py`): gate ALL /tmp/rune-ck-final/checkpoint_step*.pt (re-pull from S3 if pruned) at multiple scalings → matched−mismatched & matched−zero curves. Ready to TRIGGER on completion.
3. **Completion trigger**: watch bgtclvv1o fires on training done → run #2 analysis → decide (specificity emerged? → continue/HPO; flat? → launch #1 contrastive).
Design (contrastive loss): per row matched_adapter(ctx_i) + hardneg_adapter(strip_feedback(ctx_i)); margin_loss = mean_editlocal relu(margin − (logp_matched(gold) − logp_neg(gold))); total = KL_diffmask + contrastive_weight·margin_loss. Forces USING the feedback (trajectory-specific) → directly optimizes matched−mismatched>margin. Hard-neg "feedback removed" is constructible from activation_text.

### 2026-05-31 — OVERNIGHT CAMPAIGN: re-grounded goals + candidate backlog (try everything → winner + PR comment by AM)
**Re-grounded (PRODUCT/CLAUDE):** north-star = **pass@1** (adapter vs base); research bet = **adapter-as-memory** (trajectory-conditioned) = the matched-vs-mismatched specificity we measure. A GENERIC booster could help pass@1 but NOT prove the bet. Kill criterion = best adapter doesn't beat base pass@1. Hard rules: GPU-OK/background+watchdog, NO install, 4-bit, deferred imports, uv run.
**Monitor boa9lo9kf** fires when training ends (completion/stop/crash) → campaign A→next trigger.
**External-research synthesis (reflections, adapter-memory training) — candidate experiments, prioritized:**
- **B1 (PRIMARY, building): contrastive margin** — matched adapter beats hard-neg (swap-feedback) on edit-local gold by margin; neg detached (no_grad). Log KL vs margin magnitudes separately. Directly optimizes matched−mismatched.
- **B2: DPO/preference-style** (TRL/CPO): implicit reward r=logp_adapter(y)−logp_generic(y); optimize matched≻neg in normalized reward space (logsigmoid). Normalizes out generic prior. Variant of B1.
- **C: dense "episode card"** (raw text = weak memory substrate): prepend structured card (requested change, key identifier/literal/API fact, target edit span) to conditioning. Data-side; expose discriminative facts the hypernet may not bind in long prompts.
- **D: hard-negative POOL + difficulty bands** (Sentence-Transformers guidance): feedback-masked / contradicted-literal / same-family-different-edit / near-nonmatch; select by margin, log type. Dynamic mining (pick negs the model scores high) once a gate exists.
- **E: separate generic skill from episodic residual** (memory-adapter): frozen generic adapter + trajectory-residual hypernet, OR penalize small W(ctx)−W(generic). Don't let generic editing consume the memory channel.
- **F: scale/staging** (D2L warns batch-1 → shallow generic optimum from low context diversity): paired pos/neg same step (B1 does this), larger effective context-token batches, staged (internalize → contrastive). GPU-limited.
- **G: capacity hygiene**: push boilerplate to base prompt/template, reserve adapter capacity for episode deltas.
**Overnight execution (GPU sequential, ~10h):** cut A@~600 → gate A trajectory → run B1 → gate → (if promising) B2 or C → gate. Compare on specificity (matched−mismatched) + preservation + edit-completion (+ tiny pass@1 if feasible). Winner = best adapter-as-memory signal (+ no preservation collapse). AM: documented decision → **PR comment**.
**Eval = `tools/gate_trajectory.py` (building):** matched/mismatched/zero edit-local logprob across checkpoints × scalings (uses canonical contrastive.edit_local_mask).

### 2026-05-31 — Campaign methodology (reflections) — MUST follow
- **ONE-VARIABLE from SHARED warm-start:** every candidate (A-comparable/B1/B2/C) starts from the SAME warm-start (hpo checkpoint, scaler_b_init0.1), SAME train split, SAME clean val gate rows, SAME scalings, SAME max_steps (=600), SAME checkpoint-selection rule. Vary only the ONE intervention. Else no attribution. (B1 = A's exact config + contrastive=True; cut A at ~600 to match.)
- **PR comment = exploratory ranking, NOT proof.** Overnight identifies the most promising direction; only call it validated if it survives rerun/2nd seed/split.
- **TWO LANES in reporting:** (1) product utility = adapter>base on tiny pass@1 / edit-completion; (2) research bet = matched>mismatched on edit-local gate. A candidate can WIN lane-1 (useful generic) while LOSING lane-2 (memory). Report both; don't let pass@1 blur the memory question.

## Notes / gotchas
- 15GB CPU RAM box; `free -g` before every model load; `offload_base=False`; GPU jobs only under `tools/run_guarded.sh`.
- `uv sync --extra gpu` (plain `uv sync` prunes trl/bnb/flash-attn).
- Magnitude (`scaler_B absmax`, ΔW norm) = tripwire only, never a promotion criterion.

### 2026-05-31 21:40 — CAMPAIGN EXECUTION: A cut, B1 fixed (advisor), smoke running
**A (generic baseline) CUT at step 600** (val_diff_agr 0.204, val_pres 0.831 — preservation DEGRADING 0.93→0.83). MLflow run c2305576f2e84d2e9b7c9d4f9afae4b7 (exp 54) marked FINISHED. Checkpoints 200/400/600 synced to s3://.../checkpoints/issue49-final/. **step200/400 also copied to /tmp/rune-ck-trajectory-safe/** (prune insurance: sidecar keeps best+2-newest → step200 would be evicted when 600 saved). Sidecar + A PIDs killed by EXACT pid (331581/331575/331573/441889), never pattern-kill.

**ADVISOR CAUGHT A REAL BUG in B1 (fixed, commit 9d03b454):** the contrastive hinge `relu(margin-(lp_m-lp_n))` had `lp_n` under `no_grad` → gradient only through matched → could lift matched UP but NEVER push the wrong-context (negative) adapter's gold logprob DOWN. Against a generic booster (inflates lp for ANY context), lp_n rises with lp_m, gap never opens → B1 would have reproduced A's generic result and I'd misread it as "contrastive doesn't help." FIX: grad flows through BOTH paths, memory-bounded so seq stays 768 (one-variable vs A): detached neg pass fixes hinge active-set → matched piece (-lp_m) backwards on KL graph → matched graph freed → neg piece (+lp_n) own forward+backward. Only one student grad-graph alive at a time. Cost: 3 student fwd/contrastive row (~slower but overnight-OK).

**Pre-flight verified contrastive ACTUALLY ENGAGES (not silent no-op):**
- feedback extraction: 800/800 rows (100%) on train corpus.
- make_hard_negative scaffold parity: token-len delta median 2.7%, p90 9.8% (NOT detectably shorter — addresses reflections caution that neg must not be distinguishable by "has review text").
- feedback-swap verified 199/200; edit_local_mask non-empty on ALL 200 sampled rows (median 10.5% of answer tokens) → contrastive term fires every row.

**SMOKE RUNNING (issue49-b1-smoke, /tmp/rune-ck-b1smoke):** A's exact config + contrastive=True, 40 steps, save@20, weight=1.0. Advisor's go/no-go: gate matched−mismatched warm-start-vs-step40. If margin DROPS but matched−mismatched FLAT → fix didn't take. Also reads KL vs margin magnitudes to CALIBRATE contrastive_weight for B1 full (weight is safety-critical: too low won't beat generic optimum, too high wrecks teacher-match/preservation).

**Combined gate plan (one base load):** gate_trajectory --ckpts [warm-start, b1smoke_step40, A_step200, A_step400, A_step600] × scalings 0.25/0.5/1.0 on clean val → smoke before/after (B1 mechanism) + A baseline trajectory in one session. Then: if smoke separates → launch B1 full (600 steps, calibrated weight) → gate B1 → PR comment (two lanes: pass@1/edit-utility vs matched−mismatched memory; exploratory ranking not proof).

### 2026-05-31 21:46 — SMOKE early signal (step 10): mechanism LIVE
`step=10 kl=8.115 margin=0.998 loss=9.24 da=0.063 pres=0.946` (weight=1.0).
- **Contrastive FIRES** (margin≠0) + **no OOM at seq768** → grad-through-neg fix runs within memory budget (memory-bounded sequential backward works).
- **Calibration:** margin≈1.0 vs kl≈8.1 → contrastive ≈11% of loss at weight=1.0 (gentle, not swamping teacher-match — addresses reflections "too high wrecks preservation").
- margin near ceiling (1.0) ⇒ lp_matched−lp_swapneg≈0 NOW = the generic no-separation start the objective must break.
- **Weight decision DEFERRED to smoke gate:** if matched−swapneg OPENS at step40 (w/ matched−zero preserved) → weight=1.0 for B1 full; if flat → bump weight (2-3) & retest before concluding mechanism broken. Smoke gate = the calibration.

### 2026-05-31 22:02 — SMOKE complete (40 steps), margin did NOT drop
KL 8.1→3.4 (learning edits), **margin pinned ~1.0 (0.998→1.022)** = lp_matched−lp_swapneg≈0 on training rows throughout. pres healthy 0.95→0.98. NO crash, NO OOM at seq768 (grad-through-neg memory-bounded design holds).
- NOT the advisor's "margin drops but separation flat" failure → ambiguous: (a) 40 steps × gentle ~11% weight too little vs KL, or (b) hypernet weights barely depend on feedback span (upstream-conditioning concern, blueprint pt 7).
- Running **smoke gate** (warm-start vs step40, scaling 0.5, n=24) on HELD-OUT clean rows (more sensitive than training-batch margin avg) → decomposition matched−swapneg / matched−zero / swapneg−zero.
**DECISION TREE on gate:**
- step40 matched−swapneg OPENS vs warm AND matched−zero≥0 → mechanism works → launch B1 full (weight 1.0, maybe →2 for 600-step signal).
- FLAT → quick higher-weight re-smoke (weight≈5, 40 steps) to disambiguate weight-too-low vs conditioning-broken. Opens at high weight → launch B1 high-weight. Still flat → bottleneck is INPUT representation not loss → pivot to candidate C (episode card) or report finding.
**A trajectory gate (200/400/600) DEFERRED to morning combined gate** (one base-load with B1-full checkpoints) — not blocking B1 launch.

### 2026-05-31 22:09 — SMOKE GATE = generic-booster CONFIRMED at contrastive objective (weight 1.0, 40 steps)
| ckpt | m−mismatch | m−swapneg | m−zero | swapneg−zero | pres |
|---|---|---|---|---|---|
| warm-start | +0.003 | −0.001 | −0.008 | −0.007 | 0.997 |
| **B1 step40** | +0.002 | **+0.001** | **+0.862** | **+0.861** | 0.950 |
**INTERPRETATION:** 40 contrastive steps → matched−zero +0.86 (strong generic edit boost) but swapneg−zero ALSO +0.86 → matched−swapneg ≈ 0. **Swapping feedback content changes the adapter's edit effect by ~0.** matched−mismatched ≈0 too (different ROW's full context boosts THIS edit equally). Pure generic booster — trajectory/feedback does NOT bind.
**WHY contrastive didn't separate:** grad-through-neg fix IS present, but hypernet maps matched-ctx and feedback-swap-ctx → adapters with ~IDENTICAL edit effect → hinge sends canceling gradients (push lp_matched up + lp_swapneg down on near-same weights → cancel). = UPSTREAM CONDITIONING failure (blueprint pt7 / reflections), confirmed EMPIRICALLY, not a loss-weight issue (probably).
Gate json saved: /tmp/rune-ck-trajectory-safe/gate_smoke_warm_vs_step40.json.
**BRANCH:** consulting advisor: higher-weight retry (weight≫1) vs go straight to representation/conditioning fix (does ||W(matched)−W(swap)||≈0? if so loss can't fix → episode-card / input-conditioning candidate C).

### 2026-06-01 — ADVISOR + USER redirect: weight-sensitivity probe BEFORE any retry
**Advisor:** swapneg−zero (0.861) == matched−zero (0.862) to 3 decimals is suspiciously tight for two different inputs. The +0.86 is consistent with (a) hypernet learned a useful CONSTANT adapter (REAL; loss has no lever) OR (b) generate_weights/extract_activations flattens/ignores context (WIRING bug, upstream of B1). I eyeballed CONTEXTS differ but never checked GENERATED WEIGHTS differ. Higher-weight smoke can't disambiguate (canceling gradients flat regardless). **Probe (no train, ~5min, tools/diag_weight_sensitivity.py):** for context pairs compare extracted FEATURES + generated WEIGHTS: feat differ & W~identical → generate_weights ignores input (upstream); feat~identical → extraction bottleneck (upstream); W differs substantially → REAL, loss has lever.
**USER (Sakana doc2lora up-scaling):** they up-scaled hypernet output to achieve base-model recall (our cont_multiplier≈1.53). I tested only scaling=0.5 → a real context-specific component could be SUPPRESSED at low scale. **Probe W_rel is SCALE-INVARIANT** (scaling is uniform scalar → relative weight diff unchanged) → probe still answers "weights depend on ctx?" regardless. THEN: if W differs → gate across **0.25→2.0** (test if specificity emerges at higher scale = Sakana recall). If W~constant → scaling moot (constant adapter ≡ at all scales).
**Morning reframe (advisor):** a rigorous NEGATIVE — "contrastive correctly implemented can't induce specificity because generated weights don't depend on conditioning ctx; evidence: [ablation]" — is a COMPLETE, valuable deliverable: redirects #49 from loss-engineering to CONDITIONING. Don't crank weights to manufacture a positive.

### 2026-06-01 05:26 — reflections refine the probe (centered/layerwise, not global cosine)
New reflection bullets on the conditioning probe:
- **Global weight cosine is NOT enough** — dominated by shared generic adapter. W(ctx)=W_generic+W_residual(ctx); big generic → global cos≈1 even with a real small high-leverage residual. Report **layerwise rel-L2, CENTERED deltas vs mean/empty ctx, gen-weight norms, induced edit-local logprob deltas**. Decisive Q: "does the context-dependent RESIDUAL move the target edit differently?" not "are weights different somewhere?"
- **Two failure modes to separate:** (1) weights ~constant across matched/swap/mismatch → conditioning/representation failure (fix BEFORE loss). (2) weights differ but logits don't → scale / rank-placement / generic-component swallows the residual ← THIS is the user's Sakana up-scaling hypothesis. Different fixes; probe must separate.
**PLAN:** read running probe (global W_rel/cos + FEATURE deltas — feature part = clean upstream signal: are extracted features even different for diff texts?). Then ENHANCED probe: per-ctx ||W||, centered residual ||W(ctx)−W_mean||/||W_mean||, layerwise rel-L2, vs empty ctx. If residual ~0 → mode(1) conditioning. If residual real but logits flat at 0.5 → mode(2) → Sakana scaling gate 0.25→2.0.

### 2026-06-01 05:33 — probe iteration AAR (two crashes fixed)
- **diag_weight_sensitivity crashed:** subtracted variable-length raw features (row0=75.6M vs row1=58.3M elements). extract_activations returns (1,L,S,H), S=seq_len varies. Raw feature subtraction invalid. **Generated WEIGHTS are fixed-size** (perceiver pools) → weight comparison valid; feature comparison needs masked mean-pool over S.
- **diag_conditioning v1 crashed on EMPTY ctx:** Qwen3.5 linear-attn can't reshape 0-length seq ([1,0,-1,128] ambiguous). → replaced "" with NEUTRAL generic code snippet (also the better "neutral baseline" reflections wanted). Features now masked-mean-pooled to fixed (L*H) (fcddbab2/99610372).
- **Probe v2 running (pid 11380):** step40 ckpt, 5 rows + neutral, reports per-ctx ||W||, centered residual ratio ||W(ctx)−W_mean||/||W_mean||, feat residual, layerwise rel-L2 matched-vs-swap / matched-vs-row1. Decisive: residual ratio ~0 → constant adapter (conditioning failure) vs substantial → Sakana scaling gate next.

### 2026-06-01 05:36 — CONDITIONING PROBE v2 RESULT (step40): signal ATTENUATED ~25-30×, not absent
||W_mean||=84.02. Per-ctx ||W|| all ≈84.0 (identical norms).
| ctx | resid/Wmean (WEIGHT) | feat_resid/featmean (FEATURE) |
|---|---|---|
| row0-2 | 0.010-0.011 | 0.29-0.31 |
| row3-4 | 0.006 | 0.25 |
| NEUTRAL | 0.021 | 0.78 |
- matched(row0) vs feedback-SWAP: **W relL2=0.0042 (0.4%)** vs **feat relL2=0.053 (5.3%)**.
- layerwise W relL2 matched-vs-DIFF-ROW: up to **0.053 @layers 21-24** (mid-late). matched-vs-swap: only 0.006 @layer27.
**DIAGNOSIS:** FEATURES carry context (25-31% across rows, 5.3% from feedback alone) → extraction FINE. But GENERATED WEIGHTS barely move (residual ~1%, feedback-swap 0.4%) → hypernet COMPRESSES 25-31% feature variation → ~1% weight variation = **~25-30× attenuation of conditioning**. 99% of adapter = shared generic; trajectory drives <1%. = the "dominated by shared generic" reflections warned of, QUANTIFIED. NOT a constant adapter (residual real, 1-5%, concentrated mid-layers 21-24 for cross-row) but FUNCTIONALLY swamped at scaling 0.5.
**→ DECISIVE NEXT (advisor mode-2 + USER Sakana):** re-gate step40 (+warm-start) across scalings 0.25/0.5/1.0/2.0 — does matched−mismatched GROW with scale (residual functionally context-specific, just needs up-scaling = Sakana recall) or stay ~0 at ALL scales (residual not functionally specific → conditioning/representation failure, redirect #49 upstream)? Watch preservation (high scale may destabilize).

### 2026-06-01 05:41 — reflection caveat (mean-pool may erase sparse feedback tokens)
Reflection: mean-pooled feature comparison can dilute sparse edit-bearing feedback tokens → "similar pooled features" alone wouldn't prove extraction ignores feedback; do token-local feedback-span check or ablate feedback span & measure weight/logprob movement. Gen-weight comparison = stronger end-to-end signal.
**Doesn't change course:** my feedback-swap showed feat relL2=5.3% (features DID move) while weights moved only 0.4% → bottleneck is perceiver→weight, NOT extraction. If pooling dilutes, true local feedback signal is even >5.3% → attenuation even worse. Conclusion holds/strengthens. (Optional later: token-local feedback-span feature delta to fully nail extraction side — not blocking.) Waiting on Sakana scaling gate (bc90vw6vc).

### 2026-06-01 05:45 — SAKANA SCALING GATE (step40): up-scaling DISCONFIRMED, conditioning-collapse CONFIRMED
| scale | m−mm | m−swap | m−zero | swap−zero | pres |
|---|---|---|---|---|---|
| 0.25 | −0.002 | −0.001 | +0.691 | +0.692 | 0.974 |
| 0.5 | +0.002 | +0.001 | +0.802 | +0.802 | 0.949 |
| 1.0 | +0.002 | −0.001 | +0.211 | +0.213 | 0.863 |
| 2.0 | +0.000 | +0.003 | −1.567 | −1.570 | 0.458 |
**matched−mismatched & matched−swapneg = NOISE (±0.003) at EVERY scale.** m−zero == swap−zero to 3 decimals at all scales → matched & wrong-ctx adapters FUNCTIONALLY IDENTICAL regardless of scale. Up-scaling does NOT expose specificity — it DESTABILIZES: generic lift peaks @0.5 (+0.80) → collapses to −1.57 @2.0, pres 0.97→0.46. **Sakana up-scaling recovery DISCONFIRMED** for step40; the ~1% weight residual is NOT functionally context-specific. = conditioning-collapse branch.
**CONFIRMATION (not a step40/B1 artifact?):** gate warm-start + A 200/400/600 across scales → is flat-at-all-scales also true at the warm-start (conditioning never there) and after A's longer GENERIC run? If yes → robust architectural finding (objective- and length-independent) → morning deliverable = rigorous NEGATIVE redirecting #49 to the perceiver→weight conditioning bottleneck (~25-30× attenuation).

### 2026-06-01 05:47 — reflections CONVERGE with scaling result + point to the FIX
Reviewer (pre-confirming my 05:45 scaling result): "uniform scaling amplifies BOTH generic + residual; if specificity still doesn't emerge, fix is NOT larger scaling — it's FACTORING OUT the generic component: generic-plus-residual heads, centered/residual loss, or penalty/reward on W(ctx)−W(neutral)." + "layers 21-24 carry largest cross-row residual, feedback-swap tiny → targeted layer analysis > global scaling."
**= exact match to my findings.** Diagnosis + forward-fix converged: conditioning attenuation, generic dominates, scaling disconfirmed → fix = factor out generic / centered-residual objective.
**HIGH-VALUE follow-up to strengthen deliverable (tests the FIX direction):** CENTERED-RESIDUAL gate — apply only W(ctx)−W_mean (generic subtracted), up-scaled, measure matched−mismatched. If centered residual ALONE (upscaled) yields matched−mismatched>0 → specificity IS in the residual, fix is architectural (factor out generic) = actionable positive direction. If still flat → residual is noise, conditioning truly collapsed. Build after confirmation gate (warm+A) confirms the negative is architectural (not step40 artifact).

### 2026-06-01 05:50 — FIRST-PRINCIPLES DATA ARCHAEOLOGY (what are we asking the adapter to embed?)
**Schema (external_codereview, S3 github PR pairs):** `activation_text` = `## Task`(repo/PR/file) + `## Current Code`(pre-edit, ~1830 chars) + `## Review Feedback`(reviewer comment). `teacher_text` = activation_text + `## Revision` + FULL revised code. _map_record: context=activation_text, answer=the `## Revision` block. **revision==post_code 400/400.**
**Quantitative (400 train rows, Qwen tok):**
| metric | median | mean | p10 | p90 |
|---|---|---|---|---|
| answer tokens | 401 | 693 | 315 | 666 |
| context tokens | 472 | 716 | 364 | 721 |
| **feedback tokens** | **23** | 36 | 9 | 71 |
| feedback/context ratio | **0.047** | — | — | — |
| **COPY frac (revision verbatim ⊂ pre_code)** | **0.892** | 0.767 | 0.31 | 0.999 |
| **EDIT-local tok frac (revision differs from pre)** | **0.097** | 0.19 | 0.005 | 0.539 |
| quality_score | 0.40 | 0.34 | 0.28 | 0.40 (p90=0.4 → looks capped/uniform, investigate) |
**THE STRUCTURAL PROBLEM (explains conditioning collapse from the DATA side):**
1. The **answer is ~89% a verbatim copy of the Current Code** already in the context; the real edit is only ~10% of tokens (one ternary/expr in row0, exactly what feedback asked).
2. The **discriminative signal (feedback) is ~5% of the context (23 tok)** driving ~10% of the output. 90%+ is generic code reproduction.
3. Student forward is ANSWER-ONLY (adapter carries context) → to reproduce the 89% copy, the adapter must **regurgitate ~472 tok of specific code from a rank-r adapter** — verbatim long-span recall, MUCH harder than Sakana doc2lora's SPARSE factual QA recall.
4. **Hard-negative (feedback swap) keeps Current Code IDENTICAL** → adapter must reproduce the SAME 89% body; contrast can ONLY live in the ~10% edit, and only if the edit truly depends on feedback. **This DIRECTLY explains the 0.4% feedback-swap weight diff & ~1% residual:** the adapter's capacity goes to the shared code body; feedback-specific signal is structurally a sliver.
**SAKANA CONTRAST:** doc2lora embeds a document, tests recovery via SPARSE factual QA. Ours = verbatim reproduction of a long edit-in-context. Different memory regime; Sakana's "knowledge recovery" ≠ verbatim long-span regurgitation.
**→ USER'S DEEPER Q: do adapters present RECOVERABLE FACTS at all?** Next: direct recall test — does base+matched-adapter recall THIS row's code (FULL-answer logprob) > base+mismatched > base? If matched≈mismatch even on full-answer → adapter encodes NO recoverable context-specific facts (most basic failure, upstream of contrast).

### 2026-06-01 05:53 — reflections converge: DATA TRANSFORMATION is the fix direction
Reviewer: "full-answer logprob mostly measures reconstructing shared code body; edit-local changed tokens = relevant memory signal" (my copy/edit split handles this). KEY forward rec: **"train/evaluate on compact EDIT PROGRAMS or PATCHES (pre_code→post_code), feedback as conditioning fact; keep full-code generation as downstream integration test, NOT the core distillation target."** = matches archaeology: full-code target dilutes feedback→edit signal (89% copy). 
**Recall probe built (diag_recall.py):** copy/edit/full matched−mismatch. Will run after confirmation gate frees GPU. copy m−mm>0 = recalls specific code; edit m−mm>0 = recalls feedback-specific edit (the bet). Both expected ~0 per prior; this DIRECTLY answers "recoverable facts?".

### 2026-06-01 06:00 — CONFIRMATION GATE (warm + A 200/400/600): NOT flat — small code-residual that GROWS with train & scale
| ckpt | sc | m−mm | m−swap | m−zero | pres |
|---|---|---|---|---|---|
| warm | 0.5/1.0 | +0.001/+0.005 | ~0 | −0.009/−0.029 (INERT) | 0.997/0.995 |
| A200 | 0.5/1.0 | +0.007/+0.018 | ~0 | +1.254/+0.415 | 0.927/0.633 |
| A400 | 0.5/1.0 | −0.001/+0.005 | ~0 | +1.102/+0.172 | 0.846/0.476 |
| **A600** | 0.5/1.0 | **+0.059/+0.134** | +0.005/+0.010 | +1.075/+0.167 | 0.814/0.454 |
**REVISED NARRATIVE (richer than "total collapse"):**
- warm-start adapter ~INERT (m−zero≈0). Generic edit-boost (m−zero→+1.1) DEVELOPS with training (A200/400/600).
- **A600 (longest generic, 600 steps) shows REAL context-specificity: m−mm +0.059@0.5 → +0.134@1.0** (~10× warm/200/400 baseline, AMPLIFIES with scale → USER's Sakana intuition PARTIALLY RIGHT). But: (a) it's ~5% of the generic effect (+0.059 vs +1.075), (b) **code-driven NOT feedback-driven** (m−swap stays ~+0.005 — feedback barely binds), (c) up-scaling exposes it but CRATERS preservation (0.81→0.45).
- = adapter recalls SOME per-row CODE info (the 89%-copy task forces code encoding), swamped by a ~95% generic component; the 5% FEEDBACK (trajectory-specific request) essentially does NOT bind.
**→ recall probe on A600 (copy/edit/full m−mm) to quantify the code-recall directly = answers "recoverable facts?".**

### 2026-06-01 06:05 — RECALL PROBE (A600) = answers "recoverable facts?": NO (doc-style), faint code-edit prior only
| sc | slice | matched | mismatch | zero | m−mm | m−zero |
|---|---|---|---|---|---|---|
| 0.5 | full | −1.677 | −1.682 | −1.390 | +0.005 | **−0.287** |
| 0.5 | **copy** | −1.689 | −1.693 | −1.375 | **+0.003** | **−0.314** |
| 0.5 | **edit** | −2.238 | −2.313 | −3.406 | **+0.075** | **+1.169** |
| 1.0 | copy | −2.779 | −2.770 | −1.375 | −0.009 | −1.403 |
| 1.0 | edit | −3.144 | −3.305 | −3.406 | +0.161 | +0.262 |
**VERDICT (first-principles, recoverable facts):**
1. **NO recoverable code-body facts:** copy m−mm≈0 AND copy m−zero NEGATIVE (−0.31) → adapter doesn't recall this row's code, it HURTS code-body likelihood vs base.
2. **Adapter = GENERIC EDIT-BOOSTER:** boosts edit tokens (+1.17 m−zero) at the COST of code body (−0.31) → net full-answer WORSE than base (−0.287). Not memory; a "make-an-edit-here" prior.
3. **Faint recoverable EDIT signal:** edit m−mm +0.075→+0.161 (grows w/ scale) — but tied to CODE context, NOT feedback (feedback-swap m−swap only +0.005). Adapter weakly knows WHICH edit from surrounding code, ~nothing from the review request.
**= Sakana doc-recall premise does NOT hold here.** No document-style fact recovery; only a faint code-conditioned edit prior swamped by generic boosting; trajectory-specific (feedback) signal does not bind.
**Reviewer decision-rule mapping:** observed = "edit separates weakly, copy does NOT" (not their copy-yes/edit-no case). Implies: capacity is spent HURTING copy to boost edits generically → **data/objective reformulation toward PATCHES (pre→post diff, feedback-conditioned) is the indicated fix** — removes code-reproduction burden, isolates feedback→edit binding, reserves capacity for the episode delta. Consulting advisor on synthesis before PR comment.

### 2026-06-01 06:08 — AFTER-ACTION REPORT (deliverable shipped)
**Advisor verdict: stop probing, write it — 5 convergent probes.** Two corrections applied: (1) give CONDITIONING attenuation co-equal billing with data-dilution (don't let "patches" bury the deeper mechanistic result); recommendation = "patches AND re-run conditioning probe to confirm feedback moves weights." (2) product lane UNMEASURED, proxy net-negative — stated plainly, no pass@1 claimed.
**SHIPPED:**
- `docs/issue49-findings-2026-06-01.md` (commit ebb858f9) — durable findings.
- PR #50 comment: https://github.com/ElixirTrials/rune/pull/50#issuecomment-4590010079
- Branch pushed (26 commits → origin/feat/issue49-d2l-retrain).
**FINAL ANSWER to "recoverable facts?":** NO document-style recall (adapter doesn't store row's code, HURTS it −0.31); it's a generic edit-booster (net-negative full-answer); only a FAINT code-conditioned (NOT feedback-conditioned) edit signal (+0.075→+0.161, feedback-swap +0.005). Sakana premise doesn't transfer: our "fact" = 10%-of-output edit driven by 5%-of-input feedback, not a dense doc.
**ROOT CAUSES:** (A) data dilution 89%copy/5%feedback/10%edit (training-length-independent, load-bearing); (B) perceiver→weight conditioning attenuation ~13× for feedback (measured). Up-scaling (Sakana lever) DISCONFIRMED — destroys preservation.
**NEXT (recommended, not yet run):** patch/edit-program supervision (pre→post diff, feedback-conditioned) + conditioning-probe re-check. If feedback→weight still flat → architectural (perceiver/head capacity, generic-vs-residual factoring).
**Campaign tasks #4-7 complete.** GPU idle/free. reflections monitor bfr7fsxyf still armed.

### 2026-06-01 06:12 — USER pushback: episodic-memory framing + Q&A-out + semi-Markovian (correct, deliverable was incomplete)
**Data structure (semi-Markovian / stepwise):** ALL 6930 rows = step_index=0, external_single_turn. NOT agent trajectories — single-turn GitHub review→revision pairs. 2420 source_task_ids, 1319 have >1 row (mean 2.86/task) but as PARALLEL review comments on same PR, NOT ordered steps. → the "episode" embedded = ONE (code+feedback→edit) turn; no cross-step trajectory state. **Mismatch: goal = encode coding TRAJECTORIES (engine decompose→plan→code→repair→integrate); data = single-turn review pairs.** = a DATA-SOURCING gap, separate from conditioning/dilution.
**Missing test (user right):** prior probes = teacher-forced EDIT reproduction. Never tested can-we-QUERY-the-episode-out (the actual episodic-memory bet). Built tools/diag_qa_recall.py: recall of episode FACTS not in training output (review FEEDBACK, FILE path) under neutral lead-in, matched/mismatch/zero + free generation. Running on A600 (pid 41644).

### 2026-06-01 06:35 — QA RECALL (A600): episode NOT recoverable (clean negative)
RECALL[feedback]: matched −3.913, mismatch −3.913, zero −4.081 → **m−mm +0.0005**, m−zero +0.168.
RECALL[file]: matched −6.140, mismatch −6.151, zero −6.625 → **m−mm +0.011**, m−zero +0.484.
Free generation (base+matched-adapter, "## Review Feedback\n") for 3 DIFFERENT episodes → near-identical GENERIC boilerplate ("I have reviewed the PR and it looks good... `--no-interactive` flag to the CLI") — hallucinates unrelated content, ignores the embedded episode.
**= no recoverable episodic content.** m−zero (+0.17/+0.48) = generic "code-review mode" boost, NOT episode recall (matched≈mismatch proves it). Direct answer to "can we get Q&A out of the adapter": NO.
Now running USER's goal/diff/last-N-lines recoverability (matched/zero/mismatch); tail = "drives next step" semi-Markov signal.

### 2026-06-01 06:45 — RECOVERABILITY 4-target (A600) = comprehensive episodic-memory NEGATIVE
| target | matched | mismatch | zero | m−mismatch | m−zero |
|---|---|---|---|---|---|
| goal | −3.913 | −3.913 | −4.081 | +0.0005 | +0.168 |
| diff | −2.238 | −2.313 | −3.406 | +0.075 | +1.169 |
| tail (drives next step) | −1.442 | −1.449 | −1.058 | +0.006 | **−0.384** |
| avoid (don't repeat fail, n=14) | −0.319 | −0.344 | −0.503 | +0.026 | +0.185 |
**NONE clear the bet (m−mismatch>0 AND m−zero>0):** goal/tail/avoid m−mismatch = noise; diff +0.075 (code-driven, not feedback). **tail m−zero NEGATIVE (−0.384)** = adapter HURTS recovery of recent state → cannot drive next step. avoid only 14/24 rows have a rejected hunk; corpus has ONE rejected form/row, NO multi-attempt failure history → can't teach/test real don't-repeat-mistakes memory.
**Handoff finalized** (docs/issue49-handoff-2026-06-01.md): §4.7 recoverability scorecard + §5 table + §7 acceptance-test = the 4-target scorecard for the reformulated data/objective. Reviewer adds: train explicit queryable-memory supervision (what request/file/change/pre→post/rejected) w/ hard-neg controls; track generation diversity (mode-collapse to review boilerplate, §4.6).

### 2026-06-01 07:08 — reflections: staged-diagnostic plan for the follow-up (oracle upper bound)
Reviewer (strengthens "merge #50 + scoped follow-up, don't grind delta-coder"):
- **Oracle per-row LoRA as UPPER BOUND**: before more hypernet training, fit a LoRA to a SINGLE episode (per-example optimization) and run the recoverability scorecard. If even an ORACLE LoRA can't pass → target/data ILL-POSED (problem upstream of everything). If oracle passes but hypernet fails → amortization/CONDITIONING problem. Separates 3 questions: can LoRA store it? / can THIS hypernet amortize it? / does our DATA teach it?
- Do NOT train current delta-coder longer on full-revision target (rewards generic boosting, hurts recall).
- **Recommended order:** (1) oracle per-row LoRA on tiny patch+QA episode set; (2) Doc2LoRA/Gemma repro = known-good fact-recall positive control on our hardware (validates the harness CAN detect recall when it exists); (3) three-initialization comparison on same tiny recoverability task; (4) only then scale winning setup to Rune trajectory data.
→ Follow-up issue DoD = §4.7 scorecard; FIRST step = oracle upper-bound + Doc2LoRA control (cheaper & more decisive than jumping to patches).
