# Rune scratchpad — Issue #52: adapter-as-trajectory-memory (staged diagnostic, gated by the recoverability scorecard)

> The #49 chronological log (704 lines) is archived in `instructions/scratchpad_backup.md`;
> the #49 reviewer log (270 lines) in `instructions/reflections_backup.md`. This file is the
> fresh #52 record. #49 conclusions live durably in `docs/issue49-handoff-2026-06-01.md` and
> `docs/issue49-findings-2026-06-01.md`.

---

### 2026-06-01 07:32Z — PLAN: Deliverable 1 = Doc2LoRA positive control + scorecard validation + Rune-episode bridge

**Issue #52 is a research epic** (4-stage diagnostic + parallel data track), DoD = the §4.7
recoverability scorecard (goal/diff/tail/avoid, each m−mismatch>0 AND m−zero>0). Stages 3–4
depend on Stage 1's outcome, so they cannot be specced yet.

**Scope decision (this deliverable): reorder to do the Doc2LoRA reproduction FIRST**, before
the oracle. Issue lists oracle as step 1; we invert because the **control de-risks the probe**.
A Stage-1 oracle *negative* is uninterpretable alone — "oracle can't pass" could mean
target/data ill-posed (the signal we want) OR our probe can't detect episode-specificity even
when it exists. Our harness has only ever been shown to detect the *generic* effect (diff
m−zero ≈ +1.17); it has NEVER been validated to detect episode-*specific* recall (m−mismatch>0),
because nothing has produced genuine recall yet. The Doc2LoRA control fixes that ambiguity up
front, so every downstream negative becomes interpretable. (User-directed reorder.)

**Success bar (user-chosen, largest of three): reproduce + scorecard + tiny Rune-episode bridge.**

**Method / components**
- **Isolation: standalone sibling repo + own uv env** (rejected: vendoring into Rune, or
  demo-only). Clone `SakanaAI/doc-to-lora` → `third_party/doc-to-lora/`, isolated `uv` venv,
  `hf download SakanaAI/doc-to-lora --local-dir trained_d2l --include "*/"` →
  `trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin`. Base = Gemma-2-2b-it (license-
  gated; auth as ET-NoahDolev — accept Google license if download 403s). Keeps the control
  INDEPENDENT of Rune's suspect pipeline and sidesteps the `ctx_to_lora` name collision with
  Rune's module (separate venvs). Gemma-2b fits 23GB GPU + 15GB CPU RAM trivially.
- **Sakana API**: `ModulatedPretrainedModel.from_state_dict(...)` → `model.internalize(doc)` →
  forward/generate → `model.reset()`. This IS the matched/mismatch/zero primitive.
- **CROSS-CUTTING (advisor catch): shared scoring core.** Extract `_span_logprob` + span
  selection — pure `(logits, ids) → mean gold logprob`, torch-only, zero Rune/Gemma deps —
  into ONE module imported by BOTH the Gemma control script AND `tools/diag_recoverability.py`.
  Without this, "validate our probe" silently degrades to "validate *a* methodology": a bug in
  the real Qwen `_span_logprob` (off-by-one in the `t-1` indexing; span aggregation) survives
  untested and we're back to "is our probe broken?" — the exact question this stage kills.
  Pure tensor math runs in both venvs → does not break isolation.

**Experiment measurements (what I will collect)**
1. **Reproduction** — Sakana NIAH recall accuracy on our hardware via `scripts/niah/2-eval.sh`
   (their reported result ≈ near-perfect at 5× context window). FIRST integration step before
   any scoring: confirm I can pull per-token *logits* with the adapter active
   (`base_model(ids).logits` after `internalize()`, NOT just `.generate()` — whole scorecard
   depends on it; README only shows `generate()`).
2. **Scorecard probe (the validation)** — over doc-fact episodes: mean gold logprob of the
   answer-fact span under `internalize(matched)` / `internalize(mismatch)` / `reset()`(zero),
   using the shared scoring core. Report **m−mismatch** and **m−zero**.
3. **Calibration scale** — the *magnitude* of m−mismatch at known-good (~100% NIAH) recall,
   tied to generation accuracy. (This is the real payoff, not the binary pass/fail — see below.)
4. **Bridge** — same probe on the Sakana checkpoint+Gemma with ~8–16 tiny Rune-style code
   episodes (small code doc/patch + QA over goal / file / pre→post diff). Open data question:
   build episodes from scratch vs reformulate existing `external_codereview` rows as patch+QA.

**How I'll interpret them (decision rules, set BEFORE running)**
- **Probe validation PASSES iff** on doc-fact episodes m−mismatch>0 AND m−zero>0 → the metric
  demonstrably detects specificity → the scorecard is trustworthy for downstream stages.
  If it FAILS on a known-good checkpoint → our scorecard metric is the problem; fix the metric
  before trusting ANY Rune negative (this stage would have caught a false-negative pipeline).
- **Calibration (load-bearing, NOT an afterthought):** record where known-good recall lands.
  If real recall ⇒ m−mismatch ≈ +2, then #49's Qwen margins (+0.075 "weak", +0.0005 "noise")
  are clearly nothing. If real recall ⇒ m−mismatch ≈ +0.1, the logprob-margin metric is barely
  sensitive and the WHOLE scorecard interpretation needs rethinking. Either outcome is
  decision-relevant; "+0.075 — is that meaningful?" is unanswerable without this reference.
- **Bridge is DIRECTIONAL-ONLY, asymmetric** (Sakana trained on doc facts; code = OOD):
  - PASS → strong/surprising positive (this architecture binds code-edit facts zero-shot).
  - FAIL → NEARLY UNINFORMATIVE (likely just OOD, not "code target ill-posed").
  The clean ill-posedness test is the **oracle** (direct per-episode optimization, no OOD gap),
  NOT this bridge. Stated so a bridge failure is not over-read later.

**Out of scope (depends on this stage):** Stage 1 oracle; Stage 3 three-init comparison;
Stage 4 scale to real Rune data; parallel real-trajectory mining track. NOTE for the oracle
stage: the Qwen `_functional_lora` 4-bit path needs its OWN numerical-equivalence unit test
(`base_out + delta` == patched forward) — Gemma never exercises it.

**Status:** design approved by user; spec written to
`docs/superpowers/specs/2026-06-01-doc2lora-positive-control-design.md`.

### 2026-06-01 07:35Z — reviewer concurrence (reflections.md), two strengthenings folded in
Reviewer reviewed the plan; largely endorses (calls the shared scoring core "the strongest
part"). Confirmed the monitor contract is FILE-based (no scratchpad write ⇒ no review event —
why this log matters). Two explicit asks, both folded into the spec:
1. **Doc2LoRA-first is a probe-validation PRELUDE, not a substitute for the oracle.** It can
   kill "probe is blind" but produces NO evidence about Rune-target well-posedness — only the
   oracle answers "can a LoRA store this Rune target at all?" Added an explicit scope-boundary
   paragraph (spec §1).
2. **Keep the bridge small and NON-GATING** so it can't blur the oracle diagnostic (spec §7).
Reviewer also: add a tiny tensor-level unit test for next-token indexing + span slicing before
trusting any m−mismatch number (already in spec §8 testing).

### 2026-06-01 07:37Z — reviewer spec-review PASS (two refinements folded in)
Reviewer did a spec review: "reorder explicit + oracle preserved → main ordering concern
resolved." Two remaining items, both now in the spec:
1. **Tighten DoD beyond sign:** known-good pass = clearly-positive, not bare `>0` — multiple
   mismatch adapters/episode, per-episode margins, generation accuracy beside logprob, SE/
   bootstrap CI; calibration = effect SIZE at known-good recall (spec §6).
2. **Operational hygiene:** no HF tokens in scripts/docs/scratchpad/shell history beyond the
   credential store; never commit the `third_party/doc-to-lora/` checkout, gated checkpoints,
   HF cache, or `.venv` (spec new §8a).
Spec is now reviewer-clean. Next: user review gate, then writing-plans.

### 2026-06-01 07:40Z — user approved spec; IMPLEMENTATION via workflow (CPU build phase)
User: "Proceed to implementation using workflow." Launched workflow `wxj8bq2yf`
(issue52-d2l-control-build). Scoping: the gated Gemma download + GPU runs (NIAH, scorecard,
bridge) and the third_party `uv sync` are kept OUT of the autonomous workflow — they need a
live GPU and possibly Gemma-license acceptance, so the orchestrator drives them with the user.
Workflow builds the deterministic/parallelizable CPU artifacts:
- Scaffold (parallel): `tools/scoring_core.py` (pure-torch shared core, lifted from
  diag_recoverability `_span_logprob`/`_diff_logprob`) + toy-tensor tests; `tools/d2l_control/
  episodes.py` (doc-fact set + `build_rune_episodes` reformulating external_codereview→patch+QA)
  + tests; clone+inspect SakanaAI/doc-to-lora → `instructions/doc2lora-api-notes.md` (CRITICAL:
  document the per-token-logits-with-adapter path, not just generate()).
- Wire (parallel): `tools/d2l_control/run_scorecard.py` (probe+bridge, imports shared core via
  sys.path, per-episode margins + bootstrap CI + generation accuracy; bridge = --episodes rune,
  non-gating header) wired to the real API; refactor `diag_recoverability.py` to delegate to the
  shared core (numeric-identical).
- Verify: ruff + pytest unit + mypy; emit the ordered GPU command sequence for me to run next.
Corpus present at /tmp/rune-corpus/external_codereview.val.clean.jsonl. Awaiting completion.

### 2026-06-01 ~08:0xZ — user: "use the GPU to the fullest" + "record everything in MLflow"; took over GPU drive
Pivoted from workflow-only-CPU to driving the GPU runs myself. Stopped workflow `wxj8bq2yf`
(A1 scoring_core + A2 episodes DONE and good — 20 unit tests pass; A3 clone barrier-stuck
because my first `hf download --local-dir third_party/doc-to-lora/trained_d2l` had created the
dir, so A3's `git clone` into a non-empty dir failed). Salvaged A1/A2; doing A3/B1/GPU myself.

**Env facts (decisions + dead-ends):**
- GPU 22.5GB free, CPU 11GB avail, disk 40GB. Gemma-2-2b-it NOT license-gated for ET-NoahDolev.
- Sakana repo = 6.5GB; checkpoints: gemma_demo/checkpoint-80000 (1.26GB, the reported NIAH
  result), qwen_4b_d2l (1.68GB — Qwen-based D2L, closer to Rune, good bridge ref), mistral_7b
  (skip). Downloaded gemma_demo + qwen_4b to third_party/doc-to-lora/trained_d2l (2.8GB).
- **MLflow LIVE at http://localhost:5000** (mlflow 3.12 in rune venv; existing exps incl. 54
  issue49-d2l-final). Will log #52 runs there. Two-venv split → run_scorecard (third_party venv)
  emits JSON; a rune-venv logger pushes params/metrics/artifacts to localhost:5000.
- **Name collision CONFIRMED**: rune venv already has `ctx_to_lora` + torch 2.11/transformers
  **5.8.1**; Sakana pins transformers **4.51.3**. Version skew would break Sakana code under rune
  venv → the spec's isolation call was right. Built a lean third_party/.venv (Sakana pin set:
  torch + transformers==4.51.3 + accelerate/datasets/einops/jaxtyping/peft/torchmetrics/bnb).
- **Checkpoint loads** (base Gemma-2-2b-it; LoRA r=8 on `down_proj`, lora_alpha=45.25). API path
  for teacher-forced logits = `model.forward(ctx_ids=…, ctx_attn_mask=…, n_queries=[1],
  input_ids=full_ids).logits` (matched/mismatch by ctx; zero = base_model(ids)); generation via
  internalize()+generate(). reset() between contexts (apply_lora_to_layers patches via partial).
- **Flash-attn dependency (dead-end chain):** perceiver/ctx-encoder resampler is flash-varlen
  (cu_seq_lens + `unpad_input`); eager attention class exists (idefics2.py:218) but the resampler
  forward (idefics2.py:652) calls unpad_input unconditionally → eager path unfinished. Patched
  the 3 trivial eager hooks (registry uncomment, assert relax, env-gated `D2L_ATTN_IMPL` in
  aggregator — all INERT unless D2L_ATTN_IMPL=eager) but the resampler is irreducibly flash →
  completing eager = risky surgery in a POSITIVE CONTROL. Decision: use the designed flash path.
  No prebuilt flash-attn wheel for torch 2.12/cu13 (PyPI flash-attn has no wheels; cuda13 source
  build slow/risky). → downgrading venv torch to **2.7.0** (mature flash-attn prebuilt wheels)
  then install the abi-matched flash-attn wheel. (in progress)
Smoke script: third_party/doc-to-lora/rune_smoke.py (load + teacher-forced logits + gen recall).

### 2026-06-01 ~08:1xZ — SMOKE PASS + user expands scope to the "why" + code recall
**Env fix chain that worked:** torch 2.7.0+cu126 (abi True, cp312) + flash-attn 2.8.3 prebuilt
wheel (cu12torch2.7cxx11abiTRUE-cp312) + bitsandbytes 0.49.2 + transformers 4.51.3. Designed
flash path (D2L_ATTN_IMPL UNSET → eager patches INERT). Bug I hit+fixed in the logits helper:
forward() only BINDS A/B on top of lora_forward (installed by patch_lora_forward in _init_model);
my reset() removed it → must `reset(); patch_lora_forward()` before the modulated forward; also
forward() needs n_ctx_chunks=[1] (combine_lora). zero = base_model(ids) after reset.

**SMOKE (gemma_demo, magic-number needle '4417'):** matched −0.008, mismatch −7.71, zero −7.14
→ **m−mismatch=+7.70, m−zero=+7.13**; gen → '4417'. CALIBRATION: known-good recall ⇒ m−mismatch
≈ **+7.7 nats**. vs #49 Qwen (goal +0.0005, diff +0.075) → #49 margins are ~0.01–1% of a real
signal = decisively NOTHING. **The scorecard metric works; the probe is NOT broken.**

**USER expanded scope:** explore WHY Sakana recalls and Rune doesn't; isolate which difference
matters; can Sakana recall a CODE snippet / answer Qs about it?

**EXP1 — CODE RECALL (gemma_demo, 2 snippets × 4 facts, rune_code_recall.py):** mean m−mismatch
**+7.12**, m−zero +9.57, gen_acc 0.75 (6/8), frac(m−mis>0)=**1.00**. Recalls constants (7,42),
modulus (9973,101), return strings (ABANDONED,TRUNCATED) by GENERATION; only misses = the two
novel multi-token fn names (garbled in gen) but still m−mis +8.7/+7.2 in logprob. → **YES,
Sakana recalls code & answers questions about it, ZERO-SHOT (trained on prose, not code).**

**EXP2 — ISOLATION: RUNE's own episodes through Sakana gemma_demo (rune_episode_recall.py,
n=12, build_rune_episodes):**
| target | Sakana m−mismatch | Rune #49 | ratio | frac>0 |
| goal | **+2.30** | +0.0005 | ~4600× | 1.00 |
| file | **+1.76** | +0.011 | ~160× | 0.75 |
| diff | **+1.01** | +0.075 | ~13× | 0.75 |
OVERALL +1.69. → **DECISIVE: the bottleneck is NOT architecture / probe / facts-unlearnable /
base model. It is RUNE'S TRAINING RECIPE.** Same perceiver family Rune's hypernet derives from
binds Rune's own goal/file/diff facts as queryable memory; Rune's #49 checkpoint ≈0 on the same.
(Sakana-on-Rune +1.7 < Sakana-on-clean-code +7.1: Rune answer spans are longer/OOD-format, diff
hardest — high cross-episode code overlap, echoing #49 "code-driven not feedback-driven".)

**WHY (diff table; ruled in/out):**
- Objective: Sakana = internalize-doc + QA/NIAH recall (queryable memory) vs Rune = code-review
  edit reproduction (89% copy/10% edit, NO queryable supervision). → PRIME SUSPECT.
- Scale: ~80k steps / large corpus vs Rune 600 steps. → contributor.
- Data shape: doc+explicit questions vs single-turn code+feedback→file. → contributor.
- Architecture (perceiver, down_proj r8 α45): RULED OUT as blocker (Sakana arch binds Rune facts).
- Probe / metric: RULED OUT (detects +7 code, +1.7 Rune).
- Base model: testing via qwen_4b_d2l (base Qwen/Qwen3-4B-Instruct-2507 — SAME family as Rune
  Qwen3.5-9B) → if it also recalls, base fully ruled out. (Qwen base downloaded; run pending.)
**Actionable:** retrain Rune's hypernet with Sakana's queryable-memory objective on patch data —
the architecture is PROVEN capable. Validates+elevates the #49 handoff recommendation.

Reviewer (reflections) reqs being honored: run UNMODIFIED Sakana NIAH (run_eval) as the
reproduction anchor (in progress — dep chain rouge_score→llmlingua→tensorboardX; checkpoint_path
must be the .bin, run_dir = path−2segs, args.yaml fetched); log provenance+versions+patch-diff to
MLflow, no model files; record D2L_ATTN_IMPL state per run (UNSET=flash for all reported runs).

### 2026-06-01 ~08:3xZ — NIAH anchor + qwen base-family control + MLflow + strategy reframe
**NIAH reproduction anchor (unmodified run_eval, gemma_demo):** rougeL.f1=**1.0** (n=40,
ctx_magic_number_512_1024). Matches Sakana's reported near-perfect NIAH under our logged env
→ reviewer's clean-control gate cleared. (dep chain to get run_eval up: rouge_score, llmlingua,
tensorboardX, wandb; checkpoint_path must be the .bin; run_dir=path−2segs; args.yaml fetched.)

**EXP3 — base-family control (qwen_4b_d2l, base Qwen/Qwen3-4B-Instruct-2507, 20k steps):**
code recall mean m−mismatch **+2.60** (m−zero +8.35, gen 0.75, frac 0.88); RUNE episodes
goal +2.24/file +1.60/diff +0.98, OVERALL **+1.60** (≈ gemma +1.69). → **BASE-MODEL FAMILY
RULED OUT.** Qwen family (same as Rune's Qwen3.5-9B) + Sakana recipe binds Rune facts just as
well. Note dose-response: gemma 80k code-recall +7.1 vs qwen-4b 20k +2.6 → recall skill scales
with training steps (the recall skill is the HEAVY part, not free).

**Scorecard so far (all logged to MLflow exp 56 `issue52-d2l-control` @ localhost:5000 with
provenance params + inert patch diff + NIAH csv artifacts; 7 runs):**
| run | code m−mm | rune goal/file/diff m−mm |
| gemma_demo (80k) | +7.12 | +2.30 / +1.76 / +1.01 |
| qwen_4b_d2l (20k) | +2.60 | +2.24 / +1.60 / +0.98 |
| Rune #49 (own ckpt) | — | +0.0005 / +0.011 / +0.075 |
Ruled OUT: probe-blindness, capacity, ill-posed facts, base-model family. LIVE suspects:
training recipe (objective ≫ data-shape ≫ scale) + Rune's specific hypernet implementation.

**USER CONJECTURE (recorded):** "Rune hypernets don't need to be so fine-tuned to embed just
trajectory. Fine-tune LIGHTLY to improve recall potential specifically for diffs, error modes,
code continuation — do even better at recording this than just docs, not how we've been
training." → VERDICT: evidence backs the core. Refinements:
1. Failure is the OBJECTIVE, not the architecture. Recall is general (docs→code→Rune facts),
   base-agnostic; Rune's edit-reproduction objective ACTIVELY destroys recall (#49 copy m−zero
   −0.31, boilerplate collapse). "Not how we've been training" = correct.
2. "Lightly" = DELTA FROM A RECALL-CAPABLE INIT, not light from Rune's current state. Recall
   skill itself is heavy (gemma 80k +7.1 vs qwen 20k +2.6). Warm-start from a recall-capable
   ckpt → light specialization plausible; from Rune's collapsed start → must install recall first.

**USER CHALLENGES (accepted, supersede my earlier framing):**
- **ONE objective = recall the EPISODE.** goal/diff/continuation/avoid-failure are all just
  QUERY FACETS of the single internalized episode. Error-mode is NOT a separate objective/track
  (I was wrong to carve it out). "Don't repeat the mistake" = same episode recall on an episode
  that contains the failed step. The only (trivial, universal) requirement: a facet is recallable
  only if it's IN the episode → use trajectory data whose episodes include the failures. Data
  COVERAGE, not a different mechanism.
- **Base model is a FREE VARIABLE — not bound to Qwen.** Options: (a) adopt Sakana's RELEASED
  checkpoint+base directly (gemma-2-2b / qwen3-4b / mistral-7b in the repo; fastest, weaker
  coders); (b) train Sakana's recipe on a stronger code base; (c) **choose base by joint
  max(coding ability, recall-hypernet availability)**. If a strong code model already has (or can
  get) a Sakana-style recall hypernet checkpoint, switch to it. A doc2lora is base-tied (reads
  that base's activations) so can't transplant weights across sizes, but the RECIPE transfers.

**CORRECTED PLAN:** Give a chosen base a doc2lora recall skill via Sakana's queryable-EPISODE
objective; specialize lightly toward the scorecard facets (diff hardest → needs contrastive hard
negatives preserving local code/altering the trajectory fact; continuation = likely easy win,
untested; avoid-failure just needs failure-bearing episodes). Single objective: recall the episode.

**NEXT TESTS (proposed):** (1) measure Sakana zero-shot CONTINUATION recall (have everything);
(2) light-finetune Sakana ckpt on a tiny episode-recall set → re-score for diff/tail GAIN +
NIAH/code RETENTION (catastrophic-forgetting check) = decisive test of "light specialization
preserves recall." Plus survey best base = max(coding, recall-hypernet availability).

### 2026-06-01 ~08:35Z — reviewer design constraints for the next step (accepted)
- **Decompose "training recipe":** objective / query-supervision / data-format / scale / batch /
  init are still entangled. Design the tiny finetune as an ABLATION vs the zero-shot Sakana
  checkpoint: SAME scorecard, SAME episodes, before/after finetune.
- **HARD RETENTION GATE on the light finetune** (not just a gain gate): must improve diff/tail/
  trajectory facets WHILE preserving NIAH/code recall. Gaining diff by forgetting NIAH = overfit
  a new small task, NOT solving adapter-as-memory. Retention is a first-class pass/fail.
- **Base choice: separate "best coding base" from "available recall hypernet."** Weak coder +
  proven recall hypernet = fastest RESEARCH path; strong coder w/o hypernet = better PRODUCT path
  but must pay the Sakana-style training cost first. Decide which lane before swapping base.
- avoid-failure facet needs episodes that actually contain failed attempts + accepted
  alternatives, else untestable regardless of recall quality (reviewer reaffirms).

### 2026-06-01 ~08:40Z — EXP4: CONTINUATION facet (Sakana gemma_demo, zero-shot)
case0 m−mis +1.95 / m−zero +1.65; case1 +2.00 / +2.37. **SUMMARY mean m−mismatch +1.98, m−zero
+2.01** — vs Rune #49 tail m−zero **−0.38**. Decisive flip: the adapter makes the SPECIFIC code
body ~2 nats more likely than no-adapter and than a different episode. NUANCE: strong LOGPROB
recall but greedy regeneration of the full multi-line body DRIFTS ('Abandoned' lowercased, "#
your logic here") — nails short facts (constants/returns) verbatim, doesn't reconstruct a long
body token-for-token. For "drives next step" the ranking signal is what matters → positive.
Logged MLflow run continuation_recall_gemma (exp 56). So Sakana recalls goal/file/diff/tail
zero-shot, every one ≫ Rune #49. (Reviewer precision: this is goal/diff/TAIL + file — NOT the
original scorecard's `avoid`, which needs failure-bearing episodes; do not claim avoid solved.
Keep ranking-recall vs verbatim-generation separate metrics.)

### 2026-06-01 ~08:45Z — user picks RESEARCH lane → light-finetune ablation (running)
rune_finetune_specialize.py: warm-start gemma_demo, freeze all but hypernet (309M trainable),
AdamW lr=2e-5, 150 steps, CE on answer span (mirrors their CrossEntropyTrainer: generate adapter
from doc → CE on query+answer), reset()+patch_lora_forward() each step (avoid partial stacking).
Train = 120 (q,a) from 40 train.jsonl episodes (goal/file/diff); eval = 12 val.clean episodes
(disjoint). Smoke (2 steps) OK: no OOM, loss ~1.8, retention perfect.
**Methodological guard (the #49 trap):** eval tracks BOTH m−zero AND **m−mismatch** (episode-
specificity, mismatch=another episode's doc). #49's whole lesson: m−zero can rise via GENERIC
boosting while m−mismatch stays flat. GAIN gate = m−MISMATCH not worse (specificity), not m−zero.
RETENTION gate = NIAH + clean-code m−zero kept ≥70% (catches catastrophic forgetting of the
recall skill). Result → /tmp/d2l_ft_result.json; will log to MLflow.

### 2026-06-01 ~08:55Z — EXP5 RESULT: light-finetune ablation (150 steps, lr 2e-5, gemma_demo)
loss 1.80→1.13. (MLflow run light_finetune_ablation_gemma, exp 56.)
| facet | m−zero (lift) | **m−mismatch (specificity)** |
| goal | +2.99→+3.58 (+0.59) | +2.30→+2.27 (−0.03 flat) |
| file | +3.38→+5.85 (+2.47) | +1.76→+2.32 (+0.56 ✓) |
| **diff** | +2.68→+3.40 (+0.72) | +1.01→**+0.76 (−0.25 ✗)** |
| retention NIAH | 7.13→7.10 (ratio **0.996**) | |
| retention code | 6.79→7.05 (ratio **1.039**) | |
**VERDICT (honest, two-sided):**
1. **Retention premise of the conjecture CONFIRMED** — 99.6%/103.9%, no catastrophic forgetting.
   Light specialization from a recall-capable init does NOT destroy the recall skill. ✓
2. **But plain answer-CE REPRODUCES THE #49 TRAP on the facet that matters most:** diff m−zero
   rose +0.72 while diff SPECIFICITY (m−mismatch) DROPPED −0.25 = generic edit-boosting, not
   episode binding. goal specificity flat. Only `file` (short path, easy) gained real specificity.
   Exactly the reviewer's warning: CE on query+answer buys answer-FORMAT familiarity, and diff
   hunks overlap across episodes so CE teaches generic emission. m−zero gains are MISLEADING here.
3. **Implication:** the warm-start+light-finetune *path* is viable, but the OBJECTIVE within it
   must be specificity-aware. Plain CE is insufficient for diff/goal. → Next: CONTRASTIVE
   supervision (matched > hard-negative on edit-local tokens; hard-neg preserves local code but
   alters the trajectory fact) — independently re-derives the #49 handoff recommendation.
GAP (reviewer): didn't capture ||Δhypernet|| this run → can't yet quantify "how light"; add to
the contrastive follow-up. Retention reported as continuous ratios (not the soft 70% floor).

### 2026-06-01 ~09:0xZ — USER CONJECTURE 2 + contrastive control (OOM) + key feedback-swap finding
USER: "Don't embed code diffs (primes base to EMIT diffs); embed goal / what-was-tried /
failure-reasons / last-N-lines instead." VERDICT: largely RIGHT, evidence-backed.
- Contrastive-diff training (CE + hinge[margin−(lp_matched−lp_feedbackswap)], feedback-swap hard
  neg = same code/file, diff feedback) **OOM'd at step 0** (2 grad forwards × Gemma+perceiver >
  22GB GPU). But the BEFORE eval (zero-shot) gave the decisive number:
  **diff: m−zero +2.68, m−mismatch(generic) +1.01, m−mismatch(FEEDBACK-SWAP) +0.174.**
  → Against a proper feedback-swap negative, diff specificity COLLAPSES (+1.01→+0.17). Even
  Sakana's strong recall of the diff is ~code-echo, NOT trajectory-fact binding. Direct evidence
  FOR the conjecture: the diff is a bad memory target (embedding it recalls code, not episode).
  = #49 "code-driven not feedback-driven", now shown on Sakana w/ the right negative.
- CONJECTURE EVALUATION (recorded): (1) diff-as-memory is bad — supported (above + EXP5 diff
  m−mismatch dropped under CE). (2) goal/last-N-lines are good memory targets — goal +2.3,
  continuation +2.0 zero-shot. (3) memory(state)/policy(emit diff) SEPARATION is sound, fixes
  the #49 conflation. CAVEATS: (a) "what-tried/failure-reasons" is DATA-GATED (no failure history
  in corpus — need real engine trajectories); (b) recall ≠ utility — we've shown recallability,
  NOT that recalling state→better next edit (the missing memory→action test); (c) "last-N-lines"
  is still code → must recall-as-context, never train base to EMIT it (re-introduces the prime).
- Running eval-only (rune_facet_negatives.py): goal vs diff m−mismatch under generic AND
  feedback-swap negatives. Hypothesis: goal (feedback-derived) HOLDS under feedback-swap; diff
  (code output) COLLAPSES → clean confirmation. (no training → no OOM.)
NEXT (proposed): memory→next-edit UTILITY test (embed goal+tail, does base generate the right
edit?) + source trajectory data with real failures. Contrastive-salvage rerun needs memory fix
(8-bit Adam + ctx truncation + expandable_segments) if we still want "can diff be salvaged".

### 2026-06-01 ~09:0xZ — EXP6 facet-negatives (eval-only): CLEAN conjecture confirmation
(MLflow run facet_negatives_goal_vs_diff, exp 56.)
| facet | m−mismatch(generic) | m−mismatch(feedback-swap) | retained |
| goal  | +2.30 | **+1.59** | 69% — HOLDS |
| diff  | +1.01 | **+0.17** | 17% — COLLAPSES |
→ When the negative differs ONLY in feedback (code/file/format identical), goal recall holds
(+1.59, ~9× diff's) but diff recall collapses to noise. **goal binds the trajectory fact; diff
is code-echo.** Conjecture confirmed: embed feedback-derived episodic facts (goal, tried,
failures), NOT the diff.
REVIEWER PRECISION (accepted): "diff is a bad target" = bad **MEMORY-SUPERVISION** target under
local-code-preserving hard negatives; the diff REMAINS a valid **downstream action/eval** target
for the memory→edit utility test. And: do NOT spend GPU OOM-salvaging contrastive diff-as-memory
— the better negative already shows it's conceptually misaligned. (So the contrastive-salvage
rerun is DROPPED.)
**STANDING NEXT STEPS:** (1) memory→next-edit UTILITY test (embed goal+tail state → does base
generate/choose the right edit? — the unmeasured product link); (2) source/mine real engine
trajectories WITH failure history (unlocks tried/failure facets, the data-gated part).
Durability loose-ends still open: consolidate probes→run_scorecard.py, refactor
diag_recoverability onto scoring_core, findings doc, PR.

---

## CONCLUSIONS — Issue #52 Deliverable 1 (for reviewer sign-off in reflections before PR)

**One-line:** The adapter-as-episodic-memory bet is achievable with the existing perceiver
architecture; Rune #49 failed because of its TRAINING RECIPE/OBJECTIVE, not the architecture,
the probe, the base model, or any ill-posedness — and the right memory target is the episode's
feedback-derived FACTS (goal/state/failures), not the code diff.

### What was established (all in MLflow exp 56 `issue52-d2l-control`, full provenance; scripts in
`third_party/doc-to-lora/rune_*.py`; shared math in `tools/scoring_core.py`; episodes in
`tools/d2l_control/`):

1. **Positive control reproduced & probe validated.** Unmodified Sakana NIAH eval → rougeL.f1
   **1.0** under our logged env (torch 2.7.0+cu126, flash-attn 2.8.3, transformers 4.51.3,
   Sakana commit baa85db4; eager patches INERT/D2L_ATTN_IMPL unset). Scorecard CALIBRATION: a
   known-good needle gives m−mismatch **+7.7 nats** → #49's Qwen margins (goal +0.0005, diff
   +0.075) are ~0.01–1% of a real signal = noise. The probe is not blind.

2. **Sakana recalls Rune's own episode facts zero-shot** (goal +2.30 / file +1.76 / diff +1.01
   m−mismatch; continuation/tail m−zero **+2.01** vs Rune #49 tail **−0.38**) — recall of
   goal/diff/TAIL (not the original `avoid`, which is untested pending failure-bearing data).

3. **Base-model family RULED OUT.** Sakana qwen_4b_d2l (Qwen3-4B, same family as Rune's
   Qwen3.5-9B) recalls Rune facts ≈ identically (overall +1.60 vs Gemma +1.69). Dose-response:
   gemma 80k code-recall +7.1 vs qwen 20k +2.6 → the recall SKILL scales with training (heavy).

4. **CAUSE of #49 isolated → training recipe/objective.** Ruled out: probe-blindness, capacity,
   ill-posed facts, base-model family. The same perceiver family binds Rune facts when trained
   with Sakana's queryable-recall objective; Rune's full-revision edit-reproduction objective
   does not (and #49 showed it actively HURTS recall: copy m−zero −0.31, boilerplate collapse).

### Conjecture verdicts (experimentally tested):

- **C1 "light-finetune a recall-capable hypernet rather than train trajectory from scratch":**
  SUPPORTED in premise, SHARPENED in practice. Light finetune (150 steps, warm-start gemma_demo)
  PRESERVES recall (NIAH retention 99.6%, code 103.9% — no catastrophic forgetting). BUT plain
  answer-CE re-primes generic emission on the hard facet: diff m−zero +0.72 while diff
  m−mismatch −0.25 (the #49 trap). So: the warm-start path is viable; the OBJECTIVE inside it
  must be specificity-aware, not plain CE. (Gap: ||Δhypernet|| not captured this run.)

- **C2 "don't embed code diffs (primes diff emission); embed goal/tried/failures/last-N-lines":**
  CONFIRMED. Feedback-swap hard negative (same code/file, different feedback) → diff m−mismatch
  collapses +1.01→**+0.17** (code-echo), while goal HOLDS +2.30→**+1.59** (binds the trajectory
  fact). PRECISION (reviewer): diff is a bad MEMORY-SUPERVISION target, but remains a valid
  downstream ACTION/OUTPUT target. Architectural principle: separate MEMORY (recall episodic
  state) from POLICY (base emits the next edit conditioned on recalled state) — fixes the #49
  conflation.

### Open caveats / not-yet-shown:
- **Recall ≠ utility.** Everything measured is recallability (logprob specificity); NOT that
  recalling state improves next-edit generation/pass@1. This is the decisive product link, unmeasured.
- **Failure/tried facets are DATA-GATED.** Current corpus has no failure history (single-turn).
  The most valuable part of C2 (tried/failure recall) needs real engine trajectories with
  tried-and-failed steps.
- "last-N-lines" is still code → recall-as-context only; never train the base to EMIT it.

### Forward plan (proposed; to be the PR's plan):
1. **memory→next-edit utility test** — embed goal+tail state, measure whether base generates/
   ranks the correct edit better than diff-embedded or no-adapter. Closes recall≠utility.
2. **Mine real engine trajectories with failure history** (decompose→plan→code→[diagnose→repair]
   →integrate) → episodes with ordered steps, prior-step queries, failure facts. Unlocks
   tried/failure recall + the `avoid` scorecard facet.
3. **Specificity-aware specialization objective** (contrastive / preference on edit-local tokens
   with constructed facet-paired hard negatives) when training Rune's own memory hypernet —
   plain CE is insufficient (C1). Memory-fix the OOM (8-bit Adam + ctx truncation) IF pursued.
4. **Base-model decision** (separate lanes): fastest-research = warm-start Sakana's released
   checkpoint; best-product = train Sakana's recipe on a strong code base. Base is a free variable.
5. **Durability:** consolidate rune_*.py probes → `tools/d2l_control/run_scorecard.py` (JSON +
   bootstrap CI); refactor `tools/diag_recoverability.py` onto `tools/scoring_core.py`; findings doc.

### Hygiene note for the PR:
Do NOT commit third_party/doc-to-lora checkout, checkpoints, HF cache, or .venv (gitignored).
Committable: tools/scoring_core.py (+test), tools/d2l_control/ (episodes/+test, log_to_mlflow),
the spec, and a findings doc. The probe scripts live under third_party (uncommitted) — decide in
the PR whether to vendor a thin copy under tools/d2l_control/ or document reproduction steps.

**STATUS: reviewer SIGNED OFF (reflections 09:23Z). PR opened.**

### 2026-06-01 ~09:3xZ — PR + orphan branch (per user: codebase-relevant only; experiment → orphan)
User: "only commit things relevant to the codebase; experimentation → orphan branch for posterity,
don't contaminate the working codebase." Done:
- **PR #53** (https://github.com/ElixirTrials/rune/pull/53) base main ← feat/issue52-doc2lora-
  positive-control. Diff = ONLY: .gitignore (ignore third_party/), docs/issue52-findings-2026-06-01.md,
  and the design spec. No experiment code. Detailed description + forward plan in the PR body
  (reviewer-approved causal wording + caveats).
- **Orphan branch `experiment/issue52-doc2lora-positive-control`** (pushed): 15 experiment-only
  files — probes/, scoring_core.py, episodes.py, log_to_mlflow.py, tests, REPRODUCE.md,
  provenance + patch diff. No Rune source. For posterity/reproduction.
- third_party/ now gitignored; Sakana checkout + checkpoints + venv never tracked (verified).
- Note: local `main` carries the spec commit (1a51c380) unpushed/ahead of origin/main; it rides
  into PR #53. (Housekeeping: can reset local main to origin/main after merge — spec is safe on
  the PR branch.)
Reviewer caveats carried into PR: avoid-failure untested w/o failure data; recall ≠ pass@1;
tail recalled as state not verbatim. Next decisive experiment (post-merge): memory→edit utility.

---

## 2026-06-01 — NEW SESSION: "begin work on PR #53" (brainstorming skill invoked)

### Re-orientation (verified against repo, PR, issue #52, reflections)
- Deliverable 1 is DONE + durable (docs on PR #53; experiment code on orphan branch; MLflow
  exp 56). PR #53 is docs-only and reviewer-signed-off (qodo: 0 bugs, 1 requirement gap).
- Both this scratchpad's last line AND the reviewer's Deliverable-1 sign-off
  (reflections, final block) name the SAME next step: **memory→next-edit UTILITY test** —
  "the next decisive experiment is memory-to-edit utility, not more diff-as-memory optimization."
- The PR's forward plan lists it as step 1; it closes the load-bearing open caveat **recall ≠
  utility** (everything so far is recallability/logprob specificity; NOT that recalling state
  improves next-edit generation or pass@1 — the decisive product link, unmeasured).

### Interpretation of the task
"Begin work on PR #53" = begin the NEXT deliverable the PR's plan lays out, designed via
brainstorming → spec → plan. Default reading: design **Deliverable 2 = memory→next-edit
utility test**. Confirming scope with the user before any design work (HARD GATE: no
implementation until a design is approved).

### Prediction (pre-design, to be checked against evidence later)
Embedding goal+tail episodic state (NOT the diff — C2) into the Sakana-recall adapter and asking
the base to generate/rank the correct next edit: I expect a measurable utility lift over
no-adapter, plausibly smaller than the raw recall margins (+2.3 goal) because utility requires
the base to *act on* recalled state, not just assign it logprob. Real risk of "recallable but
not useful," which is exactly the gap this test must resolve. Ranking-based utility (matched
edit vs distractor edits) likely cleaner than free-generation utility (greedy drift seen in EXP4).

### RESOLVED scope (user + advisor)
User: "look at the scratchpad to understand where we left off. This PR has not been worked on
yet. Now we need to build." → The record names the next step twice (scratchpad close + reviewer
sign-off): **memory→next-edit UTILITY test**. Advisor: don't re-ask scope; the utility test IS
the first concrete build AND the gate on the expensive recipe — cannot responsibly build Rune's
memory-training recipe (specificity objective + real-trajectory mining) until recall→utility is
confirmed, else we'd optimize a capability that may not move next-edit/pass@1. So:
**Deliverable 2 = memory→edit utility test, framed as step 1 of the build.**

### Load-bearing design axes (advisor defaults, grounded in the record)
1. **Metric: ranking PRIMARY, generation SECONDARY.** (reviewer: keep separate; EXP4 greedy
   drift) — does memory-conditioning raise the correct next-edit's logprob/rank vs distractors;
   secondary = does it actually generate the right edit.
2. **Memory = goal+tail, NOT diff** (C2): condition on recalled episodic STATE, measure the edit
   (memory/policy separation).
3. **Specificity guard = the #49 trap again: matched vs MISMATCH vs zero**, not just vs zero.
   matched≈mismatch on edit selection ⇒ generic boosting, not utility. Include a diff-embedded
   arm as the C2 contrast (expected to help only via code-echo, if at all).
4. **Recalled facts must be genuinely OUT-OF-PROMPT** (mirror scorecard "episode NOT in prompt")
   — goal/trajectory state absent from prompt, supplied only by adapter, else test is trivial.
5. **Cheapest decisive run: Sakana checkpoint (gemma_demo / qwen_4b_d2l) zero-shot** on existing
   `build_rune_episodes`, NO training. Hard distractors = feedback-swap (same file, different
   trajectory fact → different correct edit).

### Genuine open design fork (single-turn data vs a real "next edit")
Our corpus is single-turn (step_index=0): there is no natural held-out "next step." The utility
task must be CONSTRUCTED. This is the real design question to settle before proposing the spec.

### USER DECISION + CHALLENGE (2026-06-01)
- Phasing: **cheap gate → synthetic probe → mine trajectories** (all three, in sequence).
- Challenge: "our adapters are episodic memory, SINGLE STEPS — are you sure our premined data
  can't work?" → **User is right; I over-claimed.** The adapter is episodic memory of ONE step.
  For a single-step episode the valid utility test is: withhold the GOAL/request from the prompt,
  keep pre_code visible, internalize the episode's facts → does recalled goal let the base emit
  the request-appropriate edit better than mismatch/zero? Premined single-turn data WORKS for
  this. PRECISION: for single-step rows the genuinely out-of-prompt memory target is the GOAL
  (+ feedback facts), NOT the tail — tail = current code is already visible in the prompt, so
  internalizing it adds nothing (would violate the out-of-prompt discipline). The MULTI-STEP
  facets (prior-step tail → next edit, avoid-what-failed) are exactly what premined single-turn
  data CANNOT test → that is what the synthetic probe + mined trajectories are for. The 3 phases
  map cleanly onto increasing facet coverage, not onto "premined is useless."

### REVIEWER input — Deliverable 2 Utility-Test Design (reflections, 09:44Z) — ACCEPTED
1. Ranking PRIMARY; free-gen = secondary realism check (drift already seen), not first pass/fail.
2. **Construct-validity arms (5):** in-context UPPER BOUND (goal+tail in prompt) · zero/no-adapter
   LOWER BOUND · matched-adapter · mismatch-adapter · feedback-swap hard-negative (same file/code).
3. **Prompt contract exact:** goal/tail live ONLY in the adapter — do not leak via filenames,
   comments, patch text, or distractor labels. pre_code may be in prompt; the requested change /
   prior trajectory fact being tested may NOT.
4. **Distractors hard but VALID:** same-file feedback-swap edits > random (random may be rejected
   on syntax/local-code, inflating "utility"). Report per-negative-type — don't average over easy.
5. **"In-context solves it" SANITY GATE (load-bearing):** before reading any adapter failure as
   evidence against memory, confirm base+prompted goal+tail CAN rank the correct edit. If the
   upper bound fails, the constructed task is ill-posed as an edit-utility benchmark — NOT
   evidence against adapter memory. (This gate is precisely what makes premined data trustworthy:
   it proves the hidden goal genuinely determines the edit.)

### USER REFINEMENT 2 (2026-06-01): "what was tried" is EMBEDDABLE in a single-step episode
User: "you missed the 'what we already tried' part that needs to be embedded for the adapter to
be single-step stateless in the true sense of an episode. Each episode can embed the latest
attempts but that shouldn't impact training." → I over-partitioned (claimed avoid/tried needs
phase 2/3). CORRECTION:
- The adapter is "stateless/single-step" = each adapter is one internalized episode; no cross-
  adapter state. But that ONE episode can be RICH: it embeds its own latest attempts ("what was
  tried"). Embedding tried-attempts is EPISODE CONTENT / data coverage — the internalize→recall
  OBJECTIVE is unchanged ("shouldn't impact training" = correct; same mechanism, richer episode).
- Premined `external_codereview` rows ARE already tried-and-corrected units: submitted/pre code
  = the ATTEMPT, review feedback = WHY it was rejected, revised/post code = the ACCEPTED form.
  → the original #52 `avoid` facet ("prefer accepted over rejected/failed form") IS reachable on
  PREMINED data, by embedding the attempt+critique into the episode the adapter internalizes.
- **Prompt-contract consequence:** to test avoid-as-MEMORY, the rejected form + critique must be
  WITHHELD into the adapter (not shown in prompt). Candidate set = {accepted form, rejected form};
  matched-memory (encodes "feedback said X was wrong") should rank accepted > rejected better than
  zero/mismatch. (If the rejected attempt is the visible pre_code in the prompt, its avoid signal
  is in-context, not memory — so the avoid arm needs a different prompt layout than the goal arm.)
- **Residual gap genuinely needing phase 2/3:** MULTIPLE sequential attempts / a real exploration
  history. One review pair = one tried attempt; real mined trajectories = many ordered attempts.

### REVISED phase coverage
| phase | data | facets reachable | withheld-into-adapter |
| 1 cheap gate | premined external_codereview (embeds 1 tried attempt) | goal→edit AND avoid (accepted>rejected) | goal+feedback; (for avoid) the rejected form+critique |
| 2 synthetic | hand-built 2-3 step | prior-state→next edit; MULTI-attempt avoid | prior trajectory fact(s) |
| 3 mined | real engine runs | same, production-faithful, long histories | real ordered prior steps+failures |
Per-facet readouts (never average goal over avoid); in-context sanity gate applied PER FACET.

### REVIEWER input — Single-Step Avoid Coverage (reflections, 10:18Z) — ACCEPTED, endorses correction
- Confirms the correction: pre_code = rejected attempt, feedback = rejection reason, post_code =
  accepted → one-step `avoid` testable on premined data before mined multi-step.
- **Avoid prompt contract:** if rejected pre_code is visible as current code, the model compares
  candidates from IN-PROMPT evidence, not memory. For avoid-as-MEMORY: internalize the rejected
  form + critique; the scoring prompt exposes only a NEUTRAL task/candidate-comparison scaffold to
  rank accepted vs rejected.
- **CRITICAL metric (difference-in-differences):** accepted code is intrinsically more likely than
  rejected code, so the signal is NOT `accepted > rejected` by itself. It is the IMPROVEMENT in
  the (accepted − rejected) preference under MATCHED memory vs ZERO and vs a MISMATCH episode with
  similar local code/edit type. Metric = Δ_matched(acc−rej) − Δ_zero(acc−rej), and likewise vs
  mismatch. Guards the intrinsic-quality confound. (Update my table's "ranks accepted>rejected".)
- Keep ONE-attempt avoid separate from MULTI-attempt avoid: passing premined review-pair avoid
  shows recall of one rejected form+critique; does NOT prove ordered exploration history across
  several failed repairs (that stays phase 2/3).

### USER: Phase-1 scope APPROVED + scope EXPANDED to a trained checkpoint by EOD (2026-06-01)
"Yes" to the revised Phase-1 scope. Plus FOUR new directives:
1. **Do NOT embed failed CODE, even as illustration of failure** — our own research (C2: embedding
   diff/code primes the base to EMIT it; feedback-swap collapse; #49 trap) says verbatim failed
   code in the adapter has NEGATIVE effects. So the `avoid`/failure MEMORY TARGET = the failure
   FACTS / critique (feedback-derived, abstracted "what was wrong / what to avoid"), NOT the
   rejected code string. (Sharpens the avoid arm: internalize the CRITIQUE, never the failed code.)
2. **Adapt the ADAPTER TEMPLATES** (engine Jinja2 context→hypernetwork) to match whatever training
   paradigm we settle on — the episode text the hypernet internalizes must carry goal/state/
   failure-FACTS in the trained format. Concrete code deliverable in `src/rune/`.
3. **GOAL: a trained, HPO-optimized checkpoint by EOD (today 2026-06-01).**
4. **Then run the pass@1 bench to confirm it now works.**

### Feasibility tension to resolve BEFORE committing (do not gloss)
- **pass@1 runs on RUNE's engine + Rune's base (Qwen3.5-9B) + Rune's hypernet** — NOT the Sakana
  control (gemma-2b/qwen-4b, different base/codepath). So "run pass@1 to see it works" REQUIRES
  training RUNE's OWN hypernet with the new paradigm. The Sakana utility gate is a fast parallel
  de-risk, not the thing that produces a pass@1 number.
- **Recall skill is HEAVY** (C1/dose-response: gemma 80k → +7.1, qwen-4b 20k → +2.6). Rune's
  existing hypernet was trained on the BAD #49 objective (collapsed recall). C1 said light-FT
  from a recall-capable init preserves recall, but "from Rune's collapsed start → must install
  recall FIRST." Installing recall from scratch by EOD may be infeasible. → Need to clarify what
  "HPO-optimized checkpoint" means: fresh Optuna search (likely can't finish EOD) vs train ONE
  checkpoint on already-HPO-tuned params / warm-start. And what we warm-start FROM.
- Tension with the gate: reviewer holds recall≠utility as the decisive unmeasured link; user
  wants to push to a checkpoint+pass@1 today. Given strong prior evidence (C1/C2) + deadline,
  likely run the cheap utility gate FAST in parallel rather than as a hard blocker.

### ASSET DISCOVERY (2026-06-01 ~10:25Z) — the corrected paradigm is LARGELY ALREADY CODED
- **`src/rune/training/hypernet_distill.py`** (41KB): distillation trainer with `contrastive`,
  `contrastive_weight`, `contrastive_margin` config; imports `make_hard_negative`; hinge
  `margin−(lp_matched−lp_neg)` on the edit-local span (lines 219–376). = the specificity-aware
  objective we concluded we need, ALREADY WIRED. EOD = configure+run, not build.
- **`src/rune/training/contrastive.py`**: feedback-swap hard-negative (replaces only the
  `## Review Feedback` body with another row's real feedback → guards "has-feedback-vs-not"),
  `edit_local_mask` (difflib pre_code vs answer). SHARED by training + probes.
- `diff_loss.py` (50KB, diff-masked KL), `collapse_metrics.py`, `gate.py` (success gate),
  `d2l_train.py`, `orchestrator.py` (HPO tunes lr/warmup_ratio/lora_rank/neftune_alpha).
- Corpus LOCAL: `/tmp/rune-corpus/external_codereview.{train,val,test}[.clean].jsonl` (+ unrolled).
- Checkpoint loader resolves `s3://` → `~/.cache/rune/checkpoints` (warm-start path works).
- **`instructions/adapter-as-memory-report.md` already found the working architecture:** TASK
  SPEC IN PROMPT, CODE CONTEXT IN ADAPTER (task_only prompt + code_template trajectory; ~0.75
  scaling). "Structural prompt drives spec compliance, not the adapter." → directly informs the
  template change (directive #2).

### REVIEWER input (10:22Z) — EOD MVC contract (ACCEPTED)
- Endorses "embed critique/failure-FACTS, never failed code verbatim" (C2 code-echo prime).
- EOD goal more ambitious than validated evidence: pass@1 needs Rune engine+base+hypernet, not
  Sakana stack. → define a MINIMUM VIABLE CHECKPOINT precisely: warm-start source, FIXED recipe,
  max steps, selection metric, stop criteria. "HPO-optimized by EOD" must mean a SMALL sweep over
  known-safe knobs, NOT a fresh open-ended Optuna campaign.
- Keep cheap utility gate in PARALLEL, not silently optional. If gate negative, a bigger Rune
  checkpoint may optimize recall w/o action → label the run EXPLORATORY/PRODUCT-RISKY if proceeding.
- Adapter-template change = part of the EXPERIMENT CONTRACT: LOG the exact episode serialization
  used for train AND inference, else a pass@1 failure could be a template mismatch, not recipe.

### DEFAULTS I'm proceeding with (advisor + reviewer; user may redirect)
- **Warm-start source = `s3://…/hypernet_hpo`** (recall-capable-ish HPO warm-start), NOT Rune's
  collapsed #49 init. If unavailable, fall back to the best available + note recall-install risk.
- **"HPO-optimized" = reuse known-good HPO params + a SMALL guarded sweep** (e.g. contrastive_weight
  / scaling / lr around known-safe), NOT a fresh 50-trial Optuna (won't finish EOD).
- **EOD deliverable = a checkpoint trained on the CORRECTED recipe + an HONEST pass@1 number
  (possibly partial), labeled exploratory if the utility gate isn't green.** Not a guaranteed win.
- pass@1 "works" = define the bench config + current baseline FIRST (memory says 1.0 post-#50 on
  some config) so success is distinguishable from noise / regression.

### REVIEWER input (10:27Z) — Existing machinery / EOD MVC (ACCEPTED, folded into plan)
- Contrastive obj is on EDIT-LOCAL spans → frame it as POLICY/ACTION supervision CONDITIONED ON
  memory facts (adapter internalizes goal/critique/state; loss asks whether those memories improve
  the edit choice). Do NOT describe as "diff memory."
- **Serialization snapshot/hash on EVERY checkpoint + MLflow run:** train template name/version,
  inference template name/version, a sampled rendered episode. Template mismatch ⇒ uninterpretable
  pass@1.
- **Before pass@1, run the success gate (matched vs mismatch) on the SAME checkpoint.** pass@1
  movement w/o matched-over-mismatch movement = generic boosting again.
- Baseline first: SAME tasks/config/checkpoint-loading path for base, previous-best, new ckpt.

### TRANSITION (user: "proceed to planning and implementation and testing with workflows;
### step by step; record to scratchpad + MLflow"): design converged + spec written
Spec: `docs/superpowers/specs/2026-06-01-memory-to-edit-utility-and-corrected-recipe-design.md`.
Proceeding brainstorming → writing-plans (the disciplined terminal transition), then
workflow-driven implementation + testing, long-pole training started ASAP in background.
Defaults I'm proceeding under (stated for user override): warm-start = s3 hypernet_hpo;
"HPO" = reuse known params + small guarded sweep (contrastive_weight/scaling), not fresh Optuna;
EOD = corrected-recipe checkpoint + honest (possibly partial) pass@1, exploratory-labeled if the
utility gate isn't green.

### USER RE-ORIENT (2026-06-01 ~10:35Z): training is the goal; pass@1 is the FAST check
"Get the training right but don't lose sight of the ultimate goals. Re-orient back to solving the
training issues. But get faster to checking that they make a pass@1 difference."
INTERPRETATION:
- The Sakana zero-shot utility gate (old Phase 1) was drifting into a DIAGNOSTIC DETOUR. DEMOTE
  it: it is NOT on the critical path. **pass@1 IS the utility test we care about.**
- Critical path = SOLVE RUNE'S OWN TRAINING (corrected recipe: warm-start + contrastive
  specificity objective + right serialization/scaling) → CHECK pass@1 FAST → iterate.
- "Faster to pass@1" levers (the loop must be tight, training-on-Qwen is the long pole):
  1. **EVAL-TIME levers need NO retrain:** adapter_scaling + prompt architecture (task-in-prompt /
     context-in-adapter). adapter-as-memory-report: scaling is decisive (0.75 task_only works;
     ≥0.49 spec-divergent); #50 memory: pass@1 0→1.0 was purely an 8× over-scaling fix. → Train
     ONCE, then sweep scaling/prompt vs pass@1 cheaply. Likely the biggest fast win.
  2. **Small adapter-SENSITIVE task subset** for the quick pass@1 signal during iteration; full
     bench only at the end.
  3. **Short training runs** for a directional pass@1 read before committing to a long run.
- matched-vs-mismatch scorecard = CHEAP PROXY to diagnose WHY pass@1 moved/didn't (generic boost
  vs real memory), not a replacement for the pass@1 check.
RESTRUCTURE: collapse plan to a tight TRAIN→pass@1 loop; Sakana utility gate → optional offline
diagnostic, off the critical path.

### USER (2026-06-01 ~10:45Z): hypernet_hpo warm-start is wrong; research + DECIDE between
(A) fine-tune deltacoder using a fine-tuned Sakana-Qwen as teacher/oracle, OR (B) move to Qwen-4B
to use Sakana directly as a base. Then clarified: "doesn't have to be compatible with Rune's
EXISTING hypernet — we're training a NEW one."

### RESEARCH (3 parallel Explore agents + direct verification + advisor)
- **Base swap 9B→4B is a CLEAN config change** on Rune's engine: model_id flows through
  AutoModel/AutoTokenizer; xgrammar is vocab-agnostic; LoRA application derives shapes from the
  loaded base; layer path `base_model.model.model.layers.{i}` works for any Qwen. Only coupling
  was the size-specific hypernet checkpoint — moot since we train a new one.
- **CORRECTION of agent misread (advisor caught it):** Rune does NOT reimplement Sakana — it
  IMPORTS `from ctx_to_lora.modeling.hypernet import HyperLoRA` (hypernetwork.py:281) and loads
  the FULL state_dict incl. the PERCEIVER: `HyperLoRA(hc); load_state_dict(weights, strict=False)`
  (:295-300). The agent's "perceiver discarded" was WRONG (consistent w/ D1 NIAH=1.0). **Recall
  lives in the perceiver; it DOES load.**
- **Feature path MATCHES:** Rune feeds base-model hidden states (`extract_activations_with_model`
  → `hypernet.generate_weights(features, attn_mask, None)`, hypernet_distill.py:770-777) — the
  same mechanism qwen_4b_d2l was trained with. layer_indices read FROM the checkpoint config
  (:184), so a qwen_4b_d2l warm-start auto-uses its trained layer indices. Load path forces EAGER
  attention (:289-294) → no flash-attn dep under Rune's venv.
- **deltacoder = `danielcherubini/Qwen3.5-DeltaCoder-9B`** (external 9B coder, warm-start for the
  now-REMOVED oracle stage). The cross-base teacher route (Sakana-4B → 9B student) = ~4-7 days of
  tokenizer/dim-bridge work; current pipeline does SELF-distillation (teacher = frozen base
  in-context), NOT separate-teacher KD. → Option A is slow + novel.

### DECISION: Option B, flavor-2 — move base to Qwen3-4B-Instruct-2507; train a NEW hypernet
### (the same ctx_to_lora HyperLoRA class) WARM-STARTED from Sakana qwen_4b_d2l; native Rune engine.
WHY: (1) recall ALREADY EXISTS in qwen_4b_d2l (proven to recall Rune facts +1.60 in D1) and its
perceiver LOADS into Rune's loader → solves the "recall is heavy to install" problem that killed
hypernet_hpo, by warm-starting instead of installing. (2) Base swap is clean → NATIVE Rune-engine
pass@1, no Sakana-stack bridge / venv-skew detour. (3) Option A = 4-7 days cross-base KD, expensive
9B, unproven 4B→9B recall transfer. TRADEOFF (stated): 4B is a weaker coder than 9B → lower pass@1
ceiling. This is the RESEARCH lane (prove recall→utility fast at 4B); REVERSIBLE — scale the recipe
to a strong coder once proven. Open impl risk to verify FIRST (cheap): qwen_4b_d2l loads cleanly
under Rune's venv (transformers 5.8.1, eager) + the warm-started perceiver actually recalls through
Rune's extract_activations path (re-measure m−mismatch via Rune's stack, not just Sakana's).

### REVIEWER input (11:15Z) — Base/Warm-Start Decision — full ENDORSEMENT of B flavor-2
- Precise wording: this is a move to a **recall-COMPATIBLE Qwen base**, NOT the best coding model.
  Trade lower coding ceiling for a clean fast recall→utility test. If it works, product lane =
  train the same Sakana recall recipe on a STRONGER coding base.
- Two cheap gates before long training (= my Phase 0): (1) qwen_4b_d2l loads through Rune's
  HyperLoRA with its OWN layer indices/target shapes; (2) warm-started model shows positive
  matched-vs-mismatch under Rune's activation-extraction/generation path (not only Sakana's repo).
- **pass@1 comparisons WITHIN-BASE first:** Qwen3-4B no-adapter vs Qwen3-4B + warm-start/adapted
  hypernet on the SAME tasks. Do NOT frame lower absolute pass@1 vs Qwen3.5-9B as a failure of the
  memory approach. (Add to the plan's bench framing.)
- Option A stays a LATER product path, not today's critical path.
Status: decision presented to user; HOLDING for user's "consider" before starting Phase 0.

### REVIEWER input (11:18Z) — Qwen3-4B coding adequacy + compatible-base survey (confirms B)
- Qwen3-4B-Instruct-2507 coding-capable enough for the research lane: LiveCodeBench v6 35.1,
  MultiPL-E 76.8, Aider-Polyglot 12.9. Lower ceiling than 9B/DeltaCoder but adequate to measure
  adapter LIFT. Trainable (LoRA/QLoRA works; Sakana trained qwen_4b_d2l on it).
- Among Sakana-released checkpoints, Qwen3-4B is the best fit (Gemma-2B weaker coder; Mistral-7B
  more engine drift, no clear gain). No "best-coder + released Sakana recall hypernet" exists →
  stronger coders (e.g. Qwen2.5-Coder-7B, HumanEval/MBPP high-80s/low-80s) would need the recall
  hypernet trained first = the PRODUCT lane, later.
- **NEW practical gate (add to Phase 0, FIRST):** base-only Qwen3-4B pass@1 SMOKE on the frozen
  adapter-sensitive subset. If base-only can't solve any task / can't follow the engine prompt →
  the research lane needs a simpler utility bench or a stronger base. Nonzero pass@1 → adequate to
  measure adapter lift. (Cheapest possible go/no-go before committing GPU to training.)
- Product lane if within-base utility is positive: train Sakana-style recall recipe on
  Qwen2.5-Coder-7B (or newer Qwen coder).

### USER (2026-06-01 ~11:20Z) two questions: (1) weak coder → less research impact? is there a
### Qwen-4B coding fine-tune, or merge Sakana recall + a coding LoRA? (2) should we fine-tune the
### Sakana checkpoint to recall coding state (goal/tried/where/code)? — Research + think.

### HF RESEARCH (hub_repo_search)
- **No dense Qwen3-4B coder on the Instruct-2507 backbone.** Qwen3-Coder ships only as 30B-A3B
  (MoE, qwen3_moe), 480B-A35B, and Qwen3-Coder-Next (qwen3_next) — ALL different base architectures
  from Sakana's dense qwen_4b_d2l → NOT recall-compatible (perceiver is base-tied).
- Community 4B "coders" (rahul7star Qwen3-4B-Thinking-2509-Genius-Coder; Jackrong Qwen3.5-4B-Python-
  Coder) are on DIFFERENT backbones (Thinking-2509 / Qwen3.5-4B) → also not recall-compatible.
- **No coding LoRA on Qwen3-4B-Instruct-2507** found (search empty). So the merge idea has NO
  off-the-shelf coding-LoRA component → we'd have to TRAIN one ourselves (coding dataset + time).

### KEY CONCEPTUAL CLARIFICATION — TWO ORTHOGONAL AXES (do not conflate)
- AXIS 1 = base CODING ability (writing code). Set by base weights. Recall does NOT change it.
- AXIS 2 = hypernet RECALL of episodic coding state. Set by fine-tuning the hypernet (warm-start
  qwen_4b_d2l + contrastive specialize). Does NOT change the base's coding skill.
Fine-tuning the Sakana checkpoint improves AXIS 2 only; it cannot turn a weak base into a strong coder.

### ANSWER to Q2 (fine-tune Sakana to recall coding state): YES — that IS Deliverable 2's core move
### and the single highest-leverage step. HOW (from our own findings, non-negotiable):
- Specificity-aware CONTRASTIVE objective, NOT plain CE (EXP5: plain CE re-creates the #49 trap on
  diff — m−zero ↑ while m−mismatch ↓). The machinery exists (hypernet_distill contrastive=true).
- Embed the FACTS (goal / where-we-are=current state+tail / what-was-tried=critique), NOT the diff
  or failed code verbatim (C2; user's earlier directive — code primes emission).
- "what we tried / failures" facet is DATA-GATED: premined review pairs give ONE tried attempt
  (pre_code+feedback); ordered multi-attempt history needs mined trajectories (later).

### ANSWER to Q1 (weak coder → less impact?): partially right for PRODUCT, wrong for the RESEARCH Q
- For proving recall→utility, HEADROOM matters more than peak ability. A near-ceiling coder leaves
  no room for memory to help (pass@1 saturates); a MID-tier coder (Instruct-2507: LiveCodeBench 35,
  MultiPL-E 77) has room for recalled state to lift edits. The only danger is a FLOOR (base too weak
  to follow the engine) — handled by the base-only pass@1 smoke gate.
- Stronger recall-compatible base: none exists off-the-shelf. Merge (idea 2) = mechanically a
  base-side coding-LoRA merged into W → shifts activations the perceiver reads → recall OOD;
  partially self-heals if we re-specialize the perceiver on the merged base, BUT (a) no coding LoRA
  exists → must train one (slow), (b) adds an uncontrolled variable to a test we want clean.
### RECOMMENDATION: fine-tune Sakana for recall on Instruct-2507 NOW (clean within-base test). Do
### NOT chase a stronger base / merge yet. If within-base recall→utility is POSITIVE, THEN product
### lane = train the recall recipe on a real coder (Qwen3-Coder-30B-A3B or Qwen2.5-Coder-7B) — which
### pays the heavy recall-install cost deliberately, for the product, once the bet is de-risked.

### REVIEWER input (11:24Z) — Cloud Training vs Local Runtime (sharpens the two-lane plan)
- **Train HW ≠ deploy HW.** Product constraint = cheap LOCAL INFERENCE (quantized coder + small
  generated LoRA), NOT cheap local training. Fine to pay a one-time CLOUD training cost for the
  recall hypernet. → 4B is NOT the product; it's the research proof.
- Product-base selection optimizes LOCAL-INFERENCE pass@1/latency after quantization (likely a
  7B-class Qwen coder in Q4/Q5), and trains its Sakana-style recall hypernet on cloud GPUs.
- Independently warns AGAINST the merge (matches my rec): merging a coding LoRA shifts perceiver
  activations → recall OOD; either use the UNMERGED 4B research lane OR deliberately train recall
  on the final coding base.
- Decision rule: prove within-base recall→utility on Qwen3-4B NOW; if positive, budget a CLOUD
  PILOT for a 7B-coder recall hypernet (2k–5k steps to measure step time / retention / early
  m−mismatch) BEFORE a full 20k–80k Sakana-scale run.
STATUS: plan settled + quadruple-confirmed (research + advisor + reviewer×4). HOLDING for user "go".

### USER: "proceed with workflow" (2026-06-01 ~11:25Z). RECON before orchestration:
- GPU 23GB free / 0 used; CPU RAM 11GB avail (tiny — CLAUDE.md applies; base loads to GPU).
- Bench = **MBPP** (`benchmarks/mbpp_tasks.json` + `mbpp_validation_tasks.json`) — real coding pass@1.
- HF cache HAS both Qwen3-4B-Instruct-2507 and Qwen3.5-9B.
- qwen_4b_d2l warm-start = `third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`.
- Distill entry = `run_hypernet_distillation(config)` (hypernet_distill.py:86); warm-starts from
  cfg.checkpoint_path (:179).
- **ModelWrapper.from_config REQUIRES checkpoint_path (raises if empty) + forces
  attn_implementation="flash_attention_2".** ⇒ (a) "base-only" pass@1 = load hypernet but
  adapter_scaling=0; (b) flash-attn availability in Rune's venv is a Phase-0 BLOCKER RISK to check
  before GPU spend. Gate-1 (load) is a prerequisite for gate-0/gate-2.
- PEFT scaling already correct (alpha/r); adapter_scaling is the runtime knob (RUNE_ADAPTER_SCALING).
LAUNCHING Phase-0 workflow: parallel CPU prep → SEQUENTIAL gated GPU checks (env/load → base pass@1
→ recall), short-circuit on fail, structured verdicts for review BEFORE training. No installs
(CLAUDE.md), no unilateral core-code edits — blockers surfaced for review.
Workflow launched: wwn13ocw3 (run wf_29be5ae3-f47).

### REVIEWER input (11:30Z) — Phase-0 gate framing (consistent w/ launched workflow; fold into interp)
- Ordering right: env/load BEFORE base pass@1 BEFORE recall. A flash-attn/checkpoint-load failure
  is a STACK-COMPAT blocker, NOT a scientific result about Qwen3-4B or adapter memory.
- **Label base-only precisely: "adapter-DISABLED baseline via scaling=0", NOT "no-adapter".** And
  VERIFY scaling=0 truly zeroes the adapter's effect on logits (else the baseline is contaminated).
  → apply when interpreting gate0; spot-check in gate2 (zero arm).
- base-only pass@1 interpreted NARROWLY: tests whether Qwen3-4B follows the Rune engine/prompt on
  the frozen MBPP subset. Failure due to wrapper/flash-attn/template/plumbing ≠ "base can't code."
- recall gate MUST go through Rune's exact extract-activations + loader with qwen_4b_d2l layer
  indices — Sakana-repo positive is already known; the NEW evidence needed is native-Rune recall.

### RESUME after 2 instance shutdowns (2026-06-01, later). State recovered from disk:
- Phase-0 workflow (wwn13ocw3) is DEAD; NOT resuming it (journal won't survive shutdowns). Running
  gates directly under run_guarded.sh, reviewing each before the next.
- PREP COMPLETE + VERIFIED (durable on disk): serialization test 3-passed; config
  configs/issue52_recipe_mvc_4b.yaml validates (contrastive=True, max_steps=300) — agent correctly
  found lora_rank/alpha are INERT on warm-start (checkpoint=r8/α45.25/36 layers/down_proj) and the
  loop reads grad_accum_steps (both set=8); benchmarks/mbpp_phase0_iter.json loads 10 tasks (via
  Path); tools/_bench_entry.py (type=Path, sets model_id/checkpoint_path/adapter_scaling overrides)
  + tools/_gate_load.py both sound.
- **flash_attn 2.8.3 imports in Rune's MAIN venv** → the env-gate's main blocker risk (wrapper forces
  flash_attention_2 on base) is cleared at import; actual base load tested by the gate now running.
- GPU free (0 MiB), RAM ~12GB avail.
- Gate sequence (direct Bash, gated): [running] env+load (/tmp/gate_load.log, bg bbcvh1uo2) →
  base-only pass@1 scaling=0 (gate0) → recall via Rune path (gate2, _gate_recall.py still TO-WRITE).
- Reflections monitor re-armed (task btgefmqom).

### GATE 1 (env+load): PASS ✅ (/tmp/gate_load.log)
load_ok=true; Qwen3-4B-Instruct-2507 + qwen_4b_d2l loaded through Rune's ModelWrapper.from_config;
layer_indices=36 (0-35); target_modules=[down_proj]; r=8 α=45 (≈45.25); scaler_B present absmean
0.0569 (NON-collapsed, #50 check). → Sakana qwen_4b_d2l loads into Rune's HyperLoRA with its own
geometry; flash-attn works for the base. Architecture-compat prediction CONFIRMED in Rune's stack.

### GATE 0 (base-only pass@1, adapter_scaling=0): RUNNING (bg bfzpysw6s, /tmp/gate0_base.log)
10-task MBPP subset; "adapter-DISABLED via scaling=0" (reviewer wording). Awaiting.

### GATE 2 (recall via RUNE path): REUSE tools/diag_recoverability.py — it IS the gate.
It uses load_hypernetwork + _generate_lora_dict (extract_activations→generate_weights) +
_functional_lora (Rune's hot-swap) → scores goal/diff/tail/avoid matched/mismatch/zero. No new
script. Command (run AFTER gate0, sequential — GPU contention):
  tools/run_guarded.sh /tmp/gate2_recall.log tools/diag_recoverability.py \
    --ckpt third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
    --model-id Qwen/Qwen3-4B-Instruct-2507 --val /tmp/rune-corpus/external_codereview.val.clean.jsonl \
    --n 12 --scaling 0.5
PASS = goal m-mismatch clearly >0 through Rune's path (D1 had +2.30 via Sakana internalize; even
materially >0 and >> #49's +0.0005 = transfer confirmed). scaling default 0.5 (may sweep 0.3/0.5/0.75).
NOTE scoring_core.py / episodes.py do NOT exist in working tree (orphan-branch only); diag_recoverability
has its own _span_logprob — fine, it's the validated Rune-path harness.

### GATE 0 (base pass@1) = NOT a coding result — PLUMBING CRASH (systematic-debugging, root cause):
pass@1=0.0 but every step raised: ValueError "PEFT rank 8 != adapter+bias rank 16" at
merge_head_bias_rank (hypernetwork.py:368) via generate_adapter (wrapper.py:125). Reviewer's "narrow
interpretation" case: plumbing, NOT "base can't code".
ROOT CAUSE (verified): qwen_4b_d2l has **use_bias=True** (r=8, alpha=45.25, head-bias on down_proj).
- ENGINE path (ModelWrapper.from_config → PEFT get_peft_model → generate_adapter_weights): when
  use_bias, combine_lora concatenates the head bias as extra rank slices → rank 16; but from_config
  built the PEFT adapter at r=hc.lora_config.r=8 → guard raises. Engine also scales by PEFT alpha/r
  (5.66) × adapter_scaling.
- FUNCTIONAL path (_generate_lora_dict + _functional_lora) used by BOTH training (hypernet_distill
  _student_logits) AND diag_recoverability: applies raw rank-8 (x·Aᵀ)·B · **scaling** — NO alpha/r,
  and **IGNORES the head bias entirely** (never calls combine_lora / get_head_bias).
WHY IT NEVER BIT BEFORE: Rune's own #49 checkpoints presumably use_bias=False → both paths agreed.
Adopting Sakana's bias-carrying checkpoint exposed the divergence.

### TWO IMPLICATIONS (recipe-relevant — for user):
1. pass@1 (engine) BLOCKED by the rank crash. Fix = size PEFT at combined rank (16) + correct alpha,
   OR disable use_bias. Scaling must be pinned (combined-adapter alpha/r) — get-wrong = meaningless pass@1.
2. DEEPER: training (functional) DROPS the head bias that the warm-start was trained with; engine
   APPLIES it → train/infer inconsistency. The clean fix is to make use_bias CONSISTENT across
   train/diag/engine. Decision hinges on: does qwen_4b_d2l recall survive WITHOUT the bias (rank-8
   only)? If yes → disable use_bias everywhere (simplest, coherent, no engine rank fix). If no →
   thread the bias through the functional/training path.

### SCALING TRAP (caught before running): _functional_lora scaling is a RAW multiplier (no alpha/r).
qwen_4b_d2l alpha/r = 45.25/8 = 5.66 → diag default --scaling 0.5 would be ~11× UNDER-scaled →
false negative. GATE 2 must run at scaling ≈ 5.66 (D1's +1.6 recall was via Sakana's OWN internalize,
NEVER via Rune's functional path — gate2 is the first native-Rune recall measurement; no-bias, rank-8).
Launching gate2 at --scaling 5.66.

### REVIEWER input (12:12Z) — Qwen4B Bias/Rank Plumbing Finding (validates analysis + 1 new gate)
- Agrees: gate0 = plumbing, NOT coding/pass@1. train/infer bias inconsistency is THE important
  discovery; until coherent, pass@1 AND training results are NOT interpretable.
- Run no-bias recall gate first AS A DIAGNOSTIC of whether bias is necessary: survives strongly at
  calibrated scaling → disable bias everywhere (simplest coherent path); collapses → thread
  combine_lora/head-bias through train+diag+engine (don't silently drop part of the checkpoint).
- Log the SCALING CONVENTION per path (Sakana/PEFT include alpha/r; Rune functional is raw) — else
  false neg/pos very plausible.
- **NEW pre-pass@1 GATE (adopt): functional-vs-PEFT PARITY CHECK** — for ONE generated adapter,
  functional logits and engine/PEFT logits must agree under the chosen bias/scaling convention.
  Else the bench measures an EXPORT bug, not adapter utility. → add as a gate before any pass@1 run.

### GATE 2 RESULT (recall via RUNE functional path, no-bias rank-8, scaling 5.66, n=12): NEAR-NEGATIVE
| target | matched | mismatch | zero | m-mismatch | m-zero |
| goal | -4.820 | -4.841 | -4.865 | +0.0205 | +0.0443 |
| diff | -3.365 | -3.375 | -3.375 | +0.0105 | +0.0109 |
| tail | -1.451 | -1.456 | -1.463 | +0.0058 | +0.0126 |
| avoid| -1.983 | -2.054 | -2.093 | +0.0704 | +0.1101 |
All POSITIVE but NOISE-LEVEL (~0.01-0.07) = same order as #49's failed margins; NOT the +1.6/+2.3
D1 saw via Sakana's OWN internalize. → qwen_4b_d2l recall does NOT transfer through Rune's native
functional path (no-bias rank-8). This is the load-bearing risk for Option B materializing.
CANDIDATE CAUSES (undisambiguated): (a) dropped HEAD BIAS (functional path omits it); (b) SCALING
convention (raw vs alpha/r — 5.66 may be off); (c) FEATURE-PATH mismatch (Rune feeds base
hidden-states via extract_activations_with_model; qwen_4b_d2l's perceiver may expect a separate
ctx_encoder's features → OOD); (d) EPISODE/FACT construction in diag differs from D1's Sakana-format.
D1's +1.6/+2.3 was via Sakana internalize (ctx_encoder→perceiver→combine_lora WITH bias), NEVER via
Rune's base-hidden-state→perceiver→raw-rank8 path. Calling advisor (stuck: result doesn't fit) before
choosing the disambiguation branch; then surface to user — this bears on Option-B viability.

### ADVISOR (narrowed the cause, ruled out 2 branches):
- m-zero is ALSO noise (+0.04) → adapter is nearly INERT vs no-adapter, not "active-but-non-specific".
- HEAD BIAS cannot explain an m-mismatch collapse: get_head_bias() is CONTEXT-INDEPENDENT (a learned
  constant) → cancels in matched−mismatch. Dropping it moves m-zero at most. ⇒ eliminate bias for the
  +2.3→+0.02 collapse. SCALING is a scalar → can't bridge a 100× gap. ⇒ deprioritize scaling.
- ⇒ The perceiver isn't producing a CONTEXT-CONDITIONED adapter through Rune's path = LOAD or
  FEATURE-PATH, not bias/scaling/episodes. Cheapest first: did the perceiver actually load?

### ROOT-CAUSE INVESTIGATION (systematic-debugging):
1. **PERCEIVER LOAD — RULED OUT.** Re-ran load_state_dict capturing keys: model=140 params, ckpt
   provides 143, **MISSING=0** (nothing random-init), all 127 perceiver/aggregator params LOADED.
   UNEXPECTED=3 = metadata only (base_model_name_or_path, ctx_encoder_args, hypernet_config). So the
   perceiver weights are correctly loaded — NOT silent random init.
2. **FEATURE-PATH — PRIME SUSPECT, mechanism identified.** Checkpoint carries
   `ctx_encoder_args = CtxEncoderArguments(ctx_encoder_type='per_layer_activations',
   ctx_encoder_last_layer=None, quantize_ctx_encoder=True, layer_idx=9[early_exit-only])`.
   Sakana `PerLayerActivations.forward` (ctx_encoder.py:91-145): output_hidden_states=True →
   `torch.stack(outputs.hidden_states, dim=1)`, on a base that has **the LAST transformer block
   REMOVED** (`layers[:-1]`), **QUANTIZED** (4-bit), **LM head stripped**.
   Rune `extract_activations_with_model` (hypernetwork.py:308): FULL model (all layers), select
   `hidden_states[i] for i in layer_indices[0..35]`, stack, under `disable_adapter`.
   → SAME tensor SHAPE/construction (stacked per-layer hidden states) — which is why generate_weights
   doesn't error — but DIFFERENT CONTENT: (a) Rune keeps the last block Sakana drops; (b) which
   hidden_states indices are stacked may be off-by-layer; (c) quant/dtype differs. The perceiver's
   modality_projection + cross-attn were trained on Sakana's exact feature distribution → Rune's
   features are OOD → near-inert adapter. CONSISTENT with the inert (m-zero≈noise) signature.

### OPINIONS (2026-06-01):
- **Most likely root cause = feature-interface mismatch**, NOT a fundamental "Qwen-4B/qwen_4b_d2l
  can't recall in Rune." The advisor's elimination (bias cancels, scaling scalar) + perceiver-loads
  ⇒ feature-path; and I've now identified the concrete divergence (last-block-drop / quant / hidden-
  state selection). CAVEAT: the construction is *nearly* identical, so a 100× collapse from a 1-layer
  drop alone would be surprising — quant + exact-index + normalization details may compound, OR the
  bias/scaling elimination is missing something. Needs the confirmatory test below, not assumed.
- **Option B is NOT dead, but the warm-start is NOT plug-and-play.** Native-Rune recall requires
  ALIGNING Rune's feature interface (and application: combine_lora/bias + alpha/r) to Sakana's, OR
  re-deriving the perceiver for Rune's interface. That's alignment work — more than a config swap,
  far less than installing recall from scratch. The reviewer (12:14Z, "Native Rune Recall Gate
  Negative") independently reached the same: warm-start invalid until Rune uses the same feature
  interface or retrains the perceiver for Rune's interface.
- **Do NOT run scaling sweeps / bias-inclusive variants / training / pass@1 yet** (advisor+reviewer):
  we'd be tuning on a mis-fed network; results uninterpretable. Phase-0 gate2 is the new BLOCKER.
- Reviewer's pre-pass@1 PARITY CHECK (functional vs PEFT logits agree) still stands for later.

### DECISIVE NEXT TESTS (cheap, before any commitment; pick on user's steer):
A. **Context-conditioning check (advisor):** generate adapter for 2 DIFFERENT contexts via Rune's
   path; compare A/B tensor norms + difference. Inert/identical → unconditioned (confirms feature/
   load) ; clearly different → conditioned (look elsewhere). ~1 small GPU run.
B. **Same-episode path A/B (reviewer #1):** reproduce recall on the SAME 12 episodes via Sakana's
   OWN internalize (third_party stack) vs Rune functional. If Sakana=+1.x and Rune=+0.02 on identical
   episodes → isolates PATH (not episodes). Then align Rune feature extraction to PerLayerActivations
   (drop last block + quant) and re-measure → if recall returns, root cause confirmed + fix known.
C. If A shows conditioning + B shows path-gap closes with feature alignment → the fix is a Rune
   feature-extraction change (match PerLayerActivations) + application alignment, then proceed.
STATUS: Phase-0 gate2 BLOCKED pending feature-interface diagnosis. Holding GPU spend; surfacing to user.

### REVIEWER input (12:22Z) — Feature Interface Diagnosis (endorses + refines test order)
- Diagnosis sound (perceiver loaded; bias/scaling can't explain collapse; near-inert → feature mismatch).
- **Lead with PATH A/B (my test B), NOT the tensor check:** Sakana internalize vs Rune functional on
  IDENTICAL episodes/facts. Sakana strong + Rune near-zero ⇒ path guilty (separates episode-construction
  from path). Reviewer ranks this ABOVE the context-conditioning tensor check (test A), which is weaker
  (different weights ≠ useful recall; similar weights only confirm inertness).
- Then align ONE feature-interface variable at a time, starting with faithful PerLayerActivations repro
  (hidden-state layer selection / drop-last, dtype/quant, masking). Do NOT combine with bias/export
  fixes in the same test (muddy attribution). Decisive evidence = does matched-vs-mismatch RETURN with
  Sakana-equivalent features.
- pass@1 paused for memory claims; any base-only run labeled "Qwen3-4B coding/prompt smoke" only.
REVISED next-test order: B (path A/B, same episodes) → feature-align one var at a time → re-measure.
Awaiting user steer (asked: run the tests vs steer differently); holding GPU.

### USER: "Proceed" → running clean PATH A/B (2026-06-01)
IMPORTANT RE-DIAGNOSIS from reading the harnesses: gate2 (diag) vs D1 (+2.3) was NOT a clean A/B —
they differ in THREE ways: (1) QUERY FORMAT — Sakana asks an explicit QA question via chat template
+ generation prompt (the format Doc2LoRA was TRAINED on); diag teacher-forces raw fact text after a
bare "## Review Feedback\n" header (NOT a question); (2) feature extraction; (3) application
(bias/alpha-r). So the +2.3→+0.02 gap conflated query-format + episode-construction + path. Restored
tools/scoring_core.py + tools/d2l_control/episodes.py from orphan branch (experiment-only, uncommitted)
to run the clean A/B holding QA-episodes/queries/scoring CONSTANT, varying ONLY the path.

PATH A/B SIDE 1 — SAKANA internalize (rune_episode_recall.py, qwen_4b_d2l, same 12 episodes):
  goal m-mismatch +2.235 (m-zero +3.756, frac 1.00); file +1.596; diff +0.983; OVERALL **+1.604**.
  → Reproduces D1 exactly. These QA episodes ARE strongly recallable via the faithful Sakana path.
PATH A/B SIDE 2 — RUNE functional twin (tools/_pathab_rune.py, SAME episodes/queries/scoring, Rune's
  extract_activations→generate_weights→_functional_lora @ scaling 5.66): RUNNING (bf5ral9yl).
DECISION RULE: twin ≈ +0.0x ⇒ PATH guilty (feature extraction + no-bias functional app) → align
features (PerLayerActivations) one var at a time. twin ≈ +2.x ⇒ path FINE → gate2's +0.02 was the
diag PROBE (bare teacher-forcing vs QA), not the path → fix the probe, recall was there all along.

### PATH A/B RESULT (2026-06-01) — DECISIVE: THE PATH IS GUILTY (and feature-path lead OVERTURNED)
SAME 12 episodes / SAME QA queries / SAME scoring, ONLY path varied:
| path | goal | file | diff | OVERALL |
| Sakana internalize | +2.235 | +1.596 | +0.983 | +1.604 |
| Rune functional (_functional_lora, raw rank-8, no combine_lora, no bias, scaling 5.66) | +0.024 | +0.032 | +0.015 | +0.024 |
~67× collapse. RULES OUT: probe/query-format (exact QA queries used both sides) AND episode
construction (same episodes). FEATURE-EXTRACTION lead OVERTURNED: Sakana PerLayerActivations stacks
`hidden_states` of `layers[:-1]` = embed+L0..L34; Rune extract_activations stacks hidden_states[0..35]
= embed+L0..L34 — SAME layers, both 4-bit → features ~equivalent. m-ZERO also tiny (+0.088) ⇒ adapter
is nearly INERT vs no-adapter.
→ REFINED ROOT CAUSE: the ADAPTER APPLICATION. Rune's _functional_lora applies the RAW rank-8 A/B
from generate_weights and SKIPS combine_lora (assembly of A/B + head bias + orientation that the
engine's generate_adapter_weights DOES for use_bias=True). The functional path was written for Rune's
OWN use_bias=False checkpoints (where generate_weights output IS the final adapter); for a Sakana
use_bias=True checkpoint it applies a mis-assembled/inert adapter. (Also note _to_peft_state_dict
TRANSPOSES B `b.t()` while _functional_lora uses B directly — orientation convention differs between
the two application paths; suspect.)
IMPLICATION: Rune's functional path (used by TRAINING + diag) does NOT faithfully apply a Sakana-format
adapter → warm-start recall is INVISIBLE to Rune's train loop → training would re-learn recall from
scratch through Rune's convention, defeating the warm-start. Option B requires making Rune's
application FAITHFUL (combine_lora + bias + orientation/alpha-r) across train+diag+engine, OR adopting
ctx_to_lora's apply path wholesale.
NEXT ISOLATION (single var): apply the SAME generated adapter via combine_lora (the engine/Sakana
assembly) functionally in Rune; if recall recovers → application is the bug + fix known. The engine's
generate_adapter_weights ALREADY does combine_lora — so FIXING the rank-16 PEFT-config bug (gate0) and
measuring recall via the engine/PEFT path doubles as this isolation AND unblocks pass@1.

### combine_lora read (lora_merger.py): appends bias as EXTRA rank slices to BOTH A and B (ranks
8-15 = bias tensor), leaves context-dependent ranks 0-7 UNCHANGED. So Rune-functional applies the
SAME ranks 0-7 A/B as Sakana, minus the bias slices. Bias term = (x@A_bias)@B_bias = context-
INDEPENDENT but INPUT-dependent linear transform (a learned constant LoRA), NOT a scalar constant.
PUZZLE: if ranks 0-7 identical + bias context-independent (cancels in matched−mismatch), why does
Rune m-mismatch collapse +2.2→+0.024? RESOLUTION (likely): logprob-margin is NONLINEAR (saturating)
in adapter-delta magnitude. m-ZERO is also tiny (+0.088) ⇒ Rune's adapter delta is tiny/under-applied.
The bias term adds large context-independent MAGNITUDE that puts logits in a responsive regime where
the context-dependent ranks 0-7 can express; without it (Rune functional) the delta is too small to
move logprobs → both m-zero AND m-mismatch tiny. So the advisor's strict "bias cancels linearly" is
incomplete — via softmax nonlinearity the bias CAN gate m-mismatch. ⇒ combine_lora+bias+correct
scaling plausibly RESTORES recall. Parity gate will confirm.

### USER Q1 "replace the Rune method?" + Q2 "doubt previous conclusions?" — DETERMINATION
**Q1 — YES, adopt ctx_to_lora's application CONTRACT as the single source of truth** (combine_lora +
head-bias + rank expansion + B orientation + alpha/r), unifying Rune's THREE divergent application
sites (training _functional_lora, diag _functional_lora, engine PEFT hotswap) onto ONE faithful impl.
Reviewer concurs: "a correctness boundary, not an abstraction preference." Rune already IMPORTS
ctx_to_lora.HyperLoRA, so this is consistent, not new coupling. CONSTRAINTS the faithful impl MUST
keep (can't just call ctx_to_lora's inference apply): (a) GRAD-FLOW for training (generated A/B stay
in autograd graph — why _functional_lora exists); (b) 4-bit base (Linear4bit forward_orig). So: make
Rune's application faithfully reproduce the ctx_to_lora adapter contract, GATED by a logit-PARITY test
(Sakana apply vs Rune-functional-fixed vs Rune-engine-PEFT-after-rank-fix must AGREE) before any
train/pass@1. CONDITIONAL: if parity logits still diverge after combine/bias/orientation/scaling
alignment, the A/B GENERATION differs too (feature encoding) — parity gate isolates that.

**Q2 — YES, this injects real doubt, narrowly but importantly:**
- The D1 headline "#49 failed on RECIPE (not arch/probe/base)" compared recall across TWO DIFFERENT
  application paths: Sakana-faithful (+1.6/+7) vs Rune-functional (#49 +0.0005). We NOW know those
  paths are NOT equivalent — Rune-functional under-reads a KNOWN-recall adapter by ~67× (+1.6→+0.024).
- So the CROSS-PATH CALIBRATION is invalid: "+7 = real recall" was Sakana-stack; it cannot be used to
  judge Rune-stack magnitudes. #49's +0.0005 (Rune-stack) vs Sakana's +7 (Sakana-stack) was
  apples-to-oranges in the application dimension.
- Likely STILL TRUE (directionally): #49's own adapter measured through its own (use_bias=False) path
  got noise → #49 trained little/no recall. But the MAGNITUDE/"definitely noise" claim needs
  re-validation: confirm #49 was use_bias=False AND Rune's functional path applies use_bias=False
  faithfully (parity), then re-read #49 + re-derive the calibration THROUGH RUNE'S FIXED PATH.
- NOT in doubt: the scoring math (mean_gold_logprob) itself; that Sakana recalls Rune facts (Sakana
  stack); that the architecture/base CAN bind the facts. The doubt is specifically about Rune-stack
  recall MAGNITUDES and the cross-path comparison.
ACTION: (1) build the application-parity gate; (2) adopt the ctx_to_lora contract on the parity-
passing impl; (3) re-validate #49/calibration through the fixed Rune path before trusting magnitudes;
(4) only then resume training/pass@1. Do NOT train on the current functional path.

### USER: "Go! be DRY." → APPLICATION-PARITY GATE (running, bq3q79w39)
Reviewer (12:35Z) fully concurs with the Q1/Q2 determination (D1 not erased; Rune-stack MAGNITUDES
+ Sakana/Rune ratio provisional; #49 negative stands IFF use_bias=False + parity; adopt ctx_to_lora
contract across all 4 sites; pass@1 = smoke-only until parity passes).
DRY parity gate = EXTENDED tools/_pathab_rune.py (no new file) to score TWO ARMS on the SAME episodes/
queries/scoring/functional-apply — differing ONLY in adapter ASSEMBLY:
  - arm "raw"      = current functional path (raw rank-8 generate_weights output) → expect +0.024.
  - arm "combined" = REUSE ctx_to_lora.combine_lora(raw, n_chunks=[1], lora_bias=hyp.get_head_bias())
                     [the SAME assembly the engine's generate_adapter_weights already calls] then the
                     SAME _functional_lora applies it. NO 4th application impl written (DRY).
DECISION: combined arm m-mismatch jumps toward Sakana's +2.2 ⇒ combine_lora+bias assembly IS the fix,
it's grad-compatible (functional) AND DRY → unify all sites to "combine_lora → _functional_lora" +
fix engine PEFT rank-16. combined arm still ~+0.02 ⇒ assembly isn't enough; the A/B generation differs
(feature encoding) — dig there next. scaling 5.66 = alpha/r; may sweep if combined is positive-but-low.

### PARITY GATE RESULT (raw vs combined+bias, qwen_4b_d2l, same 12 episodes):
  raw      : goal m-mismatch +0.024, m-zero +0.088
  combined : goal m-mismatch +0.025, m-zero +0.139  (combine_lora + head bias)
→ combine_lora+bias did NOT restore recall (+0.024→+0.025). ASSEMBLY/BIAS hypothesis RULED OUT
  (confirms advisor: context-independent bias cancels in matched−mismatch; m-zero rose slightly only).
→ apply MATH verified correct (B stored [r,out]; _lora_delta (x@Aᵀ)@B matches; _to_peft transposes for
  PEFT — both consistent). So the remaining suspect is FEATURE ENCODING: the A/B that Rune's
  generate_weights produces from RUNE-style features differ from what Sakana's ctx-encoder features
  produce. PRIME sub-suspect: CONTEXT TOKENIZATION — Sakana uses tokenize_ctx_text (ctx-specific
  affixes/special tokens, CTX_AFFIXES) for the doc; Rune's extract_activations uses plain
  tokenizer(text). Mismatched ctx encoding → perceiver sees OOD context → near-inert A/B. (Layer
  content itself matches: Sakana layers[:-1] stack == Rune hidden_states[0..35] = embed+L0..L34.)

### #49 IS use_bias=True TOO (checkpoint_step600.pt: use_bias=True, r=8, alpha=16) — bears on USER Q
### "could our checkpoint work + ctx_to_lora be the problem?"
DETERMINATION (nuanced):
- **CONFIRMED IN GENERAL (yes):** Rune's functional application CAN make a working checkpoint look
  broken — PROVEN: qwen_4b_d2l recalls +2.2 via Sakana's stack but only +0.024 via Rune's functional
  path. So "the checkpoint works, Rune's application was wrong" is REAL — for a checkpoint trained
  under a DIFFERENT (Sakana) convention than Rune applies.
- **FOR #49 SPECIFICALLY (most likely NO, but must verify):** #49 was TRAINED via Rune's functional
  path (_student_logits → _functional_lora → RAW generate_weights, NO combine_lora) AND MEASURED via
  diag's same functional path → SELF-CONSISTENT (perceiver trained on Rune's feature convention,
  measured on Rune's feature convention). The bug we found is a CROSS-CONVENTION mismatch (Sakana-
  trained ckpt applied under Rune's convention); #49 has no cross-convention gap. Also #49 DID learn
  generic edit-emission (real m-zero/diff behavior) → its apply path isn't totally dead → its lack of
  SPECIFICITY (m-mismatch≈0) is consistent with the recipe explanation, not a dead application.
- **DEEPER reframe to keep honest:** if Rune's feature ENCODING (ctx tokenization) is wrong in an
  ABSOLUTE sense, #49's TRAINING signal through the adapter would be weakened too → could contribute
  to #49 not learning recall (a "broken application infrastructure" cause, distinct from "recipe").
  But #49's perceiver was trained on whatever features Rune fed, so train/infer match for #49.
- **VERIFY, don't assume (reviewer):** confirm #49 train apply == diag apply (both raw functional, no
  combine_lora) from the training code; if a faithful/Sakana-equivalent application of #49 ever shows
  recall, the #49 verdict flips. Cleanest: once Rune's feature encoding is fixed + parity-gated,
  re-measure #49 through it. Until then, #49's MAGNITUDE remains provisional (Q2).
NEXT DECISIVE TEST (assembly ruled out → isolate feature encoding): in the SAKANA venv (working path),
swap ONLY the ctx feature extraction to Rune-style (plain tokenization + full-model hidden states) and
see if recall collapses; OR compare generated A/B tensors (Rune-features vs Sakana-features) through
the same perceiver. If Rune-style features collapse recall → ctx encoding/tokenization is the root.

### FEATURE-ISOLATION RESULT (feat_isolation.py, Sakana venv, n=3) — OVERTURNS feature hypothesis
Same perceiver (generate_weights), Sakana-ctx-features vs Rune-style-features (plain tok + full base
hidden states), per episode:
  tokens: sakana 545/412/455 vs rune 532/399/442 (~13 fewer — ctx affixes, minor)
  FEATURES very different: feat cos(sakana,rune) = +0.15/+0.16/+0.19; |feats_s|~45-49k vs |feats_r|~35-42k
  but A/B NEARLY IDENTICAL: **A cos +0.93, B cos +0.79, norms ~equal** (|A_s|~34 vs |A_r|~37; |B|~0.74 both)
→ The perceiver L2-normalizes (forward:422) and is ROBUST to the feature differences → Rune-style
  features produce essentially the SAME adapter Sakana's do. FEATURE-ENCODING is NOT the dominant cause.
→ So the +2.2 vs +0.024 recall gap is in the APPLICATION (A/B are ~correct), yet combine_lora/bias
  did NOT fix it and the apply MATH is verified correct (B=[r,out]; _lora_delta (x@Aᵀ)@B correct).
→ ONLY remaining free variable = SCALING/application magnitude in _functional_lora. Running scaling
  sweep 5.66/16/45/128 (bevh02zjs). Caveat: m-zero at 5.66 = +0.088; reaching Sakana's m-zero +3.7
  may need implausibly high scaling (~linear→~40×→~226) → if sweep PLATEAUS low, scaling is NOT it
  and there's a deeper application difference (then: apply Rune-feature A/B via SAKANA's own apply
  path in-venv to localize generation-vs-application definitively).
PER USER Q (#49): this WEAKENS the "ctx_to_lora application was the whole problem" theory — the
adapters Rune generates are ~correct; something in HOW Rune applies/scores them is weak. Still being
localized. #49 magnitude remains provisional (Q2 stands).

### SCALING SWEEP (tools/_pathab_rune.py raw arm, qwen_4b_d2l, same 12 episodes) — ROOT CAUSE = SCALING
| scaling | goal m-mismatch | m-zero | frac>0 |
| 5.66 (alpha/r) | +0.024 | +0.088 | 0.75 |
| 16 | +0.068 | +0.275 | 0.92 |
| 45 | +0.200 | +1.138 | 0.83 |
| 128 | +0.512 | +2.101 | 0.92 |
MONOTONIC steep climb toward Sakana (+2.235 / +3.756). The SAME (correct) A/B recover recall when
applied STRONGLY. → Rune's functional path applies the adapter ~50-90× TOO WEAKLY at alpha/r=5.66.
ROOT CAUSE = a SCALING/MAGNITUDE convention mismatch in _functional_lora (the A/B are right; the
scalar is wrong). Advisor's "scaling can't bridge 100×" REFUTED (softmax nonlinearity → it can).
Likely a missing constant factor (perceiver L2-normalizes the lora_emb → head output is ~unit-norm;
Sakana's apply must carry a large compensating scale that Rune's _functional_lora drops). Need the
PRINCIPLED factor (read Sakana's lora_layer apply), not a magic number — it matters for train+engine.
TENSION to resolve: engine/#50 needed LOW adapter_scaling (~0.75 / alpha-r) or structured-gen broke;
recall needs HIGH (~150+). Sakana uses ONE scale and gets both recall + coherent gen → there IS a
correct convention we're missing.

### THIS REVIVES THE USER'S #49 QUESTION (now the priority test):
diag_recoverability DEFAULT --scaling=0.5; gate2/this used 5.66. At 5.66 a KNOWN-recall adapter
(qwen_4b_d2l, true +2.2) reads +0.024 = NOISE. So **#49's near-zero diag margins may be a LOW-SCALING
MEASUREMENT ARTIFACT, not absence of recall.** #49 might recall much better at high scaling — exactly
the user's hypothesis ("our checkpoint works; the application/measurement was wrong"). DECISIVE TEST:
re-run diag on a #49 checkpoint (/tmp/rune-ck-final/checkpoint_step600.pt) across a SCALING SWEEP. If
#49 recall climbs like qwen_4b_d2l did → #49's "recipe failure" verdict is substantially WRONG (it was
a scaling/measurement artifact). If #49 stays flat at all scalings → #49 genuinely lacks recall (recipe
verdict holds). This single cheap experiment resolves Q2 + the user's question. RUN NEXT.
(Caveat: must confirm the PRINCIPLED scaling separately; but the sweep SHAPE answers "is recall there".)

### REVIEWER (13:01-13:02) — concurs; key framings to carry:
- Feature-interface hypothesis OVERTURNED (A/B ~same from either features). Problem is downstream of
  A/B: application magnitude/scaling/placement/scoring. (My sweep climbed sharply → scaling IS it.)
- KEEP TWO QUESTIONS SEPARATE: (1) empirical EXISTENCE of recall under high scaling (the sweep);
  (2) the PRINCIPLED scaling convention for train/engine. High-scale sweep reveals hidden recall but
  is NOT itself the production fix.
- #49 verdict: if #49 recall RISES with scaling → "recipe failure" conclusion must be REWRITTEN
  substantially (it was application/measurement-SCALE, maybe + generation instability at production
  scales). If #49 stays FLAT while qwen rose → recipe-failure SURVIVES. (Running: b2178uad1.)
- Extra rigor (optional): logit-level parity on SAME A/B (Sakana-apply vs Rune-apply) to confirm
  application-not-scoring; and "if same A/B + same scores but low recall, revisit whether the Sakana
  run used the same A/B." (My sweep already implicates magnitude strongly.)
- BEFORE pass@1: need a PRINCIPLED application/scale contract + a GENERATION-STABILITY check —
  high scaling may recover logprob memory while BREAKING structured generation. Recall and usable
  pass@1 remain SEPARATE GATES (resolves the engine-low-scaling vs recall-high-scaling tension:
  they may be irreconcilable at a single scale → that itself is a key finding if so).
Running #49 scaling sweep (5.66/45/128/256) via same QA harness; awaiting.

### #49 SCALING SWEEP RESULT (checkpoint_step600.pt, Qwen3.5-9B, same QA harness) — DECISIVE, OPPOSITE OF QWEN
| scaling | goal m-mismatch | m-zero | frac>0 |
| 5.66 | -0.119 | -8.605 | 0.50 |
| 45   | -0.020 | -7.577 | 0.50 |
| 128  | -0.112 | -8.968 | 0.50 |
| 256  | -0.193 | -9.004 | 0.42 |
TWO decisive differences from qwen_4b_d2l:
1. m-mismatch stays at NOISE (~0, frac~0.5 coin-flip) at ALL scalings — does NOT climb (qwen:
   +0.024→+0.512). No hidden episode-specificity for scaling to reveal.
2. m-zero is hugely NEGATIVE (-8 to -9), WORSENS with scale — #49's adapter ACTIVELY DESTROYS the QA
   answer likelihood. Signature of an EDIT-EMISSION adapter (scaling it pushes harder toward emitting
   code, degrading QA), NOT a memory adapter (qwen's m-zero was POSITIVE and grew).

### RESOLUTION OF USER'S QUESTION + Q2 (definitive):
- qwen_4b_d2l: recall EXISTS, was HIDDEN by low scaling through Rune's functional path (climbs with
  scaling). The scaling/application bug is REAL → relevant to Option B warm-start.
- #49: recall ABSENT at EVERY scaling (m-mismatch≈0 coin-flip) + adapter ANTI-QA (m-zero≈-9). The
  scaling artifact does NOT rescue #49. **#49 genuinely lacks queryable recall — "recipe failure"
  verdict SURVIVES and is STRENGTHENED** (the adapter doesn't merely fail to recall, it degrades
  recall, consistent with the full-revision edit-emission objective).
- So: "is it possible our checkpoint works + ctx_to_lora was wrong?" → NO for #49 (verified by direct
  re-measurement across scalings); YES in general that Rune's LOW-SCALING application can hide a
  real-recall adapter (proven on qwen). Q2 cross-path/calibration doubt is real for general magnitude
  comparison, but #49's negative is NOT a measurement artifact.
STANDING: Option B (warm-start qwen_4b_d2l) viable IF we fix the application SCALING (principled
factor, not magic number) AND pass a generation-stability gate (high scaling vs structured-gen).
Two separate questions remain: (1) principled scaling convention; (2) recall-high-scale vs
engine-low-scale reconcilability.

### CONSOLIDATED POSITION — training-paradigm migration (user greenlit recording, 2026-06-01):
MIGRATE: YES. Objective = stateless queryable EPISODIC RECALL of feedback-derived facts
(goal / current-state / tried-critique); data = patches+facts (not full-file revisions); the DIFF is
DEMOTED from memory-supervision to a downstream ACTION/EVAL target (memory/policy separation).
- "Away from diffs" half = EMPIRICALLY CLINCHED: #49's edit-emission adapter is ACTIVELY ANTI-recall
  (m-zero≈-9, worsening with scale) — diff-emission and recall are in direct tension.
- "Toward recall" half = validated in principle (Sakana recall objective binds goal/state +1.6-2.3),
  but realizing it in Rune is gated by: (a) the APPLICATION/SCALING fix (below); (b) MINING failure-
  bearing trajectories for the "what we tried" facet (current corpus single-turn; only goal +
  one-attempt + current-state available now).
NECESSARY-BUT-NOT-SUFFICIENT: paradigm migration alone won't show up unless the application is fixed.

### "WHY NOT JUST REPLICATE SAKANA" (user) — YES, and the mechanism is now PINNED:
Sakana's lora_forward (lora_layer.py) math = `(x@Aᵀ)@B * scaling` — IDENTICAL to Rune's _lora_delta
(same einsums, same layer indexing). The differences that make Sakana work and Rune under-apply:
1. **scaling = lora_alpha = 45.25** (patch_lora_forward:528 `scaling=self.peft_config.lora_alpha`),
   NOT alpha/r=5.66. Rune's _functional_lora + diag used alpha/r → **8× (=r) too small.** ← prime bug.
2. **combine_lora** (rank-16 + bias) — Sakana applies the assembled adapter; Rune functional skipped it.
3. **base precision**: Rune twin/diag load base 4-bit NF4; Sakana bf16. 4-bit noise washes margins in
   BOTH feature-extraction and scoring (suspected 3rd factor: raw@45 gave +0.200 vs Sakana +2.235).
PLAN = REPLICATE Sakana's apply wholesale (DRY): adopt ctx_to_lora's lora_forward/apply_lora_to_layers
+ combine_lora + scaling=lora_alpha across Rune's train/diag/engine (one path, retire the 3 bespoke
reimplementations). Bonus: this also kills the rank-16 PEFT crash. GRAD-FLOW is NOT a blocker —
Sakana's own CrossEntropyTrainer trains through this exact apply, so it's grad-compatible.
VALIDATION (running, bb709c6uy): twin with bf16 + combine_lora + scaling 45.25 on qwen_4b_d2l →
expect ≈ Sakana's +2.2. Grid (raw/combined × 5.66/45.25, bf16) attributes the gap to scaling vs bias
vs precision. If confirmed → replicate Sakana's apply in Rune = THE fix (no magic constant).
REMAINING GATE after replication: generation-stability — scaling=alpha (45.25) is what #50 found broke
structured-gen; must check recall-scale vs engine structured-gen reconcile (Sakana does coherent gen
at alpha, so there IS a workable contract — find why Rune's engine differed).

### REPLICATION GRID RESULT (bf16 base, qwen_4b_d2l, same 12 episodes) — FIX VALIDATED, gap attributed
| arm | scaling | goal m-mm | goal m-zero | frac>0 |
| raw | 5.66 | +0.016 | +0.042 | 0.58 |
| combined | 5.66 | +0.030 | +0.034 | 0.92 |
| raw | 45.25 | +0.191 | +0.855 | 0.92 |
| combined | 45.25 | **+0.823** | +1.852 | **1.00** |  (Sakana ref goal +2.235 / overall +1.604)
ATTRIBUTION (clean):
- SCALING alpha/r→alpha (8×) is DOMINANT: noise +0.024 → +0.823 (34×), frac 1.00 = real episode-
  specific recall through RUNE's stack.
- combine_lora+bias MATTERS at high scale: +0.191→+0.823 (4.3×). Advisor's "bias cancels in
  m-mismatch" REFUTED — at high scaling the bias shifts the regime (softmax nonlinearity) so the
  context part expresses. We DO need combine_lora.
- PRECISION irrelevant: raw@45 4-bit +0.200 vs bf16 +0.191 ≈ same. (4-bit fine → OK for the 9B.)
- RESIDUAL +0.823 vs Sakana +2.235 (~2.7×) = almost certainly the CTX FEATURE PIPELINE
  (tokenize_ctx_text affixes + PerLayerActivations; earlier A cos 0.93 ≈7% off → enough in the
  sensitive margin). Full faithful replication (apply + ctx features) should close it.
CONCLUSION: "replicate Sakana" WORKS. The collapse was an APPLICATION bug (scaling + combine_lora),
not architecture/feature-fundamentals. +0.742 overall is ALREADY real recall (vs +0.024 noise) —
likely sufficient to proceed even before closing the residual.
THE DRY FIX (decided): adopt ctx_to_lora's apply contract in Rune — lora_forward(scaling=lora_alpha) +
combine_lora + (for full fidelity) tokenize_ctx_text/PerLayerActivations ctx features — as the SINGLE
application path across train/diag/engine, retiring the 3 bespoke reimplementations. Then: gen-
stability gate (alpha-scale vs structured-gen) + the 3 anchors (qwen recall, #49 anti-QA, func-vs-
engine parity) before training.

### IMPLEMENTATION WORKFLOW LAUNCHED (wco9emwfp, run wf_92328cf4-a17) — user: "proceed with workflow"
Structured for SAFETY (delicate core code), not fan-out:
- Map (parallel read-only): functional site (hypernet_distill _functional_lora/_generate_lora_dict/
  _student_logits + diag callers) + engine site (generate_adapter_weights/_to_peft_state_dict/
  merge_head_bias_rank + wrapper.from_config PEFT config).
- Implement (TDD, equivalence-gated): NEW src/rune/model/adapter_contract.py (assemble_adapter =
  combine_lora+head-bias; effective_scaling = lora_config.lora_alpha) + tests/unit/
  test_adapter_contract.py (toy-tensor numerical equivalence to ctx_to_lora.lora_forward). THEN
  parallel: refactor functional path (train+diag → scaling=lora_alpha + combine_lora; diag --scaling
  becomes optional override) + fix engine export (PEFT r_peft=2r when use_bias [fixes rank-16 crash];
  lora_alpha_peft = ckpt_lora_alpha * r_peft so PEFT alpha/r == lora_alpha).
- Verify: full unit suite + ruff + mypy; EMIT (not run) the 4 GPU anchor commands.
GPU anchors kept OUT (I drive + review after): (1) qwen recall via fixed path; (2) #49 anti-QA;
(3) functional-vs-engine logit parity; (4) gen-stability/MBPP pass@1 smoke at contract scaling.
Will REVIEW the diffs before running GPU anchors.

### REVIEWER (13:37) — review-time checks for the implementation (apply when diffs return):
1. ENGINE PEFT scaling: `lora_alpha_peft = lora_alpha * r_peft` is correct ONLY if PEFT bakes in no
   extra scaling on the assembled tensors. REQUIRE a TOY (CPU) logit-parity test: PEFT-export apply ==
   functional-contract apply on the same A/B/input — not just the alpha/r arithmetic check.
2. NAME the two scales explicitly in adapter_contract: checkpoint lora_alpha vs runtime multiplier.
   Do NOT reuse `adapter_scaling` for both (alpha vs alpha/r vs runtime conflation = past bugs).
3. diag --scaling stays an OVERRIDE; DEFAULT must be the checkpoint contract (lora_alpha), never a
   silent 0.5/alpha-r (that recreates the measurement bug).
(Workflow mid-flight; fold these into the diff review + add the toy logit-parity test if the engine
agent didn't.)

### IMPLEMENTATION WORKFLOW COMPLETE (wco9emwfp, 6 agents) + MY DIFF REVIEW (2026-06-01):
CHANGED: NEW src/rune/model/adapter_contract.py (assemble_adapter=combine_lora+head-bias;
effective_scaling=lora_config.lora_alpha; lora_delta re-exports _lora_delta — DRY, single math path).
src/rune/model/wrapper.py: new peft_scaling_params(alpha,r,use_bias)->(r_peft=2r if use_bias else r,
alpha_peft=alpha*r_peft) so PEFT alpha_peft/r_peft == checkpoint lora_alpha; from_config uses it.
hypernet_distill.py + diag_recoverability.py: functional path now combine_lora + scaling=lora_alpha;
diag --scaling default=None→effective_scaling(hyp) (override optional). +2 unit tests.
TESTS (verified by me): full unit suite 312 passed; mypy clean; ruff clean on touched files; my own
re-run of the 2 new tests = 17 passed.
REVIEW FINDINGS:
- ✅ DRY/clean/correct-per-contract. Reviewer checks #2 (scale names: checkpoint_alpha vs alpha_peft vs
  runtime adapter_scaling — distinct) and #3 (diag default=contract) satisfied.
- ⚠️ CAUGHT AGENT ERROR: verify agent claimed qwen use_bias=False (conflated with use_per_rank_bias).
  RE-VERIFIED: qwen_4b_d2l use_bias=TRUE (alpha=45.25, r=8); #49 use_bias=TRUE (alpha=16, r=8). So
  combine_lora is NOT a no-op for qwen; engine builds PEFT r=16 for it. Corrected.
- 🔴 BIG FLAG (recall-vs-generation tension now CONCRETE in code): the engine fix makes PEFT realize
  effective scaling = checkpoint lora_alpha (qwen 45.25). The OLD #50 comment said setting scaling=alpha
  "8× too strong... structured generation never closed the JSON." So the engine now applies EXACTLY the
  scale #50 found breaks xgrammar structured-gen. → ANCHOR #4 (generation-stability / MBPP pass@1) is THE
  critical open test and likely the real remaining hurdle. Recall needs alpha; engine gen may break at
  alpha. (Sakana does coherent gen at alpha → a contract exists; may need decode/policy integration.)
- ⚠️ Anchor #3 (functional-vs-ENGINE logit parity on a REAL model) UNVERIFIED — engine unit test is
  arithmetic-only (reviewer check #1 not fully met). Needs a NEW small GPU harness (agent flagged, didn't
  fabricate): one trajectory → engine PEFT logits vs functional-contract logits → torch.allclose.
- ⚠️ TRAINING risk (Map agent): combine_lora now routes bias_A/bias_B into the training autograd graph
  for the FIRST time → scaler_B-collapse tripwires (commit c3a83217) may need retuning when we train.
- #49 contract scaling = its lora_alpha = 16 (not 45.25); my earlier sweep (5.66/45/128) backets it ~flat
  → #49 stays flat/anti-QA at its own contract scale → recipe-failure verdict holds.
GPU ANCHORS READY (I drive, under review): 1) qwen recall via fixed path (uv run python tools/_pathab_rune.py
--bf16; expect goal m-mismatch ~+0.823 = PASS, residual to +2.2 = feature path); 2) #49 anti-QA (diag,
auto-scale=16); 3) BUILD functional-vs-engine parity harness + run; 4) gen-stability/MBPP smoke (the
critical one). STATUS: implementation durable + CPU-green; ready for GPU validation.


### GPU VALIDATION SESSION START (2026-06-01, fresh agent via /workflows handoff)
Read handoff + scratchpad + reflections. Confirmed state: impl committed (2cb22a22), CPU-green;
GPU free (0/23GB), RAM 11GB free; all local-only tools + both checkpoints present.
DECISION (advisor-confirmed): do NOT fan out the 4 GPU anchors into a workflow — they are
physically serial (one model per 23GB GPU), review-gated ("review each before the next"), anchor #3
must be BUILT first, anchor #4 is expected-maybe-to-break with an interpretation a verifier can't make.
Driving them manually, reviewing each. /workflows intent honored later: anchor #3 parity-harness BUILD
is the one piece that legitimately benefits from orchestration (TDD+review) — offer when we reach it.
Armed monitor on reflections.md.

ANCHOR #1 LAUNCHED: uv run python tools/_pathab_rune.py --bf16 (4B bf16, contract scaling auto=lora_alpha=45.25)
  log: /tmp/anchor1_pathab.log
  PASS = goal m-mismatch ~+0.823 on COMBINED arm (interpret vs +0.823, NOT Sakana +2.2; residual=ctx feature path).

### ANCHOR #1 RESULT = PASS (2026-06-01)
combined arm (scaling=45.25, bias=on): goal m-mismatch=+0.824 frac=1.00; file +0.889; diff +0.523; OVERALL +0.745.
raw arm (bias off): goal +0.195 -> combined +0.824 (combine_lora+bias essential at high scale).
goal +0.824 == handoff PASS target +0.823 (exact). frac(m-mis>0)=1.00 -> real episode-specific recall through FIXED Rune stack.
Residual to Sakana +2.2 = known ctx-feature pipeline gap (not a failure). Replication grid reproduced end-to-end. -> proceed to anchor #2.

### ANCHOR #2 RESULT = PASS (2026-06-01) — #49 anti-QA sanity holds
diag at #49 contract scaling=16: goal m-mismatch +0.083 / m-zero -8.570; diff +0.241/-9.312; tail +0.047/-11.310; avoid +0.348/-0.175.
m-mismatch FLAT (~0), m-zero STRONGLY NEGATIVE (-8..-11) -> adapter actively anti-recall. Scaling fix rescued qwen (+0.824) but NOT #49.
Recipe-failure verdict survives, strengthened. 9B loaded 4-bit OK, watchdog never tripped. -> proceed to anchor #3 (BUILD parity harness).

### ANCHOR #3 BUILD via WORKFLOW (user chose orchestrate; run wf_a123a0d2-b54)
CPU-only workflow (GPU forbidden inside): Map (parallel: engine apply path wrapper.from_config/generate_adapter/
hotswap + functional path adapter_contract/_functional_lora, ref tools/_pathab_rune.py) -> Build TDD
(COMMITTABLE tests/unit/test_engine_functional_parity.py toy-tensor PEFT-vs-functional equivalence [reviewer
check #1] + LOCAL tools/_parity_engine_vs_functional.py real-model GPU harness) -> adversarial Review.
I run the GPU parity check myself after, then review. Awaiting workflow completion.

### ANCHOR #3 GPU RUN launched SIMULTANEOUSLY with workflow Review (user request)
Build phase wrote tools/_parity_engine_vs_functional.py; I read+verified it is GENUINE (engine = real
ModelWrapper.from_config/generate_adapter/hotswap PEFT path; functional path runs FIRST while PEFT B=0 so
no contamination; same DOC->same adapter; same input ids; assemble_adapter+_functional_lora@effective_scaling
vs PEFT@lora_alpha; atol0.5/rtol0.02 + last-token argmax). Launched GPU run (4B bf16, run_guarded) in
PARALLEL with the still-running Review agent. PASS = allclose True (+ argmax match). Awaiting both.

### ANCHOR #3 RESULT = PASS (2026-06-01) — real-model engine-vs-functional logit parity
effective_scaling=45.25 (lora_alpha), use_bias=True, 36 layers. max_abs_diff=0.500 mean_abs_diff=0.0409
allclose(atol0.5,rtol0.02)=True; last-token argmax engine==functional (73594) MATCH. Exit 0.
Engine PEFT hotswap path == functional contract path to bf16 precision. Reviewer check #1 (scaling proven
only by arithmetic) now CLOSED by real-model parity. mean 0.04 + argmax match decisive; 8x scaling error
would give huge divergence not 0.04. (max outlier == atol is bf16 rounding.) Awaiting workflow Review verdict to cross-check harness.

### RESPONSE to reviewer "Functional-vs-Engine Parity Pass" (14:59Z)
Reviewer endorses anchor #3 (parity closes the app-export blocker; pass@1 now measures the corrected
contract, not a PEFT export bug). Accepted — carrying forward the key guidance into anchor #4: IF gen-
stability/MBPP breaks at the corrected alpha-scale, PRESERVE the parity result (adapter application is
correct) and treat it as a decode/policy/runtime-multiplier-scheduling problem, NOT evidence against the
contract or recall. Recall and usable pass@1 remain SEPARATE gates (handoff anchor #4 note + #50 history).

### WORKFLOW REVIEW reconciled (anchor #3): verdict=issues(all MINOR), parity_is_genuine=True, confidence=high.
Review confirms harness genuine (real PEFT path, same base/input/DOC, disable_adapter clean forward, bias-half
exercised: ckpt use_bias:true r=8->r_peft=16, combine_lora bias layout correct). CPU test 2 passed, ruff+mypy clean.
Key minor: harness compared TWO independently-generated adapters (determinism assumption, adapter_diff printed not
asserted). Applied 3 cheap hardenings to tools/_parity_engine_vs_functional.py: assert adapter_max<1e-3 + no key
mismatch; stamp use_bias/r_peft regime; mean-drift backstop (PASS = allclose AND mean_abs<0.05). Re-running hardened
version (note: the reviewed/extended on-disk version adds disable_adapter + adapter_diff vs the simpler version my
first run used). Committable CPU toy test at tests/unit/test_engine_functional_parity.py (reviewer check #1 artifact).

### ANCHOR #4 PLAN (per user steer "orient to what Sakana showed worked"; advisor-endorsed)
Lead with Sakana-PROVEN baseline: third_party/doc-to-lora/.venv/bin/python rune_code_recall.py with
D2L_CKPT=trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin (Sakana venv, transformers 4.51.3).
= internalize(doc)+free-form greedy generate at lora_alpha -> coherent gen + correct fact recall. This is the
SAME effective scale (lora_alpha) Rune now applies (proven by anchor #3 parity). PASS(a)=coherent gen + recall
(expected, Sakana-proven). THEN probe Rune xgrammar structured-gen/MBPP (_bench_entry adapter_scaling=1.0) = the
OPEN product gate; a break there = decode/policy integration (adapter_scaling knob = lever), NOT contract/recall.
Gates SEPARATE. Report baseline first, let user decide spend on structured probe.

### ANCHOR #4 BASELINE (Sakana "what worked") = PASS (2026-06-01)
rune_code_recall.py on qwen_4b_d2l (Sakana venv, internalize+free-form greedy generate @ lora_alpha, doc NOT in prompt):
mean m-mismatch=+2.597 mean m-zero=+8.354 gen_accuracy=0.88 frac(m-mis>0)=0.88 (n=8). 7/8 facts hit.
Generations FULLY COHERENT fluent English (e.g. "...the number used as the modulus is **9973**."), NO degeneracy.
KEY: the lora_alpha contract scale is GENERATION-VIABLE (Sakana-proven on our ckpt). + anchor #3 (Rune apply==Sakana
apply at this scale, argmax-identical) => Rune free-form gen at contract scale should be coherent by construction.
THEREFORE if Rune xgrammar/MBPP structured-gen breaks at alpha => isolated to structured-decode/policy integration
(xgrammar x strong adapter), NOT adapter scale/recall. Recall+gen-viability at alpha now POSITIVELY controlled.
NEXT (open product gate): Rune xgrammar/MBPP structured smoke (_bench_entry adapter_scaling=1.0) +/- Rune free-form eyeball. Asking user how far to push.

### ANCHOR #4 RUNE FREE-FORM EYEBALL = PASS (2026-06-01)
tools/_rune_freeform_gen.py (engine path: from_config -> generate_adapter -> hotswap -> greedy generate @ adapter_scaling=1.0=lora_alpha; NO xgrammar). (Fixed apply_chat_template BatchEncoding extraction first.)
Adaptered gens FULLY COHERENT/fluent over 64 tok, NO degeneracy: fn-name->"...is `rebase`."; return-str->'...returns the string **"RECOVERED"**...'; sum->clean ```python def add(a,b): return a+b```.
Adapter active (answers about internalized doc w/ doc absent; emits code). Wrong facts (rebase/RECOVERED) = known ctx-feature recall RESIDUAL, NOT a coherence problem.
=> Rune's OWN stack generates coherently at the contract scale (closes anchor#3 single-forward->multi-token gap). Only open Q: does xgrammar STRUCTURED gen survive at alpha. If it breaks => isolated to structured-decode layer.
NEXT: xgrammar/MBPP smoke (_bench_entry, qwen_4b_d2l + Qwen3-4B, adapter_scaling=1.0=alpha, 10 phase0 tasks).

### ANCHOR #4 XGRAMMAR/MBPP SMOKE = STRONG POSITIVE (2026-06-01)
_bench_entry full Rune engine, qwen_4b_d2l + Qwen3-4B, adapter_scaling=1.0 (=lora_alpha=45.25), 10 phase0_iter tasks:
  RESULT: pass@1=0.7 (7/10). TRUNCATION/JSON-close warnings across all 10 tasks = 0. retry-exhaustion = 3 (the failures).
INTERPRETATION: The #50-feared break (xgrammar never closes JSON at alpha) DID NOT HAPPEN. Structured gen STABLE at
the contract scale (0 truncation, JSON closed every task) AND useful pass@1=0.7. The recall-vs-generation tension the
handoff braced for does NOT manifest for qwen_4b_d2l: same lora_alpha scale gives recall (anchors 1/3/4-baseline) AND
stable structured gen AND strong pass@1. (Run was slow ~45min due to retry-exhaustion on the 3 failing tasks; not a break.)
Prior base scal=0.0 (12:06 MLflow) = 0/10 -> large adapter delta, CONFIRMING now.
CONFIRMATION RE-RUN (user-approved, instrumented): added per-task code dump to _bench_entry; 3-task subset (mbpp/11,12,14)
at scal=1.0 vs scal=0.0 -> validate delta + SEE generated code (closure=real valid Python + correctness). Awaiting.

### ANCHOR #4 CONFIRMATION RE-RUN LANDED (2026-06-01) — baseline OK, specificity now decisive
3-task subset (mbpp/11,12,14), per-task code dump, contract scal=1.0 vs base scal=0.0:
  scal=1.0 (adapter): 1/3. mbpp/11 PASS clean 415ch; mbpp/12 FAIL clean but name `sorted_matrix`!=`sort_matrix`; mbpp/14 FAIL clean but `find_volume`!=`find_Volume`. -> all CLEAN TIGHT code, misses = name-casing.
  scal=0.0 (base): 1/3. mbpp/11 FAIL 2815ch DEGENERATE (rambling multi-fn, markdown fences, syntax error); mbpp/12 PASS 59ch terse correct `sort_matrix`; mbpp/14 FAIL 2172ch degenerate.
REINTERPRETATION (honest tempering of the 7/10 "strong positive"):
 - Baseline comparability CONFIRMED: base emits REAL code (not harness artifact) -> 0/10 is a FAIR base. Good.
 - Base failure mode = OUTPUT DEGENERATION (2-3k char rambling/non-closing). Adapter@contract = clean tight code every task.
 - So the adapter's VISIBLE benefit = generation DISCIPLINE / anti-degeneration. The full-slice 0->7 is plausibly "adapter
   suppresses base degeneration" -> may be a GENERIC prior (any adapter at this scale disciplines output), NOT episodic memory.
 - Subset TIED 1/3 each (small sample); adapter even MISSED names base got right (mbpp/12). Don't over-read 7/10 as memory.
 - DECISIVE GATE (goal 3): MISMATCH-ADAPTER arm on full 10 tasks. matched>mismatch => episodic; matched~=mismatch => generic discipline prior.
Handoff (D3) updated: TL;DR tempered, results +4d row, open-risks specificity rewritten, next-step 1 = mismatch arm (confirmation done).

### RESPONSE to reviewer "Confirmation Rerun Tempering" (16:53Z) — folded into D3 handoff
Accepted all three; handoff edited:
1. Goal 3 next-step -> THREE arms (matched/mismatch/zero) on full frozen 10, per-task code dumps, tagged by
   FAILURE CLASS (JSON/truncation/syntax/name-API/semantic). Added parallel lever: prompt/schema enforcement of
   the target function signature (current misses are name-contract errors, not memory) — cheap, independent of goal 4.
2. Goal 4 / next-step 2 -> training is now PURPOSE-GATED, not auto-next: 7/10 already shows warm-start is product-
   useful, so DECIDE after the goal-3 arms whether to train and FOR WHAT (specificity/holdout pass@1, facts/critique
   serialization), not merely "prove the contract works."
3. Reproducibility (reviewer's named weak point) -> added MLflow run IDs to Key paths: full matched 2ccd712ebf49 (7/10),
   base 8a98f9deb10a (0/10); subset matched 6c5063ec7f43, base 7486081b9d13; per-task code in run logs; seed=cfg.seed;
   caveat that decisive harnesses are local-only scratch -> commit minimal repro before final PR.
Phase-0 scoping (reviewer pt 3) already explicit in Goals (1-2 closed on slice; 3-5 open) + 4d results row.

### D3 SESSION 2 START (2026-06-01, fresh agent via subagent-driven-dev handoff)
Read D3 handoff + scratchpad tail + all reflections. State confirmed: anchors 1-3 PASS, anchor 4
xgrammar smoke 7/10 (matched) vs 0/10 (base), confirmation rerun tempered it -> base failures =
DEGENERATION, adapter benefit looks like generation DISCIPLINE which may be a GENERIC prior, not
episodic memory. DECISIVE GATE = goal 3 specificity: MISMATCH-adapter arm on full frozen 10 tasks.
Env: GPU 0/23GB free, RAM 11GB free. Both checkpoints present. matched 7/10 = MLflow 2ccd712ebf49,
base 0/10 = 8a98f9deb10a (can reuse or rerun for apples-to-apples).

PLAN (goal 3 specificity gate):
- MISMATCH construction = DERANGEMENT of task->adapter-trajectory mapping. Each task i's adapter is
  conditioned on a DIFFERENT task's description (deranged partner), generation still targets task i.
  Holds conditioning DISTRIBUTION constant (real MBPP descriptions, just permuted) -> isolates the
  task<->adapter BINDING. Stronger than constant/OOD mismatch.
- Interception point: rune.engine.graph.render_training_format_trajectory(task=...) -- called
  consistently for initial `code` action AND continuations; task arg always = ctx["task_description"]
  = state["task"][:1200] (all 10 tasks short, so = full description). Monkeypatch in a new local tool
  to swap task->partner via a precomputed derangement keyed on description.
- Run 3 arms on full 10 @ adapter_scaling=1.0, per-task code dump + FAILURE-CLASS tag
  (JSON/truncation / syntax / wrong-name-API / semantic): matched / mismatch / zero. Log to MLflow.
- VERDICT: matched > mismatch => episode-specific memory (issue #52 claim supported);
  matched ~= mismatch => 7/10 is a generic anti-degeneration prior, not memory.

### ADVISOR STEER (D3 session 2) -- probe-first + 3 traps + interpretation caveat
1. PROBE FIRST (minutes vs ~2hr gen run): for each of 10 tasks, gen matched + deranged-partner
   adapter; record (a) weight-space ||dA||/||A_matched||, (b) logprob of REFERENCE MBPP SOLUTION
   under matched/mismatch/zero. _pathab_rune.py-style scoring (already have). anchor1's +0.824 was on
   QA episodes; SHORT MBPP descriptions are a NEW conditioning set -> confirming they make
   distinguishable adapters is genuinely new + ~free. If ||dA|| tiny or solution-logprob margin ~0 ->
   matched~=mismatch on pass@1 is guaranteed; skip/sharpen the 2hr run.
2. THREE TRAPS (must-fix before any run):
   (a) RAISE on derangement lookup miss -- NO identity fallback (silent identity = matched
       contamination biasing toward matched~=mismatch). Assert key found.
   (b) SEED keyed to GENERATION-task index, NOT adapter. matched vs mismatch must differ in ONLY the
       adapter. Verify run_benchmark/cfg.seed.
   (c) APPLES-TO-APPLES: all 3 arms through the ONE new harness. Do NOT compare new mismatch vs old
       run IDs (2ccd712ebf49/8a98f9deb10a, different code path). Reproduce matched via IDENTITY
       permutation through new tool on 2-3 tasks first.
3. PRE-REGISTERED INTERPRETATION (scratchpad verdict was too strong): task IS in the prompt here, so
   adapter task-conditioning is partially REDUNDANT. matched~=mismatch reads NARROWLY = "adapter
   conditioning adds no task-specific generation utility beyond discipline, in a SINGLE-TURN setting
   where the task is already in the prompt." NOT a refutation of #52's episodic-memory bet (multi-turn
   feedback goal/tried/critique NOT in prompt). Write this BEFORE running.
4. Continuation loop conditions deranged adapter on partner-task + REAL accumulated code -> not "pure
   partner" once cont fires. First `code` action (empty current_code) = cleanest signal. Expect
   name-casing binary noise in pass@1 that the logprob probe sidesteps.
PLAN REVISED: build probe (reuse _pathab_rune scoring) -> run probe -> decide on gen run.

### SPECIFICITY PROBE RESULT (2026-06-01) — goal 3 answered at the LOGPROB level (cheap, pre-gen-run)
tools/_specificity_probe.py (bf16 4B + qwen_4b_d2l, scaling=45.25, use_bias=True, 36 layers).
DERANGEMENT i->(i+1)%10 (no fixed points, asserted). Adapter conditioning = EXACT engine surface
(render_training_format_trajectory). AUDIT artifact confirmed: matched=mbpp/11 traj, mismatch=mbpp/12
traj (gen still targets mbpp/11). Reference solutions held constant across arms.

(A) WEIGHT-SPACE ||dA||/||A|| matched-vs-deranged: per-task 0.39-0.47, MEAN=0.4313. -> adapters are
    GENUINELY task-specific in weight space (~43% rel diff). Necessary condition MET: there IS signal.

(B) REF-SOLUTION LOGPROB margins (matched - X), n=10:
  regime=PRESENT (task IN prompt, faithful to bench): m-mismatch=-0.1834 frac(>0)=0.00 (ALL 10 neg);
    m-zero=+1.2205. -> adapter >> base (discipline), but matched <= mismatch consistently.
  regime=ABSENT (task NOT in prompt, NIAH-style): m-mismatch=+1.1702 frac(>0)=1.00 (ALL 10 pos);
    m-zero=+2.0129. -> adapter STRONGLY task-specific when task is hidden.

INTERPRETATION (coherent, pre-registered read realized):
1. The adapter DOES encode task-specific memory — ABSENT regime: matched beats deranged on ALL 10
   tasks (+1.17), beats zero (+2.01). qwen_4b_d2l recall mechanism is REAL through Rune even from
   SHORT MBPP descriptions (new conditioning set vs anchor-1 QA episodes). POSITIVE for #52 mechanism.
2. In the FAITHFUL bench setting (task in prompt), that specificity is REDUNDANT with the visible
   prompt AND matched is slightly WORSE than mismatch on the ref-solution logprob (all 10 negative,
   small -0.02..-0.66). Adapter's net bench benefit = generic DISCIPLINE (m-zero=+1.22, adapter>>base),
   NOT matched-specific utility. -> the 7/10 pass@1 lift is plausibly a GENERIC anti-degeneration prior.
3. CAVEAT on present-negative: likely "matched commits to its OWN internalized task-i solution" (poss.
   different phrasing/naming than my arbitrary reference) -> lowers logprob of MY ref string while a
   deranged adapter imposes no task-i solution. So logprob-on-ref does NOT fully predict FUNCTIONAL
   pass@1; matched could still differ functionally (cf. confirmation-rerun: matched made clean code but
   missed fn-name CASING sorted_matrix!=sort_matrix == "committing to own naming").
4. This does NOT refute #52's episodic-memory bet for HIDDEN multi-turn feedback/tried/critique facts —
   the ABSENT regime is direct evidence FOR the recall mechanism. It scopes the NEGATIVE narrowly:
   single-turn, task-already-in-prompt MBPP gives the adapter no ADDITIVE task utility.

DECISION POINT: probe gives a clean RANKING-level answer (ranking = primary per reflections). The 2hr
3-arm generation run would mainly CONFIRM matched~=mismatch pass@1 (predicted) BUT pt-3 caveat means it
could still surface a functional diff the logprob misses. -> taking to advisor/user before the spend.

### SPECIFICITY PROBE v2 — SIGNATURE/BODY SPLIT (2026-06-01) — sharpens goal-3 verdict
Re-ran probe with the ref-solution span split into the `def <name>(...)` SIGNATURE line vs BODY
(advisor's name-contract discriminator). (A) weight ΔA unchanged (mean 0.4313).
(B) margins (matched - X), n=10, per span:
  PRESENT (task IN prompt):  full m-mm=-0.18/mz=+1.22 | sig m-mm=-0.28 frac0.30 mz=+2.26 | body m-mm=-0.09 frac0.10 mz=+0.64
  ABSENT  (task hidden):     full m-mm=+1.17/mz=+2.01 | sig m-mm=+3.84 frac1.00 mz=+5.25 | body m-mm=+0.14 frac0.60 mz=+0.84

SHARPENED MECHANISM (split is decisive):
1. The adapter's task-specific recall is DOMINATED by the function NAME/SIGNATURE: ABSENT sig
   m-mismatch=+3.84 (ALL 10), vs ABSENT body m-mismatch=+0.14 (weak). The strongly recallable content
   is the signature (name+args), NOT the solution algorithm. -> shallow NAME-recall >> deep
   solution-shaped memory. (NB: body mean is diluted by generic Python tokens return/for/if/ws, so
   body-low is partly by construction; but the contrast sig+3.84 vs body+0.14 is stark.)
2. CONFLICT w/ ADVISOR side-note: advisor predicted absent +1.17 "can't be name-only -> broadly
   elevated body = real solution-shaped memory." The SPLIT REFUTES that: signal is name-dominated.
   (Surfacing to advisor per protocol — primary evidence contradicts the prediction.)
3. PRESENT sig m-mismatch=-0.28 frac0.30: matched assigns LOWER logprob to the CORRECT signature than
   a deranged adapter on 7/10 tasks -> matched memory mildly FIGHTS the in-prompt name. Directly
   explains the confirmation-rerun casing misses (sorted_matrix!=sort_matrix, find_volume!=find_Volume):
   adapter recalls the name with imperfect casing and at contract scale can OVERRIDE the correct prompt name.

GOAL-3 VERDICT (well-supported, ranking-level — primary metric per reflections):
- 7/10 vs 0/10 base = generic DISCIPLINE / anti-degeneration (m-zero positive everywhere, adapter>>base),
  NOT additive matched episodic memory.
- The adapter carries task-specific info but it's dominated by surface SIGNATURE recall, which is
  (a) REDUNDANT with the in-prompt task and (b) can HURT via name-casing override.
- NOT a refutation of #52 for HIDDEN feedback/tried/critique facts; but tempers even "memory exists" ->
  what's recalled HERE is shallow (name) > deep (algorithm).
ACTIONABLE (converges w/ reviewer): (i) cheap independent lever = prompt/schema ENFORCEMENT of target
fn signature (kills the casing override) — try before/independent of training; (ii) if training (goal 4),
objective must target DEEP solution/algorithm recall, not surface name binding.
DECISION: 2hr 3-arm generation run NOT run (advisor+reviewer agree: ranking probe is the primary answer;
n=10 pass@1 too noisy to resolve small matched-vs-mismatch gap). Span-split closed the open functional
question at ranking level. Taking the advisor-conflict + verdict to advisor, then user.

### PROBE v3 — PER-TASK SIG/BODY RESOLVER (2026-06-01) — body-depth question RESOLVED
Per-task ABSENT body m-mismatch (short-body tasks = least generic-token dilution = cleanest read):
  sort_matrix `sorted(M,key=sum)` +0.4265 (key=sum recalled); square_perimeter `4*a` -0.3088 (trivial
  formula, nothing task-specific); test_duplicate `len!=len(set)` +0.0511 (~flat). Larger bodies:
  text_lowercase_underscore(regex) +0.62, find_Max_Num +0.60; negatives find_Volume -0.13, is_woodall
  -0.03. RANGE -0.31..+0.62, mean +0.14, frac 0.60.
RESOLVED CLAIM (advisor middle, grounded — REPLACES earlier "shallow name>deep algo" overreach):
  fn NAME/SIGNATURE = most task-DISCRIMINABLE recalled content (ABSENT sig +3.84 ALL 10). BODY/ALGORITHM
  recall = REAL-BUT-WEAK-AND-INCONSISTENT: present where body has a discriminative token (sort_matrix
  key=sum, regex, find_Max_Num), absent/negative for trivial-formula bodies (square_perimeter, find_Volume).
  NOT zero (refutes pure name-recall) but NOT robust deep memory. CONFOUND ack: body margin cancels generic
  Python tokens shared by both real-task arms -> under-counts; per-task short-body read corrects dilution.
GOAL-4 INPUT: algorithm recall "real but faint & uneven" -> warm-start would DEEPEN a faint existing signal
  (between advisor poles, leaning "something to deepen"); NOT from-scratch (name binding strong), NOT a sure
  thing (body weak). Reviewer adds: track sig vs body spans separately in training/selection; consider
  MASKING the signature so selection rewards solution-state not labels.
GOAL-3 VERDICT (final, ranking-level, durable):
  (1) PRESENT full -0.18 (matched<=mismatch w/ task in prompt) => 7/10 = generic DISCIPLINE, NOT additive
      episodic memory. (2) PRESENT sig -0.28/frac0.30 => matched mildly FIGHTS in-prompt name (casing misses).
  (3) ABSENT positivity (sig +3.84 all10; body weak+uneven) => task-specific info EXISTS (pro-#52 for HIDDEN
      facts), name-dominated, algorithm-depth real-but-faint.
NEXT (cheap, converges advisor+reviewer): (A) signature/schema ENFORCEMENT lever (kill casing override),
  independent of training; (B) HIDDEN-FACT utility probe (hide a fact unsolvable by name-recall: required
  branch cond / prior failed approach / critique constraint / state invariant) = the real #52 memory test.
  2hr generation run SKIPPED (ranking primary; n=10 too noisy). Reporting to user for direction.

### LEVER (A) SIGNATURE ENFORCEMENT — plan (user chose "A then B") (2026-06-01)
Data flow check: engine sees ONLY task.description (make_initial_state(task.description,budget));
entry_point NOT threaded. Output is xgrammar JSON-schema constrained -> can't cleanly regex the fn name
INSIDE the code string. So enforcement = PROMPT-level. Cleanest impl = NO src change, NO new tool:
augment each task DESCRIPTION with an explicit exact-name instruction -> run EXISTING _bench_entry.py on
the augmented tasks file = apples-to-apples w/ the 7/10 baseline (same code path, same scaling=1.0).
Augmented description flows to BOTH adapter-conditioning (render_training_format_trajectory) AND the
gen prompt (realistic "signature is part of the task" setup). MEASURES: does explicit prompting beat the
adapter's name-casing override at contract scale? PASS = previously name-failing tasks (mbpp/12 sort_matrix,
mbpp/14 find_Volume) now pass -> pass@1 > 7/10. If still 7/10 -> adapter override beats prompt -> need
schema-level enforcement (harder) — informative either way. Baseline ref = 7/10 (2ccd712ebf49, same path).

### LEVER (B) ORIENTATION (while A runs) — hidden-fact utility probe
Corpus: /tmp/rune-corpus/external_codereview.{val,test}.clean.jsonl. Episode infra (tools/d2l_control/
episodes.py _build_episode_from_row): doc=activation_text (Task header w/ file + Current Code=pre_code +
Review Feedback); post_code=accepted revision. Facets present: goal(=feedback/critique), file, diff(hunk).
_pathab_rune.py ALREADY scores goal/file/diff matched-vs-(next-episode)mismatch on the CORRECTED Rune
contract at contract scale. NEW for (B): (1) a HARD negative = FEEDBACK-SWAP (same local code, different
feedback) instead of generic next-episode mismatch -> tests if recall is FEEDBACK-bound vs code-echo (D1
found diff collapses under feedback-swap on Sakana stack; re-test through FIXED Rune contract); (2) an
ACTION-relevant hidden fact unsolvable by name (critique constraint / accepted-vs-rejected preference),
doc ABSENT from prompt. Reviewer (B) guidance: low label-leakage, high action-relevance; track sig/body
separately; the avoid facet needs episodes w/ rejected attempt + critique + accepted (external_codereview
single row = pre_code rejected / feedback critique / post_code accepted -> one-step avoid testable).
DEFER building until A reported (user: "A then B").

### LEVER (A) — validity gate + interpretation reframe (advisor, mid-run)
GATE CHECK (CPU, _is_simple_task on orig vs augmented, all 10): CHANGED=0 -> PATH-CLEAN. Augmentation
only changes prompt/adapter-conditioning, NOT the engine path (no decompose-vs-_main flip). Comparison is
prompt-confounded only, not path-confounded. Good.
INTERPRETATION TRAP (defused): the 7/10 BASELINE per-task code was NOT saved (only the 3-task subset
anchor4_sub_s0.log). So baseline failures = {mbpp/12, mbpp/14, ONE UNKNOWN}. CANNOT detect regressions on
the other 7, CANNOT id the 3rd baseline failure. => Do NOT read lever-A as an aggregate 7->N delta (n=10
noisy + #52-orthogonal). CLEAN claim from THIS single run = failure-class TRANSITION on KNOWN tasks: do
mbpp/12 (sort_matrix) and mbpp/14 (find_Volume) now emit the CORRECT name and PASS? + overall code
well-formedness. Decision: take the narrow 12/14 read; do NOT rerun baseline (poor ROI); then -> lever B
(the real #52 advance). If 12/14 flip->pass: "lever A validated on the name-casing failure mode." If not:
"soft prompt pressure loses to the adapter's recalled signature" -> schema enforcement = separate harder
lever (reviewer branch). Both clean without a baseline rerun.

### LEVER (A) RESULT = name-casing failure mode FIXED (2026-06-01) — MLflow 895ccda7fc12473d889a002e2e42fabf
Signature enforcement (augmented descriptions, path-clean, scaling=1.0, same harness/tasks): pass@1=0.9 (9/10).
PER-TASK (all dumped): 11 ok, 12 ok, 14 ok, 16 ok, 17 ok, 18 ok, 19 ok, 20 ok, 56 ok, 57 FAIL.
CLEAN CLAIM (narrow defensible read, advisor+reviewer-scoped):
- mbpp/12 sort_matrix: baseline FAIL (name sorted_matrix) -> NOW PASS, correct `def sort_matrix(matrix)`. FLIPPED.
- mbpp/14 find_Volume: baseline FAIL (name find_volume) -> NOW PASS, correct `def find_Volume(length,height,width)`. FLIPPED.
=> Explicit signature text OVERCOMES the adapter name-casing override. Lever A VALIDATED on the name-contract mode.
  (Confirms PRESENT sig -0.28 was real & FIXABLE; soft prompt BEAT the recalled signature here.)
- ONLY remaining fail mbpp/57 find_Max_Num = SEMANTIC/type (name CORRECT; returns str '321' != int 321). Different
  failure class; signature enforcement not expected to fix. Next lever for THAT = return-type contract, not name.
CAVEATS: aggregate 9/10 vs 7/10 = SUGGESTIVE not clean controlled delta (baseline per-task unsaved; 3rd baseline fail
unknown). No VISIBLE regression (8/9 non-57 pass) -> reviewer's "directive text regresses others" worry didn't manifest.
Slow (~50min) despite 1 failure -> orthogonal retry-exhaustion bug. NOT a broad pass@1 claim (frozen 10).
=> LEVER A DONE. Moving to LEVER B (hidden-fact utility probe = the real #52 advance).

### LEVER (B) STEP 1 = IN-CONTEXT CEILING GATE = FAIL/WEAK (2026-06-01) — avoid task ill-posed on this corpus
tools/_avoid_ceiling_probe.py (base 4B bf16 ONLY, no hypernet). 30 avoid episodes = external_codereview
replace-hunks (rejected pre-side / accepted post-side / feedback critique). Neutral scaffold = accepted-file
lines before hunk (leaks neither rejected region nor feedback). DiD cancels acc/rej intrinsic-likelihood.
RESULT: mean pref_nocrit=+0.170, mean pref_crit=+0.147, mean gate_DiD=-0.0229, frac(DiD>0)=0.47.
=> Critique IN THE PROMPT does NOT shift accepted-over-rejected preference (coin flip). One-step avoid task
on this corpus is ILL-POSED: even with ORACLE in-prompt critique the model can't pick the accepted edit ->
a flat ADAPTER result would be UNINTERPRETABLE (cannot separate "fact unstorable" vs "task unsolvable").
PROBE-FIRST PAYOFF: spared building the adapter/feedback-swap apparatus on a dead task.
DIAGNOSIS: DiD BIMODAL (strong + some: OpenHands_9666 +1.79, GDevelop +0.74/+0.47; strong - others:
OpenHands_11127 -1.92, _12398 -1.54). Real GitHub review feedback often non-directive ("I don't understand
why...", "could have been worth naming...") and/or targets a DIFFERENT hunk than difflib's first-replace ->
critique doesn't determine THIS edit. Confirms LONG-STANDING reviewer caveat: external_codereview is
single-turn and LACKS action-relevant, critique-DETERMINED failure-bearing episodes.
FORK (-> advisor): (a) DECISIVE -> corpus can't support a clean avoid-utility test; redirect #52 avoid facet
to MINING multi-turn engine trajectories (deferred goal-4 pacing item); OR (b) try a CLEANER subset
(rename-style / directive-critique only) before concluding. Note: critique-RECALL (mechanism) is SEPARATE &
already +0.82 in _pathab (goal facet); untested increment = feedback-SWAP negative — but recall!=utility, and
utility is what just proved untestable here.

### LEVER (B) STEP 1b = SINGLE-HUNK CEILING (a-priori filter, advisor) = BORDERLINE/WEAK (2026-06-01)
Re-ran ceiling on EXACTLY-ONE-REPLACE-HUNK diffs only (structural, leakage-free, chosen blind to DiD sign).
RESULT (n=30 single-hunk): mean pref_nocrit=+0.388, mean pref_crit=+0.557, mean gate_DiD=+0.169, frac(DiD>0)=0.53.
vs first cut (mixed hunks): DiD -0.0229 -> +0.169. => WRONG-HUNK CONFOUND WAS REAL (advisor right); removing it
moves the mean clearly positive. Avoid task is NOT dead. BUT still WEAK: frac 0.53 (<0.6 threshold), bimodal.
PATTERN in residual negatives: concentrated in HIGH-base-pref episodes (OpenHands_11127 pref_nocrit=+3.92 ->
crit +2.00 DiD-1.92; taipy_1813 +2.6/+2.4 -> DiD-1.09/-1.13). Base ALREADY strongly prefers accepted there
(accepted edit is the "obvious" continuation) -> CEILING EFFECT, critique has no headroom, only perturbs.
Strong positives where base is neutral/wrong: taipy_2289 -1.29->+1.40 DiD+2.68; OpenHands_9666 +1.79;
pocket-casts +1.76. => critique DOES determine the edit on the low-base-pref subset.
INTERPRETATION (borderline, -> advisor): single-hunk CORRECTED the confound (mean +0.17, predicted direction)
but ceiling is WEAKLY positive + noisy. Even with ORACLE in-prompt critique the avg accepted-pref lift is small;
an ADAPTER (weaker delivery than prompt) would face an even smaller effect vs noise -> full apparatus likely
UNDERPOWERED here. Also high base pref = low headroom (accepted often obvious regardless of critique). Leans:
corpus is MARGINAL for a clean avoid-utility test -> mining purpose-built failure-bearing trajectories (critique
= binding constraint by construction) is the cleaner path. NOT adding a 2nd filter (advisor: single-hunk alone;
stacking = tuning-knob trap). Decision (build apparatus vs redirect-to-mining) -> advisor.

### LEVER (B) RESOLUTION (advisor) = STOP utility facet; corpus quality walls it off (2026-06-01)
EXACT headroom split (base pref measured in NO-critique cond = outcome-independent of DiD sign):
  ALL n=30: mean DiD +0.169 frac 0.53.
  base pref_nocrit <=0 (HEADROOM, model uncertain/wrong) n=10: mean DiD +0.519 frac 0.60.
  base pref_nocrit >0  (model already leans accepted)    n=20: mean DiD -0.006 frac 0.50.
  base pref_nocrit >=1.5 (CEILING, accepted obvious)     n=3:  mean DiD -1.381 frac 0.00.
=> Avoid task IS well-posed WHERE feedback is directive AND there is headroom (+0.52); dead where the
accepted edit is already the obvious continuation (critique only perturbs). Honest conclusion is SHARPER
than "ill-posed": external_codereview MIXES directive + non-directive feedback, and the clean-signal subset
CANNOT be isolated STRUCTURALLY (single-hunk doesn't capture directiveness) — only by CONTENT filtering that
would LEAK the answer into a memory test. So this corpus cannot yield a CLEAN avoid-UTILITY number.
GUARD (honored): the headroom split is DIAGNOSTIC/suggestive ONLY. Pre-registered bar = frac>=0.6 on the
a-priori single-hunk set; got 0.53 -> MISSED. Do NOT retroactively declare pass / do NOT build apparatus on
the headroom subset (post-hoc-selection trap).
WHY NOT BUILD: oracle in-prompt delivery = +0.17 mean; adapter is a LOSSIER channel for the same fact ->
matched-vs-swap <= that vs +-1-2 nat/episode noise -> underpowered at n=30 + averages two populations
(directive/not). DECISION: STOP lever-B utility facet. Redirect-to-mining now EARNED (not assumed): mined
multi-turn trajectories have the critique as the binding constraint BY CONSTRUCTION.
SESSION ARC COMPLETE + HONEST: goal 3 answered (discipline not memory; recall exists, name-dominated);
lever A validated (name-casing fixed, 9/10); lever B utility walled off by data quality (DIAGNOSED, not
assumed). USER-PRIORITIES FORK (report, don't auto-continue):
 (i) MINE purpose-built failure-bearing trajectories -> clean avoid-utility test (proper #52 path, real invest).
 (ii) CHEAP+CLEAN on this corpus: critique-RECALL feedback-swap (is _pathab's +0.82 goal recall feedback-bound
      or code-echo, a la D1 diff collapse). Reuses _pathab, no ceiling dependence. BUT = mechanism; recall!=utility.
 (iii) PIVOT: mbpp/57 return-type lever, or goal-4 warm-start training.

### NEXT INVESTMENT (user-chosen) = MINE FAILURE-BEARING TRAJECTORIES (2026-06-01)
GOAL: build a corpus where the avoid-utility test is well-posed BY CONSTRUCTION (critique = binding constraint),
fixing what external_codereview lacks (ceiling-gate showed mixed directive/non-directive feedback).
INFRA (exists): src/rune/mining/{miner.py,session_log.py}. Session schema v2 per step: {step,action,target,
trajectory,prompt,output,feedback{exit_code,stderr}}. miner.extract_trajectories -> per-ACTION SFT shards
(for distillation) w/ STaR filter (failed run -> keep only diagnose steps of RECOVERED subtasks). bench
run_benchmark(sessions_dir=...) -> write_session per task.
AVOID EPISODE (causally aligned) = within one session, for a subtask target: a code/repair step with
feedback.exit_code!=0 (FAILED attempt; output=failed code; stderr=ERROR=the critique) THEN a later repair
step same target with exit_code==0 (ACCEPTED fix; output=passing code). rejected=failed output, accepted=
repair output, critique=stderr/diagnosis. The error CAUSED the fix + fix VERIFIED to pass -> directive by
construction (solves the directiveness problem that killed external_codereview).
PLAN: (1) GENERATE sessions: bench w/ sessions_dir on a task set that fails-then-fixes. (2) EXTRACT avoid
episodes (NEW extractor, distinct from SFT-shard miner). (3) CEILING-GATE the mined episodes (same gate;
should CLEAR by construction = the validation that mined > external_codereview). (4) if clear -> adapter
feedback-swap UTILITY probe.
RISKS (-> advisor before GPU spend): (a) RECOVERY RATE — my 7/10,9/10 runs did NOT set sessions_dir (no
existing minable sessions) AND their failures EXHAUSTED retries (didn't recover); MBPP-easy at this scale
mostly passes first-try or never -> recovery episodes may be SPARSE. Best source = tasks that PASS but only
after a failed attempt (intermediate fail->repair in the retry loop). (b) TASK SET / COST — full MBPP val
(257) to harvest recoveries = hours (retry-exhaustion). Targeted harder subset? (c) does the engine's retry
loop even emit distinct failed-code + repair StepRecords, or collapse them? Need to verify session contents.

### MINING — STRUCTURAL CHECKS 1-2 (cold, pre-GPU, advisor) — reshapes the plan
CHECK 1 (WIN): repair shard record (/tmp/smoke_shards/repair_smoke.jsonl) ALREADY co-locates the avoid triple
in `trajectory`: DIAGNOSIS:<critique> + YOUR CURRENT CODE:<failed/rejected attempt> ; completion=<accepted fix>.
Current engine render = "## Task / ## Current Code <failed> / ## Review Feedback <fix_guidance|error>". Extractor
nearly FREE. (Smoke is old prompt-format but schema_v2; structure confirmed.)
CHECK 2 (DECISIVE, reshapes plan): mid-loop exec = run_in_sandbox(strip_self_tests(code)) with NO held-out tests.
strip_self_tests removes MODULE-LEVEL tests only (keeps in-body asserts); on syntax-err returns original.
parse.py:117 passed = exit_code==0 = "code RUNS". => failure(exit_code!=0) fires on SYNTAX / IMPORT / NameError /
load-time raise — NOT on wrong-but-runnable LOGIC. A clean correct function def -> runs -> passes internally ->
NO repair. So MINED avoid episodes on MBPP = STRUCTURAL-error->fix (cf smoke "remove extra indentation"), which are:
 (+) causally aligned (error caused fix, fix verified to run);
 (-) likely SPARSE on easy MBPP (adapter discipline -> clean defs -> few syntax errs -> few episodes);
 (-) LESS aligned w/ #52 "tried-approach-&-why-it-failed" (that is SEMANTIC/logic, not typos).
FORK (-> advisor): (i) proceed to check-3 small run, measure structural-err YIELD, accept syntax-avoid episodes;
(ii) INSTRUMENT mid-loop SEMANTIC feedback — MBPP description carries an example assert (`>>> assert fn(...)==...`);
run impl+that assert mid-loop -> SEMANTIC failure->repair episodes (far more #52-relevant) — small harness change;
(iii) reconsider task set / synthetic logic-failure injection.

### MINING PLAN FINALIZED (advisor) — option (ii) semantic-signal instrumentation, yield-first
RULE OUT (i) by REASONING (no run): syntax-error episodes give a FALSE GREEN on the ceiling (given
"IndentationError L5", dedented form wins regardless of memory) + measure string-edit storage, not #52
semantic memory. Sharper: function bodies don't exec at load -> ONLY syntax/import/module-raise can fire ->
MBPP harvest = pure syntax-repair = vacuous for #52. Don't even measure its count.
DO (ii) = make diagnose->repair fire on SEMANTIC signal (faithful to user's ask). GUARDRAILS:
- MINING-ONLY monkeypatch (no src/rune change; product engine untouched; can't leak to pass@1).
- mid-loop oracle = the VISIBLE example assert from the DESCRIPTION (`>>> assert fn(...)==...`); NOT held-out
  test_code (= leakage). Weak single-case oracle (wrong-but-passes-example won't fire) — fine for YIELD probe,
  don't claim coverage.
- monkeypatch rune.engine.graph.run_in_sandbox -> append current task's example assert to the stripped code
  before exec; set per-task assert global (run_benchmark is sequential per task -> safe).
MERGED PROBE (= check 3): instrument + run 10 frozen tasks once w/ sessions_dir; MEASURE YIELD: (a) fail->
repair->pass chains, (b) diagnoses semantic?, (c) VALID episodes (same target; failed output present; passing
repair present; critique non-empty; repair changes failed region). Build extractor+apparatus ONLY after nonzero
valid yield (shape depends on real records). PRE-REGISTERED SPARSE BRANCH (expect it; sig-enforced 9/10 = most
pass first try): if near-zero, do NOT scale to 257 -> (iii-a) harder multi-step tasks [scope call -> user] OR
(iii-b) sample N candidates/task @ temperature, pair test-failing w/ test-passing, critique=the failure
(higher yield, no engine-recovery dependence; NOT a true multi-turn trajectory — flag it).

### MINING YIELD PROBE RESULT (2026-06-01) — nonzero chains BUT wrong KIND (still structural)
tools/_mine_semantic_sessions.py (semantic instrumentation: append visible example assert to mid-loop exec),
10 frozen tasks, sessions_dir=/tmp/mine_sem_sessions. YIELD: 10 sessions, 7 fail->repair->pass chains, 7 VALID
(same target, failed+passing outputs present, non-empty critique, changed region). VERDICT-by-count = green.
BUT FAILURE-TYPE CHECK (advisor "are diagnoses semantic?"): stderr classes across failed steps =
SyntaxError x7 + NameError x5, ZERO AssertionError. => The injected assert almost NEVER adjudicated semantics —
the model's code fails on SYNTAX (unterminated string) / NameError ('success' not defined) BEFORE the assert runs.
So the 7 "valid" episodes are STILL STRUCTURAL-error repairs, NOT semantic-logic repairs. Instrumentation (ii)
did NOT rescue semantic episodes on easy MBPP: the model either emits clean-correct code (passes, assert passes
too) or syntactically-broken code (fails early) — little "syntactically-clean-but-semantically-wrong" middle that
an AssertionError would catch. CONFIRMS the advisor/reviewer worry at a deeper level: MBPP mining yields
structural-repair memory (vacuous for #52) EVEN WITH the assert oracle.
=> PRE-REGISTERED (iii) FORK is now forced (scope decision -> user):
 (iii-a) HARDER/multi-step tasks where the model produces clean-but-WRONG logic (assert fires as AssertionError).
 (iii-b) TEMP-SAMPLE N candidates/task, FILTER to syntactically-valid, find one that FAILS the assert (semantic
   reject) + one that PASSES (accept); critique=assertion failure. Sidesteps syntax-preemption + engine recovery —
   but needs clean-but-wrong candidates to EXIST (rare on easy MBPP -> may also need harder tasks). NOT a true
   multi-turn trajectory.
DURABLE: sessions at /tmp/mine_sem_sessions (structural episodes; honestly labeled, NOT semantic-avoid evidence).

### NEXT (user-chosen) = TEMP-SAMPLED CANDIDATE PAIRS (iii-b) (2026-06-01)
GOAL: per task, sample N candidates @ temperature, FILTER to syntactically-valid, find one that FAILS the
example assert (semantic REJECT, AssertionError not Syntax/Name) + one that PASSES (ACCEPT); critique=assertion
failure. Build clean (reject, critique, accept) triples for the avoid-UTILITY test. Sidesteps syntax-preemption
+ engine-recovery. NOT a multi-turn trajectory (flag). KNOWN RISK: clean-but-wrong may be rare on easy MBPP
(model gets easy tasks right or breaks syntax) -> may need harder tasks too.
PROPOSED PROBE-FIRST DESIGN (-> advisor before GPU): candidate-sampling YIELD probe on the 10 frozen tasks,
classify each candidate (ast-valid? assert ec? AssertionError vs Syntax/Name), measure per-task yield of
>=1 clean-semantic-fail AND >=1 pass. OPEN: candidate SOURCE = base (diverse but degenerates->syntax waste)
vs adapter@contract (cleaner->more syntactically-valid, but more CORRECT->fewer wrong). N, temperature.

### ADVISOR RESHAPE (temp-sample) — held-out-as-labeling-oracle + redundancy trap (2026-06-01)
TRAP: critique = VISIBLE example-assert failure -> critique recoverable from prompt -> avoid-utility test
collapses to goal-3 PRESENT redundancy (matched~=all). Same wall lever B hit, relocated.
FIX: reject must PASS the visible example assert but FAIL a HIDDEN held-out case; critique = the HIDDEN failure.
Then adapter carries a fact the prompt doesn't -> matched-vs-swap can isolate avoid-memory.
LEAKAGE RULE (clarified): held-out tests OFFLINE to LABEL candidates + form the critique-that-becomes-adapter-
memory = LEGIT corpus construction. Leakage = held-out in engine mid-loop SIGNAL or in the SCORING/model PROMPT.
Never in model prompt; fine as labeling oracle.
MERGED ONE-RUN PROBE (yield + ceiling, build apparatus only if BOTH clear): per task sample N=16 cands @
adapter@contract temp~0.8-1.0; keep syntactically-valid; classify vs BOTH example assert AND held-out tests ->
{pass-both=ACCEPT; pass-example/FAIL-hidden=GOOD reject; fail-example=leaked-reject DISCARD}. Report (a) yield of
accept+good-reject pairs; (b) in-context CEILING on those pairs = accept-vs-reject DiD with HIDDEN-failure critique
in prompt vs not (base-only scoring, adapter off). PRE-REGISTER: pass-example/fail-hidden RARER than clean-wrong
-> frozen 10 likely NEAR-ZERO -> signal = need HARDER tasks (user scope call). Probe returns clean yes/no on "can
frozen 10 support this", not grind a number. Source=adapter@contract (don't tune; low yield=task difficulty).

### PR #53 COMMENT POSTED (2026-06-01, user-requested) — issuecomment-4596059919
Summarized: (1) findings so far [contract validated anchors1-3; goal3=discipline-not-memory, name-dominated;
lever A 9/10 name-casing fixed; lever B walled off by corpus quality; mining=structural-only on easy MBPP];
(2) doing now [temp-sample avoid-pair yield+ceiling probe, hidden-failure critique, held-out as offline oracle];
(3) what's needed to train hypernet + HPO [prereqs: contract DONE, FAILURE-BEARING corpus = the blocker,
collapse smoke; objective: contrastive + feedback-swap hard negs + MASK signature; success = matched>mismatch on
body/hidden-fact-utility + retention + gen-stability gates; HPO = small guarded sweep over safe knobs w/ per-facet
selection metric, NOT open Optuna; configs/issue52_recipe_mvc_4b.yaml; 2-5k pilot before full run].

### TEMP-SAMPLE AVOID PROBE RESULT (2026-06-01) — sparse yield + WEAK ceiling (tool says PASS, I disagree)
N=16 cands/task @ adapter@contract temp0.9. YIELD = 3/10 tasks with accept+good-reject pair (mbpp/16,20,56).
Good-reject (pass visible example, FAIL hidden held-out) counts: mbpp/16=14 (regex uppercase edge cases),
mbpp/20=1, mbpp/56=2; OTHER 7 tasks = 0 good-rejects (model passes all held-out or only leaked-rejects).
CEILING on the 3 pairs (base-only, hidden-failure critique in prompt vs not): per-task DiD +0.030/+0.207/+0.017,
mean +0.0847, frac 1.00. Tool VERDICT=PASS (mean>0 & frac>=0.6).
MY READ (cautious, consistency w/ lever B): tool PASS OVERSTATES it. +0.085 mean is SMALLER than the +0.17 I
called borderline-WEAK for external_codereview; 2/3 are ~flat (+0.03,+0.017); n=3 is far too small for a robust
ceiling. frac 1.00 reassuring but unreliable at n=3. => NOT a green light to build the apparatus.
REAL SIGNAL = confirms advisor pre-registration: easy MBPP yields too FEW hidden-failure pairs (3, dominated by
one regex task) -> need HARDER tasks. Build-vs-harder-tasks -> advisor calibration (I rejected +0.17; must be
consistent), then user (scope: harder tasks vs accept the cheap-probe arc as the session deliverable).

### CORRECTION (advisor) — reject on YIELD-SPARSITY + n=3, NOT on magnitude
DROP the +0.085-vs-+0.17 comparison: different constructions. external_codereview = verbose feedback @ frac 0.53
(direction ~random -> ill-posed). Here = terse hidden assert @ frac 1.00 (direction consistent). Decision axis =
DIRECTION-CONSISTENCY not magnitude -> on that axis this is BETTER, not weaker. BUT frac1.00 @ n=3 ~ 1-in-8 under
coin-flip = NOT significant. Honest: n=3 UNINFORMATIVE; neither tool-PASS nor "weaker-than-rejected" supported.
THE BLOCKER = YIELD, not ceiling: 3/10, good-rejects = mbpp/16 14-of-17 (+ mbpp/20=1, /56=2 flukes) = ONE task
with real "passes-example/fails-edge" structure. WON'T grow with more N: the 7 zero-yield tasks are TEST-DESIGN-
bound (hidden tests must probe a distinct case the visible example misses), not sample-count-bound. Don't spend
GPU on more N on frozen 10. Anchoring guard: reject because yield is the wall, NOT "to stay consistent w/ +0.17".
=> HARDER-TASKS signal cleanly confirmed. Cheap-probe arc (goal3 + leverA + leverB/mining feasibility) = complete
honest deliverable. FORK -> user (don't auto-continue): (a) harder task DISTRIBUTION [new non-frozen set + GPU;
"harder" = hidden tests add a DISTINCT case the model misses while passing the visible example (mbpp/16 structure
generalized), NOT raw difficulty that breaks syntax]; (b) BANK the arc as the session deliverable.

### BANKED (2026-06-01) — commit 8a17f8aa; moving to harder task distribution (user: "bank then move on")
Committed: tests/unit/test_engine_functional_parity.py (clean, 2 passed; full unit suite 314 passed) + D3 handoff.
GPU-scratch underscore tools stay LOCAL (16 ruff errors -> would break `ruff check .`; convention; handoff records
their paths/commands/MLflow IDs for repro). instructions/ is gitignored (local experimental log).
NEXT = HARDER TASK DISTRIBUTION (advisor spec): tasks whose HIDDEN tests add a DISTINCT case the model misses
while PASSING the visible example (mbpp/16 structure generalized) — a TEST-SUITE property, NOT raw difficulty
(raw difficulty -> syntax breaks -> structural-failure regression). PLAN: (1) cheap CPU pre-filter the larger MBPP
pool (mbpp_tasks.json 257 / mbpp_validation_tasks.json) to tasks with >=3 diverse asserts (single-assert can't
yield good-rejects); (2) extend _temp_sample_avoid_probe.py to take --tasks-file/--limit; harvest good-reject
pairs across ~40-60 filtered tasks (yield probe first); (3) report total yield + ceiling on POOLED pairs (n now
meaningful, unlike n=3). Build feedback-swap apparatus only if yield + ceiling clear.

### NORTH STAR (user) = TRAINED CHECKPOINT w/ GOOD METRICS in MLflow for pass@1 HPO (2026-06-01)
Pushed 8a17f8aa; PR comment posted. Milestones tracked (tasks #1-6). Infra READY: tools/_distill_entry.py
<yaml> [--max-steps]; config configs/issue52_recipe_mvc_4b.yaml (warm-start qwen_4b_d2l, base frozen 4bit,
contrastive=True feedback-swap on edit-local, eff_scaling=effective_scaling=lora_alpha [CORRECTED contract
confirmed L193/609], scaler_B collapse re-init + tripwires, MLflow + checkpoint_best + _save_checkpoint artifact).
GAPS to fix: config val_corpus_path="" -> no val/best-ckpt; val_steps=200 vs max_steps=300 -> 1 val point.
ADVISOR METRIC CORRECTION (critical): val_diff_agreement = matched-vs-BASE = the DISCIPLINE CONFOUND (goal-3).
The loop OPTIMIZES the right thing (contrastive hinge on edit-local w/ make_hard_negative) but SELECTS on the
wrong one. SUCCESS METRIC = matched > FEEDBACK-SWAP on edit-local gold logprob, improved over warm-start baseline
(IN-DISTRIBUTION external_codereview val). = the shelved lever-B feedback-binding test, now the training success
metric. REQUIRED: (1) BASELINE on warm-start ckpt BEFORE training; (2) periodic save_steps, pick final post-hoc on
matched-vs-swap+retention (don't trust checkpoint_best=val_diff_agreement); (3) RETENTION gate = specificity probe
ABSENT regime on trained ckpt (sig/QA recall not collapsed); (4) keep distributions distinct (train success=
in-dist; MBPP probe + pass@1 = generalization; log honestly, NO pass@1 promise); (5) SMOKE gate = scaler_B stable
+ hinge active-set NON-EMPTY + margin MOVING + memory fits (if margin flat -> objective/data broken, fix first).
STOP BAR (pilot): matched-vs-swap improved over baseline + no collapse + retention held -> log ckpt+metrics -> HPO.
NOT a converged production ckpt (user staged this).
PLAN: harvest finish -> BUILD _feedback_swap_eval.py (matched/swap/zero edit-local margin, same shaping as train)
-> baseline on warm-start -> fix config -> smoke 60 -> full 300 -> eval trained + retention -> compare -> HPO-ready.

### RESUME AFTER CRASH (2026-06-01, later) — baseline DONE, launching smoke-train
Crash hit right after the warm-start feedback-swap BASELINE completed (/tmp/fbswap_baseline.log, 20:54).
DURABLE BASELINE (warm-start ckpt qwen_4b_d2l/checkpoint-20000, 4bit, n=60 external_codereview.val):
  matched-SWAP = +0.0185  frac(>0)=0.48   <-- SUCCESS-METRIC BASELINE (≈ coin-flip: no feedback-binding yet)
  matched-zero = +0.0870  (discipline, secondary)
=> Training must push matched-SWAP meaningfully > +0.0185 (and frac > 0.5) to count as feedback-binding.
STATE: _feedback_swap_eval.py built (165L, reuses training internals); config fixed (val_corpus_path,
val_sample=40, val_steps=100, save_steps=100); avoid harvest done (5/35 pairs, complementary, seed at
/tmp/avoid_pairs_val.json). Config loads clean into DistillConfig. GPU idle, RAM ok (4bit frozen base, no offload).
NEXT (pre-registered) = SMOKE TRAIN 60 steps: tools/run_guarded.sh <log> tools/_distill_entry.py
--config configs/issue52_recipe_mvc_4b.yaml --max-steps 60. SMOKE GATE (advisor): scaler_B stable (no collapse,
watched in distill_metrics.jsonl), hinge active-set NON-EMPTY + margin_loss MOVING, memory fits. If margin flat
-> objective/data broken, fix before any long run. log_steps=10 -> 6 points; metrics ->
./checkpoints/issue52-recipe-4b/distill_metrics.jsonl + watchdog stdout log. Then: full 300 -> eval trained vs
baseline (_feedback_swap_eval.py --ckpt <trained>) + retention gate -> log ckpt+metrics to MLflow -> HPO.

### SMOKE-TRAIN ATTEMPT 1 = OOM @ step 0 (2026-06-01) -> FIXED seq 2048->768
First end-to-end training launch in the session OOM'd at hypernet_distill.py:343
(log_softmax(student_logits[:-1].float()) over a 2048-tok seq): 21.94/22.03 GiB in use,
tried +352MiB. NOT a leak — training retains base-over-CONTEXT forward graphs for the perceiver
grad (matched AND hard-neg contexts) + student/teacher/base answer passes + fp32 full-vocab
(~152k) log_softmax. Inference eval ran fine at 2048 (no_grad, no graphs); training can't.
ROOT CAUSE = config/intent mismatch: the contrastive loop's own comment says it "keeps seq=768,
peak ~ single-path" — i.e. the memory-bounding (detached-neg-pass + sequential matched/neg
backward) was DESIGNED for seq=768, but the config set max_seq_length=2048.
FIX: max_seq_length 2048->768 (p50 episode ctx+ans~=774 fits; longer truncate = designed regime;
rows that lose all edit-local tokens are skipped by the loop) + PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True (frag guard; only 85MiB was free at crash). Fallback if still tight:
gradient_checkpointing:true (loop supports it, line 163). Relaunched pid 431083.

### SMOKE-TRAIN 60 STEPS @ seq768 = HEALTHY, mechanical gate GO (2026-06-01)
margin_loss: 3.44 -> 2.07 -> 1.93 -> 1.28 -> 0.96 -> 0.99 (closed ~2.4 nats, plateau ~1.0; = the
in-train matched-minus-swap hinge on edit-local tokens moving the RIGHT way). scaler_B=1.0 all 6 pts
(NO collapse); scaler_A=1.0575 stable. bias_A/bias_B grad_l2 = 0.06-0.15 NON-ZERO (combine_lora bias
params getting gradient = c3a83217 tripwire OK, flowing not dying). gpu_peak 10.9-12.8GB. kl 16->12.
Final checkpoint.pt (842MB) saved at step 60 -> can eval held-out matched-swap directly.
SMOKE GATE (advisor: scaler_B stable + hinge active & margin moving + memory fits) = PASS on all.
Reflection failure-case ("active loss but no movement") NOT triggered. NEXT: eval matched-swap on the
60-step ckpt (held-out val) vs baseline +0.0185 -> advisor -> launch full 300 (saves+val @100/200/300).

### ROOT-CAUSE on smoke ckpt eval (2026-06-01) — scaler_B CLOBBERED at warm-start init -> FIXED
Eval of 60-step smoke ckpt (held-out val, n=60): matched-SWAP=+0.0019 (baseline +0.0185, NO movement,
frac 0.48->0.38) AND matched-zero=-8.81 (baseline +0.087 -> CRASHED, uniform ~-8.8 across ALL episodes).
Uniformity => not data-dependent training degradation. CPU state_dict diff smoke-vs-warm:
ALL params identical (scaler_A, bias_A/B, head match exactly) EXCEPT scaler_B.down_proj:
  warm  = mean|.|0.057, std 0.070 (LEARNED, structured)   smoke = 1.0, std 0.0 (uniform).
ROOT CAUSE = hypernet_distill.py:186 called reinit_scaler_b_nonzero(hypernet, scaler_b_init=1.0)
UNCONDITIONALLY after warm-start load — violating the fn's OWN docstring ("never when loading a trained
checkpoint; its learned scaler_B must be preserved"). Clobbered 0.057->1.0 = ~17x B-side inflation at
effective_scaling=lora_alpha(45.25) -> adapter massively over-perturbs -> matched-zero collapse; matched-
swap stays ~0 because BOTH matched+swap adapters equally over-scaled. Baseline (+0.087) proves 0.057 is sane.
=> NOT a broken objective. FIX: added scaler_b_is_collapsed() (all |scaler_B|<1e-4 = zero-init basin);
reinit ONLY when collapsed, else PRESERVE + log mean|.|. Safe both paths (from-scratch=0=collapsed->reinit;
warm-start=0.057->preserve). The margin "moving" 3.44->1.0 on TRAIN was the hinge closing by joint
suppression of the over-scaled adapters, not real binding — explains why held-out didn't move.
NEXT: re-run 60-step smoke with the fix -> scaler_B preserved -> eval matched-swap vs baseline +0.0185.

### FIXED 60-STEP SMOKE — VERDICT + held-out eval (2026-06-01)
scaler_B PRESERVED (ckpt mean|.|=0.05703 std=0.06992 == warm-start EXACTLY). MLflow: fixed run cbe9da363c
(margin0.968 kl1.994 loss3.759 diff_agr0.364 +ckpt artifact) vs broken 43cf8526ee (kl11.866 loss23.622
diff_agr0.020). Regression tests added (tests/unit/test_scaler_b_init.py: 4 pass) per reflection.
HELD-OUT matched-swap eval (n=60, external_codereview.val):
  matched-SWAP = +0.0687  frac0.65   (baseline +0.0185 frac0.48)  -> MOVED UP ~3.7x, frac 0.48->0.65
  matched-zero = +0.5021             (baseline +0.087; broken was -8.81)  -> SANE, adapter helps
IN-TRAIN margin (smoke60b): 1.007,0.936,1.041,1.025,1.03,0.968 = FLAT ~1.0 (noisy per-step, diff rows).
=> Held-out improved despite flat train-margin proxy. MILD POSITIVE smoke (modest, <0.1 nats, n=60, 60 steps).
NOT yet a green light for full Sakana-scale; user requested a facts+research+advisor deep-dive BEFORE the
long-duration run. Clobbered ckpt at /tmp/smoke_broken_scalerB1.pt (documented failed run, do NOT compare).

### DEEP-DIVE + RESEARCH + ADVISOR (2026-06-02) — long run NOT yet justified on unfiltered corpus
Facts dossier: docs/issue52-pretraining-facts-dossier-2026-06-02.md (facts §0-10 + research synthesis A-F).
ADVISOR REFRAME (decisive): the FIXED smoke is the #49 pattern in MINIATURE, not a green light.
Decompose: matched-zero=+0.50 (adapter vs none), matched-swap=+0.0687 (specific feedback). Only ~14% of
the lift is feedback-SPECIFIC; ~86% = generic boosting that fires WITH THE WRONG FEEDBACK = exactly the
#49 signature (m-zero rises via generic boost, specificity flat). 
RIGOR GAP: +0.0687 @ n=60 (per-ep -0.5..+1.2) ~ 1.4 SE from 0 -> aggregate move +0.0185->+0.0687 likely
NOISE. frac 0.48->0.65 (39/60 ~2.3 sigma) is the only suggestive bit. MUST compute PAIRED per-episode
delta (trained - baseline on SAME 60 val eps) + paired test before any go/no-go. (Both ckpts exist:
warm-start Sakana + checkpoints/issue52-recipe-4b/checkpoint.pt.)
THREE INDEPENDENT MEASURES CONVERGE on weak feedback->edit signal in external_codereview:
  (a) oracle IN-PROMPT ceiling (lever B) +0.17 frac0.53 (<0.6 bar); (b) in-train margin FLAT ~1.0;
  (c) held-out matched-swap +0.0687. Lever-B stratification already explained: edit-obvious -> critique
  can't move it (-1.38); base-uncertain -> feedback determines edit (+0.52). Research names it:
  FALSE NEGATIVES (Huynh: swapped fb that doesn't change the correct edit caps the margin + slows conv),
  single-exposure FT binding weak needs many paraphrases (Ovadia), SHORTCUT learning (Du: signature
  co-occurs w/ answer -> adapter binds NAME not body = our sig +3.84 vs body +0.14), Parametric Memory
  Law (loss power-law in RANK -> r=8 holds low-entropy signatures not high-entropy bodies).

GATING TESTS BEFORE LONG RUN (each ~1 smoke of GPU; partition data-vs-objective-vs-capacity):
 T0 (now, CPU+1 eval pair): PAIRED per-episode matched-swap delta trained-vs-baseline + paired t/sign test.
 T1 DIRECTIVE-FEEDBACK STRATIFICATION (DECISIVE, run first): filter train+eval to base-uncertain/
   feedback-determines-edit subset (the +0.52 stratum). Short-train on it; re-measure matched-swap.
   jump -> false negatives diluted signal -> long run trains on FILTERED data. flat -> corpus is the WALL,
   long run does NOT proceed on external_codereview.
 T2 ORACLE per-episode LoRA capacity probe at r=8 vs higher rank (reflection + Parametric Memory Law):
   oracle recovers body/fb facts but hypernet can't -> training/data problem; oracle ALSO fails at r=8 ->
   RAISE RANK before long run.
HYPOTHESES (H/test/discriminator):
 H1 corpus feedback->edit MI intrinsically low -> T1 oracle ceiling on directive subset; still flat=wall.
 H2 false negatives cap the gap -> label pairs by whether swapped fb changes gold edit; train/eval on
    true-negs only; signal rises = confirmed.
 H3 r=8 can't hold body recall -> T2 oracle r=8 vs higher; oracle fails @r8 = raise rank.
 H4 single-exposure binding too sparse -> augment K paraphrased feedbacks/episode, short-train; rises=lever.
GO/NO-GO ANCHORED TO CALIBRATION (not baseline): real recall ~+7.7, hidden-task specificity ~+1.17;
 matched-swap ~+0.1 is ~1-2% of real binding. Set the bar in THESE units + retention + gen-stability gates,
 WRITTEN DOWN before the run. LEAD the plan with T1 (most changes the answer).

### OPINION — next step given the trajectory-MEMORY product framing (2026-06-02, advisor-pressure-tested)
USER REFRAME (decisive): the adapter is EPISODIC TRAJECTORY MEMORY for cheap long-horizon continuation —
encode {GOAL (where we're going) + TRIED-AND-WHY-FAILED + LAST-ACTION/CONTINUATION (esp. resuming a
mid-cutoff generation)} into params so long tasks escape O(T^2) attention. This RECASTS the session:
feedback->edit specificity on external_codereview was a POLICY signal on a PROXY corpus, not the MEMORY
signal the product needs. So the right move is the "separate memory (recall episodic state) from policy
(frozen base emits next step conditioned on it)" principle from PR #53 — and STOP pushing the feedback-
swap contrastive long run on external_codereview (already judged unjustified; also a category error here).

MY OPINION = TWO PARALLEL TRACKS (not one; do NOT serialize):

TRACK A (cheap, now): CONTINUATION **UTILITY** confirmation — the unmeasured "recall != utility" link.
 - WHY it's only table-stakes: continuation = recall-the-prefix = NIAH, which the warm-start ALREADY
   nails (tail m-zero +2.01). A logprob "win" just re-derives Doc2LoRA. The contribution is UTILITY:
   does the FROZEN base actually GENERATE/PASS the correct suffix with the prefix in PARAMS?
 - DECISIVE BASELINE (advisor): not no-adapter — it's PREFIX-IN-PROMPT (the in-context CEILING). Product
   claim = "adapter ~= prompt but O(1) state." Run at a MODERATE length where the prompt baseline is
   still feasible -> show matched-adapter ~= ceiling -> argue it extends to lengths where prompt isn't
   feasible. Training baseline-to-beat = ZERO-SHOT WARM-START (+2.01/+1.17), not no-adapter.
 - Score on INFORMATIVE continuation tokens (edit-local-mask analog), else the signature/surface shortcut
   (Du; sig +3.84 vs body +0.14) wins again and inflates the number.
 - FIRST ACTION (maybe no new GPU): check whether third_party/.../rune_episode_recall.py already produced
   continuation matched/mismatch/CEILING numbers; §1 only logged m-zero +2.01, so specificity+ceiling are
   likely unmeasured but may be one script-run from existing runs.
 - CAVEAT TO STATE LOUDLY: a continuation-utility win validates the MECHANISM on the EASY facet only — it
   is NOT product validation. Sakana already cleared continuation.

TRACK B (the real investment, IN PARALLEL): FAILURE-BEARING TRAJECTORY DATA where the failure-reason is
 ACTION-DETERMINING. "What we tried & why it failed" IS critique/feedback binding — so the SAME walls we
 hit (false negatives, weak feedback->action MI, single-exposure binding; Huynh/Ovadia/Du) WILL RECUR on
 the tried-failed facet. Therefore: (a) construct the corpus so the failure-reason genuinely DETERMINES
 the next action (T1's directive-vs-non-directive lesson now = CORPUS-DESIGN requirement, not a discarded
 test); (b) this data-generation is the CRITICAL PATH for the product's differentiation vs Sakana — it
 must run now, not "after continuation." Sources: real engine trajectories (mining lane) + synthetic
 action-determining failures with provenance labels. Easy MBPP can't supply it (structural-only finding).

OBJECTIVE/SELECTION (when a training run IS justified): trajectory-RECALL/reconstruction over goal /
 last-action / tried-failed facets with HARD NEGATIVES = OTHER EPISODES (derangement; the hidden-task
 +1.17 regime), select on matched-vs-MISMATCHED-EPISODE on informative tokens, anchored to calibration
 (real recall ~+7.7). Keep retention (NIAH/QA) + generation-stability (xgrammar pass@1) gates. NOT the
 feedback-swap-vs-base metric (policy/discipline confound).

DE-PRIORITIZE: T0 paired feedback-swap test = OFF the critical path (don't spend GPU re-confirming a
 metric we're pivoting away from). T1 demoted AS a feedback->edit test but its construction lesson is
 LOAD-BEARING for Track B's corpus. T2 (oracle capacity at r=8 vs higher; Parametric Memory Law: loss
 power-law in RANK) stays relevant for body/continuation recall — fold into Track A if continuation
 utility is weak.

ONE-LINE: stop optimizing feedback->edit on the proxy corpus; (A) cheaply prove recall->GENERATION utility
on continuation against the in-context ceiling, and (B) in parallel build action-determining failure-bearing
trajectory data — the facet that is the product's actual differentiation and where the data, not the model,
is the wall.

### CORRECTION (2026-06-02) — NOT a pivot; the substrate goal was always the north star; CODE-ACCESSIBILITY is the core
USER: "This is not a pivot." Rune's goal = a system UNBOUNDED by context window (max_length) that ITERATES
UNTIL SOLVED, where EACH trajectory step is ORIENTED BY A SWAPPABLE ADAPTER. The hypernetwork adapter must
serve OPTIMALLY as that SUBSTRATE — s.t. "code embedded along with the broader context is ACCESSIBLE to the
model." The feedback->edit investigation was a DIAGNOSTIC that refined corpus/objective design, not a wrong
turn. Restating: the success criterion is the adapter-as-substrate, always has been.

WHAT THE EMPHASIS ON "CODE ... ACCESSIBLE" CHANGES (vs my prior entry):
The hard, central capability is CODE-CONTENT ACCESSIBILITY: the model must, at step t, USE the embedded
code (continue a cutoff function body, call a helper defined earlier, recall what a prior block does, avoid
a tried approach) WITHOUT it being in the prompt. Our own evidence already localizes the problem: the
substrate exposes LABELS/signatures well (+3.84) but CODE BODIES barely (+0.14). That body/label gap — not
feedback->edit — IS the thing to attack. Two first-order gates:

GATE 1 = CAPACITY (now FIRST, was "fold in"): can the substrate HOLD high-entropy code content at all?
 Parametric Memory Law: loss ~ power-law in RANK; r=8 likely can't hold code bodies (consistent w/ +0.14).
 KEY ARCHITECTURE INSIGHT: Doc2LoRA's CHUNKING already scales effective rank (per-chunk adapters concat
 along rank dim) — so "capacity for code" is governed by CHUNK COUNT, not a fixed r=8. TEST: measure
 body/code-content recall + accessibility as a FUNCTION OF CHUNKS/RANK (oracle per-episode LoRA at r=8 vs
 higher, AND warm-start D2L at 1 vs K chunks). If code becomes accessible only at higher effective rank ->
 the substrate must allocate rank to code (chunking/rank is the lever) BEFORE any training objective work.

GATE 2 = UTILITY/ACCESSIBILITY (does embedded code translate to correct ACTION): continuation/generation
 test, but specifically on CODE-CONTENT-DEPENDENT next steps (resume a cutoff body; call a previously-
 defined helper), scored on CODE-CONTENT tokens (not signature — else the +3.84 shortcut inflates it),
 against the IN-CONTEXT CEILING (prefix-in-prompt) at a moderate length, with the ZERO-SHOT WARM-START as
 the training baseline-to-beat. Product claim = "adapter ~= prompt but O(1) state, extending past
 max_length." Confirms recall->GENERATION utility (the unmeasured PR#53 link), now at BODY granularity.

THE FAILURE/AVOID FACET ("what we tried & why it failed") is still critique-binding and still data-gated;
 its corpus must be built so the failure-reason is ACTION-DETERMINING (T1's lesson = corpus design). Run
 failure-bearing data generation IN PARALLEL — it's the product differentiation vs Sakana. But the
 *immediate* technical wall is CODE-CONTENT ACCESSIBILITY + CAPACITY (Gates 1-2), because if the substrate
 can't make code bodies accessible at sufficient rank, the goal/last-action/avoid facets all inherit that
 ceiling.

REVISED NEXT STEP (supersedes the "two parallel tracks" framing above):
 (1) GATE 1 capacity sweep FIRST — cheap, decides architecture (rank/chunks). Oracle r=8 vs higher + D2L
     1-vs-K-chunk body/code-content recall. Possibly reuses rune_episode_recall.py (no/low new GPU).
 (2) GATE 2 code-accessibility UTILITY vs in-context ceiling, scored on code-content tokens.
 (3) THEN, conditioned on Gates 1-2: train the substrate on real trajectories with cross-episode hard
     negatives, select on BODY/code-content matched-vs-mismatch (anchored to calibration +7.7), retention +
     gen-stability gates. (4) Failure-bearing corpus generation in parallel throughout.
ONE-LINE: the always-goal is the adapter as an unbounded-context substrate that makes embedded CODE
accessible for the next step; the binding wall is code-BODY accessibility at sufficient RANK/CHUNKS
(Gate 1) and whether that accessibility yields correct GENERATION vs the in-prompt ceiling (Gate 2) — settle
those before training, and build action-determining failure data in parallel.

### EVAL (2026-06-02) — doc-Q&A ORIGIN hypothesis: rank is NOT the only wall; representation + directionality argue for fine-tuning
USER: the hypernet was built for doc Q&A, so it may not optimally embed (a) CODE and (b) DIRECTIONALITY
(where-we-were -> where-we're-headed). This is WHY fine-tuning may help, ALONGSIDE the rank argument.

VERDICT: well-founded, and crucially it is a SEPARATE AXIS from rank. Three confounded axes were all hiding
behind the single number "body recall +0.14":
 AXIS-1 CAPACITY (rank/chunks): how much the substrate CAN hold. Parametric Memory Law: loss ~ rank.
 AXIS-2 REPRESENTATION/OBJECTIVE: WHAT the learned text->weights compressor chooses to keep. Rate-distortion
   logic: a compressor trained on doc-QA/NIAH keeps QA-answerable facts and is REWARDED FOR DISCARDING the
   verbatim middle — exactly the high-entropy code-body detail a continuation/cross-reference needs. Rank
   can't fix this: more capacity in a function that doesn't encode code-bodies still won't encode them.
   Fine-tuning RE-OPTIMIZES the function -> the right lever for this axis.
 AXIS-3 DIRECTIONALITY (the deepest, and the STRONGEST a-priori case FOR fine-tuning): doc-QA is ORDER-
   AGNOSTIC RETRIEVAL ("what does the doc say about X"), a SET representation. Trajectory orientation needs a
   VECTOR/PROCESS representation: did A -> failed b/c B -> therefore heading to C. The doc-QA objective never
   rewarded encoding the arrow. Rank supplies tokens, not relational/causal order -> directionality is
   intrinsically an OBJECTIVE/architecture property -> fine-tuning (or arch change), never rank.

EVIDENCE FOR (partial): zero-shot warm-start binds goal +2.30, diff +1.01, tail +2.01, signature +3.84 but
 body +0.14 — i.e. it KEEPS labels/answerable-facts, LOSES code-body detail = the rate-distortion signature
 of a QA compressor. The feedback->edit weakness (matched-swap +0.069, flat margin) RE-READS as a
 DIRECTIONALITY failure: "the critique (why the last act failed) should REDIRECT the next edit" is a causal/
 directional relation, and the substrate doesn't bind it. The avoid facet is inherently directional.
EVIDENCE AGAINST / caveat: partial transfer (+2.30/+1.01/+2.01) means doc-QA training is NOT useless for
 code/recency — so fine-tuning is ADAPTATION, not from-scratch. And tail +2.01 = recency sensitivity, which
 is NOT the same as directionality/causality (untested). So AXIS-3 is hypothesized, not yet measured.

DISCRIMINATING EXPERIMENT (separates the 3 axes; this REFRAMES Gate 1 from a capacity sweep into a 2x2+probe):
 E1 (capacity vs representation): ORACLE per-episode LoRA vs HYPERNET-GENERATED adapter, AT MATCHED RANK,
    scored on CODE-BODY tokens.
      oracle GOOD @ r8 + hypernet BAD @ r8  -> REPRESENTATION/objective wall (doc-QA compressor) -> FINE-TUNE.
      oracle BAD @ r8, GOOD @ higher        -> CAPACITY wall -> RAISE RANK/CHUNKS.
      both BAD even high rank               -> data/architecture.
    (this is the reflection's "optional capacity check" + advisor H3, now the LEAD discriminator.)
 E2 (directionality — NEW, current probes never ran it): matched vs DIRECTION-SCRAMBLED context (swap
    where-were <-> where-headed; or time-reverse the trajectory; or goal-swap) -> does the adapter's next-
    step prediction change APPROPRIATELY? If scrambled ~= correct-direction -> substrate is order/direction-
    agnostic (doc-QA-like) -> directional fine-tuning objective REQUIRED. Score on action/next-step tokens.
 E3 (capacity sweep): body/code accessibility vs CHUNKS (D2L 1 vs K) and vs rank, to size the AXIS-1 lever.

RE-EVALUATION OF MY PRIOR COMMENTS:
 - I put "capacity FIRST, then train" and treated fine-tuning as downstream. CORRECTION: fine-tuning has its
   OWN representational rationale (Axes 2-3) INDEPENDENT of rank; it is not merely "after capacity." E1
   decides rank-vs-finetune; they may BOTH be needed (raise rank AND fine-tune for code+direction).
 - My Gate-1 "capacity sweep" is REPLACED by E1 (oracle-vs-hypernet @ fixed rank) as the lead — it's the
   single experiment that tells us whether the next step is rank, fine-tune, or both.
 - DIRECTIONALITY (E2) was ENTIRELY ABSENT from my prior framing and from every probe to date; given the
   trajectory product it is plausibly THE central gap and the clearest justification for fine-tuning. Add it.
 - The continuation/utility test (prior Gate 2) stays, but now explicitly as the BODY+DIRECTION utility
   check vs the in-context ceiling, not prefix-echo.
 - CAVEAT retained: fine-tuning the warm-start risks forgetting the strong NIAH/QA recall prior (retention
   gate) and is delicate (scaler_B lesson) — gate any fine-tune on retention + gen-stability.

REVISED IMMEDIATE NEXT STEP: run E1 (oracle-vs-hypernet @ matched rank on code-body tokens) + E2
 (direction-scramble probe) — both cheap, both likely reuse rune_episode_recall.py / the specificity-probe
 harness with little new GPU. Their 2x2 + directional readout tells us EXACTLY which of {raise rank, fine-
 tune for code, fine-tune for directionality, build failure data} to invest GPU in — before any long run.
ONE-LINE: the doc-Q&A origin means body+directionality are likely REPRESENTATION/OBJECTIVE gaps fine-tuning
must fix, distinct from the RANK capacity gap; E1 (oracle vs hypernet @ fixed rank) + E2 (direction-scramble)
separate rank-vs-finetune-vs-both and should run before committing to a long training session.

### EVAL (2026-06-02) — ENTROPY/CAPACITY EFFICIENCY: don't embed every char; encode the RESIDUAL (4th lever)
USER: in training, maximize capacity + reduce entropy — must we embed each character of code, or is there a
more efficient way (more code per chunk -> better detailed recall by the base)?
VERDICT: well-founded; answer = NO, don't embed verbatim. Code is LOW-entropy given the frozen base's strong
code prior, so most tokens are base-predictable and waste the limited rank. The literature converges on the
mechanism AND the caveat. This is a FOURTH lever (objective-level per-token weighting / residual encoding),
distinct from + complementary to: AXIS-1 capacity(rank/chunks), AXIS-2 representation(code), AXIS-3 direction.

MECHANISM (the lever): PER-TOKEN INFORMATION/SURPRISAL WEIGHTING of the recall/reconstruction loss.
 - CaMeLS (Hu 2023) names OUR exact failure: naive FT on documents has "low information uptake" because
   "gradient from important (factual) tokens is DROWNED OUT by inherently noisy tokens." Fix = meta-learn
   per-token loss weights to maximize post-update QA. -> spend capacity on high-info tokens, not boilerplate.
 - Token Weighting for Long-Range LM (Helm 2025): non-uniform loss weights (long- vs short-context model
   confidence) improve long-context recall. 
 - MemFT / Parametric Memory Law (2605.30260): reallocate gradient budget from MASTERED to below-threshold
   tokens; phase transition p>0.5 -> verbatim recall. Goal: push ACTION-RELEVANT tokens above threshold,
   don't spend budget on already-mastered ones.
 - LLMLingua (info-entropy prompt compression): low-PPL tokens carry little info; "50-95% of tokens are
   low-value"; 20x compression near-lossless. Analog: adapter should encode the RESIDUAL (high-surprisal:
   identifiers, literals, branch conditions, the cutoff point), not syntax/boilerplate the base reconstructs.

CAVEAT (decisive for "DETAILED recall"): gist/aggressive compression FAILS on fine-grained synthetic recall
 (Gist-token study 2412.17483; GistPool 2504.08934 capacity limits; UniGist flags "detail-recalling tasks").
 So we CANNOT gist code into a blob — the product needs exact identifier / exact continuation point. =>
 design = TIERED / TASK-AWARE fidelity, NOT uniform compression:
   - HIGH-info / action-determining (exact signatures, literals, ACTIVE/cutoff code, branch conds) -> VERBATIM
     fidelity (drive above recall threshold).
   - LOW-info / base-predictable (boilerplate, whitespace, std idioms) -> drop/gist (base reconstructs).
   "More code per chunk" = stop spending rank on the 50-95% low-value tokens -> same rank/chunk holds far more
   SEMANTICALLY-relevant code -> better detailed recall per unit capacity.

HOW IT INTERACTS:
 - Primarily an OBJECTIVE/training-target change (per-token loss weighting) -> folds into the SAME doc-QA->
   code/direction fine-tune, just a better loss. Highest-leverage TRAINING change (no arch change).
 - MULTIPLIES AXIS-1: effective code-per-chunk rises -> fewer chunks for a codebase -> capacity goes further.
 - = AXIS-3 directionality: "action-determining" == "directionally/causally relevant to the next step", so
   surprisal-weighting and action-relevance weighting are the same lever aimed at the trajectory.
 - CANONICALIZATION (cheap pre-proc): normalize formatting/whitespace/comments (entropy down, semantics kept)
   but DO NOT normalize identifiers (exact-name recall = the +3.84 asset). Optional: AST/signature+contract
   for DISTANT/reference code, verbatim for ACTIVE code.

MEASUREMENT: recall-fidelity-PER-RANK under {verbatim vs surprisal-weighted vs canonicalized} encoding,
 scored on HIGH-SURPRISAL / action-determining tokens (the informative-token metric). "bits of useful code
 recalled per unit rank" is the efficiency objective. Refines E1: score oracle-vs-hypernet on INFORMATIVE
 tokens; add lever-test E4 = does surprisal-weighted training beat uniform on recall-per-rank?
RISK: over-compression breaks the detailed recall the product needs (gist-fails-on-recall) -> keep
 conservative + task-aware; gate by retention + continuation-utility.

RE-EVAL OF PRIOR COMMENTS: adds a 4th lever I'd missed — OBJECTIVE-LEVEL per-token weighting / residual
 encoding. Likely the highest-ROI training change: directly raises EFFECTIVE capacity with no arch change,
 and CaMeLS shows it's the known fix for low information-uptake in exactly the fine-tune-on-documents regime.
 Sequencing now: E1 (rank vs representation) + E2 (directionality) DIAGNOSE; E4 (surprisal-weighted vs
 uniform recall-per-rank) is a TRAINING lever to test once we fine-tune. Verbatim-everything is rejected.
ONE-LINE: don't store every character — train the substrate to encode the high-surprisal RESIDUAL (exact ids/
literals/active-code/branch-conditions) verbatim and let the frozen base reconstruct the predictable rest, via
per-token surprisal/importance-weighted loss (CaMeLS/MemFT/LLMLingua-style); this multiplies rank/chunk
capacity and is the same lever as action-relevance, but must stay conservative because aggressive gisting
breaks the fine-grained recall the product needs.

### HANDOFF WRITTEN (2026-06-02) — docs/issue52-deliverable4-handoff-2026-06-02.md (context will clear next)
Self-contained experimentation-phase handoff: goal (substrate, not a pivot), what's done (commit 1553026f:
scaler_B fix + seq + tests + dossier), the 4-levers model (capacity/representation/directionality/entropy),
the critique-hardened experiment plan (T0 paired-stats, E1 oracle-vs-hypernet @matched-rank +cross-over
control, E2 directionality via minimal-edit counterfactuals scored on action-consequences, E3 chunk/rank
sweep, E4 utility-per-rank w/ 3 weighting families + small-token-big-effect control + local-state-aware
compression), failure-bearing corpus in parallel, calibration LADDER (not NIAH+7.7), retention+gen-stability
gates, leakage rule + positive controls, infra/run commands, pitfalls.
RESPONDED TO reflections.md 2026-06-02 pushback — ALL THREE entries folded into the handoff:
 - "Pushback on Gate Reframe": '#49 in miniature' softened to "insufficient+confounded"; 14/86 split =
   warning not attribution; T0 = bootstrap CI+sign test+scatter, byte-identical rows; T1 freeze-before-deltas
   +report filtered&full; calibration LADDER not +7.7; positive-control row added.
 - "Body/Directionality Gate Controls": E1 oracle=upper-bound + same masks/negs/prompts + CROSS-OVER tiny
   hypernet finetune; E2 minimal-edit counterfactuals + same-bag control + score on action-consequences;
   retention gate on any finetune.
 - "Pushback on Residual Encoding": high-surprisal != useful; E4 = 3 weighting families targeting downstream
   UTILITY; utility-per-rank not bits-per-rank; small-token-big-effect negative control; canonicalization
   dangerous (never normalize identifiers); local-state-aware compression.
Handoff is durable on disk (survives context clear). Not committed (user said "write"); offer to commit/push.

### INFRA + T0 LAUNCH (2026-06-02, post-idle-restart) — checkpoints→S3, RAM/disk guard, spec frozen
RESUMED after instance idle-shutdown killed the scout workflow. Recovered: 4 scouts had cached;
re-ran synthesis (resumeFromRunId). Outputs durable.

INFRA (per user: "ram guard so you never crash", "only store checkpoints on S3", "use log_artifact
not manual"):
 - tools/run_guarded.sh: added DISK floor (RUNE_DISK_MIN_KB, default 3GB on / /tmp /workspaces) +
   pidfile registration alongside the existing RAM watchdog.
 - tools/instance_guard.sh: NEW always-on daemon (RUNNING, pid logged to /tmp/rune-guard/instance_guard.log)
   polling RAM+disk every 5s; kills registered guarded jobs (pidfile dir) on breach, never random procs.
 - hypernet_distill.py _save_checkpoint: now uploads via mlflow.log_artifact -> VERIFIES present in the
   artifact store (active_run + artifacts.list_artifacts) -> DELETES local staging copy. Keeps local only
   if no handle / upload fails / unverified (never loses a checkpoint). keep_local=True to force-retain.
   MLflow server (localhost:5000) is S3-backed via infra/docker-compose.yml --artifacts-destination
   s3://elixirtrials-949678234935-eu-west-2-artifacts/mlflow/artifacts/ ; verified end-to-end upload->S3.
 - DISK: was 97% (5.8GB). Preserved /tmp/smoke_broken_scalerB1.pt (documented-FAILED, handoff-flagged) +
   fixed-smoke checkpoint.pt to s3://.../checkpoints/issue52/{documented-failed,fixed-smoke}/ (byte-verified),
   deleted stale /tmp scratch (ckpt_candidates 11G, rune-ck-final/b1smoke/trajectory-safe, v3b). Now 70% used.
   KEPT local: warm-start (third_party input), fixed-smoke checkpoint.pt (T0 needs it), /tmp/rune-corpus,
   /tmp/*.log result dumps, /tmp/rune-docs-only (git checkout, not a checkpoint).

SCOUT/SYNTHESIS WORKFLOW caught a LOAD-BEARING attribution error in deliverable4 handoff:
 - headline warm-start goal +2.30 / file +1.76 / diff +1.01 / tail +2.01 are GEMMA (gemma_demo), NOT qwen.
 - TRUE qwen warm-start (/tmp/d2l_qwen_ep.log): goal +2.235 / file +1.596 / diff +0.983; code +2.597.
 - NO qwen continuation/tail number on disk; any body/tail CEILING must come from a FRESH run.
 - Calibration ladder re-anchored to qwen rungs (body +0.14 .. signature +3.84). Frozen in
   docs/issue52-predeclared-spec-T0-E1-E2-2026-06-02.md (FROZEN before any trained delta; leakage rule).

T0 (running, MLflow-free eval; under guard): tools/_feedback_swap_eval.py REWRITTEN — --ckpt2 second arm,
 --out per-episode JSONL, ONE max_seq_length=768 threaded to BOTH conditioning + scored span, ckpt-INDEP
 eligible-set precompute, ctx_hash, NaN pairing -> shared denominator (assert n_arm1==n_arm2). arm1=warm-start,
 arm2=trained recipe-4b. Stats in tools/_feedback_swap_stats.py (CPU, from dump): paired bootstrap 95% CI +
 sign test + scatter. FORBIDDEN: comparing historical +0.0185 (2048-path) vs +0.0687 (768-path).
 Log /tmp/t0_run.log, dump /tmp/t0_dump.jsonl.

### T0 RESULT (2026-06-02) — significant but sub-threshold; NULL/NO-GO per frozen bar
Controlled paired re-run, ONE process, seq=768, byte-identical 60 val rows, NaN-paired shared denom
(n_arm1==n_arm2==60, 0 pre-skipped). MLflow issue52-T0-paired run db7dc17e (dump artifact on S3).
 arm1 warm-start matched-swap +0.0188 (frac 0.48); arm2 trained +0.0691 (frac 0.65). REPRODUCES the
 historical +0.0185/+0.0687 almost exactly -> the feared 2048-vs-768 length-regime confound was NOT the
 driver; the +0.0019 outlier was the scaler_B-collapse run, not a length artifact.
 PAIRED d = arm2-arm1 = +0.0503; bootstrap 95% CI [+0.0101,+0.0961] EXCLUDES 0; sign test +37/-19
 p=0.0222; scatter broad (top-5 |d| = 43%, not outlier-driven). => the training gain is REAL and
 statistically SIGNIFICANT per-episode, not sample-composition noise.
 VERDICT (frozen go/no-go): NULL/NO-GO. arm2 mean +0.069 < rung-1 body (+0.14). Significant but
 magnitudinally trivial (~1-2% of real binding). Does NOT justify a long run on unfiltered
 external_codereview (matches handoff: stop optimizing feedback->edit on the proxy corpus). The lever
 decision moves to E1 (capacity vs representation) / E2 (directionality). T0 = significance closed only.

### REFLECTIONS RESPONSE (2026-06-02) — T0 rigor requirements satisfied
The 2026-06-02 "Pushback on the Gate Reframe" demanded: NOT t-test alone (used bootstrap CI + sign test +
row-level scatter); byte-identical rows across arms (ctx_hash dumped, shared recs, eligible-set precomputed
checkpoint-independently, NaN-paired denominator with n_arm1==n_arm2 assert); do NOT compare historical
+0.0185 vs +0.0687 (re-measured both fresh in one process under one seq=768); calibration LADDER not NIAH
+7.7 (bar = rung-1 body +0.14, qwen-anchored). ALL met. The "#49 in miniature / 14-86 split" was NOT used
as attribution; T0 reports the paired effect + its CI honestly: real, significant, sub-threshold.

### E1 RESULT (2026-06-02) — REPRESENTATION wall, NOT capacity (r8 holds body; hypernet doesn't encode it)
Oracle r8 + hypernet body arm, BOTH 4-bit, ABSENT regime, same 10 frozen MBPP episodes, same
derangement, BODY span [hi,len) only (signature-hardened span_bounds; 0 excluded). Train==score
surface for the oracle (advisor fix: train on ABSENT, not context+answer).
 ORACLE r8 body: lp_m=-0.216 (overfit-PC PASS, memorized), base lp_z=-1.673, m-mismatch=+21.75, frac 1.00.
   => an r8 down_proj LoRA CAN hold episode-specific code body. CAPACITY IS NOT THE WALL.
 HYPERNET body: lp_m=-0.999, base lp_z=-1.645, m-mismatch=+0.141 (frac 0.70), m-zero=+0.646.
 HYPERNET signature: lp_m=-1.993, m-mismatch=+4.089. HYPERNET full: m-mismatch +1.246.
 DECISIVE asymmetry WITHIN the hypernet (no oracle needed): signature episode-specific +4.089 vs body
   episode-specific +0.141. The adapter lifts body over base (+0.646 m-zero) but ~GENERICALLY (m-mismatch
   only +0.141) — it binds the NAME to the episode, not the BODY. = the doc-Q&A rate-distortion signature
   (keep answerable labels, discard the verbatim middle).
 VERDICT (frozen decision table): oracle GOOD @ r8 + hypernet BAD @ r8 -> REPRESENTATION/OBJECTIVE WALL.
   Lever = FINE-TUNE / re-objective the hypernet for body, NOT raise rank/chunks. SKIP the r16/r32 capacity
   sweep (E3 deprioritized). NEXT = cross-over control: does a tiny hypernet fine-tune on these exact 10
   facts move body m-mismatch toward oracle territory? moves -> trainability/objective gap (fine-tune is the
   lever); doesn't move -> architecture/conditioning attenuation back on the table.
 CAVEAT (advisor): oracle = per-instance-gradient UPPER BOUND with a harder negative (different memorized
   answer); +21.7 is NOT the bar. The robust claim is (a) r8 capacity suffices, (b) the hypernet encodes
   signatures not bodies episode-specifically — a representation gap, not capacity.
 CODE: tools/_e1_oracle.py (train-on-ABSENT, overfit-PC, body-span); _specificity_probe.py (--out lp_m dump,
   span_bounds hardened to raise+exclude). Dumps /tmp/e1_oracle_r8.jsonl, /tmp/e1_hypernet_body.jsonl.

### CROSS-OVER DESIGN GAP (2026-06-02) — _distill_entry objective != body recall
The spec's cross-over (REUSE _distill_entry + tiny.yaml on these facts) has an objective mismatch:
_distill_entry optimizes the FEEDBACK-SWAP contrastive hinge on external_codereview edit-local tokens, NOT
body RECALL on MBPP refs in the absent regime. A faithful cross-over must fine-tune the HYPERNET so its
GENERATED adapter improves body m-mismatch on these 10 episodes (CE/recall on body span + derangement
negative, backprop into the hypernet). Needs a small dedicated trainer, not _distill_entry as-is. Consult
advisor before building.

### E1 PRECISION CAVEAT (2026-06-02, user Q: "could FP accuracy play a role?")
scoring_core.mean_gold_logprob uses fp32 log_softmax + fp64 accumulation -> scoring math is NOT the
limit. The limit is UPSTREAM: 4-bit nf4 base + bf16 compute + flash-attn produce the logits. Matched vs
mismatch share the same 4-bit base (adapter delta differs) so static quant error is ~common-mode and
cancels in the margin; but bf16+flash-attn rounding differs between the two adapter forwards = residual
noise that can blur a SMALL margin. The hypernet BODY margin (+0.14, per-episode -0.31..+0.63, 3/10
negative, lp_m~lp_x within 0.1-0.6) sits in that danger zone. Precision CANNOT explain the asymmetry:
signature +4.09 and oracle +21.7 are orders of magnitude above any bf16 floor, and body-lift-over-base
(m-zero +0.18..+1.49) is robust. RESIDUAL RISK: can't yet distinguish "+0.14 = true small body binding"
vs "+0.14 = larger binding attenuated to the 4-bit noise floor." TEST (running): re-run hypernet body+sig
arm in BF16. body stays ~+0.14 -> representation wall holds; body jumps -> 4-bit suppressed real signal
(and product runs 4-bit per CLAUDE.md, so that's itself a finding). This gates the cross-over premise.

### E1 PRECISION CHECK RESULT (2026-06-02) — FP is NOT the explanation; representation wall confirmed
Re-ran hypernet body+sig arm in BF16 (vs the 4-bit run), same episodes/derangement/ABSENT/body-span.
 body m-mismatch: 4-bit +0.1412 (frac 0.70) -> bf16 +0.1370 (frac 0.60). IDENTICAL (Δ 0.004).
 sig  m-mismatch: 4-bit +4.089 -> bf16 +3.837. Stable (~+4).
 Per-episode body tracks tightly across precision (mbpp/16 +0.63->+0.62, /57 +0.60->+0.60, /12 +0.32->
 +0.43); 1/10 sign flip and it's +0.06->-0.009 (both ~0, not meaningful). The 3 negative episodes
 (/14,/17,/20) stay negative under both -> mixed signs = REAL per-episode variation (some bodies just
 aren't bound), not a noise floor.
 => higher precision reveals NO hidden body signal. +0.14 is the TRUE small episode-specific body
 binding, not a 4-bit artifact. Representation wall holds at both precisions; product runs 4-bit so the
 operative number is confirmed. Cross-over premise stands. (MLflow: log alongside E1.) dump
 /tmp/e1_hypernet_body_bf16.jsonl.

### PRECISION REGIME CORRECTION (2026-06-02, user Q: "more GPU room now we switched models?") — USE BF16
Model switch is real: CLAUDE.md base = Qwen3.5-9B (the 18GB base the 4-bit/OOM rules were written for),
but ALL issue-52 runs use Qwen3-4B-Instruct-2507. The 4-bit nf4 in the EVAL PROBES (_specificity_probe,
_feedback_swap_eval, _e1_oracle defaults) is a LEFTOVER from the 9B regime, NOT a current need.
CORRECTION to my earlier claim "product runs 4-bit": WRONG. The ENGINE loads the 4B base in BF16
(src/rune/model/wrapper.py:123 dtype=torch.bfloat16, offload_base=False, NO quantization_config). So
bf16 is the OPERATIVE precision; the 4-bit eval probes were a MISMATCH with the deployed engine. (4-bit
survives only in the distill TRAINING yaml = QLoRA memory choice, also a 9B carryover.)
ROOM: 4B bf16 ~8GB weights on 23GB GPU -> ~15GB free; ample for hypernet + activations + cross-over
grads/optimizer. (fp32 ~16GB would fit but bf16 matches the engine = the right call.)
CONVERGENCE w/ the FP check: the bf16 E1 run is BOTH the higher-precision check AND the engine-matching
measurement -> body +0.137 ~= 4-bit +0.141. Representation wall robust + now at the operative precision.
DECISION: run the cross-over in BF16, and treat the bf16 numbers (body m-mismatch +0.137, sig +3.837) as
the E1 baseline-to-beat. Eliminates the quantization variable AND matches the engine. (Eval-probe 4-bit
defaults should be flipped to bf16 going forward for engine-parity; flag CLAUDE.md base-model drift.)
