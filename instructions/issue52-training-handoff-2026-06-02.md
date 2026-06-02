# Issue #52 — Training handoff (BF16 hypernet fine-tune for code-body recall) — 2026-06-02

Self-contained handoff for the **training phase**. Companion durable docs (committed on the feature
branch): `docs/issue52-deliverable4-results-2026-06-02.md` (T0/E1 numbers),
`docs/issue52-predeclared-spec-T0-E1-E2-2026-06-02.md` (frozen scoring rules),
`docs/issue52-pretraining-facts-dossier-2026-06-02.md` (facts + research + 4-levers model).
Working logs (this file, scratchpad, reflections) are gitignored here and snapshotted on the
`scratch/issue52-research-tools` orphan branch for posterity.

PR #53; branch `feat/issue52-doc2lora-positive-control`; last commit `ac2f0022`.

---

## 1. North star (unchanged)
Rune is a system **unbounded by the base model's context window** that **iterates until solved**, where
**each trajectory step is oriented by a swappable adapter**. The hypernetwork adapter must be the
**substrate**: the embedded **code + context** must be **accessible to the frozen base at each step**
(continue a cutoff body, call a helper defined earlier, recall what a prior block does, avoid a tried
approach) **without it being in the prompt**. Design principle: **separate memory from policy** — the
adapter recalls episodic state; the frozen base emits the next step conditioned on it.

The central, measurable capability is **code-content accessibility**: our substrate exposes
labels/signatures well but code **bodies** barely. That body gap is the thing this training attacks.

---

## 2. Background — the experiments we ran and what they showed

System under test: base `Qwen/Qwen3-4B-Instruct-2507` (frozen), ctx-to-lora HyperLoRA perceiver warm-start
`qwen_4b_d2l/checkpoint-20000` (r=8, lora_alpha=45.2548, 36 layers, target_modules={`down_proj`}). Adapter
contract: `effective_scaling = lora_alpha = 45.2548` applied **un-divided**. Single 23GB GPU, ~15GB CPU RAM.

**Calibration ladder (qwen warm-start, the yardstick — NOT NIAH +7.7):** signature +3.84/+4.09 · goal
+2.235 · file +1.596 · code-recall +2.597 · diff +0.983 · **body +0.14** (the floor we attack) · feedback
chance ~+0.018. (Correction: the deliverable-4 handoff's goal +2.30 / diff +1.01 / tail +2.01 were GEMMA,
not qwen; there is **no qwen continuation/tail number on disk**.)

### T0 — feedback-swap paired significance (MLflow `issue52-T0-paired`)
Controlled paired re-run, one process, seq=768, byte-identical 60 val rows (**4-bit eval path — a legacy
diagnostic for the feedback→edit objective we are abandoning; not re-run in bf16, do not treat as a bf16
engine-parity gate**). Warm-start matched−swap
+0.0188 → trained +0.0691; paired d=+0.050, bootstrap 95% CI [+0.010,+0.096] (excludes 0), sign test
+37/−19 p=0.022. **Real and significant, but +0.069 << rung-1 body (+0.14) → NULL/NO-GO.** The
feedback→edit objective on the unfiltered `external_codereview` proxy is **not** worth a long run. It was
a *diagnostic*, not the product signal.

### E1 — capacity vs representation (MLflow `issue52-E1-capacity-vs-repr`) — the decisive result
Oracle per-episode LoRA vs hypernet adapter at **matched r=8**, `down_proj` only, scored on the **BODY
span only** (`[hi,len)`; signature span hardened to raise+exclude), ABSENT (hidden) regime, derangement
negative. Oracle trains on the **same ABSENT surface it is scored on** (train==score ⇒ a weak oracle is a
true capacity limit, not a transfer artifact); overfit positive-control passed (oracle matched `lp_m`≈−0.22).

| | matched body `lp_m` | base `lp_z` | body m−mismatch (episode-specific) |
|---|---|---|---|
| Oracle r8 | −0.22 | −1.67 | **+21.7** |
| Hypernet | −1.00 | −1.65 | **+0.14** (frac 0.70) |
| Hypernet **signature** | −1.99 | | **+4.09** |

**Verdict: REPRESENTATION/OBJECTIVE wall, not capacity — scoped to this 10-episode MBPP absent/body
micro-probe.** An r8 `down_proj` LoRA *can* memorize these short bodies (oracle, train==score) — which
scopes the capacity claim to the micro-probe, NOT to longer trajectories, multi-file state, or
failure-history, where rank/chunk capacity may re-emerge (keep as a later interaction test). The hypernet
binds the **signature** episode-specifically (+4.09)
but the **body** only generically (+0.14 specific; the +0.65 lift over base is non-specific). The
sig-vs-body asymmetry *within the same hypernet at the same rank* is the decisive, oracle-independent
evidence — the rate-distortion signature of a doc-Q&A compressor that keeps answerable labels and discards
the verbatim body.

### Precision check — FP is not the explanation
bf16 vs 4-bit: body m−mismatch +0.137 ≈ +0.141; sig +3.84 ≈ +4.09; per-episode values track tightly.
Scoring is fp32 log_softmax + fp64 accumulation. The +0.14 is the **true** small body binding, not a
4-bit noise-floor artifact. The wall holds at both precisions.

### Precision regime correction (why BF16)
The engine loads the 4B base in **bf16** (`src/rune/model/wrapper.py`), not 4-bit — the 4-bit nf4 in the
eval probes / distill config was a 9B-era leftover (CLAUDE.md still lists `Qwen3.5-9B`). The 4B base in
bf16 (~8GB on 23GB) leaves ample room. **bf16 is the engine-operative precision and the training regime.**

---

## 3. Why training is warranted (and what is NOT)

**Warranted:** **fine-tuning the hypernet's representation for code-body / trajectory recall.** E1 shows
capacity is sufficient *for the 10-episode micro-probe* (r8 oracle memorizes the short bodies) and the gap
is **representational** — the doc-Q&A objective never rewarded encoding the verbatim body. The warm-start
transfers partially (goal/file/diff/signature bind), so this is **adaptation, not from-scratch**.
Re-optimizing the function is the indicated lever for a compressor that discards bodies; raising rank alone
cannot fix *what the function chooses to encode* — though rank/chunk capacity may still re-emerge as a
bottleneck once we move from 10 short bodies to realistic trajectory/multi-file contexts (a later
interaction test, not the first move).

**NOT warranted:** a long feedback-swap run on unfiltered `external_codereview` (T0 NULL); raising rank /
chunks *as the first move* (E1: capacity is not the wall **for the micro-probe** — revisit only if a working
body fine-tune then plateaus on realistic contexts); selecting on `val_diff_agreement` (matched-vs-base
discipline confound).

**The cheap gate before any long run — E1 cross-over (run this FIRST, ~1 smoke):** does a tiny
**contrastive** body-span hypernet fine-tune on the exact 10 facts *move* body m−mismatch?

**Trainer-sanity gate (mandatory, evaluated FIRST):** the cross-over is interpretable only if the trainer
is shown to optimize the intended loss — log the hinge/loss curve and confirm it decreases / overfits the
10 facts. A flat body m−mismatch is **ambiguous** until then: inspect gradient flow through functional-LoRA,
scaler_B preservation, adapter assembly, hinge margin, steps/LR, and the tiny heterogeneous 10-row corpus
**before** concluding anything about the architecture.

Predeclared bar (vs the hypernet's own signature binding +4.09, NOT the oracle's +21.7) — a **decision
threshold, not a truth boundary**:
- **+0.14 → ≥ +1.0**, with matched rising more than mismatch/zero and signature retained = gap is
  gradient-reachable ⇒ scale to a better-designed pilot, then the full fine-tune.
- a **smaller but statistically broad** move (matched > mismatch, signature retained, paired sign test
  positive) = still reachable ⇒ iterate the pilot; do **not** declare null.
- **stays ~+0.14 *and the trainer-sanity gate passed*** = the conditioning/representation is hard to move
  ⇒ rethink the conditioning path. This is NOT proof that fine-tuning or rank can never work.

This is a **trainability** probe (trained-on-test by design), not product-generalization. Only a real,
controlled move (matched rises, controls held) launches the long run.

---

## 4. Training plan (BF16)

### Objective (the three knobs that change vs the current distill loop)
- **Span:** edit-local → **BODY / informative tokens** (never signature — the +3.84 shortcut inflates any
  aggregate; never full-span).
- **Negative:** feedback-swap → **cross-episode derangement** (the hidden-task +1.17 regime): matched
  episode's body recall must beat *another episode's* adapter on the same body.
- **Loss:** **CONTRASTIVE hinge** pushing matched-body-lp above derangement-partner-body-lp — **NOT CE**
  (CE raises matched and mismatch together = the generic-boost confound this project exists to detect; the
  derangement negative must be IN the loss, not just eval).
- **Hinge safeguards (log all four every checkpoint):** matched, mismatch(derangement), zero(no-adapter
  base), and the hinge value — plus a no-finetune warm-start readout. A hinge can be satisfied by pushing
  the *deranged partner down* or destabilizing both paths; the desired signal is **matched-body rising with
  mismatch/zero held**, not margin movement alone. Generic body boosting (matched and mismatch rise
  together) is a FAIL, not a pass.
- **Corpus:** cross-over pilot = the 10 frozen MBPP body facts. Full run = real engine trajectories +
  action-determining failure-bearing episodes (the product differentiator; build in parallel), with
  cross-episode hard negatives. Facets: goal / last-action / tried-and-why-failed / **body**.
  **Corpus-quality gate before any long run (do NOT launch just because the 10-row cross-over moved):**
  yield, causal alignment of failure-reason → next action, an in-prompt CEILING per episode,
  positive-control episodes, and provenance labels. Validate the data, not just the mechanism.
- **(Phase 2 lever, E4)** per-token **importance/surprisal weighting** (CaMeLS/MemFT/LLMLingua-style) to
  spend rank on the high-information residual, with the **small-token-big-effect** negative control
  (flipped inequality, missing `not`, int-vs-str return, boundary, which exception). Conservative
  canonicalization only; **never normalize identifiers** (exact-name recall is our +3.84/+4.09 asset).

### Regime / config changes (BF16)
- Base in **bf16** (`load_in_4bit: false`), `offload_base: false`, flash_attention_2 — matches the engine
  and removes the quantization variable. bf16 *training* is a **deliberate objective change** (log it as
  such), distinct from bf16 eval parity — the distill loop historically used a 4-bit frozen base for memory.
- **bf16 memory is NOT proven by arithmetic.** ~8GB weights leaves apparent room, but the prior OOM came
  from **graph-retaining contrastive training** (optimizer state + dual forward/backward), not weights.
  REQUIRE a guarded smoke with a **logged GPU peak** before any multi-hour run; fall back to a 4-bit-base
  training regime (still valid QLoRA) if bf16 peaks too high.
- `max_seq_length: 768` (the contrastive loop's designed memory regime; revisit upward only after a
  memory check — seq 2048 OOM'd in the old 4-bit path). Export
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Never reinit the warm-start `scaler_B`** (collapse-guarded; the unconditional reinit inflated B ~17×
  and produced the −8.8 collapse). Warm-start preservation is automatic now.
- Checkpoints are **S3-only** via `mlflow.log_artifact` (upload→verify→delete local staging;
  `_save_checkpoint`). **Verification must check byte size** (not just artifact-path existence) before
  deleting — reusable names (`checkpoint.pt`, `checkpoint_best.pt`) can list-present while bytes are
  stale/zero (known code gap: `_artifact_uploaded` checks path only — fix before relying on auto-delete).
  Keep a manifest: local path, S3 URI, size, step, purpose.
- Periodic saves; **post-hoc selection** on body/informative matched-vs-mismatched-episode (NOT
  `val_diff_agreement`, NOT signature spans).

---

## 5. How we measure success (predeclared; freeze before looking at trained deltas — leakage rule)

**Primary (does the body get bound?)** body (and goal / last-action / tried-failed) **matched-vs-
mismatched-EPISODE** m−mismatch on **BODY / informative tokens**, moving up the ladder:
- cross-over gate: **+0.14 → ≥ +1.0 = reachable** — a *decision threshold, not a truth boundary* (see §3:
  a smaller statistically-broad move with matched>mismatch + signature retained also counts; a flat result
  is meaningful only after the trainer-sanity gate confirms the loss actually moved).
- meaningful long-run win: clear a ladder rung and approach the within-model proof that binding is
  achievable (signature +4.09; facet refs goal +2.235, diff +0.983). Real recall ~+7.7 is the ultimate,
  not the near-term bar.
- Movement that stays in the +0.14 noise band = NULL, regardless of sign.

**Retention gate (every trained checkpoint; must not regress vs warm-start CI):**
- episode recall goal/file/diff matched−mismatch ≥ warm-start (+2.235 / +1.596 / +0.983);
- code-recall (+2.597) not regressed; signature binding (+4.09) **not traded away**;
- matched−zero discipline **> 0** (the −8.8 collapse is the scaler_B failure mode — gate on it).

**Generation-stability gate:** xgrammar-constrained **pass@1** (rune bench) not degraded vs warm-start. A
representation gain that breaks structured generation is not a win.

**Statistics + hygiene:** per-episode **paired bootstrap CI + sign test** (margins heavy-tailed; not a
t-test/eyeballed mean). **Predeclare subset/scoring rules before any trained delta.** Every comparison
carries a **positive control** (e.g. code-recall +2.597 proves the substrate can bind code) so a null
distinguishes weak-signal from broken-harness. Log everything to MLflow (S3-backed).

---

## 6. Infra / run commands
- **Env:** `uv run` for all Python; `uv sync --extra gpu` (plain `uv sync` prunes trl/bnb/flash-attn).
- **Guards (mandatory):** the standalone `tools/instance_guard.sh` daemon should be running (RAM+disk
  watchdog; kills guarded jobs on breach, never the session/MCP). Launch every multi-minute GPU job under
  `tools/run_guarded.sh <log> <script> [args]` (RAM floor + disk floor + pidfile registration).
  **Known gap to fix before the long run:** the guards kill the parent PID only; `uv run python &` can
  leave an orphan Python/CUDA child holding GPU/RAM after a kill. Switch to a **process-group** launch/kill
  (`setsid` + `kill -- -PGID`) and verify no descendant survives, or the guard can report success while
  pressure continues.
- **Cross-over pilot:** a small contrastive body-span hypernet fine-tune trainer (reuse
  `hypernet_distill`'s generation→apply→backprop core — do NOT greenfield the gradient path or the
  scaler_B guard; swap span/negative/corpus), then re-score with the E1 body-span probe (bf16). Scratch
  harnesses are on the `scratch/issue52-research-tools` orphan branch.
- **Distill entry:** `tools/run_guarded.sh /tmp/run.log tools/_distill_entry.py --config <bf16 yaml>
  [--max-steps N]`; metrics → MLflow experiment `issue52-recipe`. Update the config to bf16 per §4.

---

## 7. Pitfalls (so they don't recur)
- Never reinit a warm-start's learned `scaler_B` (guarded + regression-tested, but easy to reintroduce).
- Don't select on `val_diff_agreement` (matched-vs-base discipline confound).
- Score on body / informative / action-determining tokens — never full-span or signature (the +3.84/+4.09
  shortcut wins any aggregate).
- The contrastive loss must use the derangement negative **in the loss** (CE alone = generic boost).
- High-surprisal ≠ useful; utility-per-rank, not bits-per-rank; keep canonicalization conservative,
  identifiers never normalized.
- seq 2048 OOM'd (old 4-bit path); keep 768 unless a bf16 memory check says otherwise.
- Every fine-tune passes the retention + generation-stability gates, written down first.

---

## 8. IMPLEMENTED 2026-06-02 (Opus 4.8 session) — cross-over built + trainer-sanity PASSED

State committed to branch `issue52-bf16-body-contrastive` so it survives an instance/region relaunch.
The cross-over is a **trained-on-test trainability probe**, so everything below is **REMOVE-BEFORE-MERGE
scaffolding** (manifest at the end of this section) — committed only to carry the in-progress probe across
the migration, NOT to land on `main`.

### What was built
- `contrastive_mode` knob in `DistillConfig` (`hypernet_distill.py`): `"feedback_swap_edit_local"` (issue #49,
  default) vs `"body_derangement"` (issue #52). The body-derangement branch reuses the existing
  generation→apply→backprop core + scaler_B guard (no greenfield gradient path), swapping only span/negative:
  - `_body_span_mask` — BODY gold tokens `[after def-<entry_point>(...): line, end)`; raises if the signature
    marker is missing (refuses to silently score the signature as body).
  - `_deranged_partner_context` — deterministic non-current episode context as the hard negative.
  - `_contrastive_logprob_readout` / `_gold_logprobs` / `_empty_contrastive_metrics` — per-checkpoint
    safeguard logging of `lp_matched`, `lp_mismatch`, `lp_zero`, `hinge_active_frac`, `contrastive_tokens`.
- `configs/issue52_body_crossover_4b.yaml` — bf16 (`load_in_4bit: false`), 10 frozen MBPP facts,
  grad_accum 5 → 2 opt-steps/epoch × 15 epochs = 30-step probe, contrastive body_derangement, margin 1.0.
- `configs/issue52_mbpp_body_crossover.jsonl` — the committed 10-row corpus (canonical reproduction input;
  `corpus_path` points here so a clean clone trains without the gitignored builder deps).
- `tools/build_issue52_mbpp_body_corpus.py` — provenance of the corpus (needs the gitignored
  `_specificity_probe.py` REFS + `benchmarks/mbpp_phase0_iter.json`; on `scratch/issue52-research-tools`).
- Unit tests in `tests/unit/test_hypernet_distill_config.py` (deranged-partner never returns current
  episode; readout ordering; `_artifact_uploaded` size check). `pytest`/`ruff` green on the changed files.

### Trainer-sanity gate (mandatory FIRST per §3) — PASSED (MLflow run `c401f0c0`, 30 steps, bf16)
- `loss` 5.13 → 0.045 (overfits the 10 facts → trainer optimizes the intended loss).
- `lp_matched` −1.31 → ~−0.4 (E1 bf16 baseline was −0.999); `lp_mismatch` (derangement) driven to −4.1;
  `lp_zero` (no-adapter base) held ~−0.77. Direction = matched↑, mismatch↓, zero held = the DESIRED shape,
  not generic body-boosting.
- `scaler_B/absmax` CONSTANT 0.177 (no collapse, no inflation); gradients flow; `gpu_peak_gb` 11.66 on 23GB
  → **bf16 contrastive training is memory-safe** (closes the §4 open worry — measured, not arithmetic).
- ⚠️ NECESSARY-NOT-SUFFICIENT: this is the IN-LOOP train==test readout. The ONLY valid cross-over number is
  re-scoring `checkpoint_step30.pt` (and the warm-start) with the FROZEN E1 body-span probe
  (`tools/_specificity_probe.py`, bf16, same 10 episodes/derangement) — **NOT YET RUN** (the go/no-go).

### Session facts to preserve (c401f0c0 — for replay/audit; binary is local-only)
- seed pinned (`DistillConfig.seed=0`; used by `_shuffled` + derangement `seed_index`), so a re-run
  reproduces the cross-over CONCLUSION modulo CUDA nondeterminism + library versions (not bitwise).
- sha256 trained `checkpoint_step30.pt`: `d296a4e22336ae2bf89d6c322993531c1218b2f010d123dac95fe0807da9da0e`
- sha256 warm-start `qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`:
  `6438b46c828dd3b5f88f21add0f7f5cacc7994d47bf15eda266786a506044591`
- final step-30 metrics (single-batch, noisy; trend is the signal): loss 0.104, lp_matched −0.768,
  lp_mismatch −3.598, lp_zero −1.566, hinge_active 0.375, scaler_B/absmax 0.1768, gpu_peak 11.94GB.
- MLflow run `c401f0c0b1ec44be97ce3c6b2ef62437`, experiment `issue52-body-crossover` (61). Metrics in the
  MLflow sqlite backend (durability depends on the container volume / litestream-S3, which may also be
  affected by the IAM block — do not assume it survives a region change).

### RESUME HERE in the new region (the pending go/no-go)
1. Restore the warm-start checkpoint (`third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/`,
   gitignored/local-only) and `uv sync --extra gpu`.
2. Re-run the 30-step probe (fast, fully reproducible from committed config+corpus):
   `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && tools/run_guarded.sh /tmp/xover.log \
    tools/_distill_entry.py --config configs/issue52_body_crossover_4b.yaml --max-steps 30`
   (the local `checkpoints/issue52-body-crossover-4b/*.pt` from this session are local-only and lost on
   recycle — regenerable, so not blocking).
3. **Run the frozen probe** on the trained checkpoint vs warm-start (bf16), report per-episode
   `body m−mismatch`, `m−zero`, and **signature retention (~+3.8)**. Bar: +0.137 → ≥ +1.0 = reachable
   (decision threshold, not truth boundary; a smaller statistically-broad move with matched>mismatch +
   signature retained also counts — see §3/§5).
4. If the in-loop readout moved but the frozen probe barely moves → diff `_body_span_mask` vs the probe's
   `span_bounds` on one episode BEFORE blaming architecture (truncation `MAX_ANS_TOK=96` + a documented
   t−1 gold-shift differ; checked harmless at `ans_len≈55`, but verify if numbers disagree).
5. Only AFTER a confirmed frozen-probe GO: corpus-quality gate → full run; HPO comes later (see below).

### MLflow logging trouble — ROOT CAUSE = AWS IAM/KMS (infra), NOT a code bug
`log_artifact` → HTTP 500 on every checkpoint; metrics/params log fine (sqlite backend). Reproduced with the
instance role `elixirtrials-dev-gpu-instance`: `PutObject` → AccessDenied `kms:GenerateDataKey` (KMS path)
and AccessDenied `s3:PutObject` (AES256 path) → **the role cannot write to the artifact bucket at all**. The
MLflow container (outside the devpod) uses the same role, so its artifact proxy surfaces the denial as 500.
Code correctly falls back to keeping checkpoints local (no data loss). FIX is infra-side (grant the role
`s3:PutObject`+`kms:GenerateDataKey`, or repoint the container `--artifacts-destination` to local FS) — being
handled by the user. In the new region this may differ; re-test `aws s3 cp - s3://…/mlflow/artifacts/_probe`
before relying on artifact upload. HPO/checkpoint persistence does not need S3 (runs on-instance).

### HPO consideration (deferred — gated on a frozen-probe GO)
`tools/hpo_train.py` / `bench/hpo.py` scaffold Optuna over the logged val metrics. Do NOT start HPO now: (a)
the cross-over go/no-go (frozen probe) is unresolved — HPO before a confirmed lever optimizes a metric we
haven't validated; (b) HPO needs working metric/artifact logging (MLflow S3 currently blocked). Sequence:
frozen-probe GO → corpus-quality gate → short full run → THEN HPO over engine params for pass@1.

### REMOVE-BEFORE-MERGE manifest (strip all of this before merging `issue52-bf16-body-contrastive`)
- `configs/issue52_body_crossover_4b.yaml` — delete (probe config).
- `configs/issue52_mbpp_body_crossover.jsonl` — delete (trained-on-test corpus).
- `tools/build_issue52_mbpp_body_corpus.py` — delete (probe corpus builder; scratch deps).
- `src/rune/training/hypernet_distill.py` — revert the `body_derangement` branch + helpers
  (`_body_span_mask`, `_deranged_partner_context`, `_contrastive_logprob_readout`, `_gold_logprobs`,
  `_empty_contrastive_metrics`) and the `contrastive_mode` field — UNLESS the cross-over validates the lever,
  in which case promote it deliberately with its own tests/docs. (Keep the `sb0.detach()` and
  `_artifact_uploaded` size-check fixes — those are real bug fixes, mergeable on their own.)
- `tests/unit/test_hypernet_distill_config.py` — keep the `_artifact_uploaded` test; drop the
  body-derangement-specific tests if the feature is reverted.
