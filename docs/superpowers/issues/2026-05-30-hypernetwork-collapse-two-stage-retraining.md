# Hypernetwork adapter collapse — revisit two-stage mining + training

**Type:** issue / epic for a new branch (`feat/two-stage-retrain` or similar).
**Status:** root cause confirmed; retraining required. Inference code is sound enough
to deploy a *good* checkpoint, but **no good checkpoint exists** — the entire trained
campaign collapsed. Do NOT attempt to "select a better checkpoint"; retrain.

## TL;DR
Every hypernetwork checkpoint in MLflow/S3 is **degenerate**: the trained `scaler_B`
(which multiplicatively gates the generated LoRA-B matrix) collapsed to ~0, so the emitted
adapter `ΔW = B·A ≈ 0` for *any* input. The adapter is **inert** — it neither encodes nor
retrieves the conditioning trajectory. The old validation gate (`adapter_has_any_effect`)
was too weak to detect this, so the system *looked* validated. Retraining must (a) prevent
the collapse, (b) gate on a **real retrieval/contrast probe**, and (c) verify the mining
corpus + conditioning format actually carry learnable trajectory signal.

## Evidence (this session, 2026-05-30)
- **Static (decisive):** in EVERY checkpoint, `scaler_B.down_proj` (shape `(1,32,8,1)`) has
  `mean ≈ 0.0055–0.0066`, `absmax ≈ 0.003–0.013` — barely off its zero-init. `bias_B` is
  exactly 0; `scaler_A ≈ 0.996` (ones-init). With `B = B_raw · scaler_B`, the LoRA-B side is
  scaled ~200× down → `ΔW ≈ 0`.

  | checkpoint | exp | top1_agreement | scaler_B absmax | verdict |
  |---|---|---|---|---|
  | deployed `checkpoints/hypernet_hpo/checkpoint.pt` | 20 (HPO) | loss 0.0024 | 0.0108 | COLLAPSED |
  | `kd-a10-t10` (`24/0a1586b2`) | 24 | **0.842 (best)** | 0.0129 | COLLAPSED |
  | `kd-a099-t20` (`25/653397c4`) | 25 | 0.826 | 0.0120 | COLLAPSED |
  | `kd-a09-t10` (`23/04b3aa82`) | 23 | 0.794 | 0.0108 | COLLAPSED |
  | `20/577269…` | 20 | — | 0.0031 | COLLAPSED |

- **Empirical (8 probes, all consistent):** with the deployed adapter, two very different
  trajectories yield adapters with **cosine ≈ 1.0** (rel_L2 ≈ 0.005); a **retrieval probe**
  (unique facts `MAGIC_OFFSET=73921`, `frobnicate_payload` embedded only in the trajectory)
  returns the fact in **0/48** cases — `real == zero == contra` at every scaling 1.5→16; a
  **continuation probe** (real vs zero vs contradictory) is byte-identical `real==zero`. The
  adapter changes greedy output only to *generic* code at high scaling, never the conditioned
  content.

## Why it wasn't caught
- `top1_agreement` does **not** detect collapse: the base model already agrees with the
  oracle teacher ~84% of the time, so a near-zero adapter still scores ~0.84. Low distillation
  loss is achievable by collapsing the adapter (student → base).
- The v1 acceptance gate (MLflow exp 39 `adapter-probe`, run `50272017`) only checked
  `gate/adapter_has_any_effect` and `contradictory_shows_contamination` — i.e. *does the output
  differ at all*. A collapsed adapter at high effective scaling perturbs output to generic
  noise → passes the gate **without retrieving content**. (A second probe run `7676e1f6`
  scored all zeros.) The gate gave false confidence.

## Root-cause hypothesis (verify during retrain)
The adapter has almost no gradient pressure to be non-trivial: most supervised next-tokens are
already predicted correctly by the base model (≈teacher), so the loss is near-minimal at
`scaler_B = 0`. The Perceiver/scaler init at zero + this weak gradient → permanent collapse.

## Retrain checklist — what to fix / be careful about

### A. Training objective & init (prevent the collapse)
- [ ] Initialize `scaler_B` **away from 0** (the collapse basin); e.g. small positive, or
      remove the multiplicative `scaler_B` gate if it only enables collapse.
- [ ] Add an **anti-collapse signal**: regularize toward non-trivial adapter contribution, or
      supervise *only the tokens where teacher ≠ base* (the diff tokens) so the adapter is the
      only way to reduce loss. Confirm the **diff-aware weighting is actually applied in the
      hypernet distillation**, not just in the stage-1 oracle SFT — in v1 the `DiffWeighted`
      collator lived in `run_distillation` (plain SFT) and the hypernet KL+CE path was NOT
      diff-weighted, and the collator was inert anyway because `StepRecord` carried no
      `pre_codes`/`post_codes` (see §C).
- [ ] Ensure the **teacher genuinely differs from base** on the supervised span (otherwise no
      gradient). Sanity-check `top1_agreement(base, teacher)` — if it's already ~0.84, the
      adapter has little to learn; weight the loss toward disagreement tokens.

### B. Acceptance gates (replace the weak gate)
- [ ] Gate checkpoint acceptance on a **retrieval probe**: embed unique unguessable facts in
      the trajectory, require the model to recall them with a lean prompt (reuse
      `tools/diag_retrieval_probe.py`). Require `real_hit ≫ zero_hit`.
- [ ] **Trajectory-sensitivity:** two distinct trajectories must produce adapters with
      **cosine well below 1.0** (reuse `tools/diag_recall_probe.py` weight-divergence).
- [ ] **Continuation contrast:** `real > nothing > contradictory` on a held-out completion
      task (reuse `tools/diag_continuation_probe.py`).
- [ ] Inspect `scaler_B` magnitude as a cheap CI tripwire (`absmax > ~0.05`).

### C. Data mining (the corpus feeding the conditioning)
- [ ] Verify the mined corpus actually contains **diffs/trajectories with signal** — task +
      prior-code diff + corrective feedback → revision (v1 `d2l_data.py::unroll_trajectory_to_pairs`,
      format `## Task / ## Current Code / ## Review Feedback / ## Revision`).
- [ ] If diff-weighting is intended, the mining `StepRecord` must carry `pre_codes`/`post_codes`
      (it currently does not → the diff collator silently no-ops).
- [ ] **Conditioning-format alignment:** inference builds the trajectory via
      `render_template('code'/'code_continue', ...)` (`ROLE/PROJECT/SUBTASK/PLAN/EXISTING CODE`),
      which is **out-of-distribution** vs the training format. Either train on the inference
      format or render the inference trajectory in the training format. (This alone did not
      explain the collapse — the inert adapter fails on *both* formats — but it must be fixed so
      a good adapter isn't fed OOD text.)
- [ ] STaR success filter (only correct traces) is correct; keep it (`success_filter.py`).

### D. Inference-application correctness (verify once the adapter is non-inert)
These are latent today (no effect with a collapsed checkpoint) but must be correct before a
good checkpoint can work:
- [ ] **Apply `combine_lora` + `get_head_bias()`** — v2 drops the trained `bias_A` (norm 1.13);
      `src/rune/model/hypernetwork.py::_to_peft_state_dict` (~line 315) and
      `generate_adapter_weights` (~line 391) call neither, unlike v1
      (`adapter_generator.py`, `ModulatedPretrainedModel.forward`).
- [ ] **Un-contaminate activation extraction** — v2 extracts on the **PEFT-wrapped** base
      (`wrapper.py:104`); after the first `hotswap_adapter` each extraction is contaminated by
      the previous adapter. Extract under `with base_model.disable_adapter():` or via a separate
      non-PEFT handle (`extract_activations_with_model`, `hypernetwork.py`).
- [ ] **Scaling:** base up-scaling is the lever; continuation is **1.5× base** by design
      (`cont_multiplier≈1.53`). The validated regime was base ≈ 7.84 → continuation ≈ 12.
      Re-measure the sweet spot once the adapter actually conditions (do NOT reuse the old
      numbers — they were measured against a collapsed adapter and an OOD format).

## Two-stage pipeline reference (v1-final `libs/model-training/`)
1. **Stage 1 — oracle QLoRA, diff-aware loss** (`trainer.py`, `diff_loss.py`): per-bin oracle
   adapters, warm-start `Qwen3.5-DeltaCoder-9B`, rank 64, hunk-weighted CE.
2. **Stage 2 — HyperLoRA distillation** (`round2_train.py`): KL+CE of student(base+gen-adapter)
   vs the per-bin oracle teacher, over the answer span. Warm-starts the Perceiver aggregator
   from `SakanaAI/doc-to-lora` (architecture only).
The repo's wired `run_distillation`/`d2l_train.py` is **plain SFT and does not train the
hypernet**; the deployed checkpoint came from the `hypernet-hpo` sweep (mostly failed/collapsed).

## Checkpoint inventory (all COLLAPSED)
- `s3://…/checkpoints/hypernet_hpo/{checkpoint,ckpt-1..8}.pt` — deployed + HPO trials.
- `s3://…/mlflow/artifacts/{20,23,24,25}/<run>/…/checkpoint.pt` — HPO + KD runs.
- exp 26 `hypernet-full-t10` (top1 0.835) logged **no checkpoint artifact**.
- Live MLflow at `http://localhost:5000` (exp 20 `hypernet-hpo`, 23-25 `kd-*`, 26
  `hypernet-full`, 39 `adapter-probe`, 27 `paper-gate2`). Local `mlflow.db` is empty (Default).

## Reusable probes (this session, under `tools/`)
`diag_retrieval_probe.py` (needle recall), `diag_continuation_probe.py` (real/zero/contra),
`diag_recall_probe.py` (weight cosine + logit KL), `diag_scaling_mode_probe.py`
(structured vs freeform × scaling), `diag_format_probe.py` (OOD vs training format).
Run under `/tmp/run_guarded.sh` (RAM watchdog; ~15GB CPU box, `offload_base=False`).
