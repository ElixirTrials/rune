# Handoff — Re-port the KD training pipeline into v2 (simplified)

## Goal

Re-implement, in `src/rune/training/`, the **knowledge-distillation (KD) training that actually produced the shipped hypernetwork checkpoint** — simplified (KISS/DRY/YAGNI), functionally **equivalent** to v1, *not* a verbatim restore. v2's training is currently a stub; this closes that gap.

**Non-goal:** restoring v1 verbatim. v1 was ~5,000+ lines across a dozen `d2l_*` modules; the v2 re-port should be a handful of focused modules.

## Why this is needed (current state vs reality)

- **What trained the checkpoint** (MLflow-confirmed; `s3://…/checkpoints/hypernet_hpo/checkpoint.pt`): experiment `hypernet-full-t10`, run `8d2f0908` (2026-05-12). A ~429M-param HyperLoRA hypernetwork (`r=8`, `target_modules=down_proj`, 32 layers) over `Qwen/Qwen3.5-9B` (nf4), trained by **round-2 distillation**: `KL(student‖teacher)@temperature + CE`, `alpha=0.999`, `temperature=1.0`, teacher = `hpo_artifacts/best_diffloss_v1`, precomputed teacher logits, `adamw-8bit`, `lr=2e-4`, `num_steps=500` (early-stopped @184), final `total_loss≈0.0267`, `kl_loss≈0.025`, `ce_loss≈1.65`, **`top1_agreement≈0.835`**.
  Lineage: diff-loss SFT HPO → teacher `best_diffloss_v1` → `hypernet-hpo` (exp 20, picked hyperparams) → `hypernet-full-t10` → `checkpoint.pt`.
- **What v2 `src/rune/training/` does today:** a stub.
  - `orchestrator.py`: 3 stages; `_run_oracle_training` **raises `NotImplementedError`** ("see libs/model-training"); gate runs on placeholder empty scores; `neftune_alpha` is read but never passed to `SFTConfig` (dead knob).
  - `d2l_train.py` (`run_distillation`): plain **diff-weighted CE SFT** via `DiffAwareSFTTrainer` — **no teacher, no KL**. This is *not* KD.
  - `diff_loss.py`: the diff-aware per-token weighting **is** ported (~1,240 lines) and reusable.
  - `oracle_cache.py`: scaffolding only; not wired into any active path.
- **The gap:** v2 ported the student-side diff-weighted CE but **not** the teacher-forward, the **KL term**, or the **oracle stage**. So v2 cannot reproduce the checkpoint.

## v1 reference (read-only, on `archive/main-v1`)

`archive/main-v1:libs/model-training/src/model_training/`:
- **`round2_train.py`** (612 lines) — the KD core. Key pieces:
  - `_training_step_round2(...)` — two-pass step: **student** forward (with grad) + **teacher** forward (oracle LoRA, `no_grad`) → loss.
  - `_teacher_forward_with_oracle(base_model, oracle_lora_dict, …)` — applies the per-bin oracle's **functional-LoRA** to the *same* base model (no PEFT module rebuild; context-managed so there's no hook leakage between teacher/student), returns teacher logits.
  - `_compute_kl_ce_loss(student_logits, teacher_logits, answer_start, config)` — `KL@temperature + CE`, restricted to answer tokens (`answer_start` offset).
  - `oracle_fallback` ∈ {`skip` (default; skip records whose bin has no oracle), `base_model` (ablation: bare base as teacher)}.
- **`oracle_cache.py`** (297) — `OracleAdapterCache` (LRU) resolving a bin key → `oracle_<bin>` functional-LoRA dict; `_bin_key_for_record` (`{phase}_{benchmark}`, `diagnose`→`diagnose_pooled`).
- **`d2l_data.py`** (984) — distillation record format (`teacher_text`, `answer_start`, activation/teacher split), precomputed-logits path.
- **`d2l_train.py`** (844) — the round-1 (bare-base teacher) two-pass step that round-2 mirrors; source of the shared `_compute_kl_ce_loss` / `_extract_activations_with_model`.
- **`diff_loss.py`** (1250) — diff-aware weighting (already ported to v2; reuse v2's).
- Supporting: `d2l_diff.py`, `d2l_pairing.py`, `d2l_config.py`, `d2l_lora.py`, `d2l_models.py`.

## Proposed simplified v2 design

Replace the orchestrator stubs with a small, real KD path. Aim for ~3 focused modules, reusing what v2 already has.

1. **`src/rune/training/distill.py`** (new, the core) — the two-pass KD step + train loop:
   - student forward (grad) + teacher forward (oracle functional-LoRA, `no_grad`) on the same base model;
   - `kd_loss = alpha · KL(student‖teacher; τ) + (1-alpha?) · diff_weighted_CE` — **reuse `diff_loss._compute_weighted_loss` for the CE term** so the diff-weighting carries over (DRY);
   - precomputed-teacher-logits path (v1 used `use_precomputed_logits=True`) so the teacher forward can be skipped when logits are cached — *strongly preferred for the simplified version* (avoids holding teacher+student on-GPU; matches how the checkpoint was actually trained).
2. **`src/rune/training/oracle_cache.py`** (already present) — finish/keep just enough to resolve a bin → oracle functional-LoRA dict. Don't re-port the LRU sprawl unless needed.
3. **`src/rune/training/orchestrator.py`** — implement Stage 1 + Stage 2 to call the above; delete the `NotImplementedError`. Keep Stage 3 gate as-is (already fixed: union-of-keys).

**Reuse / DRY:** v2 `diff_loss.py` (weighting), `d2l_train.py` scaffolding (corpus load, `SFTConfig`, trainer construction), `config.py`.
**Drop (YAGNI — not part of KD):** `encoder_pretrain/`, `rag_pipeline`, `github_client`, `merging`, `reconstruction/`, `d2l_external`, `d2l_licenses`, `d2l_probe`, `d2l_quality`, `model_pool`, `kill_switch`.

## Equivalence & verification

1. **Code-level:** the KD loss math must match v1 — `KL(softmax(student/τ) ‖ softmax(teacher/τ))·τ² + diff_weighted_CE`, answer-token-restricted. Diff this against `round2_train._compute_kl_ce_loss`.
2. **CPU unit tests (no GPU):** feed synthetic student/teacher logits → assert KL term, τ scaling, CE reduction, and the all-equal-weights identity (CE reduces to standard mean CE). Test `_bin_key_for_record` and oracle resolution.
3. **GPU equivalence run (you run + log — CLAUDE.md):** a short round-2 run; compare to the MLflow `hypernet-full-t10` targets: `total_loss≈0.027`, `kl_loss≈0.025`, `ce_loss≈1.65`, `top1_agreement≈0.835`, early-stop ≈ step 184. I'll prepare the exact command + a metrics-diff script.

## Open decisions (resolve before/while implementing)

- **Oracle stage scope:** implement live per-bin QLoRA oracle training, *or* (simpler, matches the actual run) consume **precomputed teacher logits / pre-trained oracle adapters** and skip live oracle training. Recommend the latter for the first cut.
- **`alpha`/`temperature` placement:** carry as `D2LTrainConfig` fields (v1: `alpha=0.999`, `temperature=1.0`).
- **Data format:** reuse v1's `teacher_text` + `answer_start` record schema (from `d2l_data`) or simplify to the v2 corpus shape — must preserve answer-token masking.
- **`neftune_alpha`:** either wire it into `SFTConfig` or drop the knob (currently dead).

## References

- Checkpoint: `s3://elixirtrials-949678234935-eu-west-2-artifacts/checkpoints/hypernet_hpo/checkpoint.pt`
- MLflow (localhost:5000): exp 26 `hypernet-full-t10` run `8d2f0908`; exp 20 `hypernet-hpo`; teacher `hpo_artifacts/best_diffloss_v1`.
- v1 code: `archive/main-v1` (tip `2cbfef01`), `libs/model-training/src/model_training/` — deleted from `main` by `9ef3a1d4` "clean slate for rune v2".
- v2 code: `src/rune/training/{orchestrator,d2l_train,diff_loss,gate,oracle_cache,config}.py`.

## Status / sequencing

- ✅ Benchmark task generator shipped (`rune gen-tasks`) so HPO can run in parallel.
- ✅ Benchmark HPO hold-out: `run_hpo` now splits the task pool (`split_tasks`, seed 42 / 0.70), optimizes on the tuning set, and scores best params once on the held-out validation set (`validation_pass_at_1`).
- ⬜ This KD re-port (above).
