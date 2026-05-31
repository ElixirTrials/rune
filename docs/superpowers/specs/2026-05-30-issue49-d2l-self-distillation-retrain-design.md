# Issue #49 — D2L self-distillation retrain (mining → validated hypernetwork checkpoint)

**Date:** 2026-05-30
**Issue:** #49 — Hypernetwork adapter collapse (`scaler_B ≈ 0`); every trained checkpoint is inert.
**Goal:** Recover a non-collapsed HyperLoRA training path and produce a checkpoint that *retrieves trajectory content* — proven by retrieval/contrast gates, not by magnitude or `top1_agreement`.

## North-star functional goal (the falsifiable spec)

A generated adapter must:
1. **Enable continuation** of partial code at constant prompt length (trajectory removed from prompt).
2. **Improve** multi-step trajectory performance over the base model.
3. **Worsen** results when the conditioning trajectory contradicts the task.

Property 3 is the decisive control: a collapsed/inert adapter cannot worsen anything; a generic-perturbation adapter worsens everything equally. Only a content-conditioned adapter is *selectively* worse under contradiction.

## Definition of done

A checkpoint that **passes the content gates** (below) plus a **2–3 task tiny benchmark** against base / zero / shuffled / contradictory controls. Full MBPP/HumanEval campaign and HPO are **out of scope** for this milestone.

---

## Root cause — confirmed at the source

`ctx_to_lora/modeling/hypernet.py`:

```python
self.scaler_A = nn.Parameter(torch.ones ((1, n_layers, r, 1)))   # ones-init
self.scaler_B = nn.Parameter(torch.zeros((1, n_layers, r, 1)))   # ZERO-init
# _to_lora_dict:  B = einsum(B_raw, scaler_B)
```

Gradient trap: with `B = B_raw · scaler_B` and `scaler_B = 0`:
- `∂L/∂B_raw = ∂L/∂B · scaler_B = 0` → `B_raw` never moves.
- Only escape is `scaler_B`, driven by `∂L/∂B · B_raw`. When teacher ≈ base (~84% token agreement on code), `∂L/∂B ≈ 0` → no push → permanent collapse.

Third accomplice: `ctx_to_lora/trainer.py::DistillationTrainer` adds `gen_lora_l1_reg_coef · ‖generated LoRA‖₁` — an active force **toward** the inert solution.

`scaler_B` *is* a `requires_grad` parameter and the V1 optimizer (`round2_train.py:428`) grabs all `requires_grad` params, so "optimizer omission" is unlikely. `strict=False` loading (`hypernetwork.py:267`) remains a suspect for silently dropping `bias_*`/head keys at **inference** (§D), and is checked statically in Stage 0.

This is a fixable **reparameterization + objective** problem, not an architectural one.

---

## Architecture — modified Sakana D2L (privileged-context self-distillation)

The real D2L mechanism is shipped in `ctx_to_lora/modeling/context_distillation.py::CtxDistillModel._distill_context`:

```python
with torch.no_grad(), self.base_model.disable_adapter():   # TEACHER = base + context, adapters OFF
    teacher_logits = logits_at_positions(teacher_outputs, t_pos)
    topk_vals, topk_idx = teacher_logits.topk(K, dim=-1)
    teacher_p = (topk_vals - teacher_denom).exp().detach()
# STUDENT = base + hypernet-generated adapter, context REMOVED from prompt
selected_student_logits = student_logits.gather(-1, topk_idx)
token_losses = -(teacher_p * student_logq).sum(dim=-1)      # top-K KL over answer span
```

- **Teacher:** frozen base model with the trajectory/diff **in the prompt** (`disable_adapter()`).
- **Student:** base model + hypernet-generated adapter, with the trajectory **removed** from the prompt.
- The adapter is forced to internalize what the in-context teacher got for free → **adapter-as-memory by construction**.

`ctx_to_lora/data/processing.py` ships `add_negative_prompt` / `neg_ctxs`: D2L *can* train with contradictory contexts. **Status (corrected per review):** the implemented `hypernet_distill` loop trains on **real context per record only**; "worse under contradiction" is therefore an **emergent, evaluation-time property**, not a trained objective. The synthetic forced-choice result (2026-05-31) showed this property emerges from positive-only training (an adapter that encodes the correct value necessarily does worse when fed a contradictory context). Negative-context training (`add_negative_prompt`) is an **optional enhancement** to be added only if the emergent contradiction-worsening proves too weak on the real corpus — not a precondition. Do not cite "trained anti-conditioning" as evidence unless negative-context training is actually enabled.

**Consequence:** **no separate Stage-1 oracle QLoRA is needed.** The teacher is the frozen base model. The dead oracle machinery (`Round2TrainConfig`, `oracle_cache.py`, `audit_oracle_coverage`, `_run_oracle_training` stub) and the plain-SFT `run_distillation`/`to_sft_columns` path are removed (lean/DRY; no backward-compat shims).

### The three anti-degeneracy modifications

1. **Fix the gate.** Init `scaler_B` to ones (mirror `scaler_A`), or reparameterize `B = B_raw · (1 + s)` with `s` zero-init. Identity-at-init, full gradient to both `B_raw` and the gate.
2. **Kill the L1 sink.** `gen_lora_l1_reg_coef = 0` (or negligible) for this objective.
3. **Diff-token masking.** Supervise only positions where teacher(with-context) ≠ base(no-context). Track `diff_agreement = mean(student_top1 == teacher_top1 | base_top1 ≠ teacher_top1)`, **not** `top1_agreement` (which cannot detect collapse — base already agrees with teacher ~84%).

---

## Data

Existing S3 corpus — **no re-mining**:
- `s3://elixirtrials-949678234935-eu-west-2-artifacts/training-data/github-pairs/all_unrolled.jsonl` — 8,517 rows, 91% with `pre_code`/`post_code`.
- `…/external_codereview.unrolled.jsonl` — 7,670 rows, 100% diff coverage.

Schema maps directly to D2L: `activation_text` → context, `teacher_text` → answer span, `pre_code`/`post_code` → diffs (unified-diff hunks, truncated ~8000 chars). STaR success filter retained. Engine-side `StepRecord` `pre_codes`/`post_codes` schema work is **deferred** (future corpora only; not a blocker here).

---

## Execution spine (cheap → decisive → expensive)

### Stage 0 — Static + synthetic discriminator (no corpus; minutes)
- **Static contracts (CPU):** assert `scaler_B.requires_grad`; assert it appears in the optimizer param groups; assert `strict=False` checkpoint load drops no `scaler_*`/`bias_*`/head keys (diff the state-dict keys, fail loud on missing).
- **Oracle-free synthetic NIAH overfit (GPU, under watchdog):** 3–5 records, unguessable needle (e.g. `MAGIC_OFFSET=73921`, `frobnicate_payload`) present only in the trajectory; held-out prompts without the fact; run the real D2L loop with the three modifications.
  - **Gate:** `real_hit > zero_hit` **and** `real_hit > contradictory_hit`.
  - **Instrumentation (JSON, saved):** per-component grad norms (`scaler_B`, head, aggregator), `scaler_B` stats, generated ΔW norm, real-vs-zero logit KL, real-vs-contradictory adapter cosine, skipped-record count, diff-token fraction, `diff_agreement`.
  - **Branch:** still collapses → mechanical bug (fix before proceeding); recalls → loop is sound, proceed. **This gate blocks all of Stage 1+.**

### Stage 1 — Real-corpus D2L training (GPU, under watchdog)
- Pull corpus from S3 to local; load via `corpus_path`. Map context/answer/diff fields. Apply STaR filter.
- Train with the modified D2L objective + negative/contradictory contexts.
- Log collapse diagnostics every N steps (same metrics as Stage 0).

### Stage 2 — §C conditioning-format alignment
- Align the **inference renderer** (`engine/graph.py` templates: `code`/`code_continue`, `ROLE/PROJECT/SUBTASK/PLAN/EXISTING CODE`) to the **training context format** (`## Task/## Current Code/## Review Feedback/## Revision`). Lower-risk than re-mining. No silent OOD mismatch.

### Stage 3 — §D inference-application correctness (now verifiable)
- Apply `combine_lora` + `get_head_bias()` in `hypernetwork.py::_to_peft_state_dict`/`generate_adapter_weights`; ensure PEFT adapter rank matches any bias-induced rank expansion (shape test first, else raise).
- Run activation extraction under `with base_model.disable_adapter():` (D2L does this itself) to prevent prior-adapter contamination; non-PEFT path still works.
- Re-measure the scaling sweet spot on the non-inert checkpoint. **Do not** reuse old base≈7.84 / cont≈12 numbers (measured against a collapsed adapter + OOD format).
- Each change validated against the §B probes (`tools/diag_retrieval_probe.py`, `diag_recall_probe.py`, `diag_continuation_probe.py`).

### Stage 4 — Gates + tiny benchmark (definition of done)
- **Promotion gates (content, not magnitude):**
  - `real_hit_rate > zero_hit_rate`, `> shuffled_hit_rate`, `> contradictory_hit_rate`;
  - trajectory-sensitivity: two distinct trajectories → adapter cosine well below 1.0;
  - continuation contrast: `real > zero > contradictory`;
  - `diff_agreement` over threshold.
  - `scaler_B absmax` is a **logged tripwire only**, never a promotion criterion.
- **Tiny benchmark:** 2–3 tasks × {base-only, zero adapter, real, shuffled, contradictory}. Require no regression vs base/zero and directional lift for real over controls. Directional smoke, not publication evidence.

---

## Cross-cutting requirements

- **RAM watchdog:** `/tmp/run_guarded.sh` is referenced by every probe but **not committed**. Recreate and commit it (e.g. `tools/run_guarded.sh`): monitors RSS, kills the job before the 15GB VM OOMs, preserves partial JSONL output. Every GPU step runs under it with `offload_base=False`. `free -g` before each model load.
- **CPU-importable invariant:** all GPU imports stay deferred inside function bodies; `pytest tests/unit/` and `mypy src/` run CPU-only.
- **Test-first per slice:** unit contracts (CPU) before GPU smoke; `ruff check .`, `mypy src/`, focused `pytest` before marking a slice done.
- **No magnitude promotion:** `adapter_diff`, `scaler_B`, "output changed" are tripwires/diagnostics only — never acceptance criteria.

## File map (indicative)

**Training (new lean path):**
- Add `src/rune/training/hypernet_distill.py` — D2L context-distillation entrypoint (teacher/student, top-K KL, diff-mask, negative contexts), wrapping `ctx_to_lora` primitives.
- Add `src/rune/training/collapse_metrics.py` — pure metric helpers (grad-norm summary, optimizer-membership assert, `diff_agreement`, ΔW norm).
- Modify `src/rune/training/orchestrator.py` — Stage-2 dispatches to `hypernet_distill`; remove oracle stage + empty-gate placeholder.
- Modify `src/rune/model/hypernetwork.py` — `scaler_B` init/reparam fix; `strict` load key audit; `combine_lora`/`get_head_bias`.
- Modify `src/rune/model/wrapper.py` — `disable_adapter()` during activation extraction.
- **Remove:** `src/rune/training/oracle_cache.py`, `Round2TrainConfig`, `_run_oracle_training`, plain-SFT `run_distillation`/`to_sft_columns`.

**Probes / gates:**
- Modify `tools/diag_*_probe.py` — JSON output for gate evaluation.
- Add `tools/diag_synthetic_overfit.py` — Stage 0 synthetic NIAH gate.
- Add `tools/run_guarded.sh` — committed RAM watchdog.
- Modify `src/rune/training/gate.py` — `evaluate_retrieval_gate()` (content gates); wire real bench scores.

**Engine (Stage 2 format alignment):**
- Modify `src/rune/engine/graph.py` + Jinja2 templates — render inference trajectory in training format.

## Risks / open checks
- `strict=False` may be masking missing keys today (Stage 0 static check resolves).
- Diff truncation (~8000 chars on `pre_code`) may weaken signal for large hunks — acceptable for this milestone; revisit if Stage 1 diff-token fraction is low.
- If Stage 0 synthetic gate fails after the gate fix + L1 removal + diff-mask, the cause is mechanical (dtype underflow, skipped records, load) — instrument and fix before Stage 1; do **not** proceed to corpus training.
