# Issue #52 Deliverable 2 — Handoff (2026-06-01)

**Branch:** `feat/issue52-doc2lora-positive-control` (PR #53). Latest commit **`2cb22a22`** = the
adapter-application contract fix (this session). Builds on Deliverable 1 (the Doc2LoRA positive
control; `docs/issue52-findings-2026-06-01.md`).
**Chronological record:** `instructions/scratchpad.md` (read its tail for the full blow-by-blow).
**Reviewer log:** `instructions/reflections.md` (file-based; weigh it).

---

## TL;DR — where we are

The session set out to migrate Rune's training paradigm toward **stateless episodic recall** and get
to a trained checkpoint + pass@1. It turned into a **deep root-cause hunt** that ended in a concrete,
committed fix. Net:

- **Decided (greenlit): migrate away from diff-as-memory → stateless episodic recall** of
  feedback-derived facts (goal / current-state / tried-critique). Diff is demoted to a downstream
  action/eval target (memory/policy separation). The "away from diffs" half is *empirically clinched*;
  the "toward recall" half is validated in principle and now has working infrastructure.
- **Found & fixed the blocker:** Rune's recall looked dead through its own stack purely because of an
  **adapter-application bug** (wrong scaling + missing `combine_lora`), NOT architecture/features.
  Fix committed in `2cb22a22`, **CPU-verified** (312 unit tests, mypy, ruff all green).
- **Next: GPU validation** (4 anchors below), then — if green — train the corrected recipe.

---

## The core finding (how the fix was derived)

Chain of elimination (all on the same 12 QA episodes, `qwen_4b_d2l` checkpoint):

1. Sakana's own stack recalls Rune facts **+1.6–2.3** (goal +2.235). Rune's functional path gave
   **+0.024** (noise). Same episodes/queries/scoring → **the application path was guilty**, not the
   probe, episodes, or features.
2. Ruled out: **probe/query-format** (identical QA queries both sides), **episode construction**
   (same episodes), **feature encoding** (Sakana-features vs Rune-features → A/B cosine **0.93**; the
   perceiver L2-normalizes and is robust to feature differences), **assembly/bias alone** (`combine_lora`
   at low scale barely moved it), **precision** (4-bit ≈ bf16).
3. **Root cause = scaling.** Sakana's `lora_forward` (`ctx_to_lora/modeling/lora_layer.py` +
   `patch_lora_forward`) applies `delta = (x·Aᵀ)·B * scaling` with **`scaling = lora_config.lora_alpha`**
   (qwen=45.25), NOT `alpha/r` (=5.66). Rune used `alpha/r` → **8× (=r) too weak**. Plus the functional
   path skipped `combine_lora`/head-bias. The apply *math* was already identical.
4. **Replication grid (bf16, qwen):** raw@5.66 +0.024 → raw@45.25 +0.191 → **combined@45.25 +0.823**
   (frac>0 = 1.00). Real episode-specific recall recovered through Rune's stack. Residual to Sakana's
   +2.235 (~2.7×) is the **ctx feature pipeline** (`tokenize_ctx_text` affixes + `PerLayerActivations`),
   the one piece not yet replicated.
5. **#49 re-examined (resolves the "was it really the recipe?" doubt):** across a full scaling sweep
   `checkpoint_step600.pt` stays **flat on m-mismatch (~0, coin-flip)** and **strongly negative on m-zero
   (−8 to −9, worsening with scale)** — its edit-emission adapter is *actively anti-recall*. The scaling
   fix rescued qwen but **not #49**: #49's "recipe failure" verdict **survives, strengthened**.

---

## What was implemented (commit `2cb22a22`, CPU-green)

Single shared application contract; the three bespoke apply paths now route through it.

- **`src/rune/model/adapter_contract.py`** (new): `assemble_adapter(hyp, lora_dict, n_chunks)` =
  `combine_lora` + head bias when `use_bias`; `effective_scaling(hyp)` = `lora_config.lora_alpha`
  (NOT alpha/r); `lora_delta` re-exports the one einsum (`hypernet_distill._lora_delta`).
- **`src/rune/model/wrapper.py`** `from_config`: `peft_scaling_params(alpha, r, use_bias)` →
  `(r_peft = 2r if use_bias else r, lora_alpha_peft = alpha·r_peft)` so PEFT's `alpha/r` equals the
  checkpoint `lora_alpha`. Also fixes the rank-16 hotswap crash.
- **`src/rune/training/hypernet_distill.py`** + **`tools/diag_recoverability.py`**: functional path
  (training + diag) now uses `combine_lora` + `scaling=lora_alpha`; diag `--scaling` defaults to the
  contract (override optional).
- **Tests:** `tests/unit/test_adapter_contract.py` (numerical equivalence to `ctx_to_lora.lora_forward`),
  `test_engine_apply_scaling.py` (PEFT arithmetic), `test_serialization_contract.py`.
- Repo-wide ruff format + deferred-GPU-import `# noqa: PLC0415`; mypy clean.

**NOT committed (experiment/diagnostic scratch, local only — per the "don't contaminate" rule):**
`tools/_pathab_rune.py`, `tools/_gate_load.py`, `tools/_bench_entry.py`, `tools/_distill_entry.py`,
`tools/scoring_core.py`, `tools/d2l_control/`, and `third_party/` (gitignored). These are needed to run
the GPU anchors — they exist locally. The D1 experiment code is on orphan branch
`experiment/issue52-doc2lora-positive-control`.

---

## NEXT: GPU validation anchors (run in order; review each before the next)

GPU rules (CLAUDE.md): `free -g` first; runs under `tools/run_guarded.sh`; 4-bit NF4 for the 9B
(bf16 fine for the 4B); kill by exact PID; log to MLflow `http://localhost:5000`.

1. **qwen_4b_d2l recall through the FIXED Rune path** (auto-resolves scaling to the contract):
   `uv run python tools/_pathab_rune.py --bf16`
   **PASS = goal m-mismatch ≈ +0.823** (climbing toward Sakana +2.2; the residual is the feature path —
   do NOT expect +2.2). This confirms the fix works end-to-end through Rune's stack.
2. **#49 anti-QA / flat-specificity anchor** (sanity that the fix didn't fabricate recall):
   `uv run python tools/diag_recoverability.py --ckpt /tmp/rune-ck-final/checkpoint_step600.pt --model-id Qwen/Qwen3.5-9B`
   **Expect still flat/negative** (contract scaling for #49 = its lora_alpha = 16).
3. **Functional-vs-engine logit parity** (UNVERIFIED — must build): no tool runs both the functional
   apply and the engine PEFT hotswap on one adapter. Build a small harness: for one trajectory,
   compute engine logits (`ModelWrapper.from_config` + `hotswap_adapter`) vs functional logits
   (`assemble_adapter` + `_functional_lora` at `effective_scaling`), assert `torch.allclose`. This is
   the reviewer's #1 check — the engine PEFT scaling (`alpha_peft = alpha·r_peft`) is only proven by
   *arithmetic* so far, not real-model parity.
4. **Generation-stability / MBPP pass@1 smoke at the contract scaling** — THE critical open test:
   `bash tools/run_guarded.sh /tmp/bench_smoke.log tools/_bench_entry.py --tasks-file benchmarks/mbpp_phase0_iter.json --model-id Qwen/Qwen3.5-9B --checkpoint-path <ckpt>`
   ⚠️ The engine now applies `scaling=lora_alpha` (qwen 45.25) — **exactly the scale the #50 work found
   "over-drives the adapter so structured generation never closes the JSON."** Recall needs `alpha`; the
   engine's xgrammar gen may break at `alpha`. If it does, treat it as a **decode/policy integration
   problem** (Sakana generates coherently at `alpha`, so a workable contract exists), NOT evidence
   against recall. Recall and usable pass@1 are **separate gates**.

---

## Open risks / things a fresh agent MUST keep in mind

- **Generation-stability is the real remaining hurdle** (anchor #4). The recall-vs-coherent-generation
  tension is now concrete in the engine config.
- **Training will wake the head-bias gradients for the first time.** `combine_lora` now routes
  `bias_A/bias_B` into the training autograd graph (they previously got zero gradient). This connects to
  the `scaler_B`-collapse history (commit `c3a83217`); the collapse tripwire thresholds may need
  retuning when training resumes.
- **Residual feature-path gap:** to fully match Sakana (+2.2 vs our +0.823), Rune would also need
  Sakana's ctx feature pipeline (`tokenize_ctx_text` + `PerLayerActivations`). Not required to proceed,
  but it's the known remaining delta.
- **Both qwen_4b_d2l and #49 are `use_bias=True`** (a prior workflow agent wrongly said qwen was
  use_bias=False by conflating `use_per_rank_bias`). Verified directly. `combine_lora` is active for both.
- **Scaling vocabulary** (don't reconflate — source of past bugs): checkpoint `lora_alpha` (the effective
  apply scaling) vs PEFT `lora_alpha_peft = alpha·r_peft` vs the runtime `adapter_scaling` knob.

---

## Forward plan beyond the anchors (the actual product goal)

1. **If anchors pass:** train Rune's hypernet with the **corrected recipe** — queryable episodic-recall
   objective (the existing `contrastive=True` specificity machinery in `hypernet_distill.py`), patches+
   facts data, warm-started from a recall-capable checkpoint, applied via the now-correct contract.
   Config draft: `configs/issue52_recipe_mvc_4b.yaml`.
2. **Data gap:** the "what we tried / failure-history" recall facet needs **mined engine trajectories**
   (current corpus is single-turn — only goal + one-attempt + current-state available). Mining
   `decompose→…→repair` runs unlocks it.
3. **Base/lane:** research lane = Qwen3-4B + `qwen_4b_d2l` warm-start (recall already exists). Product
   lane (later, if research lane is positive) = train a Sakana-style recall hypernet for a stronger
   *locally-quantizable* coder (~7B class) on cloud GPUs (train/deploy are decoupled).

## Key paths & env
- Checkpoints: qwen_4b_d2l = `third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`;
  #49 = `/tmp/rune-ck-final/checkpoint_step600.pt` (Qwen3.5-9B); warm-start HPO in S3 (`hypernet_hpo`).
- Sakana stack: `third_party/doc-to-lora/.venv` (transformers 4.51.3, flash-attn 2.8.3); harness
  `third_party/doc-to-lora/rune_episode_recall.py`. Rune venv = `uv run` (transformers 5.8).
- Corpus: `/tmp/rune-corpus/external_codereview.{train,val.clean}.jsonl`. Bench: `benchmarks/mbpp_*.json`.
- CPU RAM ~15GB (tiny) — `offload_base=False`, load the 9B in 4-bit.
- ALWAYS write your observations, plans, considerations, interpretation of results to instructions/scratchpad.md 
- Arm a monitor for instructions/reflections.md and respond to the critiques there. 