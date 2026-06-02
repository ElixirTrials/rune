# Issue #52 Deliverable 3 — Handoff (2026-06-01)

**Branch:** `feat/issue52-doc2lora-positive-control` (PR #53). Builds on Deliverable 2
(`docs/issue52-deliverable2-handoff-2026-06-01.md`, commit `2cb22a22` = adapter-apply contract fix;
`a2b31275` = D2 diagnostic harness). This session **executed D2's GPU-validation phase** (the 4 anchors)
and produced a new positive product result.
**Chronological record:** `instructions/scratchpad.md` (tail). **Reviewer log:** `instructions/reflections.md`.

---

## Goals (enumerated)

**Product goal (north star):** a local-first coding agent that encodes coding trajectories into LoRA
adapters via the ctx-to-lora hypernetwork, runs the single-loop engine, and scores useful pass@1.

**Issue #52 goal:** migrate Rune's training paradigm from *diff-as-memory* to **stateless episodic
recall** of feedback-derived facts (goal / current-state / tried-critique), with diff demoted to a
downstream action/eval target (memory/policy separation).

Sub-goals, in dependency order:
1. **Prove recall is recoverable through Rune's stack** (not dead) — i.e. the apply contract is correct. ✅ this session
2. **Prove the corrected contract is engine-compatible** — structured (xgrammar) generation stays stable and
   yields useful pass@1 at the contract scale. ✅ this session (10-task slice)
3. **Prove the pass@1 lift is episode-specific** (matched-adapter > mismatch), not a generic coding-prior
   boost. ✅ **answered at ranking level (this session)** — *negative for the bench setting:* the 7/10
   lift is generic generation **discipline**, not additive matched memory (PRESENT-regime matched ≤
   mismatch). Task-specific info genuinely **exists** in the adapter (ABSENT-regime matched ≫ mismatch)
   but is **name/signature-dominated** and redundant with the in-prompt task. See specificity probe below.
4. **Warm-start fine-tune Rune's hypernet (the HyperLoRA perceiver) from `qwen_4b_d2l` on the same Qwen3-4B
   base (base frozen)**, corrected recipe: queryable episodic-recall objective (`contrastive=True`) over the
   goal / current-state / tried-critique facets, Rune's own feature extraction, via the corrected contract.
   *Not* a from-scratch retrain or a new base model — that is the separate, deferred cloud "product lane". Record every experiment (and associated traces, metrics) to MlFlow. ⏳ next
5. **Close the recall residual** (Rune +0.823 vs Sakana +2.235) if/when recall — not pass@1 — is the binding constraint. ⏳ deferred lever

---

## TL;DR — UPDATE (specificity gate resolved at ranking level)

**Goal 3 is answered (ranking-level, the primary metric).** A cheap logprob **specificity probe**
(`tools/_specificity_probe.py`, derangement matched-vs-mismatch on the frozen 10, both task-in-prompt
and task-hidden regimes, reference-solution span split into signature vs body) **replaced** the 2-hr
3-arm generation run (advisor + reviewer agreed: ranking is primary, n=10 pass@1 too noisy to resolve a
small gap). Findings:
- **The 7/10 is generic generation discipline, not matched episodic memory.** With the task in the prompt
  (faithful to the bench), matched ≤ mismatch on every task (full-span m-mismatch **−0.18**, frac 0.00),
  while both beat zero (m-zero **+1.22**) — i.e. the adapter's net bench benefit is anti-degeneration that
  a *mismatched* adapter supplies just as well.
- **Task-specific info genuinely exists, but it is name/signature-dominated.** With the task hidden
  (NIAH-style), matched ≫ mismatch on all 10 (full +1.17), and the split shows this is concentrated in the
  **`def <name>(…)` signature** (m-mismatch **+3.84**, all 10) far more than the **body/algorithm**
  (m-mismatch **+0.14**, range −0.31…+0.62, real-but-weak-and-inconsistent — present where the body has a
  discriminative token like `key=sum`/regex, absent for trivial formulas like `4*a`).
- **Matched memory mildly *fights* the in-prompt name.** PRESENT-regime signature m-mismatch **−0.28**
  (frac 0.30) — the adapter recalls the name with imperfect casing and at contract scale can override the
  correct prompt name. This *mechanistically explains* the confirmation-rerun casing misses
  (`sorted_matrix`≠`sort_matrix`, `find_volume`≠`find_Volume`).
- **Scope:** this does **not** refute #52's episodic-memory bet for **hidden** multi-turn
  feedback/tried/critique facts (not in the prompt at all) — the ABSENT regime is positive evidence the
  recall mechanism works through Rune. It scopes the negative narrowly: *single-turn, task-already-in-prompt
  MBPP gives the adapter no additive task utility, and its strongest recalled content is the label.*

## TL;DR — where we are (prior, retained)

The D2 contract fix is now **validated end-to-end on GPU** (anchors 1–3 PASS). The product signal is
encouraging: at the corrected contract scale (`adapter_scaling=1.0` ⇒ effective `lora_alpha`=45.25 =
Sakana's apply convention) the engine recovers recall, keeps structured generation **stable** (0
JSON-close failures), and scores **pass@1 = 7/10 on the frozen 10-task Phase-0 slice** — the feared
recall-vs-generation break did **not** reproduce. **Baseline comparability is now confirmed** (the
same-setup `scaling=0.0` re-run emits *real* generated code, so `0/10` is a fair base): the base's failures
are output **degeneration** — 2000+-char rambling, markdown fences, multiple functions, non-closing/syntax
errors — whereas the adapter at the contract scale produces clean, tight, well-formed code every time. **But
that sharpens the remaining gate rather than closing it:** the adapter's *visible* benefit is generation
**discipline (anti-degeneration)**, which may be a **generic prior**, not episode-specific memory — and on
the 3-task confirmation subset both arms in fact **tied 1/3** (the adapter's two misses were function-name
*casing*, e.g. `sorted_matrix`≠`sort_matrix`, not bad code; the base passed the one it kept terse). So (3)
**episode-specificity** (matched-adapter > mismatch on the same tasks) is the **decisive** next gate. Goals
1–2 are met **on this slice**; goal 3 is the gate.

---

## What we validated this session (the 4 anchors)

All GPU runs under `tools/run_guarded.sh` (RAM watchdog), 4-bit for the 9B / bf16 for the 4B, `free -g` first.

| # | anchor | result | verdict |
|---|--------|--------|---------|
| 1 | `qwen_4b_d2l` recall via FIXED Rune path (`_pathab_rune.py --bf16`) | goal m-mismatch **+0.824** (combined arm, scaling=45.25), frac(m-mis>0)=**1.00**; raw arm +0.195 | **PASS** (target +0.823; residual to Sakana +2.2 = feature path) |
| 2 | #49 anti-QA sanity (`diag_recoverability.py`, 9B, scaling=16) | goal +0.083 / m-zero **−8.57**; diff +0.24/−9.31; tail +0.05/−11.31 | **PASS** — flat + anti-recall; #49 recipe-failure verdict holds, strengthened |
| 3 | engine-PEFT vs functional-contract logit parity (`_parity_engine_vs_functional.py`) | **adapter_diff=0**, max_abs 0.50 / mean **0.041**, allclose=True, last-token argmax **match**; regime use_bias=True, r_peft=16, scaling=45.25 | **PASS (airtight)** — engine apply == functional contract on the real model |
| 4a | Sakana free-form baseline ("what worked") on `qwen_4b_d2l` (`rune_code_recall.py`, Sakana venv) | mean m-mismatch **+2.597**, m-zero +8.35, **gen_accuracy 0.88** (7/8 facts), coherent | **PASS** — contract scale is generation-viable (Sakana-proven) |
| 4b | Rune free-form eyeball at contract scale (`_rune_freeform_gen.py`, engine path) | fluent coherent gen over 64 tok; wrong facts = recall residual, NOT degeneracy | **PASS** — Rune's own stack generates coherently at `lora_alpha` |
| 4c | **xgrammar/MBPP smoke** (`_bench_entry.py`, `adapter_scaling=1.0`, 10 Phase-0 tasks) | **pass@1 = 0.7 (7/10)**, **0** truncation/JSON-close failures; base (`scaling=0.0`) = 0/10 | **POSITIVE** — structured gen stable; the #50 break did not reproduce |
| 4d | confirmation re-run (3-task subset, code dump): contract `1.0` vs base `0.0` | both **1/3** on subset; base failures = **degeneration** (2–3k-char rambling, non-closing), adapter = clean tight code; adapter misses = function-name casing | **comparability OK; specificity OPEN** — `0/10` base is fair (real code); adapter benefit looks like anti-degeneration discipline, mechanism not yet shown episodic |

### The core finding
`adapter_scaling=1.0` **is** Sakana's scaling: PEFT bakes effective scaling = `lora_alpha` (`lora_alpha_peft/r_peft`),
the runtime knob multiplies on top, so total = `lora_alpha × adapter_scaling` → at 1.0 = `lora_alpha` = 45.25.
Anchor #3 proves this at the logit level (engine == functional, argmax-identical). At that scale the engine
gives both recall and stable structured gen + 7/10 — so the corrected contract is engine-compatible.

---

## Artifacts produced (uncommitted)

- **`tests/unit/test_engine_functional_parity.py`** (NEW, committable) — CPU toy-tensor PEFT-vs-functional
  equivalence test = the reviewer's check #1. 2 passed; ruff/mypy(src) clean. **Recommend committing.**
- **`tools/_parity_engine_vs_functional.py`** (NEW, local-only/underscore) — anchor-#3 real-model GPU parity
  harness (asserts `adapter_diff<1e-3`, stamps regime, mean-drift backstop). Built via a CPU-only workflow +
  adversarial review (verdict: parity genuine, high confidence, all issues minor → fixed).
- **`tools/_rune_freeform_gen.py`** (NEW, local-only) — Rune-side free-form generation eyeball.
- **`tools/_specificity_probe.py`** (NEW, local-only) — the goal-3 ranking probe: derangement
  matched-vs-mismatch on the frozen 10, task-present + task-hidden regimes, reference-solution span split
  into signature/body, weight-space ‖ΔA‖ sanity (mean 0.43). One model load (~2 min, bf16). Deterministic
  derangement `i→(i+1)%10`; raises on any reference-solution/derangement miss (no silent identity fallback).
  Prints an AUDIT pair (matched vs partner rendered trajectory). Reuses `tools/scoring_core.py`.
- **`tools/_bench_entry.py`** (MODIFIED) — added a per-task code/pass/stderr dump (diagnostic visibility).
- mypy: enforced scope (`mypy src/`, what CI + Makefile run) is **clean**; new test + tools' substantive type
  errors fixed (remaining `import-untyped` are scope artifacts from running mypy outside `src/`, never in CI).

---

## Open risks / things to keep in mind

- **Specificity (goal 3) — RESOLVED at ranking level (this session).** The mismatch control was run as a
  cheap **logprob probe** (derangement, not a 2-hr generation arm): **matched ≈ mismatch with the task in
  the prompt** (full −0.18, both ≫ zero) ⇒ the 7/10 is a **generic anti-degeneration prior, not matched
  episodic memory**. Adapter task-specificity *exists* but is **name/signature-dominated** (hidden-task sig
  +3.84 vs body +0.14) and matched even mildly **fights** the in-prompt function name (casing misses). The
  episodic-memory claim for **hidden** facts is *not* refuted (hidden-task matched ≫ mismatch is positive
  evidence) — but it must be tested on facts that **cannot be solved by function-name recall** (required
  branch condition, prior failed approach, critique constraint, state invariant), not on task labels.
- **The 10-task Phase-0 slice is adapter-sensitive — freeze it.** Valid for fast iteration; NOT a broad pass@1
  claim. Keep distinct from any holdout / full-bench result.
- **Recall residual = feature mismatch, not inferiority.** `qwen_4b_d2l`'s perceiver was trained on Sakana's
  `ctx_encoder` features (`tokenize_ctx_text` affixes + `PerLayerActivations`, a separate encoder model); Rune
  feeds it base-model hidden states (one model, no encoder). A/B cosine 0.93 → the +0.823-vs-+2.2 gap is
  off-distribution warm-start, not a proven ceiling. Native training (goal 4) should dissolve it; importing
  Sakana's encoder is the fallback (goal 5), weighed against the tiny-RAM / single-base-model GPU constraint.
- **Training will wake head-bias gradients for the first time** (`combine_lora` routes `bias_A/bias_B` into
  autograd). Watch the `scaler_B`-collapse tripwires (commit `c3a83217`) in the first steps when training resumes.
- **Retry-exhaustion speed bug:** failing tasks burn all 4 retries with long generations (~45 min for 10 tasks).
  Real, but orthogonal to the contract — a separate engine-loop efficiency issue.

---

## Next steps (enumerated)

1. **Specificity gate (goal 3) — DONE (ranking probe; see TL;DR update + `tools/_specificity_probe.py`).**
   Verdict: 7/10 = generic discipline, not matched memory; adapter task-specificity is name/signature-
   dominated; matched mildly fights the in-prompt name. The 2-hr 3-arm generation run was **skipped** (ranking
   is primary; n=10 pass@1 too noisy to resolve the small matched-vs-mismatch gap the probe already
   characterized). The two forward levers below replace it:
   - **(A) Signature enforcement — DONE, validated on the name-contract mode.** Augmenting each task
     description with an explicit *exact-name* instruction (path-clean: `_is_simple_task` unchanged on all 10,
     so prompt-confounded not path-confounded) → **pass@1 = 9/10** (MLflow `895ccda7fc12473d889a002e2e42fabf`).
     The two *known* name-casing failures **flipped to pass with the correct name**: mbpp/12 `sorted_matrix`→
     `sort_matrix`, mbpp/14 `find_volume`→`find_Volume`. Soft prompt pressure *beat* the adapter's recalled
     signature. The lone remaining failure (mbpp/57) is **semantic/type** (correct name, returns `'321'`≠`321`)
     — a different class, pointing to a return-type-contract lever, not name. *Caveat:* 9/10-vs-7/10 is
     suggestive, not a clean controlled delta (baseline per-task code was not saved); the defensible claim is
     the mbpp/12+14 transition, with no visible regression on the other tasks.
   - **(B) Hidden-fact utility probe — ATTEMPTED; walled off by corpus quality (this session).** Built the
     one-step `avoid` task on external_codereview (rejected pre-side / accepted post-side / review critique)
     and ran the **in-context ceiling gate first** (`tools/_avoid_ceiling_probe.py`, base-only): does the
     critique *in the prompt* shift the accepted-over-rejected preference (DiD vs no-critique)? First cut was
     flat (mean DiD −0.02) but **bimodal**; an a-priori **single-hunk** filter (exactly one replace hunk —
     removes the wrong-hunk confound) moved it to **mean +0.17, frac 0.53** — short of the pre-registered
     frac ≥ 0.6 bar. Diagnostic stratification (suggestive only, not a retroactive pass): where the base is
     uncertain (headroom) the critique *does* determine the edit (DiD **+0.52**, n=10); where the accepted edit
     is already obvious (ceiling, base pref ≥1.5) it can't (DiD **−1.38**, n=3). **Conclusion:**
     external_codereview mixes directive and non-directive feedback, and the clean-signal subset can't be
     isolated *structurally* — only by content filtering that would leak the answer into a memory test. So this
     corpus **cannot yield a clean avoid-utility number**; even oracle in-prompt delivery is only +0.17, so an
     adapter (lossier channel) would be underpowered at n=30. **Redirect: mine purpose-built failure-bearing
     trajectories** (critique = binding constraint by construction). *Mechanism (separate, cheap, still clean):*
     the critique-**recall** feedback-swap test on `_pathab`'s +0.82 goal facet (feedback-bound vs code-echo) —
     but recall ≠ utility. *Optional pilot:* a predeclared directive/low-headroom + normalized-critique arm.
2. **Warm-start fine-tune the hypernet (goal 4) — now *purpose-gated*, not auto-next.** The 7/10 already shows
   the warm-start is product-useful, so training is no longer needed merely to "prove the fixed contract works."
   **After the goal-3 arms, decide whether to train and for what** — sharper purposes: lift *specificity* /
   *holdout* pass@1, or adapt *facts/critique serialization* — rather than as the default next move. If pursued:
   continue-train the perceiver from `qwen_4b_d2l` on Qwen3-4B (base frozen), corrected recipe (`contrastive=True`
   in `hypernet_distill.py`), patches+facts data, Rune's own feature extraction (train == inference), via the
   corrected contract. Draft config: `configs/issue52_recipe_mvc_4b.yaml`.
   - **Trigger (all three):** goal 3 green · failure-history corpus mined (step 3) · collapse smoke clean.
   - **Start now, in parallel (none block on goal 3):** finalize `issue52_recipe_mvc_4b.yaml`; run a few-step
     **collapse smoke-train** (warm-started; watch `scaler_B` / `collapse_metrics` tripwires from step 0, since
     `combine_lora` now wakes the bias gradients); kick off the trajectory mining in step 3.
3. **Mine engine trajectories for the failure-history facet** (the pacing item for goal 4). Current corpus
   (`external_codereview`) is single-turn — goal + one-attempt + current-state only; the "what we tried / why it
   failed" facet needs mined `decompose→…→repair` engine runs.
4. **Close the recall residual only if it's the bottleneck (goal 5):** cheap first — port `tokenize_ctx_text`
   affixes; fall back to Sakana's `PerLayerActivations`/`ctx_encoder` only if native training + affixes don't close it.
5. **Housekeeping:** commit `tests/unit/test_engine_functional_parity.py` (real artifact); keep the underscore
   tools local. Confirm the `scaling=0.0` baseline is apples-to-apples on the frozen subset.

---

## Key paths & env

- Checkpoints: `qwen_4b_d2l` = `third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`
  (base `Qwen/Qwen3-4B-Instruct-2507`); #49 = `/tmp/rune-ck-final/checkpoint_step600.pt` (`Qwen/Qwen3.5-9B`).
- Sakana stack: `third_party/doc-to-lora/.venv` (transformers 4.51.3) — NOT `uv run` (transformers 5.8).
- Bench: `benchmarks/mbpp_phase0_iter.json` (10 tasks, frozen slice; subset = `mbpp/11,12,14`). MLflow
  `http://localhost:5000`, experiment `issue52-recipe` (id 57). **Run IDs (reproducibility):**
  full 10-task — matched `2ccd712ebf49` (7/10), base `8a98f9deb10a` (0/10); 3-task subset code-dump —
  matched `6c5063ec7f43` (1/3), base `7486081b9d13` (1/3). Per-task generated code is dumped to the run
  logs (`_bench_entry.py` stderr); the engine seed comes from `cfg.seed` (deterministic per task index).
  **Reproducibility caveat:** the decisive harnesses (`_pathab_rune.py`, `_parity_engine_vs_functional.py`,
  `_bench_entry.py`) are local-only scratch — before final PR, commit minimal repro scripts or these IDs/commands.
- Anchor commands (in order): `uv run python tools/_pathab_rune.py --bf16`;
  `tools/run_guarded.sh <log> tools/diag_recoverability.py --ckpt /tmp/rune-ck-final/checkpoint_step600.pt --model-id Qwen/Qwen3.5-9B`;
  `tools/run_guarded.sh <log> tools/_parity_engine_vs_functional.py --bf16`;
  `tools/run_guarded.sh <log> tools/_bench_entry.py --tasks-file benchmarks/mbpp_phase0_iter.json --model-id Qwen/Qwen3-4B-Instruct-2507 --checkpoint-path <qwen_4b_d2l> --adapter-scaling 1.0`.
- GPU rules: `free -g` first; tiny CPU RAM (~15GB) → `offload_base=False`, 9B in 4-bit; runs under `run_guarded.sh`.
- Always log observations to `instructions/scratchpad.md`; respond to critiques in `instructions/reflections.md`.
