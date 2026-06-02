# Memory→edit utility + corrected-recipe checkpoint (Issue #52, Deliverable 2)

**Date:** 2026-06-01
**Issue:** [#52](https://github.com/ElixirTrials/rune/issues/52) — adapter-as-trajectory-memory
**Follows:** PR #53 (Deliverable 1 — Doc2LoRA positive control; cause of #49 isolated to the
training recipe; memory target = feedback-derived facts, not the diff).
**Status:** design approved (user); proceeding to plan + implementation under EOD time pressure.
**Full reasoning trail:** `instructions/scratchpad.md` (chronological).

---

## 1. Goal (two-part, gated, step-by-step)

**A. Memory→edit utility gate (cheap, fast, parallel).** Prove the load-bearing open caveat from
Deliverable 1 — *recall ≠ utility* — one way or the other: does conditioning on recalled
episodic **state** (goal / failure-facts) make the base produce the **request-appropriate edit**
better than no-adapter or a mismatched-episode adapter? Zero-shot on the Sakana control + read on
Rune episodes. This de-risks but does **not block** the training run.

**B. Corrected-recipe Rune checkpoint + honest pass@1 (the EOD deliverable).** Train Rune's *own*
hypernetwork (Qwen3.5-9B, the only stack the pass@1 bench runs on) with the **already-wired
specificity-aware contrastive objective**, the corrected episode serialization, and aligned
adapter templates; warm-start from the HPO checkpoint; small guarded sweep; then run the pass@1
bench. **Deliverable = a checkpoint on the corrected recipe + an honest pass@1 number (possibly
partial), labeled exploratory if the utility gate is not yet green.** Not a guaranteed win — the
heavy "install recall from a non-recall init" cost is real and may not fully resolve by EOD.

## 2. Settled design decisions (from #52 evidence + reviewer + advisor)

1. **Memory target = feedback-derived facts (goal / current-state / failure-critique), NOT the
   diff and NOT verbatim failed code.** C2: embedding code strings primes the base to *emit*
   them (the #49 trap; feedback-swap collapse +1.01→+0.17). The diff stays a downstream
   action/eval target, never a memory target. **Failure is internalized as the abstract critique
   ("what was wrong / what to avoid"), never as the rejected code string** (user directive; C2).
2. **Architecture = task spec in the PROMPT, episodic state in the ADAPTER** (already found in
   `adapter-as-memory-report.md`; structural prompt drives spec compliance, adapter supplies
   context the prompt omits). Memory(recall state) / policy(emit next edit) separation.
3. **Single-step episode is a true episodic-memory unit** that may embed its latest tried attempt
   as facts — this is data coverage, not an objective change (the internalize→recall objective is
   unchanged). One review pair = one tried attempt (`pre_code`=attempt, feedback=why rejected,
   `post_code`=accepted).
4. **Specificity guard (the #49 trap) is mandatory everywhere:** every measurement and the
   training objective use matched **vs mismatch vs zero**, never matched-vs-zero alone. Generic
   boosting (m−zero up, m−mismatch flat) is failure, not success.

## 3. Phased plan (step by step; build on each result; record partials)

### Phase 0 — Pin the contract (before any GPU spend)
- Confirm the pass@1 **bench config + current baseline** (memory: ~1.0 post-#50 on some config) so
  "works" is distinguishable from noise/regression. Pick the bench config that actually exercises
  adapter-supplied context (task-in-prompt, context-in-adapter), not a trivial one.
- Confirm **warm-start checkpoint** resolves (`s3://…/hypernet_hpo`); record its SHA + the loaded
  scaler_B (must be preserved — #50 collapse fix).
- **Log the exact episode serialization** (train + inference) as the experiment contract.

### Phase 1 — Memory→edit utility gate (cheap, parallel, non-blocking)
Five arms, **ranking primary**, free-gen secondary realism check:
1. in-context **upper bound** (goal+facts in prompt, no adapter) — the "is the task well-posed"
   sanity gate; if this fails the constructed task is ill-posed, NOT evidence against memory.
2. **zero** (no adapter, state withheld) — lower bound.
3. **matched** adapter. 4. **mismatch** adapter. 5. **feedback-swap** hard negative.

Two separated sub-tests on the same premined episodes (never averaged):
- **goal→edit:** prompt = `pre_code` with the request removed; correct next edit = the diff;
  distractors = same-file feedback-swap edits. Utility = matched ranks correct edit > zero/mismatch
  and approaches the upper bound.
- **avoid (difference-in-differences):** internalize the **critique** (not the failed code);
  scoring prompt = neutral candidate-comparison scaffold only; candidates = {accepted, rejected}.
  Signal = improvement in the (accepted−rejected) margin under **matched** memory vs **zero** and
  vs **mismatch** with similar local code/edit type (cancels the intrinsic-quality confound).
  This is **one-attempt** avoid; multi-attempt ordered history stays for synthetic/mined phases.

Report per-facet, per-negative-type, with bootstrap CI. Tooling reuses `tools/scoring_core.py`
and the Sakana harness (orphan branch) for the zero-shot read.

### Phase 2 — Corrected-recipe Rune training run (the long pole; start ASAP, background)
- **Data reformulation:** episodes carry goal + current-state + (when present) failure-critique
  facts; target framed toward the **edit/patch** with the edit-local mask, not full-file copy.
- **Objective:** `contrastive=True` in `hypernet_distill.py` (feedback-swap hard negatives, hinge
  on the edit-local span) + diff-masked term. This is the existing wiring.
- **Templates (directive #2, experiment contract):** align the engine's adapter-context
  serialization (`render_training_format_trajectory` / `code_template`) and the prompt templates
  (task-only prompt) to the trained episode format. Log both serializations.
- **Recipe = MVC:** warm-start = HPO checkpoint; fixed lora_rank/lr/warmup from known-good HPO
  params; **small guarded sweep** over `contrastive_weight` / `adapter_scaling` only (NOT a fresh
  50-trial Optuna). Max steps + selection metric (held-out m−mismatch on goal/edit, + retention)
  + stop criteria fixed in advance.
- **CLAUDE.md GPU rules:** `free -g` first; `offload_base=False`; runs under
  `tools/run_guarded.sh`; background multi-minute jobs; kill by exact PID only. 4-bit NF4 base.

### Phase 3 — pass@1 bench + verdict
- Run the pinned pass@1 bench on the new checkpoint vs the baseline.
- **Verdict honesty:** report the number even if partial; if Phase 1 gate was not green, label the
  checkpoint exploratory/product-risky. Distinguish "recipe is right but undertrained" (recall
  margins moved, pass@1 flat) from "recipe wrong" (margins flat) using the scorecard on the new
  checkpoint, not pass@1 alone.

## 4. Recording (every step, including partials)
- **`instructions/scratchpad.md`** — chronological decisions, numbers, dead-ends, partials.
- **MLflow** (exp `issue52-d2l-control` lineage / a new `issue52-recipe` exp) — params, metrics,
  artifacts, provenance for every run; no model files. Adapt as partial results arrive.
- Watch `instructions/reflections.md` for reviewer input; weigh and fold in.

## 5. Out of scope
Synthetic multi-step episodes (Phase 2 of the utility ladder), real-trajectory mining (Phase 3 of
the utility ladder), fresh open-ended HPO, base-model swap. These follow the EOD checkpoint.

## 6. Risks (stated, not hidden)
- **Recall is heavy** (dose-response: gemma 80k→+7.1, qwen-4b 20k→+2.6). Warm-start from a
  non-recall init may not install queryable recall in an EOD-sized run → pass@1 may not move even
  with the right recipe. Mitigation: report scorecard margins on the new checkpoint to separate
  "undertrained" from "wrong recipe."
- **Template mismatch** between train and inference serialization would masquerade as a recipe
  failure → logged as a first-class contract artifact (Phase 0).
- **Utility gate negative** → training optimizes recall without action; proceed only with the
  exploratory label.
