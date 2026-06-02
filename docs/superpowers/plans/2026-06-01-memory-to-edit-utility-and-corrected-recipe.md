# Memory→edit utility + corrected-recipe checkpoint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Rune's own hypernetwork (Qwen3.5-9B) with the already-wired specificity-aware contrastive objective on the corrected episode serialization, warm-started from the HPO checkpoint, and report an honest pass@1 — gated by a cheap memory→edit utility check and a matched-vs-mismatch success gate.

**Architecture:** The corrected objective already exists (`hypernet_distill.py` `contrastive=True` → `make_hard_negative` feedback-swap hinge on the edit-local span). The episode the adapter internalizes is `activation_text` (Task + Current Code + Review Feedback); the action target is `teacher_text − activation_text`. We (a) pin the experiment contract (bench config, baseline, warm-start ckpt, serialization hash), (b) run a cheap zero-shot utility gate in parallel, (c) run the contrastive distillation MVC, (d) success-gate then pass@1 it. Memory(facts)/policy(edit) separation; never embed failed code verbatim.

**Tech Stack:** Python 3.12 / uv, PEFT + transformers, the HyperLoRA hypernetwork, MLflow (localhost:5000), `tools/run_guarded.sh` RAM watchdog, pytest/ruff/mypy.

**Working method:** record every step + partial to `instructions/scratchpad.md` and MLflow; watch `instructions/reflections.md`; GPU rules (CLAUDE.md): `free -g` first, `offload_base=False`, runs under `run_guarded.sh`, background multi-minute jobs, kill by exact PID. Defaults (user may override): warm-start = `s3://…/hypernet_hpo`; "HPO-optimized" = reuse known params + a small guarded sweep over `contrastive_weight`/`adapter_scaling`, NOT a fresh 50-trial Optuna; EOD deliverable = corrected-recipe checkpoint + honest (possibly partial) pass@1, exploratory-labeled if the utility gate is not green.

---

## Phase 0 — Pin the experiment contract (no heavy GPU; do first, fast)

### Task 0.1: Identify the pass@1 bench config + current baseline

**Files:**
- Inspect: `src/rune/bench/runner.py`, `src/rune/config.py`, any `bench/tasks*.json`
- Record: `instructions/scratchpad.md`

- [ ] **Step 1:** Find the tasks file(s) and the bench config the runner expects.

Run: `ls -R tests/ bench/ 2>/dev/null | grep -iE "task|bench" ; grep -rnE "load_tasks|tasks_file|pass_at_1|to_thread|generate" src/rune/bench/runner.py | head`
Expected: the tasks JSON path + how `run_benchmark` scores pass@1.

- [ ] **Step 2:** Establish the baseline on the SAME tasks/config/checkpoint-loading path you will use for the new checkpoint: run pass@1 with (a) no adapter and (b) the previous-best checkpoint if available.

Run (GPU, background, after `free -g`):
`tools/run_guarded.sh /tmp/bench_baseline.log -m rune.cli bench --tasks-file <TASKS> --config <BASE_CFG>`
(If `run_guarded.sh` only takes a script, wrap the CLI call in a tiny `tools/_bench.py` that calls `rune.cli`.)
Expected: a baseline pass@1 number logged. **If the remembered "1.0 post-#50" is on a trivial/contaminated config, state that plainly and pick a config that exercises adapter-supplied context.**

- [ ] **Step 3:** Record baseline + exact config (tasks file, model_id, checkpoint_path, scaling) to scratchpad + MLflow (`issue52-recipe` experiment).

### Task 0.2: Confirm warm-start checkpoint resolves + scaler_B preserved

**Files:**
- Inspect: `src/rune/model/hypernetwork.py` (`_resolve_checkpoint_path`, `audit_checkpoint_keys`)

- [ ] **Step 1:** Resolve the warm-start checkpoint locally.

Run: `uv run python -c "from rune.model.hypernetwork import _resolve_checkpoint_path as r; print(r('s3://<HPO_CKPT>'))"`
Expected: a cached local path (downloads to `~/.cache/rune/checkpoints`). If it fails, fall back to best available checkpoint and note recall-install risk in scratchpad.

- [ ] **Step 2:** Verify the checkpoint's `scaler_B` is non-collapsed (the #50 fix) and that `audit_checkpoint_keys` reports no missing collapse-critical groups.

Run: `uv run python -c "import torch; sd=torch.load('<CACHED>', map_location='cpu'); import re; ks=[k for k in sd if 'scaler_B' in k]; print(len(ks), 'scaler_B keys; sample norm:', float(sd[ks[0]].abs().mean()) if ks else 'NONE')"`
Expected: non-zero scaler_B norm. Record SHA256 + scaler_B norm to scratchpad.

### Task 0.3: Serialization-contract hash helper (NEW, TDD)

**Files:**
- Create: `src/rune/training/serialization_contract.py`
- Test: `tests/unit/test_serialization_contract.py`

- [ ] **Step 1: Write the failing test**

```python
from rune.training.serialization_contract import episode_serialization_fingerprint

def test_fingerprint_is_stable_and_sensitive():
    a = {"train_template": "code_template", "infer_template": "code_template",
         "sample_episode": "## Task\nfoo\n## Current Code\nbar\n## Review Feedback\nbaz"}
    fp1 = episode_serialization_fingerprint(**a)
    fp2 = episode_serialization_fingerprint(**a)
    assert fp1 == fp2                          # stable
    assert isinstance(fp1, str) and len(fp1) == 64  # sha256 hex
    b = dict(a, infer_template="code_continue")
    assert episode_serialization_fingerprint(**b) != fp1  # sensitive to mismatch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_serialization_contract.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**

```python
"""Experiment-contract fingerprint: train/inference episode serialization must match."""
import hashlib


def episode_serialization_fingerprint(
    train_template: str, infer_template: str, sample_episode: str
) -> str:
    """SHA256 over the (train template, inference template, one rendered episode).

    Logged to every checkpoint + MLflow run so a pass@1 failure can be attributed
    to a template mismatch vs a recipe failure (reviewer, 2026-06-01).
    """
    h = hashlib.sha256()
    for part in (train_template, infer_template, sample_episode):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_serialization_contract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rune/training/serialization_contract.py tests/unit/test_serialization_contract.py
git commit -m "feat(#52): episode-serialization fingerprint for the train/infer template contract"
```

---

## The fast loop (re-orient, 2026-06-01): training is the goal; pass@1 is the fast check

The critical path is **fix Rune's own training → check pass@1 → iterate**. Training Qwen3.5-9B is the long pole (hours/run), so the loop is made fast by checking pass@1 *early and cheaply*, exploiting levers that need **no retrain**:

- **Eval-time levers are the biggest fast win.** `adapter_scaling` + prompt architecture (task-in-prompt / context-in-adapter) move pass@1 *without retraining*. Evidence: `adapter-as-memory-report.md` (scaling ≥0.49 → spec-divergent, 0.75 task_only works) and #50 (pass@1 0→1.0 was purely an 8× over-scaling fix). **Train once, then sweep scaling/prompt against pass@1.**
- **Small adapter-sensitive task subset** for the quick pass@1 signal during iteration; full bench only at the end.
- **Short training runs** for a directional pass@1 read before committing to a long run.
- **matched-vs-mismatch scorecard = cheap diagnostic** of *why* pass@1 moved (real memory vs generic boosting) — not a substitute for the pass@1 check itself.

**Guardrails (reviewer 10:35Z) — to keep the fast loop honest:**
- **Freeze the iteration subset before iterating.** Three distinct task sets: a tiny *iteration* subset (fast signal), a separate *holdout mini-bench* (never tuned on), and the *full/standard* bench (final). Choosing/revising the subset after seeing results turns pass@1 into a tuning target.
- **Keep one canonical fixed (scaling, prompt) setting across all checkpoints**, reported alongside the per-checkpoint sweep — so "progress" can't be just a different decode/prompt/scaling choice on the same checkpoint.
- **Interpretation forks:** pass@1 ↑ but matched-vs-mismatch flat = a *generic utility* win, not episodic memory (still useful, different claim). matched-vs-mismatch ↑ but pass@1 flat = memory present, engine/prompt can't exploit it yet → feed **prompt/action design**, not another blind training run.

The Sakana zero-shot utility probes are **off the critical path** — see Appendix A (optional offline diagnostic), run only if a pass@1 result is ambiguous and we need to confirm the probe/architecture isn't the issue.

---

## Phase 2 — Corrected-recipe distillation MVC (the long pole; start ASAP, background)

### Task 2.1: Define the MVC training config (NEW yaml; no placeholder values)

**Files:**
- Create: `configs/issue52_recipe_mvc.yaml`
- Inspect: `src/rune/training/d2l_train.py` (`D2LTrainConfig`), `hypernet_distill.py` (`DistillConfig`)

- [ ] **Step 1:** Confirm the distill entry point and which fields `DistillConfig` exposes beyond `D2LTrainConfig` (`contrastive`, `contrastive_weight`, `contrastive_margin`, `max_steps`, `early_stop_warmup`).

Run: `grep -nE "contrastive|max_steps|early_stop|adapter_scaling|val_diff_agreement|def run_distill|def main|__main__" src/rune/training/hypernet_distill.py | head`

- [ ] **Step 2:** Write the config (fixed recipe, warm-start, contrastive ON):

```yaml
# configs/issue52_recipe_mvc.yaml — corrected-recipe MVC (issue #52 Deliverable 2)
model_id: "Qwen/Qwen3.5-9B"
checkpoint_path: "s3://<HPO_CKPT>"        # warm-start (Task 0.2 resolved path)
corpus_path: "/tmp/rune-corpus/external_codereview.train.jsonl"
checkpoint_dir: "./checkpoints/issue52-recipe-mvc"
experiment_name: "issue52-recipe"
learning_rate: 2.0e-5                      # known-good HPO param
warmup_ratio: 0.1
lora_rank: 8
lora_alpha: 16
batch_size: 1                              # GPU/RAM safety; grad-accum carries effective batch
gradient_accumulation_steps: 8
max_seq_length: 2048
fp16: true
# corrected-recipe levers:
contrastive: true                          # specificity-aware: matched > feedback-swap hard-neg
contrastive_weight: 1.0                    # swept in Task 2.4
contrastive_margin: 1.0
max_steps: 600                             # MVC budget (issue49 generic run used 600)
early_stop_warmup: 150
```

- [ ] **Step 3: Commit**

```bash
git add configs/issue52_recipe_mvc.yaml
git commit -m "feat(#52): corrected-recipe MVC distill config (contrastive, warm-start)"
```

### Task 2.2: Align + snapshot the adapter-template serialization contract

**Files:**
- Inspect: `src/rune/engine/graph.py:65` (`render_training_format_trajectory`), `src/rune/templates/*.j2`, `src/rune/model/wrapper.py`
- Modify (if mismatch found): the engine's adapter-context render so inference internalizes the SAME episode serialization (`activation_text`: Task + Current Code + Review Feedback) that training does.
- Record: write the fingerprint (Task 0.3) into the run.

- [ ] **Step 1:** Render one training episode (`activation_text` from a corpus row) and one engine-inference adapter context for the same logical inputs; diff them.

Run: `uv run python -c "import json; r=json.loads(open('/tmp/rune-corpus/external_codereview.train.jsonl').readline()); print(r['activation_text'][:800])"`
Then render the engine side via `render_training_format_trajectory` with matched inputs and compare structure (headers, order, feedback block).

- [ ] **Step 2:** If they differ, align the engine inference serialization to the training one (architecture: **task spec in prompt, code+facts in adapter** — per `adapter-as-memory-report.md`). Keep diffs minimal; do not embed failed code verbatim.
- [ ] **Step 3:** Compute and log `episode_serialization_fingerprint(train_template, infer_template, sample_episode)` to MLflow + scratchpad for this run. **This is the experiment contract.**
- [ ] **Step 4: Commit** any template change with the rendered-sample + fingerprint noted in the commit body.

### Task 2.3: Pre-flight the run on a tiny budget (catch OOM / wiring before the real run)

- [ ] **Step 1:** `free -g` (need base+hypernet in 23GB GPU, `offload_base=False`; CPU RAM ~15GB).
- [ ] **Step 2:** Smoke run `max_steps=4`, `contrastive=true` under the watchdog:

Run: `tools/run_guarded.sh /tmp/recipe_smoke.log tools/_distill_entry.py --config configs/issue52_recipe_mvc.yaml --max-steps 4`
(Create `tools/_distill_entry.py` if the distill trainer has no script entry: a thin `if __name__=="__main__"` that loads the yaml into `DistillConfig` and calls the trainer.)
Expected: no OOM; contrastive hinge active (both `lp_matched` and `lp_neg` computed); loss finite; a checkpoint writes. **Known OOM risk:** 2 grad forwards × Qwen3.5-9B+perceiver. Mitigations if it OOMs: 8-bit Adam, `max_seq_length` ↓, ctx truncation, `expandable_segments`.

- [ ] **Step 3:** Record smoke result to scratchpad.

### Task 2.4: Launch the MVC run + small guarded sweep (background)

- [ ] **Step 1:** Launch the full `max_steps=600` run (background, watchdog). Selection metric = best held-out `val_diff_agreement` AND **m−mismatch** on goal/edit (specificity, not just m−zero) → `checkpoint_best.pt`.
- [ ] **Step 2:** Small guarded sweep ONLY over `contrastive_weight ∈ {0.5, 1.0, 2.0}` and `adapter_scaling` at eval (not a fresh Optuna). Each run logs to MLflow `issue52-recipe` with the serialization fingerprint + provenance.
- [ ] **Step 3:** As partials arrive, record m−mismatch / m−zero trajectory + retention to scratchpad; adapt (stop early if specificity is flat = generic boosting recurring).

---

## Phase 3 — Success gate, then pass@1 (verdict with honesty)

### Task 3.1: Matched-vs-mismatch success gate on the NEW checkpoint (before pass@1)

**Files:**
- Reuse: `tools/diag_recoverability.py` / `tools/scoring_core.py`, `src/rune/training/gate.py`

- [ ] **Step 1:** On the held-out clean split, score goal/diff/tail under matched / mismatch / zero for `checkpoint_best.pt`. **Gate:** m−mismatch must move positive on goal/edit, not just m−zero.
- [ ] **Step 2:** Log to MLflow + scratchpad. If m−mismatch is flat while m−zero rose → generic boosting (the #49 trap); pass@1 is then uninterpretable as memory — record and stop, do not over-claim.

### Task 3.2: FAST pass@1 loop — eval-time scaling/prompt sweep on the new checkpoint

This is the fast loop: one trained checkpoint, swept cheaply against pass@1 with **no retrain**.

- [ ] **Step 1:** On the **small adapter-sensitive subset**, sweep `adapter_scaling ∈ {0.1, 0.3, 0.5, 0.75, 1.0}` × prompt architecture {task-only, structural} for the new `checkpoint_best.pt`. (Scaling is an eval-time knob — confirmed in Task 2.4.)

Run (GPU, background): `tools/run_guarded.sh /tmp/bench_sweep.log tools/_bench.py --tasks-file <SMALL_TASKS> --config <NEW_CFG> --sweep-scaling`
Expected: pass@1 per (scaling, prompt) cell. Pick the pass@1-max cell.

- [ ] **Step 2:** Run the FULL bench at the best (scaling, prompt) on the SAME tasks/config/loading path as the Task 0.1 baseline.

Run: `tools/run_guarded.sh /tmp/bench_new.log tools/_bench.py --tasks-file <TASKS> --config <NEW_CFG>`
Expected: a pass@1 number directly comparable to baseline.

- [ ] **Step 3:** Record base / previous-best / new pass@1 side by side + the best (scaling, prompt) + the serialization fingerprint to MLflow + scratchpad.

- [ ] **Step 4 (iterate):** If pass@1 flat at every eval-time cell, the training lever is the issue, not scaling — adjust ONE training lever (contrastive_weight ↑, more steps, serialization fix) and re-run the short training → fast sweep. Record each iteration's pass@1 + m−mismatch so we build on partials.

### Task 3.3: Verdict (honest, possibly partial)

- [ ] **Step 1:** Write the verdict to scratchpad + a short findings note:
  - If pass@1 ↑ AND m−mismatch ↑ AND utility gate green → corrected recipe validated.
  - If m−mismatch ↑ but pass@1 flat → recipe right, likely undertrained (recall is heavy) — state as such.
  - If m−mismatch flat → recipe still wrong; do NOT claim success.
  - Label exploratory/product-risky if the Phase-1 utility gate was not green.
- [ ] **Step 2:** Note any follow-up (synthetic multi-step episodes, real-trajectory mining, longer training) for the next deliverable.

---

## Self-review notes (gaps to watch during execution)
- `run_guarded.sh` takes a **script**, not a module — Phase-0/2/3 GPU tasks may need thin `tools/_bench.py` / `tools/_distill_entry.py` wrappers; create them as the first sub-step of the relevant task.
- Exact distill entry point + whether `adapter_scaling` is a train-time or eval-time knob must be confirmed in Task 2.1/2.4 before the sweep.
- The utility-gate scripts live under `third_party/` (uncommitted, gitignored); mirror to the orphan experiment branch for posterity (do not commit to the Rune working tree).

---

## Appendix A — Sakana zero-shot utility probes (OPTIONAL, off critical path)

Run ONLY if a pass@1 result is ambiguous and we need to rule out "the probe/architecture, not the recipe." Zero-shot on the Sakana control via the orphan-branch harness + `tools/scoring_core.py`, reusing `build_rune_episodes`. Five arms (in-context upper bound / zero / matched / mismatch / feedback-swap), ranking primary.
- **goal→edit:** prompt = `pre_code`, request removed; candidates = correct diff + same-file feedback-swap distractors; matched should rank correct edit > zero/mismatch, approaching the upper bound. Sanity gate: if the upper bound can't rank it, the task is ill-posed (not anti-memory).
- **avoid (difference-in-differences):** internalize the **critique, never the failed code**; neutral scaffold prompt; candidates = {accepted, rejected}; signal = improvement in (accepted−rejected) under matched vs zero AND vs mismatch. One-attempt only.
Scripts live under `third_party/` (uncommitted, gitignored); mirror to the orphan experiment branch.
