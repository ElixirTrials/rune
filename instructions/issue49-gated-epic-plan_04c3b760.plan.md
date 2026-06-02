---
name: issue49-gated-epic-plan
overview: "Revised peer-reviewed plan for Issue #49. The plan is test-first, treats collapse cause as a hypothesis until instrumented, blocks HPO/full benchmarks until retrieval gates pass, and separates cheap unit contracts from GPU smoke gates."
todos:
  - id: evidence-and-red-tests
    content: Capture current broken behavior with red tests and baseline probe artifacts.
    status: pending
  - id: corpus-schema
    content: Persist and mine canonical trajectory, prompt, completion, pre_codes, and post_codes fields.
    status: pending
  - id: hypernet-kd
    content: Replace the plain SFT distillation path with real hypernetwork KD plus collapse instrumentation.
    status: pending
  - id: oracle-stage
    content: Wire Stage-1 oracle artifact production and registry coverage before Stage-2 training.
    status: pending
  - id: inference-contracts
    content: Fix adapter application contracts with unit tests and defer empirical scaling until a non-inert adapter exists.
    status: pending
  - id: retrieval-gates
    content: Promote retrieval, trajectory-sensitivity, continuation, and pass@1 controls to first-class gates.
    status: pending
  - id: tiny-then-scale
    content: Run synthetic overfit, tiny benchmark, and only then long benchmark/HPO work.
    status: pending
isProject: false
---

# Issue #49 Gated Two-Stage Retrain Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover a non-collapsed HyperLoRA training path and prove that generated adapters retrieve trajectory content, not merely perturb outputs.

**Architecture:** Keep Rune's existing single-package layout. Use one canonical mined/training record schema, restore Stage-1 oracle training before Stage-2 hypernetwork KD, and gate promotion on retrieval/contrast controls before pass@1 runs. Treat `scaler_B` collapse as confirmed behavior but keep the causal mechanism as a hypothesis until gradient and optimizer instrumentation prove it.

**Tech Stack:** `uv`, Python 3.12, pytest, ruff, mypy, PEFT, transformers, ctx-to-lora HyperLoRA, existing `tools/diag_*.py` probes.

---

## Peer Review Findings Fixed In This Revision

- **The original plan over-promised 10-15 minute gates.** Some checks are cheap unit contracts; model-load and synthetic-overfit gates are GPU jobs and must run under `free -g` plus the RAM watchdog. This plan labels those separately.
- **The plan treated the root cause as fully known.** `scaler_B approx 0` and adapter inertness are confirmed; the causal path still needs `requires_grad`, optimizer membership, and gradient-norm evidence for `scaler_B`, output heads, and aggregator parameters.
- **The wired repo path is not HyperLoRA KD.** `src/rune/training/d2l_train.py::run_distillation()` currently trains a PEFT SFT model after `to_sft_columns()` drops `trajectory`, `pre_codes`, and `post_codes`. The revised plan turns that into the first red test instead of assuming the diff collator can be patched in place.
- **Diff side channels are not produced.** `StepRecord`, `session_log.write_session()`, and `miner.extract_trajectories()` do not persist `pre_codes` or `post_codes`. The plan now makes the corpus schema the first implementation slice.
- **Inference fixes were ordered as runtime-verified even though the adapter is inert.** Adapter application fixes stay in scope, but their pre-training gate is shape/key/contamination unit coverage. Empirical scaling and retrieval validation wait until a synthetic non-inert adapter exists.
- **Acceptance thresholds were vague or unsafe.** `scaler_B absmax > 0.05` is a tripwire only, never a promotion criterion. Promotion requires content retrieval over zero, shuffled, and contradictory controls.
- **Conditioning-format alignment was missing.** Training records must preserve the serve-time trajectory format or the inference renderer must intentionally match the training format. No silent OOD mismatch.
- **HPO was premature.** Current HPO optimizes a constant empty success gate. HPO stays blocked until synthetic retrieval and tiny pass@1 gates are meaningful.

## File Map

**Corpus and schema**
- Modify: `src/rune/engine/state.py`
- Modify: `src/rune/engine/graph.py`
- Modify: `src/rune/mining/session_log.py`
- Modify: `src/rune/mining/miner.py`
- Modify: `src/rune/training/d2l_train.py` or delete the SFT-only projection path if replaced
- Test: `tests/unit/test_graph_records.py`
- Test: `tests/unit/test_session_log.py`
- Test: `tests/unit/test_miner.py`
- Test: `tests/unit/test_d2l_train.py`

**Training**
- Modify: `src/rune/training/orchestrator.py`
- Modify: `src/rune/training/config.py`
- Modify: `src/rune/training/gate.py`
- Modify: `src/rune/training/oracle_cache.py`
- Add: `src/rune/training/hypernet_distill.py`
- Add: `src/rune/training/collapse_metrics.py`
- Test: `tests/unit/test_orchestrator.py`
- Add: `tests/unit/test_hypernet_distill.py`
- Add: `tests/unit/test_collapse_metrics.py`
- Test: `tests/unit/test_gate.py`

**Inference application**
- Modify: `src/rune/model/hypernetwork.py`
- Modify: `src/rune/model/wrapper.py`
- Modify: `src/rune/model/adapter.py` only if rank/scaling helpers are shared
- Test: `tests/unit/test_wrapper.py`
- Add: `tests/unit/test_hypernetwork_peft_mapping.py`

**Probes and benchmark gates**
- Modify: `tools/diag_recall_probe.py`
- Modify: `tools/diag_retrieval_probe.py`
- Modify: `tools/diag_continuation_probe.py`
- Modify: `src/rune/bench/runner.py` only after retrieval gates pass
- Test: `tests/unit/test_gate.py`
- Test: `tests/unit/test_bench_runner.py`

## Global Execution Rules

- [ ] Before any model load, run `free -g`. Keep `offload_base=False` unless RAM headroom is proven; this VM has little CPU RAM.
- [ ] Long GPU jobs run under the RAM watchdog used by the repo, not bare Python.
- [ ] Use `uv run` for every Python command.
- [ ] After each implementation slice, run focused unit tests first, then `uv run ruff check .`, `uv run mypy src/`, and relevant pytest suites before marking the slice complete.
- [ ] Do not run HPO, full MBPP/HumanEval, or scaling sweeps until the synthetic retrieval gate passes.
- [ ] Do not use magnitude-only checks (`adapter_diff`, `scaler_B`, "output changed") as promotion criteria.

## Task 0: Baseline Evidence And Red Tests

**Purpose:** Freeze the broken current behavior so later changes cannot accidentally reintroduce plain SFT, side-channel loss, or empty success gates.

**Files:**
- Modify: `tests/unit/test_d2l_train.py`
- Modify: `tests/unit/test_orchestrator.py`
- Modify: `tests/unit/test_gate.py`
- Read-only evidence: `tools/diag_recall_probe.py`, `tools/diag_retrieval_probe.py`, `tools/diag_continuation_probe.py`

- [ ] **Step 0.1: Add a red test proving `to_sft_columns()` currently drops training signal**

Add a test that expects trajectory and diff side channels to survive projection:

```python
def test_training_projection_preserves_hypernet_signal_columns() -> None:
    records = [{
        "trajectory": "ROLE: coder\nneedle=73921",
        "prompt": "implement",
        "completion": "done",
        "pre_codes": ["def f(): pass"],
        "post_codes": ["def f(): return 73921"],
        "metadata": {"phase": "code", "benchmark": "synthetic"},
    }]

    rows = to_sft_columns(records)

    assert rows == [{
        "trajectory": "ROLE: coder\nneedle=73921",
        "prompt": "implement",
        "completion": "done",
        "pre_codes": ["def f(): pass"],
        "post_codes": ["def f(): return 73921"],
        "metadata": {"phase": "code", "benchmark": "synthetic"},
    }]
```

Expected now: FAIL, because current code returns only `prompt` and `completion`.

- [ ] **Step 0.2: Add a red test proving the success gate cannot accept empty score maps**

`tests/unit/test_orchestrator.py` should assert that `run_training_pipeline()` never calls `_run_success_gate({}, {})` in a real run. The current code logs a warning and does exactly that; the test should fail before implementation.

- [ ] **Step 0.3: Capture cheap probe output as baseline artifacts**

Run only if a checkpoint is already available locally. Do not download or train.

```bash
free -g
uv run python tools/diag_recall_probe.py --help
uv run python tools/diag_retrieval_probe.py --help
uv run python tools/diag_continuation_probe.py --help
```

Expected: probe CLIs are callable. If any probe lacks a JSON/output mode needed by later gates, fix the probe harness before training code.

## Task 1: Canonical Corpus Schema And Format Alignment

**Purpose:** Make every mined record carry the information the hypernetwork needs: conditioning text, supervised target, exact serve-time prompt, and pre/post code diffs when code changed.

**Files:**
- Modify: `src/rune/engine/state.py`
- Modify: `src/rune/engine/graph.py`
- Modify: `src/rune/mining/session_log.py`
- Modify: `src/rune/mining/miner.py`
- Modify: `src/rune/training/d2l_train.py`
- Test: `tests/unit/test_graph_records.py`
- Test: `tests/unit/test_session_log.py`
- Test: `tests/unit/test_miner.py`
- Test: `tests/unit/test_d2l_train.py`

- [ ] **Step 1.1: Extend `StepRecord` with optional diff side channels**

Add fields:

```python
pre_code: str | None = None
post_code: str | None = None
```

For `code` and `repair` actions, `pre_code` should be the subtask's prior code context and `post_code` should be extracted generated code. For `integrate`, use the integrated pre/post body. For non-code actions, both fields remain `None`.

- [ ] **Step 1.2: Test graph record population**

Add assertions to `tests/unit/test_graph_records.py` that a code-producing step records:

```python
assert record.trajectory_text
assert record.prompt_text
assert record.output_text
assert record.pre_code is not None
assert record.post_code == record.generated_code
```

Expected before implementation: FAIL on missing fields.

- [ ] **Step 1.3: Serialize schema version 3**

Update `SESSION_SCHEMA_VERSION` to `3` and write `pre_code` / `post_code` in `session.jsonl`.

Add `tests/unit/test_session_log.py` coverage that a serialized code step contains the new keys and old caller-supplied schema versions cannot override the producer's version.

- [ ] **Step 1.4: Mine `pre_codes` and `post_codes` as per-record lists**

Update `extract_trajectories()` to emit:

```python
{
    "trajectory": step["trajectory"],
    "prompt": step["prompt"],
    "completion": step["output"],
    "pre_codes": [step["pre_code"]] if step.get("pre_code") else [],
    "post_codes": [step["post_code"]] if step.get("post_code") else [],
    "metadata": {
        "phase": step.get("action", "unknown"),
        "target": step.get("target"),
        "step": step.get("step"),
        "benchmark": benchmark,
        "problem_id": problem_id,
        "pass_at_1": pass_at_1,
        "schema_version": SESSION_SCHEMA_VERSION,
    },
}
```

Keep one canonical schema. Do not add compatibility shims for old sessions; schema-version mismatch should continue to fail fast.

- [ ] **Step 1.5: Preserve side channels through training dataset construction**

Change or replace `to_sft_columns()` so it no longer drops `trajectory`, `pre_codes`, `post_codes`, or `metadata`.

Run:

```bash
uv run pytest tests/unit/test_graph_records.py tests/unit/test_session_log.py tests/unit/test_miner.py tests/unit/test_d2l_train.py -q
```

Expected after implementation: PASS.

## Task 2: Hypernetwork KD Skeleton And Collapse Instrumentation

**Purpose:** Stop routing Stage-2 through plain PEFT SFT and introduce unit-testable HyperLoRA distillation primitives before any GPU training run.

**Files:**
- Add: `src/rune/training/hypernet_distill.py`
- Add: `src/rune/training/collapse_metrics.py`
- Modify: `src/rune/training/orchestrator.py`
- Modify: `src/rune/training/config.py`
- Test: `tests/unit/test_hypernet_distill.py`
- Test: `tests/unit/test_collapse_metrics.py`
- Test: `tests/unit/test_orchestrator.py`

- [ ] **Step 2.1: Add pure metric helpers first**

In `collapse_metrics.py`, add pure functions with no GPU imports at module import time:

```text
summarize_named_tensors(named_tensors) -> dict[str, float]
  Returns per-name mean, absmax, and l2 stats for watched tensor groups.

assert_optimizer_covers(parameters, optimizer) -> None
  Raises RuntimeError listing trainable parameter names absent from optimizer groups.
```

Tests must cover missing optimizer membership for `scaler_B`, output heads, and aggregator parameters.

- [ ] **Step 2.2: Add diff-token metric helpers**

In `hypernet_distill.py`, add pure helpers:

```text
compute_diff_positions(base_top1, teacher_top1, labels) -> boolean mask
  True where labels are supervised and base_top1 differs from teacher_top1.

summarize_diff_agreement(student_top1, teacher_top1, diff_positions) -> dict[str, float]
  Returns diff_token_frac and diff_agreement over the diff_positions mask.
```

Tests must assert that ordinary `top1_agreement` can be high while `diff_agreement` is zero when the student equals base on disagreement tokens.

- [ ] **Step 2.3: Replace orchestrator Stage-2 entrypoint**

Change `_run_hypernetwork_distillation()` to call the new HyperLoRA KD entrypoint, not `run_distillation()`. Keep `run_distillation()` only if it remains the Stage-1 oracle SFT path; otherwise rename it so the stage boundary is not misleading.

Add `tests/unit/test_orchestrator.py` coverage that Stage-2 dispatches to `rune.training.hypernet_distill.run_hypernet_distillation`.

- [ ] **Step 2.4: Add one-step synthetic gradient test before full model training**

Use tiny fake modules or monkeypatched tensor producers. The test should prove:

- diff-token loss is non-zero when teacher differs from base;
- at least one hypernetwork trainable parameter receives a non-zero gradient;
- `scaler_B`/head/aggregator parameter groups are included in the optimizer.

This is the acceptance gate for editing the real GPU training loop.

Run:

```bash
uv run pytest tests/unit/test_hypernet_distill.py tests/unit/test_collapse_metrics.py tests/unit/test_orchestrator.py -q
```

Expected: PASS.

## Task 3: Restore Stage-1 Oracle Artifacts

**Purpose:** Make Stage-2 learn against real per-bin oracle adapters instead of the base model or a plain SFT model.

**Files:**
- Modify: `src/rune/training/orchestrator.py`
- Modify: `src/rune/training/config.py`
- Modify: `src/rune/training/oracle_cache.py`
- Add or modify only if needed: `src/rune/training/oracle_train.py`
- Test: `tests/unit/test_orchestrator.py`
- Test: `tests/unit/test_oracle_cache.py` if split from `test_orchestrator.py`

- [ ] **Step 3.1: Define oracle stage contract**

`_run_oracle_training(config, corpus_dir)` must either:

- train/register the missing oracle adapters for the corpus bins; or
- detect existing complete coverage and skip training with a clear log.

It must not raise `NotImplementedError` after this task.

- [ ] **Step 3.2: Add coverage gate before Stage-2**

Use `audit_oracle_coverage()` before HyperLoRA KD. Abort if coverage is below `Round2TrainConfig.min_oracle_coverage`.

Test:

```python
with pytest.raises(RuntimeError, match="oracle coverage"):
    _run_hypernetwork_distillation(config_with_missing_oracles)
```

- [ ] **Step 3.3: Keep oracle fallback explicit**

Default stays `oracle_fallback="skip"`. `base_model` fallback may remain only for ablation, and it must log that the run is not a promotion candidate.

Run:

```bash
uv run pytest tests/unit/test_orchestrator.py tests/unit/test_config.py -q
```

Expected: PASS.

## Task 4: Inference Adapter Application Contracts

**Purpose:** Fix latent adapter application issues without pretending they are empirically validated before a non-inert adapter exists.

**Files:**
- Modify: `src/rune/model/hypernetwork.py`
- Modify: `src/rune/model/wrapper.py`
- Modify: `src/rune/model/adapter.py` only if shared validation helpers are needed
- Add: `tests/unit/test_hypernetwork_peft_mapping.py`
- Test: `tests/unit/test_wrapper.py`

- [ ] **Step 4.1: Add PEFT shape/key tests before implementation**

The tests should build a fake `lora_dict` with `A`, `B`, and optional head-bias pieces and assert:

- every emitted key matches PEFT's expected `lora_A.weight` / `lora_B.weight` pattern;
- rank expansion from bias handling is either supported by `LoraConfig.r` or rejected with a clear `ValueError`;
- no silent truncation or transpose mismatch occurs.

- [ ] **Step 4.2: Integrate `combine_lora` / `get_head_bias` only if rank contracts pass**

Do not list this as a mechanical fix. If combining bias changes effective rank, update `ModelWrapper.from_config()` to create the PEFT adapter with the same rank or raise before hotswap.

- [ ] **Step 4.3: Prevent activation contamination**

Update `extract_activations_with_model()` or its caller so PEFT adapters are disabled during activation extraction when the model supports `disable_adapter()`.

Test with a fake model:

```python
model.disable_adapter.assert_called_once()
```

Also test that non-PEFT models still extract activations without requiring the context manager.

Run:

```bash
uv run pytest tests/unit/test_hypernetwork_peft_mapping.py tests/unit/test_wrapper.py -q
```

Expected: PASS.

## Task 5: Retrieval And Contrast Gates

**Purpose:** Replace weak "adapter has any effect" acceptance with content and control gates.

**Files:**
- Modify: `src/rune/training/gate.py`
- Modify: `tools/diag_recall_probe.py`
- Modify: `tools/diag_retrieval_probe.py`
- Modify: `tools/diag_continuation_probe.py`
- Test: `tests/unit/test_gate.py`

- [ ] **Step 5.1: Add structured probe result schema**

Each probe should be able to emit JSON with enough fields for gate evaluation:

```json
{
  "real_hit_rate": 0.0,
  "zero_hit_rate": 0.0,
  "shuffled_hit_rate": 0.0,
  "contradictory_hit_rate": 0.0,
  "adapter_cosine": 1.0,
  "diff_agreement": 0.0,
  "scaler_b_absmax": 0.0
}
```

- [ ] **Step 5.2: Add `evaluate_retrieval_gate()`**

Promotion gate should require all of:

- `real_hit_rate > zero_hit_rate`;
- `real_hit_rate > shuffled_hit_rate`;
- `real_hit_rate > contradictory_hit_rate`;
- trajectory adapters are not near-identical by the probe's own cosine/L2 signal;
- continuation contrast trends in the expected direction when that probe is run.

Keep `scaler_b_absmax` as a logged tripwire only. It may fail the smoke run for investigation, but it cannot by itself promote a checkpoint.

- [ ] **Step 5.3: Test that generic output perturbation fails**

Add a fixture where `real_hit_rate == zero_hit_rate == contradictory_hit_rate` but `adapter_cosine` or output text changes. The gate must fail.

Run:

```bash
uv run pytest tests/unit/test_gate.py -q
```

Expected: PASS.

## Task 6: Synthetic Overfit GPU Gate

**Purpose:** Prove the training loop can create a non-inert, content-retrieving adapter before MBPP/HumanEval work.

**Files:**
- Add or modify: `tools/diag_synthetic_overfit.py`
- Modify: `src/rune/training/hypernet_distill.py`
- Test: unit tests from Tasks 2 and 5

- [ ] **Step 6.1: Create a tiny unguessable-facts corpus**

Use 3-5 synthetic records with facts present only in `trajectory`, for example `MAGIC_OFFSET=73921`, and held-out prompts that do not include the fact.

- [ ] **Step 6.2: Run under memory guard**

```bash
free -g
uv run python tools/diag_synthetic_overfit.py --max-steps 20 --json-out /tmp/rune-issue49-synth.json
```

Expected gate: `real > zero`, `real > shuffled`, and `real > contradictory` on content retrieval. If model loading or the watchdog fails, stop and fix harness; do not continue to benchmarks.

- [ ] **Step 6.3: Inspect collapse diagnostics**

The JSON/log must include:

- `scaler_B` stats;
- generated `Delta W` norm;
- real-vs-zero logit KL;
- real-vs-contradictory adapter cosine;
- gradient norms for `scaler_B`, output heads, and aggregator;
- skipped-record count;
- diff-token fraction and diff agreement.

Any flat gradients or skipped-record spike blocks downstream work.

## Task 7: Tiny Benchmark Then Full Campaign

**Purpose:** Validate pass@1 only after retrieval is real.

**Files:**
- Modify: `src/rune/bench/runner.py` only if needed for tiny benchmark selection or JSON artifacts
- Modify: `src/rune/training/orchestrator.py`
- Modify: `src/rune/training/gate.py`
- Test: `tests/unit/test_bench_runner.py`
- Test: `tests/unit/test_gate.py`

- [ ] **Step 7.1: Wire real baseline/new scores into the success gate**

Remove placeholder empty score maps from the training pipeline. A missing baseline or missing new score for a configured benchmark should fail loudly.

- [ ] **Step 7.2: Run a 2-3 task tiny benchmark**

Controls required:

- base-only;
- zero adapter;
- real trajectory adapter;
- shuffled trajectory adapter;
- contradictory trajectory adapter.

Expected: no regression vs base/zero and directional lift for real over controls. This is directional smoke, not publication evidence.

- [ ] **Step 7.3: Scale only after tiny gate passes**

Run full benchmark campaign only after Tasks 0-7.2 pass. HPO remains blocked until the objective includes retrieval gate metrics and pass@1 lift over controls.

## Final Verification Before Declaring Issue #49 Fixed

- [ ] `uv run ruff check .`
- [ ] `uv run mypy src/`
- [ ] `uv run pytest tests/unit/ -q`
- [ ] Synthetic retrieval gate JSON passes and is saved with the checkpoint.
- [ ] Tiny benchmark gate passes against base, zero, shuffled, and contradictory controls.
- [ ] Full benchmark run logs real baseline/new scores; no placeholder empty dicts.
- [ ] Final report states which hypotheses were confirmed, which were falsified, and which remain ablation candidates.
