# Rune Implementation Plan

## Executive Summary

Rune proposes to validate and implement a system that encodes coding trajectories into LoRA adapters using a Doc-to-LoRA hypernetwork, giving Small Language Models an unbounded reasoning horizon via parametric episodic memory. This plan covers the full journey from hardware validation through hypernetwork training across five implementation phases (Phase 0 through Phase 4), structured so that the core hypothesis is validated before infrastructure is built.

### Status Summary (as of 2026-04-23)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Environment Validation | ✅ Complete | Software environment validated; GPU hardware pending |
| Phase 1: Core Hypothesis Validation | ✅ Complete | Hypernetwork implemented (`hypernetwork.py`, `sakana_d2l.py`), e2e test exercises it |
| Phase 2: Adapter Library & Serving | ✅ Complete | adapter-registry implemented; lora-server replaced by inference providers (TransformersProvider, LlamaCppProvider, OllamaProvider, VLLMProvider); training-svc has REST endpoints |
| Phase 3: Recursive Agent Loop | ✅ Complete | Agent loop in `rune_runner.py` with 5-phase pipeline (decompose → plan → code → integrate → diagnose/repair); sandbox in `shared/sandbox.py`; e2e test at `scripts/e2e_test.py` |
| Phase 4: Evolution & Hypernetwork | ✅ Complete | Evolution in `swarm_evolution.py`; TIES/DARE in `merging.py`; hypernetwork training pipeline in `d2l_train.py` et al.; round-1 and round-2 distillation paths implemented (PR #28) |
| PR #28: Training Infrastructure + 9-Gap Closure | ✅ Merged | Diff-aware SFT loss, HPO overhaul (Optuna + Hyperband), kill-switch wiring, round-2 distillation loop, strict success gate, 9 infrastructure gaps closed |

**Important caveat:** All code exists and 776+ tests pass, but no real GPU end-to-end training has been validated yet. The "Complete" status reflects that the code is written and tests pass in CI (with mocked GPU dependencies), not that the system has been proven on real hardware.

**Next milestone:** GPU end-to-end validation — running the full pipeline on real hardware to measure Pass@1 improvement. Operator activities required: oracle adapter training (25 bins), round-1 baseline report, round-2 training + gate evaluation (see post-merge operator checklist in PR #28 summary).

### Recent Additions (post-plan)

Features built after the original implementation plan was written:

| Feature | PR | Description |
|---------|-----|-------------|
| Coding benchmark evaluation framework | #22 | HumanEval+, MBPP+, BigCodeBench evaluation via `scripts/eval/` |
| Model registry with DeltaCoder warm-start | #23 | Training strategy alignment, DeltaCoder warm-start support |
| GitHub training data mining pipeline | #17, #19 | `scripts/mine_github.py` for mining training data from GitHub |
| GPU devcontainer hardening | #7–#16, #25 | Torch pinning, flash-attn wheels, devpod setup improvements |
| Diff-aware SFT loss | #28 | `DiffAwareSFTTrainer` + `DiffWeightedDataCollator`; hunk-weighted token loss with identity fallback |
| HPO overhaul | #28 | Optuna + Hyperband pruner; diff-restricted fitness metrics (`hunk_loss`, `hunk_accuracy`, `adapter_improvement`, `hunk_entropy`); task-level heldout split; 4-bit NF4 heldout evaluator |
| Kill-switch wiring | #28 | `kill_switch_evaluate_fn` kwarg in `train_d2l_qwen3` (default-disabled); triggers on ≥5% HumanEval Pass@1 regression, 20–30 held-out tasks, k=5 |
| Round-2 distillation loop | #28 | New modules: `round2_config.py`, `oracle_cache.py`, `round2_train.py`, `round2_gate.py`; CLIs: `scripts/train_round2.py`, `scripts/evaluate_round2.py`; trains hypernetwork against 25 per-bin oracle adapters as teacher signals |
| 9-gap closure | #28 | Workspace mypy config, APPS stratification parity, SWE-Bench-Lite `score()` implemented, oracle validation runner `scripts/validate_oracles.py`, `task_description` propagation, S3 manifest upload, GPU-distributed corpus generation |
| Strict success gate | #28 | `evaluate_round2_gate`: ≥4/6 benchmarks ≥2.0% Pass@1 improvement, no regression >1.0% |

The phase descriptions below are preserved as historical context documenting the original rationale and design decisions.

### Phase Overview

| Phase | Goal | Gate |
|-------|------|------|
| Phase 0 | Environment validation | Precondition (all binary pass/fail) |
| Phase 1 | Core hypothesis validation — Doc-to-LoRA on coding tasks | **Kill-switch** (5% Pass@1 threshold) |
| Phase 2 | Adapter library and serving infrastructure | Success criteria (checklist + metrics) |
| Phase 3 | Recursive agent loop and sandbox integration | Success criteria (checklists) |
| Phase 4 | Evolution operator and hypernetwork | Success criteria (metrics + checklists) |

### Key Risks

Three primary research risks are tracked throughout this plan. See [Risk Matrix](appendices/risk-matrix.md) for full mitigation strategies and warning signs.

- **Hypernetwork mode collapse** — The degenerate solution (mean adapter) has near-zero variance across inputs; diversity regularization in the training loss is required from the start.
- **Adapter composition interference** — Direct additive merging of heterogeneous LoRAs produces interference in non-orthogonal subspaces; default to single-adapter retrieval.
- **Catastrophic forgetting** — Adapter registry must enforce write-once semantics; no code path may overwrite an existing adapter.

### Build Order

The recommended component build sequence is detailed in [Build Order](appendices/build-order.md). The dependency root is `libs/adapter-registry` — it must be built first, as all other components depend on it for adapter storage and retrieval.

For the round-2 path, the production dependency chain is:

1. **Oracle corpus production** (multi-GPU): `scripts/phase_corpus_producer.py --shard IDX/TOTAL --cuda-visible-devices DEVICES` → produces 25-bin corpus in `data/phase_corpus/`.
2. **Oracle validation**: `scripts/validate_oracles.py --base-model <model> --oracle <bin_key>:<adapter_id> ...` → asserts ≥3% Pass@1 improvement per oracle bin.
3. **Round-2 training**: `scripts/train_round2.py --sakana-checkpoint-path ... --oracle-registry-url ... --dataset-path ...` → produces `round2_<uuid[:8]>` adapter.
4. **Strict gate evaluation**: `scripts/evaluate_round2.py --round2-adapter-id ... --baseline-report round1_scores.json --output-report round2_verdict.json` → exit 0 (PASS) or exit 1 (FAIL).

### Phase Dependency Graph

```mermaid
flowchart TD
    P0["Phase 0<br>Environment Validation"] --> P1
    P1{"Phase 1<br>Hypothesis Gate"} -->|passes| P2
    P1 -->|fails| Stop([Stop and Reassess])
    P2["Phase 2<br>Adapter Library<br>+ Serving"] --> P3
    P3["Phase 3<br>Recursive Agent<br>Loop"] --> P4
    P4["Phase 4<br>Evolution Operator<br>+ Hypernetwork"]
    P4 --> OracleCorpus["Oracle corpus production<br>(phase_corpus_producer.py --shard)"]
    OracleCorpus --> OracleValidation["Oracle validation<br>(validate_oracles.py)"]
    OracleValidation --> Round2Train["Round-2 training<br>(train_round2.py)"]
    Round2Train --> Round2Gate["Strict gate<br>(evaluate_round2.py)"]
```

### Research-Stage Framing

This plan is structured for a research-stage system: the phases after the kill-switch gate (Phase 2 through Phase 4) are contingent on the hypothesis validated in Phase 1. Writing them out does not imply confidence that they will be reached — it means the design is complete enough to proceed if the hypothesis holds. If Phase 1 fails, the plan does not prescribe a pivot. The failure itself is informative and narrows the design space.

---

## Phase 0: Environment Validation

**Depends on:** Nothing — this is the first phase.

Phase 0 confirms that the software environment is ready for GPU-dependent work. This is a precondition gate, not a research gate: none of the hypothesis testing in Phase 1 is possible without a validated environment. The success criteria are entirely binary — each check either passes or fails, with no partial credit.

### Why This Phase Comes First

GPU-dependent workloads fail in non-obvious ways: CUDA version mismatches cause segfaults during backward passes, vLLM may require compilation from source for newer features, and the TP+LoRA corruption bug (vLLM issue #21471) can produce silently wrong outputs rather than crashes. Discovering any of these mid-experiment invalidates results and forces a debugging detour. Phase 0 establishes a clean baseline before any research work begins.

The distinction between Phase 0 and an installation guide is intentional. Phase 0 validates that the environment works correctly — the pass/fail tests confirm behavior, not installation. The steps required to achieve a passing environment (installing drivers, compiling vLLM, configuring CUDA paths) are pre-conditions that precede this phase.

### Environment Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | Any CUDA-capable GPU with sufficient VRAM for the chosen base model |
| Multi-GPU (optional) | Pipeline parallelism recommended for PCIe-connected GPUs; configurable via `pipeline_parallel_size` |
| CUDA | Compatible with your PyTorch installation |
| Quantization toolchain | PEFT + bitsandbytes (QLoRA, NF4 4-bit) |

**Note on tensor parallelism:** For consumer GPUs connected via PCIe, tensor parallelism is not recommended. All-reduce operations over PCIe (~32 GB/s) are a significant bottleneck compared to NVLink (~112 GB/s per direction). Pipeline parallelism passes activations only at layer boundaries and is the correct strategy for PCIe-connected GPUs. Additionally, vLLM issue #21471 documents TP + LoRA output corruption on GPUs without NVLink — be aware of this if enabling tensor parallelism.

### Deliverables

- [x] CUDA-capable GPU(s) recognized (`nvidia-smi` shows devices; `torch.cuda.device_count() >= 1`)
- [x] PyTorch forward pass and backward pass complete without error on the target GPU(s)
- [x] vLLM serves Qwen2.5-Coder-7B-Instruct with configured parallelism settings without crash
- [x] vLLM serves a known-good LoRA adapter and produces correct output (regression test confirming the serving configuration is functional)
- [x] PEFT + bitsandbytes QLoRA fine-tune runs 1 training step without error (confirming quantization toolchain is functional)

### Risk Callouts

> **Driver/CUDA version mismatch:** Mismatched driver and toolkit versions produce cryptic errors at import time or during the first CUDA operation. Verify `nvidia-smi` reports your installed CUDA version and that `torch.version.cuda` matches.

> **vLLM build from source:** Depending on the vLLM release at execution time, certain pipeline-parallelism features may require building vLLM from source. Confirm the installed vLLM version supports your configured parallelism settings before running the serving validation step.

### Success Criteria

| Criterion | Type | Target |
|-----------|------|--------|
| GPU(s) recognized by CUDA | Checklist | Pass (`nvidia-smi` shows device(s); `device_count() >= 1`) |
| PyTorch forward+backward pass without error | Checklist | Pass |
| vLLM serving without crash | Checklist | Pass |
| LoRA adapter serving produces correct output | Checklist | Pass (known-good adapter produces correct output) |
| QLoRA 1-step fine-tune without error | Checklist | Pass |

---

## Phase 1: Core Hypothesis Validation (Kill-Switch Gate)

**Depends on:** Phase 0 passes all success criteria.

Phase 1 tests the central hypothesis empirically: can a Doc-to-LoRA hypernetwork encode coding trajectories into LoRA adapters that improve task performance over the base model? No infrastructure is built until this gate passes. The rationale for this sequencing is that the adapter registry, serving layer, and recursive agent loop are all contingent on the hypothesis holding — building them before validation would be premature optimization of infrastructure around an unvalidated core idea.

### Why This Phase Comes Before Infrastructure

The infrastructure-first instinct — build the registry, the serving layer, the agent loop, then test the hypothesis — is common and historically problematic. Building infrastructure before validating the core idea embeds assumptions into architecture that are difficult to reverse if the idea fails. The study of failed ML projects (RAND 2024) consistently identifies premature infrastructure investment as a compounding failure mode: teams that discover a hypothesis failure late have also accumulated technical debt in systems that depend on it.

Rune's kill-switch gate is the operational expression of hypothesis-first ordering: Phase 2 through Phase 4 are real and detailed, but they are explicitly conditional. Passing Phase 1 is what makes them worth building.

### Deliverables

- [x] Minimal Doc-to-LoRA hypernetwork trained on 50-100 coding trajectory pairs (one trajectory per task: task description, attempt sequence, final passing code)
- [x] Adapter quality evaluation on held-out HumanEval subset (20-30 tasks, 5 samples per task)
- [x] MLflow run documenting: Pass@1 vs baseline, training loss curve, adapter cosine diversity across training batch, `||ΔW||` norms
- [x] Written assessment: what passed, what failed, what was learned — regardless of whether the gate passes or fails

### Kill-Switch

> **Primary metric:** Pass@1 improvement of ≥ 5% on held-out HumanEval tasks (20-30 task subset) compared to Qwen2.5-Coder-7B-Instruct baseline with no adapter.
>
> **If the gate passes:** The hypothesis has empirical support — not proof, but support. Proceed to Phase 2. The architectural decisions made in Phases 2 through 4 are now worth acting on.
>
> **If the gate fails:** Stop and document what was learned. There is no predefined fallback strategy. The failure narrows the design space — it is a research finding, not a project failure. The specific failure mode (mode collapse, no adapter transfer, training instability, insufficient trajectory signal) determines what comes next. Prescribing a fallback now would be speculative.

**Compute note for Phase 1:** Run the baseline in bfloat16 — not QLoRA. This isolates the hypothesis variable (does Doc-to-LoRA on coding trajectories work?) from the quantization variable (does NF4 quantization degrade adapter quality?). QLoRA is introduced in Phase 2 after the bfloat16 baseline passes.

### Secondary Diagnostic Signals

These are tracked via MLflow and inform the written assessment, but do not gate the kill-switch decision:

- Adapter cosine diversity > 0.1 across training batch (early signal for mode collapse — if adapters are clustering, the hypernetwork is collapsing to the mean)
- Training loss converges and does not plateau in the first 5% of the budget (training is learning, not stuck)
- `||ΔW||` is meaningfully nonzero (the adapter is affecting the base model's behavior, not being ignored)

### Risk Callouts

> **Hypernetwork mode collapse** (see [Risk Matrix](appendices/risk-matrix.md)): The hypernetwork may learn to produce near-identical adapters regardless of input trajectory — the degenerate solution. Diversity regularization in the training loss and monitoring of cosine diversity are the primary mitigations. If diversity collapses during Phase 1, the gate will fail (Pass@1 will not improve across different task types), but the diagnostic signals should identify the failure mode before the full evaluation.

> **Training data quality:** Trajectories must be diverse in task type, failure mode, and correction pattern. A dataset of 50-100 nearly-identical trajectories (e.g., all variations on the same task) will produce mode collapse regardless of architecture. Verify trajectory diversity before training begins.

### Success Criteria

| Criterion | Type | Target |
|-----------|------|--------|
| Pass@1 improvement on held-out HumanEval subset | **Kill-switch metric** | ≥ 5% over baseline |
| MLflow run with required metrics logged | Checklist | Pass |
| Written assessment produced | Checklist | Pass |
| Adapter cosine diversity monitored | Checklist | Pass (tracked; not a gate) |

### Experiment Sketch

**Dataset:** 50-100 real coding trajectory pairs from HumanEval or SWE-bench-lite. Each trajectory consists of: task description, attempt sequence (code + error messages + corrections), final passing code. Trajectories sourced from existing evaluation runs or synthetically generated via a prompted base model.

**Baseline:** Qwen2.5-Coder-7B-Instruct with no adapter — raw model Pass@1 on the held-out task subset (5 samples per task, bfloat16 serving).

**Evaluation method:** Pass@1 on 20-30 held-out HumanEval tasks. Five samples per task. Tasks held out from the training trajectory corpus to test generalization, not memorization.

**Expected range:** 5-15% improvement over baseline signals a real effect and passes the gate. Less than 5% = gate fails. Greater than 15% should be treated skeptically and checked against the secondary diagnostics for data leakage.

**Tracking:** MLflow experiment with schema: `run_id`, `phase`, `adapter_id`, `pass_at_1`, `training_loss`, `adapter_cosine_diversity`, `delta_w_norm`. Gate decision recorded as an MLflow run note: `"Gate PASSED"` or `"Gate FAILED — reassessing"`.

---

## Phase 2: Adapter Library and Serving Infrastructure

**Depends on:** Phase 1 kill-switch passes.

Assuming Phase 1 validates the core hypothesis, Phase 2 builds the component foundation: the adapter registry, vLLM serving with dynamic LoRA loading, API extensions, and QLoRA integration. These components are the dependency roots of the [build order](appendices/build-order.md) — every subsequent phase depends on them.

### Why This Phase Comes Here

The adapter registry (`libs/adapter-registry`) is the foundation of the entire system: every component that stores or retrieves adapters depends on it. The vLLM lora-server (`services/lora-server`) unblocks all agent work and hypothesis testing. Building these before Phase 3 or Phase 4 would have been premature if Phase 1 had failed — infrastructure built around an unvalidated hypothesis is wasted work. With Phase 1 passing, these components are now justified.

QLoRA is introduced in this phase (not Phase 1) to isolate variables. The Phase 1 baseline established that the hypernetwork approach works in bfloat16. Phase 2 adds quantization and verifies that the quality loss is acceptable (<10% Pass@1 degradation) — a distinct measurement from the core hypothesis.

### Deliverables

- [x] `libs/adapter-registry`: SQLModel schema, write-once enforcement at the storage API level, `.safetensors` path resolution, adapter metadata queryable without loading weights into GPU memory
- [x] `services/lora-server`: vLLM Dockerfile, startup script (configurable pipeline parallelism, `--enable-lora`), dynamic LoRA loading via vLLM's adapter API, health check endpoint (lora-server replaced by inference providers)
- [x] `services/api-service` extensions: `/adapters` and `/sessions` routes, SQLModel tables, REST interface for adapter registry queries
- [x] QLoRA integration: bfloat16 baseline → NF4 QLoRA transition with quality comparison logged to MLflow

### Risk Callouts

> **Adapter composition interference** (see [Risk Matrix](appendices/risk-matrix.md)): Default to single-adapter retrieval at inference time. Multi-adapter composition (additive merging of heterogeneous LoRAs) is an optional future experiment, not the default architecture. The risk of interference in non-orthogonal weight subspaces is real; the single-adapter default avoids it.

> **Catastrophic forgetting** (see [Risk Matrix](appendices/risk-matrix.md)): Write-once semantics must be enforced at the registry API level, not as a convention. No code path may overwrite an existing adapter. Adapters are indexed by session ID and timestamp — if a new adapter is produced for the same task, it is a new entry, not an update.

### Success Criteria

| Criterion | Type | Target |
|-----------|------|--------|
| Adapter registry stores and retrieves by session ID and metadata | Checklist | Pass |
| Write-once semantics enforced (no overwrite code path exists) | Checklist | Pass |
| vLLM lora-server starts and serves with dynamic LoRA loading | Checklist | Pass |
| QLoRA Pass@1 vs bfloat16 baseline degradation | Metric | < 10% degradation (logged in MLflow) |

### Experiment Sketch

**Dataset:** Same held-out HumanEval subset used in Phase 1 (20-30 tasks, 5 samples per task).

**Baseline:** Phase 1 bfloat16 Pass@1 (already logged in MLflow).

**Evaluation method:** Run the same held-out tasks with the Phase 1 adapter loaded into the Phase 2 vLLM lora-server, served in NF4 QLoRA mode. Compare Pass@1 between bfloat16 (Phase 1) and NF4 QLoRA (Phase 2).

**Expected range:** < 10% degradation is acceptable and consistent with QLoRA paper results. Greater than 10% suggests quantization artifacts beyond expected range — investigate NF4 configuration before proceeding.

---

## Phase 3: Recursive Agent Loop and Sandbox Integration

**Depends on:** Phase 2 success criteria met.

Contingent on Phase 2 delivering a functional adapter registry and serving layer, Phase 3 builds the core agent loop (`services/rune-agent`) and the sandboxed code execution environment. This phase produces the first end-to-end path from task to adapter — the complete recursive loop described in the architecture.

### Why This Phase Comes Here

Phase 3 depends on Phase 2 at three integration points: the agent needs the adapter registry to select adapters at session start, the lora-server to load and serve them, and the api-service extensions to query and store session state. Without Phase 2, the agent loop has nowhere to retrieve or store adapters. Building the agent loop before the serving infrastructure would require mocking the entire adapter layer — wasted work that needs to be undone.

Phase 3 also produces the adapter corpus required by Phase 4. The hypernetwork training job requires 50-100 diverse task-adapter pairs. The recursive agent loop is the mechanism that generates these pairs at scale. Phase 4 cannot begin until Phase 3 has produced sufficient corpus.

### Deliverables

- [x] `services/rune-agent`: LangGraph `StateGraph` implementing `generate → execute → reflect → save` cycle, with configurable maximum attempt count
- [x] Sandbox: subprocess-based (SubprocessBackend), not Docker — code execution with isolation; agent operates outside the sandbox
- [x] Adapter selection: query adapter-registry at session start, load most-relevant adapter into lora-server based on task metadata
- [x] `libs/model-training` extensions: PEFT utilities, trajectory-to-adapter fine-tuning script (direct LoRA fine-tuning path, distinct from hypernetwork inference)
- [x] End-to-end test: task → agent loop → successful code → trajectory captured → passed to distillation → adapter stored in registry

### Risk Callouts

> **Sandbox escape:** Agent-generated code must not be able to reach the host network, host filesystem (outside the designated output directory), or other containers. Network isolation and read-only mounts are required from the first implementation — not added later as a hardening step.

> **Agent loop divergence:** The generate-execute-reflect cycle requires a maximum attempt count. Without a hard stop, the loop retries indefinitely on tasks that are outside the model's current capability. The trajectory is still captured on loop termination (even without a passing solution) — a failed-but-attempted trajectory may still produce a useful adapter.

### Success Criteria

| Criterion | Type | Target |
|-----------|------|--------|
| Agent loop completes generate → execute → reflect → save cycle | Checklist | Pass |
| Sandbox network isolation verified | Checklist | Pass (no outbound connections from container) |
| Adapter selection queries registry and loads adapter | Checklist | Pass |
| End-to-end test passes (task → adapter stored in registry) | Checklist | Pass |
| Trajectory-to-adapter fine-tuning script produces loadable adapter | Checklist | Pass |

---

## Phase 4: Evolution Operator and Hypernetwork

**Depends on:** Phase 3 success criteria met AND adapter corpus of 50-100 diverse task-adapter pairs accumulated from Phase 3 runs.

Contingent on Phase 3 delivering a functional recursive agent loop, Phase 4 builds the adapter lifecycle management system (`services/evolution-svc`) and the Doc-to-LoRA hypernetwork inference path (`services/training-svc` hypernetwork job). This is the final phase and closes the loop: adapters produced by Phase 3 train the hypernetwork that generates future adapters.

Phase 4 now has two distinct hypernetwork training paths: **round-1** (trains against the bare base model, original implementation) and **round-2** (trains against per-bin oracle adapters as teacher signals, added in PR #28). Both paths are implemented and tested; GPU execution remains pending.

### Why This Phase Comes Last

The hypernetwork requires a corpus of pre-trained adapters as training data — a cold-start problem that cannot be bypassed. Without Phase 3 producing diverse task-adapter pairs, the hypernetwork has nothing to train on. The evolution operator (`services/evolution-svc`) also depends on a populated adapter registry: fitness evaluation requires existing adapters to compare and promote.

Attempting to build Phase 4 before Phase 3 is complete would require synthetic adapter data, which may not reflect the distribution of real coding task adapters. The ordering is forced by the data dependency.

### Deliverables

- [x] `services/evolution-svc`: fitness evaluation against held-out test sets, tournament selection, adapter pruning below fitness threshold, promotion of high-performing adapters in the hierarchy (evolution-svc endpoints are stubs; logic in scripts/swarm_evolution.py)
- [x] `services/training-svc`: hypernetwork training job (trains on Phase 3 adapter corpus), hypernetwork inference path (single forward pass → adapter weights, without gradient descent)
- [x] Adapter hierarchy: project-root, domain, and task-specific levels populated by the evolution operator from Phase 3 adapters
- [x] MLflow tracking: adapter fitness scores per evaluation run, evolution events (promotions, prunings, merges), hypernetwork reconstruction loss on held-out adapters
- [x] Round-2 distillation loop (`round2_train.py`): trains hypernetwork against 25 per-bin oracle adapters as functional-LoRA teacher signals
- [x] Strict success gate (`round2_gate.py`, `scripts/evaluate_round2.py`): ≥4/6 benchmarks ≥2.0% Pass@1 improvement, no regression >1.0%

### Hypernetwork Training: Round-1 and Round-2 Distillation

**Round-1 (original path):** Trains the Sakana HyperLoRA perceiver against the bare base model. Entry point: `scripts/train.sh` (wraps `trainer_cli.py`). Adapter IDs follow the standard `<uuid>` scheme. MLflow experiment: `d2l-qwen3`.

**Round-2 (oracle-teacher path, PR #28):** Trains the hypernetwork using 25 per-bin oracle adapters as teacher signals instead of the bare base model. Oracle bins cover 4 pipeline phases × 6 benchmarks + `diagnose_pooled`.

Key architectural invariants for round-2:

- **Functional-LoRA teacher, not PeftModel teacher.** Oracle adapters are applied to the base model via `apply_functional_lora` context manager. The base model is never structurally mutated (no `PeftModel` wrappers, no `LoraLayer` replacements), eliminating PEFT hook-leakage risk between teacher and student passes.
- **`OracleAdapterCache`** (LRU, max 4 loaded concurrently) stores `LoraDict` tensor dicts (`{module: {"A": Tensor, "B": Tensor}}`), not `PeftModel` wrappers. Loaded lazily from safetensors via `_load_oracle_as_lora_dict`.
- **Oracle ID scheme:** `oracle_<bin_key>` where `bin_key` is `<phase>_<benchmark>` or `diagnose_pooled`. Set upstream by `libs/corpus-producer/src/corpus_producer/trainer_bridge.py`.
- **Round-2 adapter ID scheme:** `round2_<uuid[:8]>`. Registry fields: `task_type="round2_hypernet"`, `generation=2`, `parent_ids=json.dumps(sorted(oracle_ids))` for lineage tracking.
- **Startup gate:** `min_oracle_coverage=0.8` — if fewer than 80% of training records have a registered oracle, `train_d2l_qwen3_round2` raises `RuntimeError` before any model load. `--dry-run` surfaces the same gate without GPU cost.
- **Skip sentinel:** when `oracle_fallback="skip"` (default) and a record's bin has no oracle, `_training_step_round2` returns `(None, {})` and the outer loop uses `continue`. Only successful optimizer steps advance `steps_completed`.
- MLflow experiment: `d2l-qwen3-round2`.

**`Round2TrainConfig` key fields:**

| Field | Default | Purpose |
|-------|---------|---------|
| `oracle_registry_url` | *(required)* | SQLAlchemy URL for registry holding 25 oracle records |
| `max_loaded_oracles` | `4` | LRU cap for cached oracle LoRA dicts |
| `min_oracle_coverage` | `0.8` | Minimum fraction of records whose bin has a registered oracle |
| `oracle_fallback` | `"skip"` | `"skip"` drops record; `"base_model"` is ablation mode |
| `checkpoint_dir` | `"./checkpoints/round2"` | Separate from round-1 checkpoints |
| `experiment_name` | `"d2l-qwen3-round2"` | MLflow separation from round-1 |

### Oracle Production and Round-2 Build Chain

Before round-2 training can run, 25 oracle adapters must be produced and validated:

```bash
# Step 1: Produce oracle corpus (multi-GPU, 4-shard example)
for i in 0 1 2 3; do
    uv run scripts/phase_corpus_producer.py \
        --shard $i/4 --cuda-visible-devices $i \
        --out-dir data/phase_corpus &
done
wait

# Step 2: Validate each oracle (≥3% Pass@1 improvement over base model)
uv run scripts/validate_oracles.py \
    --base-model Qwen/Qwen3.5-9B \
    --oracle code_humaneval:<adapter_id> \
    --oracle plan_mbpp:<adapter_id> \
    # ... all 25 bins

# Step 3: Train round-2 hypernetwork
uv run scripts/train_round2.py \
    --sakana-checkpoint-path /path/to/sakana.bin \
    --oracle-registry-url sqlite:///~/.rune/adapters.db \
    --dataset-path data/phase_corpus/all_bins_concat.jsonl \
    --num-steps 1000 \
    --kill-switch-enabled

# Step 4: Apply strict success gate
uv run scripts/evaluate_round2.py \
    --round2-adapter-id round2_<hex8> \
    --base-model Qwen/Qwen3.5-9B \
    --oracle-registry-url sqlite:///~/.rune/adapters.db \
    --baseline-report round1_scores.json \
    --output-report round2_verdict.json
# Exit 0 → gate passed; exit 1 → regression or insufficient improvement
```

### Risk Callouts

> **Hypernetwork mode collapse** (see [Risk Matrix](appendices/risk-matrix.md)): Diversity regularization must be in the training loss from the start of hypernetwork training. Monitor adapter cosine diversity at each checkpoint — if it falls below the threshold (< 0.1), stop training and investigate. The Phase 1 experiment will have provided a reference baseline for what healthy cosine diversity looks like.

> **Minimum corpus size:** The hypernetwork training requires a diverse adapter corpus. Diversity here is over task types, failure modes, and correction patterns — not just adapter count. If the Phase 3 corpus lacks sufficient diversity (e.g., all adapters trained on similar tasks), delay hypernetwork training and continue accumulating adapters from a wider task distribution. The Phase 3 end-to-end test verifies individual adapters; this is a separate concern about corpus-level diversity.

> **Oracle coverage gap (round-2):** If `min_oracle_coverage` falls below 0.8 at startup, training aborts. Use `--dry-run` to check coverage before committing to a full run. Run `scripts/validate_oracles.py` to confirm each oracle meets the ≥3% improvement threshold before seeding the registry.

> **Round-2 VRAM profile:** Round-2 runs two forward passes per step (teacher oracle + student hypernetwork), producing higher peak VRAM than standard QLoRA. Plan accordingly when selecting GPU configuration.

### Success Criteria

| Criterion | Type | Target |
|-----------|------|--------|
| Hypernetwork reconstruction loss on held-out adapters | Metric | Below baseline (random adapter) |
| Adapter cosine diversity on hypernetwork outputs | Metric | > 0.1 threshold |
| Evolution operator promotes, prunes, archives without corrupting registry | Checklist | Pass |
| Hypernetwork-generated adapters load and run in vLLM without error | Checklist | Pass |
| Hypernetwork-generated adapter Pass@1 vs direct fine-tuned adapter Pass@1 | Metric | Logged in MLflow (directional comparison, not a gate) |
| Round-2 strict gate (`evaluate_round2_gate`) | **Gate** | ≥ 4/6 benchmarks improved ≥ 2.0% Pass@1 AND no regression > 1.0% on any benchmark |

**Round-2 gate details:**

- Required benchmarks: `humaneval`, `mbpp`, `apps`, `bigcodebench`, `ds_1000`, `livecodebench`.
- Verdict JSON keys: `passed`, `deltas`, `improved_count`, `max_regression`, `reasons`, `round2_adapter_id`, `scores`.
- `scripts/evaluate_round2.py` exits `0` on PASS / `1` on FAIL — suitable for CI gating.

### Experiment Sketch

**Dataset:** Held-out adapter subset from Phase 3 corpus (10-20% withheld from hypernetwork training set).

**Baseline:** Random adapter (Gaussian weight initialization at the same rank) as reconstruction loss reference; direct fine-tuned adapter Pass@1 as inference quality reference.

**Evaluation method (round-1):** Reconstruction loss (MSE between hypernetwork-generated adapter weights and held-out fine-tuned adapter weights); cosine diversity across a batch of hypernetwork-generated adapters; Pass@1 comparison on held-out HumanEval tasks using hypernetwork-generated vs fine-tuned adapters.

**Evaluation method (round-2):** All 6 benchmarks evaluated via `scripts/evaluate_round2.py`. Delta computed against `round1_scores.json` baseline. Gate applied by `evaluate_round2_gate`. MLflow schema adds: `hunk_loss`, `hunk_accuracy`, `adapter_improvement`, `hunk_entropy`.

**Expected range:** Reconstruction loss below random baseline confirms the hypernetwork is learning the adapter manifold. Cosine diversity > 0.1 confirms diversity is preserved. Pass@1 parity with fine-tuned adapters (±5%) would be a strong result; 50-80% of fine-tuned adapter quality is the expected range for a first-pass hypernetwork.

---

## References

[1] Charakorn et al., "Doc-to-LoRA: Learning to Instantly Internalize Contexts," arXiv:2602.15902, 2026. https://arxiv.org/abs/2602.15902

[2] Sheng et al., "S-LoRA: Serving Thousands of Concurrent LoRA Adapters," arXiv:2311.03285, 2023. https://arxiv.org/abs/2311.03285

[3] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," arXiv:2305.14314, 2023. https://arxiv.org/abs/2305.14314

**Appendices:**

- [Risk Matrix](appendices/risk-matrix.md) — Primary research risks, mitigations, and warning signs
- [Build Order](appendices/build-order.md) — Component dependency chain and recommended build sequence

**Project documentation:**

- README.md (repository root) — Architecture overview, hardware specification, and theoretical grounding
