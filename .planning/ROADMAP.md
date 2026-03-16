# Roadmap: Rune

## Milestones

- ✅ **v1.0 Documentation & Implementation Plan** — Phases 1-3 (shipped 2026-03-02)
- ✅ **v2.0 Repo Restructuring & Scaffold** — Phases 4-7 (shipped 2026-03-03)
- ✅ **v3.0 Scientific Article Documentation** — Phases 8-12 (shipped 2026-03-03)
- ✅ **v4.0 API Wireframes & TDD Foundation** — Phases 13-17 (shipped 2026-03-05)
- ✅ **v5.0 First Implementation** — Phases 18-24 (shipped 2026-03-06)
- 📋 **v7.0 Hypernetwork Training** — Phases 25-29 (planned)

## Phases

<details>
<summary>✅ v1.0 Documentation & Implementation Plan (Phases 1-3) — SHIPPED 2026-03-02</summary>

- [x] Phase 1: README (1/1 plans) — completed 2026-03-02
- [x] Phase 2: Implementation Plan (2/2 plans) — completed 2026-03-02
- [x] Phase 3: Architecture Docs (1/1 plans) — completed 2026-03-02

</details>

<details>
<summary>✅ v2.0 Repo Restructuring & Scaffold (Phases 4-7) — SHIPPED 2026-03-03</summary>

- [x] Phase 4: Cleanup (3/3 plans) — completed 2026-03-02
- [x] Phase 5: Foundation Libraries (3/3 plans) — completed 2026-03-02
- [x] Phase 5.1: Template Artifact Cleanup (2/2 plans) — completed 2026-03-03
- [x] Phase 6: Service Scaffolds (4/4 plans) — completed 2026-03-03
- [x] Phase 7: Configuration & Quality Gate (3/3 plans) — completed 2026-03-03

</details>

<details>
<summary>✅ v3.0 Scientific Article Documentation (Phases 8-12) — SHIPPED 2026-03-03</summary>

- [x] Phase 8: MkDocs Infrastructure (2/2 plans) — completed 2026-03-03
- [x] Phase 9: References Skeleton + Background (2/2 plans) — completed 2026-03-03
- [x] Phase 10: Methods Section (1/1 plan) — completed 2026-03-03
- [x] Phase 11: Results & Discussion Outlines (2/2 plans) — completed 2026-03-03
- [x] Phase 12: Abstract + Quality Audit (2/2 plans) — completed 2026-03-03

</details>

<details>
<summary>✅ v4.0 API Wireframes & TDD Foundation (Phases 13-17) — SHIPPED 2026-03-05</summary>

- [x] Phase 13: Test Infrastructure (2/2 plans) — completed 2026-03-03
- [x] Phase 14: Core Library Wireframes (3/3 plans) — completed 2026-03-03
- [x] Phase 15: New & Reworked Library Wireframes (2/2 plans) — completed 2026-03-03
- [x] Phase 16: Service Wireframes (4/4 plans) — completed 2026-03-03
- [x] Phase 17: Quality Gate (2/2 plans) — completed 2026-03-04

</details>

<details>
<summary>✅ v5.0 First Implementation (Phases 18-24) — SHIPPED 2026-03-06</summary>

- [x] Phase 18: Adapter Registry (2/2 plans) — completed 2026-03-05
- [x] Phase 19: Inference Provider Abstraction (3/3 plans) — completed 2026-03-05
- [x] Phase 20: Agent Loop (2/2 plans) — completed 2026-03-05
- [x] Phase 21: QLoRA Training Pipeline (2/2 plans) — completed 2026-03-05
- [x] Phase 22: Kill-Switch Gate (3/3 plans) — completed 2026-03-06
- [x] Phase 23: Integration Fix & Quality Gate (1/1 plan) — completed 2026-03-06
- [x] Phase 24: Phase 22 Verification & Dep Hygiene (1/1 plan) — completed 2026-03-06

</details>

### v7.0 Hypernetwork Training (Planned)

**Milestone Goal:** Build the KL-divergence context distillation training pipeline for HyperLoRA on Qwen3-Coder-Next, enabling the hypernetwork to compress procedural coding knowledge from trajectories into LoRA adapter weights.

- [x] **Phase 25: Configuration & Data Pipeline** — Config helpers and full data pipeline with zero GPU dependencies (completed 2026-03-13)
- [x] **Phase 26: Architecture Probe & Activation Extraction** — Dynamic attention layer discovery and pre-loaded model activation extraction (completed 2026-03-13)
- [x] **Phase 27: Partial Weight Transfer** — Freeze aggregator, reinitialize head for Qwen3-Coder-Next output dimensions (completed 2026-03-13)
- [x] **Phase 28: Functional LoRA Injection** — Autograd-safe context manager for hypernetwork training (completed 2026-03-16)
- [x] **Phase 29: Training Loop Integration** — KL+CE distillation training script with dry-run and smoke-test modes (completed 2026-03-16)

## Phase Details

### Phase 25: Configuration & Data Pipeline
**Goal**: The config helpers and data pipeline are fully operational with all 5 data tests passing on CPU in CI, providing the config dict and dataset functions every downstream phase depends on.
**Depends on**: Nothing (first phase of v7.0)
**Requirements**: CFG-01, CFG-02, DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, TEST-02
**Success Criteria** (what must be TRUE):
  1. `get_d2l_qwen3_config()` returns a dict with correct Qwen3-Coder-Next dimensions (hidden_size, attention layer indices, GQA heads) without loading model weights
  2. `build_qwen3_hypernet_config()` constructs a HypernetConfig with exactly 12 attention layer indices discovered dynamically from config
  3. `format_for_distillation()` produces records with separate `activation_text` (no answer tokens) and `teacher_text` (with answer) — verifiable by inspecting output fields
  4. JSONL round-trip: a dataset written with `save_jsonl()` loads back with `load_jsonl()` byte-for-byte identical
  5. Train/test split has zero task-ID overlap: `set(train_ids) & set(test_ids) == set()` after the full pipeline runs
**Plans:** 2/2 plans complete
Plans:
- [x] 25-01-PLAN.md — Config helpers + core data pipeline (JSONL, needle, split, distillation format) with tests
- [ ] 25-02-PLAN.md — Trajectory generation from HumanEval, LLM augmentation, GitHub mining stubs

### Phase 26: Architecture Probe & Activation Extraction
**Goal**: The architecture probe correctly identifies exactly 12 standard attention layers in Qwen3-Coder-Next (not DeltaNet), and activation extraction accepts a pre-loaded model reference rather than loading weights per call.
**Depends on**: Phase 25
**Requirements**: ARCH-01, ARCH-02
**Success Criteria** (what must be TRUE):
  1. `build_qwen3_hypernet_config()` discovers exactly 12 attention layer indices (`[3, 7, 11, ..., 47]`) from `config.layer_types` without loading 80GB of weights
  2. `extract_activations_with_model()` accepts a pre-loaded model and returns hidden states only from the specified attention layer indices (not from DeltaNet layers)
  3. Tests using `sys.modules` injection pass on CPU in CI with no GPU installed
**Plans:** 1/1 plans complete
Plans:
- [x] 26-01-PLAN.md — Probe module (probe_model, cache, extract_activations_with_model) + config/extraction wiring

### Phase 27: Partial Weight Transfer
**Goal**: Hypernetwork aggregator weights transfer from the `qwen_4b_d2l` checkpoint with aggregator parameters frozen and head re-initialized for Qwen3-Coder-Next's output dimensions, verified by key-presence assertions.
**Depends on**: Phase 26
**Requirements**: XFER-01, XFER-02
**Success Criteria** (what must be TRUE):
  1. After loading, all `aggregator.*` parameters have `requires_grad=False` — confirmed by iterating named parameters
  2. After loading, `head.*` parameters are absent from the checkpoint (in `missing_keys`) and have the correct output shape for 12 Qwen3-Coder-Next attention layers
  3. A post-load assertion explicitly fails if any expected aggregator key is missing from the loaded state dict
**Plans:** 1/1 plans complete
Plans:
- [ ] 27-01-PLAN.md — Weight transfer functions (transfer_aggregator_weights, get_aggregator_config, assertions) + tests

### Phase 28: Functional LoRA Injection
**Goal**: The `apply_functional_lora` context manager patches target modules using `F.linear` with live hypernetwork tensor graph nodes, preserving autograd continuity through the hypernetwork, and restores original forward methods on exit.
**Depends on**: Phase 27
**Requirements**: LORA-01, LORA-02, LORA-03
**Success Criteria** (what must be TRUE):
  1. Generated LoRA tensors have non-None `.grad` after `loss.backward()` — hypernetwork parameters receive gradient signal
  2. All hypernetwork parameters have non-None `.grad` after a backward pass through the functional LoRA path
  3. After exiting the context manager, target module forward methods are restored to their originals with no side effects (verified by calling the module again outside the context)
**Plans:** 1/1 plans complete
Plans:
- [x] 28-01-PLAN.md — apply_functional_lora context manager + autograd/restoration TDD tests

### Phase 29: Training Loop Integration
**Goal**: The complete KL-divergence context distillation training script assembles all prior components, with dry-run mode verifying shapes on CPU in under 30 seconds and smoke-test mode confirming finite, decreasing loss over 5 real training steps.
**Depends on**: Phase 28
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, TEST-01
**Success Criteria** (what must be TRUE):
  1. `--dry-run` completes in under 30 seconds on CPU, validates all tensor shapes in one forward pass, and exits 0 with no optimizer step taken
  2. `--smoke-test` runs 5 training steps, reports finite loss at every step, and step-5 loss is lower than step-1 loss
  3. The KL loss unit test returns zero when `student_logits == teacher_logits` (formula correctness verified)
  4. Optimizer is scoped exclusively to trainable parameters (head + projections only) — aggregator parameters have zero gradient after backward
  5. Checkpoint files written every N steps contain model state, config, step count, and attention layer indices — readable back without errors
**Plans:** 2/2 plans complete
Plans:
- [ ] 29-01-PLAN.md — D2LTrainConfig + _compute_kl_ce_loss + 7 unit tests + mlflow dep + exports
- [ ] 29-02-PLAN.md — Full training loop (train_d2l_qwen3, checkpoint, MLflow, dry-run, smoke-test, CLI) + 7 remaining tests

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. README | v1.0 | 1/1 | Complete | 2026-03-02 |
| 2. Implementation Plan | v1.0 | 2/2 | Complete | 2026-03-02 |
| 3. Architecture Docs | v1.0 | 1/1 | Complete | 2026-03-02 |
| 4. Cleanup | v2.0 | 3/3 | Complete | 2026-03-02 |
| 5. Foundation Libraries | v2.0 | 3/3 | Complete | 2026-03-02 |
| 5.1. Template Artifact Cleanup | v2.0 | 2/2 | Complete | 2026-03-03 |
| 6. Service Scaffolds | v2.0 | 4/4 | Complete | 2026-03-03 |
| 7. Configuration & Quality Gate | v2.0 | 3/3 | Complete | 2026-03-03 |
| 8. MkDocs Infrastructure | v3.0 | 2/2 | Complete | 2026-03-03 |
| 9. References Skeleton + Background | v3.0 | 2/2 | Complete | 2026-03-03 |
| 10. Methods Section | v3.0 | 1/1 | Complete | 2026-03-03 |
| 11. Results & Discussion Outlines | v3.0 | 2/2 | Complete | 2026-03-03 |
| 12. Abstract + Quality Audit | v3.0 | 2/2 | Complete | 2026-03-03 |
| 13. Test Infrastructure | v4.0 | 2/2 | Complete | 2026-03-03 |
| 14. Core Library Wireframes | v4.0 | 3/3 | Complete | 2026-03-03 |
| 15. New & Reworked Library Wireframes | v4.0 | 2/2 | Complete | 2026-03-03 |
| 16. Service Wireframes | v4.0 | 4/4 | Complete | 2026-03-03 |
| 17. Quality Gate | v4.0 | 2/2 | Complete | 2026-03-04 |
| 18. Adapter Registry | v5.0 | 2/2 | Complete | 2026-03-05 |
| 19. Inference Provider Abstraction | v5.0 | 3/3 | Complete | 2026-03-05 |
| 20. Agent Loop | v5.0 | 2/2 | Complete | 2026-03-05 |
| 21. QLoRA Training Pipeline | v5.0 | 2/2 | Complete | 2026-03-05 |
| 22. Kill-Switch Gate | v5.0 | 3/3 | Complete | 2026-03-06 |
| 23. Integration Fix & Quality Gate | v5.0 | 1/1 | Complete | 2026-03-06 |
| 24. Phase 22 Verification & Dep Hygiene | v5.0 | 1/1 | Complete | 2026-03-06 |
| 25. Configuration & Data Pipeline | v7.0 | 2/2 | Complete | 2026-03-13 |
| 26. Architecture Probe & Activation Extraction | v7.0 | Complete    | 2026-03-13 | 2026-03-13 |
| 27. Partial Weight Transfer | 1/1 | Complete    | 2026-03-13 | - |
| 28. Functional LoRA Injection | v7.0 | 1/1 | Complete | 2026-03-16 |
| 29. Training Loop Integration | 2/2 | Complete   | 2026-03-16 | - |

---
*Last updated: 2026-03-16 after Phase 29 planning complete*
