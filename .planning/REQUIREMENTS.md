# Requirements: Rune v7.0 Hypernetwork Training

**Defined:** 2026-03-13
**Core Value:** A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.

## v7.0 Requirements

Requirements for KL-divergence context distillation training pipeline. Each maps to roadmap phases.

### Configuration

- [ ] **CFG-01**: `get_d2l_qwen3_config()` returns Qwen3-Coder-Next dimensions (hidden_size, attention layer indices, GQA heads) via AutoConfig
- [ ] **CFG-02**: `build_qwen3_hypernet_config()` constructs HypernetConfig targeting Qwen3-Coder-Next with dynamic layer discovery via `config.layer_types`

### Data Pipeline

- [ ] **DATA-01**: `format_for_distillation()` converts trajectory JSON to `{trajectory, query, answer}` JSONL records with full procedural history
- [ ] **DATA-02**: `generate_needle_dataset()` creates needle-in-haystack synthetic records for smoke testing context distillation
- [ ] **DATA-03**: `save_jsonl()` and `load_jsonl()` provide round-trip JSONL persistence
- [x] **DATA-04**: `generate_trajectory_dataset()` generates coding trajectories from benchmark datasets (HumanEval, MBPP, etc.)
- [x] **DATA-05**: `augment_trajectories()` produces 3-5 augmented variants per validated trajectory
- [ ] **DATA-06**: Train/test split enforced at task-ID level before augmentation, with no task-ID crossing the boundary

### Architecture Probe

- [x] **ARCH-01**: Dynamic attention layer discovery identifies exactly the standard attention layers (not DeltaNet) from hybrid model config
- [x] **ARCH-02**: `extract_activations_with_model()` extracts hidden states from pre-loaded model at specified attention layer indices only

### Weight Transfer

- [x] **XFER-01**: Partial weight transfer from `qwen_4b_d2l` checkpoint freezes `aggregator.*` parameters (`requires_grad=False`)
- [x] **XFER-02**: Head (`head.*`) is re-initialized with correct output dimensions for Qwen3-Coder-Next's 12 attention layers

### Functional LoRA

- [x] **LORA-01**: `apply_functional_lora()` context manager patches target modules with `F.linear` preserving autograd graph
- [x] **LORA-02**: Generated LoRA tensors have non-None `.grad` after backward pass (autograd continuity verified)
- [x] **LORA-03**: Original forward methods restored on context manager exit with no side effects

### Training Loop

- [x] **TRAIN-01**: KL-divergence context distillation training script with CLI arguments matching specification
- [x] **TRAIN-02**: Two-pass teacher/student separation: activations from `[trajectory+query]` only, teacher logits from `[trajectory+query+answer]`
- [x] **TRAIN-03**: Blended loss `α*KL + (1-α)*CE` with configurable temperature and blending coefficient
- [x] **TRAIN-04**: AdamW optimizer scoped to trainable parameters only (head + projections), with cosine LR schedule and warmup
- [x] **TRAIN-05**: Checkpoint saving every N steps with model state, config, step count, and attention layer indices
- [x] **TRAIN-06**: Dry-run mode (`--dry-run`) validates all tensor shapes in one forward pass, exits 0 on success
- [x] **TRAIN-07**: Smoke-test mode (`--smoke-test`) runs 5 training steps, verifies finite loss and non-None gradients

### Testing

- [x] **TEST-01**: 14 training tests covering shape verification, gradient flow, frozen/trainable params, functional LoRA, KL loss, two-pass separation
- [ ] **TEST-02**: 5 data pipeline tests covering needle format, JSONL round-trip, augmentation count, task-ID split integrity

## v7.x Requirements

Deferred to after baseline pipeline is validated working.

### Observability

- **OBS-01**: TensorBoard/wandb integration for loss curve visualization
- **OBS-02**: Per-step structured logging (kl_loss, ce_loss, total_loss, grad_norm, gpu_mem_gb)

### Advanced Data

- **ADATA-01**: LLM-assisted needle generation via Ollama for improved diversity
- **ADATA-02**: Multi-document batched distillation with attention masking

## Out of Scope

| Feature | Reason |
|---------|--------|
| MSE loss on LoRA weights | Weight space permutation symmetry makes L2 distance mathematically unsound |
| PEFT `get_peft_model` during training | Severs autograd graph; silent failure mode where hypernetwork never updates |
| Including DeltaNet layers in activation extraction | Different module structure, incompatible with Perceiver trained on standard attention |
| Per-document gradient-based context distillation | Defeats purpose of amortized hypernetwork; minutes per document vs <1s |
| Concurrent training + inference | CUDA OOM confirmed; sequential scheduling required |
| Modifying `DocToLoraHypernetwork` in `hypernetwork.py` | Hash-based path unchanged; all work on Sakana HyperLoRA perceiver |
| Augmentation before train/test split | Inflates evaluation metrics via task-family leakage |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CFG-01 | Phase 25 | Pending |
| CFG-02 | Phase 25 | Pending |
| DATA-01 | Phase 25 | Pending |
| DATA-02 | Phase 25 | Pending |
| DATA-03 | Phase 25 | Pending |
| DATA-04 | Phase 25 | Complete |
| DATA-05 | Phase 25 | Complete |
| DATA-06 | Phase 25 | Pending |
| ARCH-01 | Phase 26 | Complete |
| ARCH-02 | Phase 26 | Complete |
| XFER-01 | Phase 27 | Complete |
| XFER-02 | Phase 27 | Complete |
| LORA-01 | Phase 28 | Complete |
| LORA-02 | Phase 28 | Complete |
| LORA-03 | Phase 28 | Complete |
| TRAIN-01 | Phase 29 | Complete |
| TRAIN-02 | Phase 29 | Complete |
| TRAIN-03 | Phase 29 | Complete |
| TRAIN-04 | Phase 29 | Complete |
| TRAIN-05 | Phase 29 | Complete |
| TRAIN-06 | Phase 29 | Complete |
| TRAIN-07 | Phase 29 | Complete |
| TEST-01 | Phase 29 | Complete |
| TEST-02 | Phase 25 | Pending |

**Coverage:**
- v7.0 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-16 after Phase 28 Plan 01 — LORA-01, LORA-02, LORA-03 complete*
