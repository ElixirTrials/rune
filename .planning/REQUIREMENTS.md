# Requirements: Rune

**Defined:** 2026-03-05
**Core Value:** A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.

## v5.0 Requirements

Requirements for first working implementation. Each maps to roadmap phases.

### Adapter Registry

- [x] **AREG-01**: User can store a new adapter record with metadata (task_type, base_model_id, rank, file_path, file_hash, source) via AdapterRegistry.store()
- [x] **AREG-02**: User can retrieve an adapter record by ID via AdapterRegistry.retrieve_by_id()
- [x] **AREG-03**: User can query adapters by task_type via AdapterRegistry.query_by_task_type()
- [x] **AREG-04**: User can list all non-archived adapters via AdapterRegistry.list_all()
- [x] **AREG-05**: AdapterRegistry uses SQLite with WAL mode and engine-parameterized constructor for shared database access

### Inference

- [x] **INF-01**: Abstract InferenceProvider interface with generate(), load_adapter(), unload_adapter(), list_adapters() methods
- [x] **INF-02**: VLLMProvider implementation with full LoRA hot-loading support (POST /v1/load_lora_adapter, /v1/unload_lora_adapter)
- [x] **INF-03**: OllamaProvider implementation for inference via Ollama API (LoRA support where Ollama supports it)
- [x] **INF-04**: Provider factory/registry for selecting backend by configuration (env var or config file)
- [x] **INF-05**: User can generate with a specific loaded adapter by passing adapter name as model parameter
- [x] **INF-06**: User can load multiple adapters simultaneously for composition (provider-dependent, graceful degradation)
- [x] **INF-07**: Per-step model/provider configuration — agent can use different models or providers for different steps (e.g., one model for generation, another for reflection)

### Agent Loop

- [x] **AGENT-01**: generate_node calls InferenceProvider.generate() with task description and optional adapter, returning generated code (backend-agnostic)
- [x] **AGENT-02**: execute_node runs generated code in a sandboxed subprocess with timeout, returning stdout/stderr/exit_code/tests_passed
- [x] **AGENT-03**: reflect_node accumulates trajectory data (attempt count, code, results) without LLM call
- [x] **AGENT-04**: save_trajectory_node persists trajectory via record_trajectory() and sets outcome
- [x] **AGENT-05**: RuneState includes session_id field for trajectory persistence
- [x] **AGENT-06**: Agent loop closes end-to-end: generate → execute → reflect → retry/save with should_retry routing

### Training Pipeline

- [x] **TRAIN-01**: User can record a coding trajectory as structured JSON via record_trajectory(session_id, steps, outcome)
- [x] **TRAIN-02**: User can convert trajectory data to SFT chat format via format_for_sft()
- [x] **TRAIN-03**: User can create a QLoRA config with NF4 quantization and bfloat16 compute dtype via build_qlora_config()
- [x] **TRAIN-04**: User can apply a LoRA adapter to a base model via apply_lora_adapter()
- [x] **TRAIN-05**: QLoRA training pipeline runs end-to-end: trajectory → SFT format → PEFT train → save safetensors → store in registry
- [x] **TRAIN-06**: training-svc exposes POST /train/lora endpoint with async background job tracking
- [x] **TRAIN-07**: training-svc pyproject.toml declares model-training as workspace dependency

### Doc-to-LoRA Hypernetwork

- [x] **DTOL-01**: DocToLoraHypernetwork module exists in model-training lib with Perceiver-based architecture
- [x] **DTOL-02**: Hypernetwork generates rank-8 LoRA adapter weights from coding trajectory in a single forward pass (<1s)
- [x] **DTOL-03**: Generated adapters are compatible with vLLM dynamic LoRA loading (standard PEFT safetensors format)
- [ ] **DTOL-04**: training-svc exposes POST /train/hypernetwork endpoint for Doc-to-LoRA training

### Evaluation & Kill-Switch

- [ ] **EVAL-01**: User can run a HumanEval subset benchmark via run_humaneval_subset()
- [ ] **EVAL-02**: User can calculate Pass@k metrics via calculate_pass_at_k()
- [ ] **EVAL-03**: Kill-switch gate compares baseline vs adapter-enhanced Pass@1 (5% improvement threshold)

### Infrastructure

- [x] **INFRA-01**: lora-server Dockerfile uses vllm/vllm-openai:v0.16.0 base image (not python:3.12-slim)
- [x] **INFRA-02**: lora-server sets VLLM_ALLOW_RUNTIME_LORA_UPDATING=True environment variable
- [x] **INFRA-03**: docker-compose resolves port conflict (api-service and lora-server on different host ports)
- [x] **INFRA-04**: model-training pyproject.toml adds GPU dependencies (peft, bitsandbytes, transformers, trl, datasets) with TYPE_CHECKING guards
- [x] **INFRA-05**: All GPU imports deferred inside function bodies (not top-level) for CPU-only CI compatibility

## Future Requirements

Deferred to v6+. Tracked but not in current roadmap.

### Evolution

- **EVOL-01**: Evolution operator scores adapter fitness based on HumanEval pass rate
- **EVOL-02**: Evolution operator performs adapter crossover via weight interpolation
- **EVOL-03**: Evolution operator archives low-fitness adapters

### Advanced Features

- **ADV-01**: Embedding-based automatic adapter selection via vector similarity search
- **ADV-02**: HITL UI with session monitoring, adapter library browser, trajectory viewer
- **ADV-03**: Docker container isolation for code execution (replace subprocess sandbox)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Hardware validation phase | User deferred — will validate hardware separately |
| Tensor parallelism (TP=2) | vLLM bug #21471 — corrupted output on PCIe GPUs without NVLink |
| Adapter merging into base weights | Loses hot-swap composability; vLLM Punica kernels handle LoRA with near-zero overhead |
| Cloud API inference | Hard constraint: local-first, no cloud dependencies for inference. Provider abstraction supports local backends (vLLM, Ollama) only |
| Concurrent training + inference | CUDA OOM — must schedule sequentially |
| Streaming responses in agent loop | Agent needs complete code, not token stream; add to API layer separately if needed |
| Multi-tenant isolation | Single-user local system; session_id provides sufficient organization |
| evolution-svc implementation | Depends on populated adapter library from multiple sessions; v6+ |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| AREG-01 | Phase 18 | Complete |
| AREG-02 | Phase 18 | Complete |
| AREG-03 | Phase 18 | Complete |
| AREG-04 | Phase 18 | Complete |
| AREG-05 | Phase 18 | Complete |
| INF-01 | Phase 19 | Complete |
| INF-02 | Phase 19 | Complete |
| INF-03 | Phase 19 | Complete |
| INF-04 | Phase 19 | Complete |
| INF-05 | Phase 19 | Complete |
| INF-06 | Phase 19 | Complete |
| INF-07 | Phase 19 | Complete |
| INFRA-01 | Phase 19 | Complete |
| INFRA-02 | Phase 19 | Complete |
| INFRA-03 | Phase 19 | Complete |
| AGENT-01 | Phase 20 | Complete |
| AGENT-02 | Phase 20 | Complete |
| AGENT-03 | Phase 20 | Complete |
| AGENT-04 | Phase 20 | Complete |
| AGENT-05 | Phase 20 | Complete |
| AGENT-06 | Phase 20 | Complete |
| TRAIN-01 | Phase 20 | Complete |
| TRAIN-02 | Phase 20 | Complete |
| TRAIN-03 | Phase 21 | Complete |
| TRAIN-04 | Phase 21 | Complete |
| TRAIN-05 | Phase 21 | Complete |
| TRAIN-06 | Phase 21 | Complete |
| TRAIN-07 | Phase 21 | Complete |
| INFRA-04 | Phase 21 | Complete |
| INFRA-05 | Phase 21 | Complete |
| DTOL-01 | Phase 22 | Complete (22-01) |
| DTOL-02 | Phase 22 | Complete (22-01) |
| DTOL-03 | Phase 22 | Complete (22-01) |
| DTOL-04 | Phase 22 | Pending |
| EVAL-01 | Phase 22 | Pending |
| EVAL-02 | Phase 22 | Pending |
| EVAL-03 | Phase 22 | Pending |

**Coverage:**
- v5.0 requirements: 37 total
- Mapped to phases: 37
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-05*
*Last updated: 2026-03-05 after 19-02 complete (INFRA-01, INFRA-02, INFRA-03 completed)*
