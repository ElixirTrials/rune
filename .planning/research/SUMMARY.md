# Project Research Summary

**Project:** Rune v5.0 — First Working Implementation
**Domain:** Local-first coding agent with LoRA adapter training, hot-loading, and Doc-to-LoRA hypernetwork
**Researched:** 2026-03-05
**Confidence:** HIGH

## Executive Summary

Rune v5.0 is a local-first coding agent that closes a full generate-execute-reflect-train loop on consumer GPU hardware (dual RTX 4090, 48 GB VRAM total). The system's research question is whether session-derived LoRA adapters — either distilled via QLoRA gradient descent or generated in a single forward pass via a Doc-to-LoRA hypernetwork — measurably improve code generation quality (Pass@1) over the unadapted base model. The entire codebase already exists as API wireframes with failing TDD tests; v5.0 is purely implementation, not design. Every interface is defined, every stub returns `NotImplementedError`, and the kill-switch gate (5% Pass@1 improvement on a 20-task HumanEval subset) is the singular go/no-go criterion for the Doc-to-LoRA hypothesis.

The recommended implementation approach follows a strict build order dictated by dependency topology: adapter-registry CRUD first (everything else writes to or reads from it), lora-server hardware validation second (PP=2 + LoRA must be empirically verified on Qwen2.5-Coder-7B before any integration work), inference lib and model-training lib third (parallel), then agent nodes and training-svc in parallel, and finally the Doc-to-LoRA hypernetwork module once the full pipeline produces real training trajectories. The stack is fully determined — vLLM 0.16.0, PEFT 0.18.1, bitsandbytes 0.49.2, transformers 5.3.0, TRL 0.29.0 — with specific version pins verified against PyPI. No technology decisions remain open.

The dominant risk is the PP=2 + LoRA compatibility question. While vLLM PR #7292 fixed the PPMissingLayer crash for Llama in August 2024, regressions have been documented in v0.12.0 (issue #30269), and Qwen2.5 architecture specifics have not been verified. If PP=2 + LoRA fails on the target hardware, the fallback is PP=1/TP=1 (single GPU, 24 GB), which still supports the core hypothesis test but halves serving capacity. This determination must be the first act of implementation — before a single stub is filled in.

## Key Findings

### Recommended Stack

The baseline stack (FastAPI, LangGraph, SQLModel, Pydantic, openai async client, uv workspace) is already in place and must not change. The new additions required for v5.0 are six packages: vLLM 0.16.0 (LLM serving with multi-LoRA, PP=2 support), PEFT 0.18.1 (LoRA adapter training via `LoraConfig` + `get_peft_model`), bitsandbytes 0.49.2 (NF4 4-bit quantization for QLoRA training), transformers 5.3.0 (model loading with `BitsAndBytesConfig`), TRL 0.29.0 (SFTTrainer for chat-format trajectory data), and datasets 4.6.1 (in-memory dataset conversion). Supporting libraries are aiosqlite (if async SQLite is needed), safetensors (adapter weight I/O), and accelerate (device map for multi-GPU model loading). The lora-server runs in the official `vllm/vllm-openai:v0.16.0` Docker image — building vLLM from source is explicitly rejected (30+ min CUDA kernel compilation, fragile).

The critical serving configuration is `--pipeline-parallel-size 2 --tensor-parallel-size 1 --enable-lora --quantization awq --max-loras 2 --max-lora-rank 64 --gpu-memory-utilization 0.80`. Tensor parallelism is hard-blocked by vLLM bug #21471 (corrupted output on PCIe-connected RTX 4090s without NVLink), and `LoraServerConfig.__post_init__` already enforces TP=1 with a `ValueError`. The two quantization systems must be kept distinct: QLoRA NF4 is used during training to produce standard PEFT adapter files; AWQ is used during serving as the base model quantization. The resulting adapter files are compatible with AWQ-based dynamic LoRA loading — they are standard PEFT format, not quantized weights.

**Core technologies (new additions):**
- **vLLM 0.16.0**: LLM serving with multi-LoRA hot-loading — only production-grade engine supporting PP=2 + dynamic LoRA concurrently
- **PEFT 0.18.1**: LoRA adapter training — official HuggingFace implementation, `LoraConfig` + `get_peft_model` + `save_pretrained`
- **bitsandbytes 0.49.2**: NF4 4-bit quantization — required by QLoRA; training path only (not serving)
- **transformers 5.3.0**: Model loading — `AutoModelForCausalLM.from_pretrained` with `BitsAndBytesConfig`
- **TRL 0.29.0**: `SFTTrainer` — direct PEFT/QLoRA integration; chat-format training data from `format_for_sft`
- **datasets 4.6.1**: In-memory dataset — `Dataset.from_dict()` converts trajectory messages to HuggingFace format

### Expected Features

The feature set is partitioned into three tiers by the research. The agent loop (generate -> execute -> reflect -> save_trajectory) plus the adapter registry and training pipeline constitute the table stakes — without all of these working end-to-end, the hypothesis cannot be tested. The Doc-to-LoRA hypernetwork and evaluation lib (for the kill-switch gate) are the differentiating features that make Rune a research contribution rather than a standard coding agent. Everything else — evolution operator, embedding-based adapter retrieval, multi-tenant isolation, streaming — is deferred to v6+ and must not be implemented in v5.0.

**Must have (table stakes — v5.0 MVP):**
- Hardware validation script — gates PP=2 + LoRA compatibility check; must pass before any implementation
- AdapterRegistry full CRUD (`store`, `retrieve_by_id`, `query_by_task_type`, `list_all`) — SQLite-backed; the hub of every other component
- `record_trajectory()` + `format_for_sft()` — trajectory persistence and SFT chat-format conversion
- `generate_node()` — calls VLLMClient, returns code string
- `execute_node()` — subprocess sandbox with 30-second timeout; returns stdout/stderr/exit_code/tests_passed
- `reflect_node()` — mechanical accumulation; increments attempt_count; appends to trajectory; no LLM call
- `save_trajectory_node()` — calls `record_trajectory()`; sets outcome
- `VLLMClient.generate()` + `load_adapter()` + `unload_adapter()` — vLLM OpenAI-compatible client
- `build_qlora_config()` + `apply_lora_adapter()` — PEFT LoraConfig instantiation with NF4 quantization
- QLoRA end-to-end training pipeline — trajectory to SFT format to PEFT train to save safetensors to registry

**Should have (kill-switch validation — v5.x, post-MVP):**
- `run_humaneval_subset()` + `calculate_pass_at_k()` — evaluation lib for the 5% Pass@1 gate
- Doc-to-LoRA hypernetwork forward pass — `DocToLoraHypernetwork` in `model-training` lib; generates adapter in <1 second
- Adapter composition (multi-adapter loading) — `state["adapter_ids"]` supporting multiple adapters

**Defer (v6+):**
- Evolution operator — fitness-based adapter promotion, crossover, pruning; depends on a populated adapter library
- Embedding-based automatic adapter selection — requires vector similarity index; obscures Phase 1 hypothesis test
- HITL UI beyond basic monitoring
- Docker container isolation for code execution (subprocess + timeout is sufficient for v5.0)

**Anti-features (explicitly rejected):**
- Tensor parallelism (TP=2) — vLLM bug #21471, enforced by `LoraServerConfig`
- Adapter merging into base weights at inference time — loses hot-swap composability
- Cloud API inference (OpenAI, Anthropic) — hard constraint in PROJECT.md; local-first
- Concurrent training + inference — CUDA OOM; schedule sequentially

### Architecture Approach

The architecture follows a library-as-business-logic, service-as-HTTP-shell pattern throughout: all core logic lives in six importable Python workspace libs (`adapter-registry`, `inference`, `model-training`, `evaluation`, `shared`, `events-py`), and five FastAPI services (`api-service`, `rune-agent`, `training-svc`, `evolution-svc`, `lora-server`) are thin HTTP wrappers. The lora-server is outside the uv workspace and runs in the official vLLM Docker image. GPU imports (`peft`, `bitsandbytes`, `transformers`) must be deferred inside function bodies throughout the workspace — top-level imports break CPU-only CI and service startup on non-GPU hosts.

All components sharing adapter data must point to the same SQLite file via a `DATABASE_URL` environment variable and pass the SQLModel engine explicitly to `AdapterRegistry(engine=engine)` — the no-argument constructor pattern would silently create per-service databases and silo records. LoRA adapter files are shared via a volume-mounted `/adapters/` directory; `AdapterRecord.file_path` stores the path as seen by the vLLM container. A critical docker-compose port conflict (both api-service and lora-server expose port 8000) must be resolved before any integration work.

**Major components:**
1. **adapter-registry lib** — canonical SQLite store for all adapter metadata; `AdapterRecord` table; engine-parameterized `AdapterRegistry` class
2. **lora-server** — vLLM GPU inference container; PP=2/TP=1/AWQ; dynamic LoRA hot-loading via `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`
3. **inference lib** — stateless vLLM OpenAI client wrapper; `generate_completion`, `generate_with_adapter`, `load_adapter`, `unload_adapter`
4. **model-training lib** — QLoRA PEFT config (`build_qlora_config`), trajectory persistence (`record_trajectory`, `format_for_sft`), and (new) `DocToLoraHypernetwork` module
5. **rune-agent** — LangGraph `StateGraph`; four node stubs to implement (`generate_node`, `execute_node`, `reflect_node`, `save_trajectory_node`); `should_retry` conditional edge already implemented
6. **training-svc** — FastAPI background job dispatcher; `POST /train/lora` calls `SFTTrainer`; `GET /jobs/{job_id}` tracks async training tasks
7. **evaluation lib** — HumanEval benchmark runner; `calculate_pass_at_k()`; required for kill-switch gate

**Build order (dependency-constrained):**

1. adapter-registry CRUD
2. lora-server hardware validation
3. inference lib
4. model-training lib (parallel with inference lib for CPU-safe functions)
5a. rune-agent nodes (parallel with training-svc)
5b. training-svc wiring
6a. api-service routers (parallel with evaluation lib)
6b. evaluation lib
7. hypernetwork module
8. kill-switch gate
9. evolution-svc (v6+)

### Critical Pitfalls

1. **PP=2 + LoRA may still be broken for Qwen2.5** — PR #7292 fixed the Llama case, but issue #30269 shows PP regressions in v0.12.0. Empirically verify `vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --pipeline-parallel-size 2 --enable-lora` generates correct output before any implementation work. Fallback: PP=1/TP=1 (single GPU).

2. **`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` is required for dynamic loading** — without it, `/v1/load_lora_adapter` returns 404 or 405 silently. Add this env var to the lora-server Dockerfile on day one; add a smoke test that POSTs a test adapter on server startup to confirm the endpoint is live.

3. **AWQ serving quantization and QLoRA training quantization are separate systems** — "QLoRA" refers only to how adapters are trained (NF4 bitsandbytes), not to their file format. The saved PEFT adapter is standard safetensors; it is compatible with AWQ-based vLLM dynamic loading. Never use `--qlora-adapter-path` in vLLM startup — that path disables dynamic loading. Never use `model.merge_and_unload()` before saving adapters.

4. **`bnb_4bit_compute_dtype=torch.float16` causes silent NaN loss** — always use `torch.bfloat16` in `build_qlora_config()`. Float16's exponent range overflows during NF4 dequantization for 7B models; loss becomes NaN after 10-50 steps; the resulting adapter saves without error but generates garbage output. Hardcode `bfloat16` as a constant, not a default parameter.

5. **max_loras VRAM pre-allocation causes OOM at startup** — with `max_loras=8` and `max_lora_rank=64`, vLLM pre-allocates approximately 24 GB for adapter slots before loading model weights. Start with `max_loras=2`, `max_lora_rank=64`, `gpu_memory_utilization=0.80` and validate startup first.

6. **SQLite without WAL mode deadlocks under concurrent FastAPI requests** — add `PRAGMA journal_mode=WAL` via a SQLAlchemy connection event to all SQLite engines before implementing any router logic. `check_same_thread=False` is not sufficient.

7. **LangGraph default recursion limit (25) obscures bugs in `reflect_node`** — if `reflect_node` fails to increment `attempt_count`, the loop cycles indefinitely until hitting the limit, and `GraphRecursionError` hides the real cause. Set explicit `recursion_limit = max_attempts * 4 + 5` in `graph.invoke()` and add a monotonicity assertion in `should_retry`.

8. **Doc-to-LoRA meta-training takes days to weeks on multiple GPUs** — the kill-switch gate must use either Sakana AI's pre-trained weights (adapted to Qwen2.5) or a compute-budgeted partial training run (e.g., 24 GPU-hours). Do not plan for full convergence in v5.0. Define pass/fail in terms of relative improvement over baseline, not absolute performance.

## Implications for Roadmap

Based on the dependency graph from ARCHITECTURE.md and the pitfall-to-phase mapping from PITFALLS.md, five phases are recommended for v5.0.

### Phase 0: Hardware Validation
**Rationale:** Hardware validation gates everything. The PP=2 + LoRA compatibility question and the max_loras VRAM calculation must be resolved before a single stub is implemented. If the hardware validation phase produces a "fallback to PP=1" outcome, the entire training-to-serving pipeline contract changes. This must come first.
**Delivers:** Verified vLLM startup configuration; confirmed PP=2 + LoRA compatibility (or documented fallback); validated VRAM headroom numbers; `VLLM_ALLOW_RUNTIME_LORA_UPDATING` smoke test passing
**Addresses:** Hardware validation script feature from FEATURES.md
**Avoids:** PP=2 + LoRA incompatibility (Pitfall 1); max_loras VRAM OOM (Pitfall 11); AWQ + QLoRA confusion (Pitfall 4)

### Phase 1: Infrastructure Foundation
**Rationale:** All other components write to or read from adapter-registry. All agent and training components call through the inference lib. These two libs plus the lora-server client must be implemented and tested before any higher-level component can be tested end-to-end.
**Delivers:** Working `AdapterRegistry` CRUD with WAL-mode SQLite; working `VLLMClient` with dynamic adapter loading/unloading; working inference lib (`generate_completion`, `generate_with_adapter`)
**Addresses:** AdapterRegistry full CRUD; VLLMClient.generate() + load_adapter() + unload_adapter()
**Uses:** SQLModel, openai async client, aiosqlite
**Avoids:** Multiple-engine SQLite silo anti-pattern; VLLM_ALLOW_RUNTIME_LORA_UPDATING missing (Pitfall 2); POST stubs missing schemas (Pitfall 9)
**Research flag:** Standard patterns — no additional research needed; all APIs verified

### Phase 2: Agent Loop
**Rationale:** The agent loop depends on inference lib (Phase 1) for generation and model-training lib for trajectory recording. The four node stubs and the model-training lib (config, trajectory, peft_utils — CPU-safe functions) can be implemented together since `save_trajectory_node` is the bridge between them.
**Delivers:** Fully functional generate -> execute -> reflect -> save_trajectory loop; trajectory JSONL files written to `/adapters/trajectories/`; all three exit paths tested (success, exhausted, exception)
**Addresses:** All four node functions; `record_trajectory()` + `format_for_sft()`; `make_initial_state()` factory function
**Uses:** LangGraph StateGraph (existing graph.py), asyncio subprocess sandbox, model-training lib
**Avoids:** RuneState missing keys on invocation (Pitfall 7); LangGraph recursion limit obscuring bugs (Pitfall 6); trajectory stored only in memory (Architecture Anti-Pattern 4)
**Research flag:** Standard patterns — LangGraph recursion loop is well-documented

### Phase 3: QLoRA Training Pipeline
**Rationale:** The training pipeline requires real trajectories from Phase 2 as training data. It is also a prerequisite for the Doc-to-LoRA hypernetwork (Phase 4 needs reference LoRA adapters as meta-training targets). This phase implements the gradient-descent path (slower, proven, fallback if hypernetwork fails kill-switch).
**Delivers:** End-to-end QLoRA training: trajectory to SFT format to PEFT train (SFTTrainer) to save safetensors to AdapterRegistry.store(); training-svc `/train/lora` endpoint with async background job tracking; adapter format validation (no embed_tokens keys)
**Addresses:** `build_qlora_config()` + `apply_lora_adapter()`; QLoRA training pipeline; training-svc job dispatch
**Uses:** PEFT 0.18.1, bitsandbytes 0.49.2, transformers 5.3.0, TRL 0.29.0, datasets 4.6.1
**Avoids:** Adapter format incompatibility (Pitfall 3 — explicit target_modules, key validation); QLoRA float16 NaN loss (Pitfall 10 — bfloat16 hardcoded); GPU imports at module level (Architecture Anti-Pattern 2)
**Research flag:** Standard patterns — PEFT/QLoRA pipeline is well-documented; specific Qwen2.5 target_modules may need verification

### Phase 4: Kill-Switch Gate (Doc-to-LoRA Validation)
**Rationale:** The Doc-to-LoRA hypernetwork is the core research contribution. It cannot be implemented until real training trajectories exist (Phase 2) and the PEFT infrastructure is wired (Phase 3). The evaluation lib is a prerequisite for measuring the gate criterion. This phase implements the hypothesis test — if it fails, the hypernetwork approach is invalidated and the project falls back to QLoRA distillation.
**Delivers:** `DocToLoraHypernetwork` module in model-training lib; `run_humaneval_subset()` + `calculate_pass_at_k()` in evaluation lib; training-svc `/train/hypernetwork` endpoint; kill-switch gate comparison (baseline vs. adapter Pass@1)
**Addresses:** Doc-to-LoRA hypernetwork (differentiator); evaluation lib; HITL kill-switch gate
**Avoids:** Meta-training timeline underestimation (Pitfall 12 — fixed compute budget, relative improvement metric); cold-start problem (use manually trained LoRA adapters from Phase 3 as reference targets)
**Research flag:** NEEDS research-phase — Doc-to-LoRA architecture adaptation to Qwen2.5 and Sakana AI pre-trained weight availability are open questions

### Phase 5: API Service + Adapter Composition
**Rationale:** The api-service HTTP layer and multi-adapter composition are useful for external visibility and experiment design but not required for the core loop or hypothesis test. They belong after the kill-switch gate is resolved.
**Delivers:** Working api-service adapter CRUD endpoints (`GET /adapters`, `GET /adapters/{id}`, `POST /adapters`); adapter composition in agent state (`state["adapter_ids"]` supporting multiple adapters); api-service schemas.py with proper Pydantic request models
**Addresses:** api-service adapter routers; adapter versioning and lineage
**Avoids:** Multiple AdapterRegistry engines (confirm shared DATABASE_URL); docker-compose port conflict (api-service must use host port other than 8000)
**Research flag:** Standard patterns — FastAPI dependency injection is straightforward

### Phase Ordering Rationale

- Hardware validation before infrastructure: PP=2 + LoRA failure changes the serving contract globally; discovering this after implementing three phases of integration would require rework
- Infrastructure before agent: agent loop has hard dependencies on inference lib (VLLMClient.generate) and adapter-registry (save_trajectory_node); neither can be tested end-to-end without Phase 1
- Agent before training pipeline: training-svc needs real trajectories as training data; deferring it produces better integration tests against real data
- Training pipeline before kill-switch: Doc-to-LoRA requires reference LoRA adapters as meta-training targets (cold-start problem); Phase 3 produces those adapters
- Kill-switch before api-service: if the hypothesis fails, priorities shift; api-service polish should not block learning whether the research question has a positive answer
- evolution-svc deferred to v6+: depends on a populated adapter library from multiple sessions; cannot be meaningfully validated until Phases 2-4 have produced a library

### Research Flags

Phases needing deeper research during planning:
- **Phase 4 (Kill-Switch Gate):** Doc-to-LoRA architecture adaptation to Qwen2.5-Coder-7B requires understanding the Perceiver cross-attention dimension alignment; Sakana AI pre-trained weight availability and licensing are unknown; meta-training cold-start strategy needs concrete dataset selection

Phases with standard patterns (skip research-phase):
- **Phase 0 (Hardware Validation):** PyTorch CUDA check API is well-documented; vLLM startup flags verified in STACK.md
- **Phase 1 (Infrastructure):** SQLModel CRUD and vLLM OpenAI client patterns are thoroughly documented
- **Phase 2 (Agent Loop):** LangGraph StateGraph recursion loop is a documented pattern; asyncio subprocess sandbox is stdlib
- **Phase 3 (QLoRA Training):** PEFT + bitsandbytes + TRL SFTTrainer stack has extensive official documentation
- **Phase 5 (API Service):** FastAPI dependency injection and Pydantic schemas are standard patterns

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All library versions verified against PyPI; vLLM PP+LoRA fix verified against GitHub issue #7253; AWQ + dynamic LoRA compatibility verified via forum (MEDIUM on that specific claim) |
| Features | HIGH | All stubs inspected directly; feature dependencies derived from actual code structure; Doc-to-LoRA paper read (arXiv:2602.15902); Rune-specific trajectory adaptation is novel (MEDIUM on hypernetwork efficacy) |
| Architecture | HIGH | Based on direct inspection of all 11 component source files; integration patterns derived from actual pyproject.toml dependency declarations |
| Pitfalls | HIGH | vLLM pitfalls verified against GitHub issues #7253, #21471, #29049, #30269; SQLite WAL pitfall verified against SQLAlchemy docs; LangGraph recursion pitfall verified against official error docs |

**Overall confidence:** HIGH

### Gaps to Address

- **PP=2 + LoRA on Qwen2.5-Coder-7B (empirical):** No verified test of this specific combination exists in the research. The PR #7292 fix applies to Llama architecture; Qwen2.5 layer naming may trigger the same `PPMissingLayer` issue. Resolution: Phase 0 hardware validation is the empirical test. Fallback documented.

- **Sakana AI Doc-to-LoRA pre-trained weights availability:** The paper references a checkpoint release, but the research cannot confirm public availability or licensing as of 2026-03-05. Resolution: Phase 4 planning must verify availability first; if unavailable, design cold-start training from scratch with a fixed GPU-hour budget.

- **`RuneState` session_id field missing:** `save_trajectory_node` calls `record_trajectory(session_id, ...)`, but `RuneState` TypedDict does not currently include a `session_id` field. This is a minor schema addition that must be made before agent node implementation.

- **docker-compose port conflict:** Both api-service and lora-server currently expose `${API_PORT:-8000}:8000`. Resolution: api-service must be moved to host port 8080 before any local integration testing.

- **training-svc missing model-training dependency:** `training-svc/pyproject.toml` does not list `model-training` as a workspace dependency (v4.0 tech debt). This must be added at the start of Phase 3 before any training-svc router implementation.

## Sources

### Primary (HIGH confidence)

- vLLM official docs (v0.8.1, stable) — LoRA adapter format, dynamic loading API, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, PP=2 configuration
- vLLM GitHub issues #7253, #21471, #29049, #30269 — PP+LoRA compatibility, TP+LoRA corruption, max_loras OOM, PP regression
- vLLM PyPI — version 0.16.0 confirmed released 2026-02-26
- PEFT PyPI + HuggingFace docs — version 0.18.1; QLoRA training path; adapter saving format
- bitsandbytes PyPI — version 0.49.2; CUDA 12.x compatibility
- transformers PyPI — version 5.3.0; `BitsAndBytesConfig`, `AutoModelForCausalLM`
- TRL GitHub releases — version 0.29.0; `SFTTrainer` with PEFT integration
- datasets PyPI — version 4.6.1; `Dataset.from_dict()`
- HuggingFace PEFT docs — QLoRA quantization, bitsandbytes config, adapter saving
- HuggingFace TRL SFT Trainer docs — chat format, PEFT integration, QLoRA path
- Direct codebase inspection — all 11 component source files, all pyproject.toml declarations
- arXiv:2602.15902 (Doc-to-LoRA paper) — Perceiver architecture, rank-8, <1s forward pass, meta-training cost
- LangGraph official error docs — GRAPH_RECURSION_LIMIT, INVALID_GRAPH_NODE_RETURN_VALUE
- SQLite WAL mode docs + SQLAlchemy event docs — WAL PRAGMA via connection event

### Secondary (MEDIUM confidence)

- vLLM forum: AWQ + LoRA dynamic switching — AWQ supports dynamic adapter switching; bnb does not (forum, not official docs)
- vLLM blog (2026-02-26) — multi-LoRA MoE kernel optimizations in 0.15.0-0.16.0
- Sakana AI Doc-to-LoRA announcement — batched vs iterative mode, chunk composition for long docs
- Dual RTX 4090 vLLM benchmark — throughput data for 14-16B models on consumer hardware

### Tertiary (LOW confidence)

- CXL P2P memory access behavior — `can_device_access_peer()` may return False even with CXL interconnect; PP=2 works through CPU memory copy fallback (acceptable latency for v5.0)

---
*Research completed: 2026-03-05*
*Ready for roadmap: yes*
