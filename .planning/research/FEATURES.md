# Feature Research

**Domain:** Local-first coding agent — first working implementation (v5.0)
**Researched:** 2026-03-05
**Confidence:** HIGH for vLLM/LoRA serving patterns (official docs verified); HIGH for LangGraph agent loop (official docs + existing wireframes); MEDIUM for Doc-to-LoRA hypernetwork (paper verified, Rune-specific adaptation is novel); MEDIUM for hardware validation requirements (patterns verified, CXL-specific checks are LOW confidence)

---

## Scope Note

This document covers **software features** for the v5.0 implementation milestone. The predecessor FEATURES.md (2026-03-03) covered scientific article content features for v4.0. That document is superseded here. All six target feature areas from PROJECT.md are mapped: hardware validation, adapter-registry CRUD, vLLM serving with hot-loading, QLoRA training pipeline, rune-agent recursive loop, and Doc-to-LoRA hypernetwork.

The existing codebase has complete API wireframes (all `NotImplementedError` stubs) for all 11 components and TDD failing tests. Implementation fills those stubs. Nothing is being designed from scratch — the interfaces already exist.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that must work for Rune to be demonstrably functional. Missing any one means the end-to-end loop cannot close.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **AdapterRegistry.store() + retrieve_by_id()** | Without persistent adapter storage, nothing else in the system works — training has nowhere to write, agent has nowhere to load from | MEDIUM | SQLModel + SQLite already wired; `AdapterRecord` table model exists; need to instantiate `SQLModel.create_all()`, add a `Session` to `AdapterRegistry.__init__`, and implement the two writes/reads. Error classes (`AdapterAlreadyExistsError`, `AdapterNotFoundError`) already defined. |
| **AdapterRegistry.query_by_task_type() + list_all()** | Agent's adapter selection and the API `/adapters` listing both require queryable registry | LOW | Same SQLite session; straightforward `WHERE task_type = ?` and `WHERE is_archived = False` queries; index on `task_type` already in the model |
| **vLLM server startup with PP=2/TP=1 + LoRA** | The inference layer cannot be tested without a running vLLM server; all agent tests depend on it | HIGH | `startup.sh` already has the correct flags (`--pipeline-parallel-size 2 --tensor-parallel-size 1 --enable-lora --max-loras 8`); complexity is hardware: requires both RTX 4090s visible, AWQ-quantized model weights pre-downloaded, CUDA version matching vLLM wheel |
| **VLLMClient.generate() with model + adapter_id** | The `generate_node` in the agent loop calls this; without it the loop cannot produce any output | MEDIUM | `AsyncOpenAI.chat.completions.create()` with `model=adapter_id` (vLLM uses adapter name as model selector); existing `VLLMClient` class already has the `_client` attribute wired; the stub just needs one `await self._client.chat.completions.create(...)` call |
| **VLLMClient.load_adapter() / unload_adapter()** | Hot-loading is the central vLLM feature; without it the adapter concept is just decoration | MEDIUM | POST to `/v1/load_lora_adapter` with `{"lora_name": ..., "lora_path": ...}`; requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` env var; response on 200 is `"Success: LoRA adapter '...' added successfully."` |
| **generate_node() — code generation** | The first node in the recursive loop; without it the graph cannot execute even one step | MEDIUM | Calls `VLLMClient.generate()` with `state["task_description"]` + system prompt; returns `{"generated_code": str}`. Dependency: `VLLMClient.generate()` must work first. |
| **execute_node() — sandboxed execution** | The second node; without execution results, reflection and retry have no signal | HIGH | Runs generated code + test suite in subprocess with timeout (e.g., `asyncio.create_subprocess_exec` + `asyncio.wait_for`); returns `{"stdout": ..., "stderr": ..., "exit_code": ..., "tests_passed": bool}`. High complexity because subprocess sandboxing on macOS requires handling encoding, timeout, and code injection risks |
| **reflect_node() — trajectory accumulation** | Appends attempt data to `state["trajectory"]`; without this, `save_trajectory_node` has nothing to persist | LOW | Increments `attempt_count`, appends `{"attempt": N, "code": ..., "exit_code": ..., "tests_passed": ...}` to `trajectory`. No LLM call in Phase 1 (reflection is purely mechanical accumulation). |
| **save_trajectory_node() — persist to registry** | Writes the trajectory so training can later consume it; closing the memory loop | MEDIUM | Calls `record_trajectory()` from `model-training` lib with `state["trajectory"]`; sets `state["outcome"]`. Depends on `record_trajectory()` being implemented first. |
| **Hardware validation script** | Phase 0 must gate everything; if GPUs, CUDA, or vLLM are misconfigured, all subsequent phases fail | MEDIUM | Needs: `torch.cuda.device_count() == 2`, both cards are RTX 4090 (24 GB each), CUDA driver version compatible with vLLM wheel, `torch.cuda.is_available()`, P2P access between GPUs (`torch.cuda.can_device_access_peer(0, 1)`), vLLM importable in Docker container |

### Differentiators (Competitive Advantage)

These are the features that make Rune distinct from a standard coding agent. They are not required for Phase 1 loop closure but define the research contribution.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Doc-to-LoRA hypernetwork (forward pass adapter generation)** | Converts coding trajectories to LoRA adapters in <1 second without gradient descent; this is the core hypothesis | HIGH | Hypernetwork architecture: frozen LLM encodes trajectory tokens → Perceiver-style compressor (~309M params) → LoRA weight matrices (rank-8, targeting MLP layers). Training: minimize KL divergence between teacher (LLM with trajectory in context) and student (LLM + generated adapter, no context in prompt). Implementation requires: a pre-trained Doc-to-LoRA checkpoint OR training from scratch on coding trajectories. Kill-switch gate: 5% Pass@1 improvement on HumanEval subset. |
| **Adapter composition — loading multiple adapters per request** | Agent can load project + domain + task adapters simultaneously; memory compounds without token budget growth | HIGH | vLLM supports multiple LoRA adapters per request via `LoRARequest`; composing rank-8 adapters from different sessions adds ranks (effective rank = r × K for K adapters); interference risk increases with K. Use single adapter (most recent task) as the conservative default for Phase 1. |
| **QLoRA fine-tuning from trajectory data** | Distills coding trajectories into LoRA adapters using PEFT + bitsandbytes; traditional path when hypernetwork is not yet trained | HIGH | Pipeline: `record_trajectory()` → `format_for_sft()` → `build_qlora_config()` → `apply_lora_adapter()` → HF Trainer → `merge_adapter()` / save safetensors → `AdapterRegistry.store()`. Requires NF4 quantization of base model (bitsandbytes), PEFT LoraConfig with `task_type=CAUSAL_LM`, rank=64, alpha=128. Training time: ~30 min per session on RTX 4090 |
| **Evolution operator — fitness-based adapter lifecycle** | Promotes, prunes, and merges adapters based on HumanEval pass rate; prevents adapter library bloat | VERY HIGH | Requires: `evaluate_fitness()` scoring, adapter weight interpolation for crossover, archival of low-fitness adapters. This is a Phase 2+ feature — do not implement in v5.0. The `evolution-svc` stubs can remain 501. |
| **Adapter versioning and lineage** | `AdapterRecord.version` field enables tracing adapter lineage; identifies whether an adapter improved from session to session | LOW | Already modeled in `AdapterRecord`; increment `version` when storing an adapter derived from a prior adapter. Query `retrieve_by_id` for the parent. No additional complexity beyond correct version assignment in `store()`. |
| **HITL kill-switch gate** | Human validates Doc-to-LoRA before committing to full implementation; prevents wasted effort on a failing hypothesis | MEDIUM | Phase 1 gate: run HumanEval 20-task subset with and without adapter; if Pass@1 improvement < 5%, the hypernetwork approach is invalidated. The `evaluation` lib `run_humaneval_subset()` and `calculate_pass_at_k()` need to be implemented for this gate. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Tensor parallelism (TP=2)** | "Use both GPUs in parallel for faster inference" | vLLM bug #21471: TP + LoRA produces corrupted outputs on consumer GPUs without NVLink; RTX 4090 uses PCIe, not NVLink. `LoraServerConfig.__post_init__` already raises `ValueError` if `tensor_parallel_size=2` | Use PP=2 (pipeline parallelism): GPU 0 hosts layers 1–N/2, GPU 1 hosts layers N/2+1–N. Slower than TP but correct. |
| **Adapter merging into base weights at inference time** | "Merge LoRA into base model for zero-overhead inference" | Loses the composability property — once merged, you cannot unload or swap adapters; the base model is permanently modified | Keep adapters separate; vLLM's Punica CUDA kernels handle LoRA computation with near-zero overhead without merging. Only merge for export/deployment of a frozen specialist model. |
| **Cloud API for inference (OpenAI, Anthropic)** | "Just use GPT-4o for better code quality" | PROJECT.md: "No cloud API dependencies for inference." Local-first is a hard constraint. Also: cloud APIs cannot load local LoRA adapters. | vLLM + Qwen2.5-Coder-7B-Instruct on local hardware. The research question is whether local SLM + adapter outperforms local SLM baseline — not whether it matches frontier models. |
| **Real-time streaming responses in the agent loop** | "Stream tokens for better UX" | Complicates the generate-node significantly; the agent loop needs complete code, not token-by-token stream. The loop is not a chat interface — it runs tests against the full output. | Complete completion (non-streaming) for the recursive loop. Streaming can be added to the API layer separately if HITL UI needs it. |
| **Multi-tenant adapter isolation** | "Make adapters private per user" | There is only one user (local-first). Multi-tenant isolation adds significant infrastructure complexity (auth, per-user SQLite databases, adapter namespacing) with no benefit on a single-user local system. | Single-user model. `session_id` in `AdapterRecord` provides session-level organization without tenancy. |
| **Automatic adapter selection (retrieval-augmented)** | "Let the system choose which adapters to load automatically based on the task" | The retrieval logic requires embeddings, cosine similarity over all adapters, and threshold tuning. For Phase 1, this is complexity that obscures whether the hypothesis (adapters improve Pass@1) is true. | Manual adapter selection via `state["adapter_ids"]` in Phase 1. Add embedding-based retrieval as a Phase 2 differentiator once the base loop is validated. |
| **Concurrent training + inference (shared GPU)** | "Run training in background while serving" | vLLM holds GPU memory exclusively. CUDA OOM is near-certain when training and inference contend for the same 24 GB VRAM. | Schedule training and inference sequentially. After a session ends and `save_trajectory_node` completes, trigger training as a separate job. The training job blocks the inference server until complete. |

---

## Feature Dependencies

```
Hardware validation
    └──gates──> Everything else (Phase 0 must pass before any implementation work)

AdapterRegistry.store() + retrieve_by_id()
    └──required-by──> save_trajectory_node() (needs a registry to write to)
    └──required-by──> generate_node() (needs registry to load adapter_ids from)
    └──required-by──> VLLMClient.load_adapter() (adapter path comes from registry)

record_trajectory() [model-training lib]
    └──required-by──> save_trajectory_node() (the node calls this function)

format_for_sft() [model-training lib]
    └──required-by──> QLoRA training pipeline (SFT needs chat-format data)
    └──requires──> record_trajectory() (must have stored trajectories to format)

build_qlora_config() + apply_lora_adapter() [model-training lib]
    └──required-by──> QLoRA training pipeline
    └──required-by──> Doc-to-LoRA training (both use PEFT infrastructure)

VLLMClient.generate()
    └──required-by──> generate_node()
    └──requires──> vLLM server running (cannot be tested without GPU + Docker)

VLLMClient.load_adapter()
    └──required-by──> generate_node() (adapter hot-loading before generation)
    └──requires──> AdapterRegistry.retrieve_by_id() (to get adapter file_path)
    └──requires──> vLLM server running

generate_node()
    └──required-by──> execute_node() (must have code to execute)
    └──requires──> VLLMClient.generate()

execute_node()
    └──required-by──> reflect_node()
    └──requires──> generate_node()

reflect_node()
    └──required-by──> save_trajectory_node()
    └──required-by──> should_retry() router (already implemented in graph.py)
    └──requires──> execute_node()

save_trajectory_node()
    └──requires──> reflect_node() (trajectory must be populated)
    └──requires──> record_trajectory()
    └──requires──> AdapterRegistry.store() (to register the resulting adapter)

QLoRA training pipeline
    └──requires──> record_trajectory() + format_for_sft()
    └──requires──> build_qlora_config() + apply_lora_adapter()
    └──produces──> AdapterRecord (stored via AdapterRegistry.store())
    └──required-by──> Doc-to-LoRA training (needs training infrastructure to be wired first)

Doc-to-LoRA hypernetwork
    └──requires──> QLoRA training pipeline (hypernetwork training uses same PEFT infrastructure)
    └──requires──> format_for_sft() (trajectory data must be in chat format)
    └──required-by──> HITL kill-switch gate
    └──enables──> Adapter composition (generate multiple adapters from multiple trajectories)

run_humaneval_subset() + calculate_pass_at_k() [evaluation lib]
    └──required-by──> HITL kill-switch gate
    └──requires──> VLLMClient.generate() (benchmark runs inference)
    └──requires──> AdapterRegistry (to load adapters for evaluation)
```

### Dependency Notes

- **Hardware validation gates everything**: If Phase 0 fails (GPU not visible, CUDA mismatch, PP=2+LoRA not working), none of the above work. Phase 0 is a strict prerequisite.
- **AdapterRegistry is the hub**: Every other feature either writes to or reads from the registry. Implement it first, completely.
- **vLLM server is an external process**: `VLLMClient` methods cannot be unit tested without a running vLLM instance. Tests for these methods should use pytest mocks or the real Docker container in integration tests. The existing `test_vllm_client.py` should mark vLLM-dependent tests with `@pytest.mark.gpu`.
- **model-training lib is a blocker for the agent loop**: `save_trajectory_node()` calls `record_trajectory()`; that function must be implemented before the node can be green-lighted. The lib → service dependency is the primary integration risk.
- **Doc-to-LoRA and QLoRA training are independent paths**: Both produce adapters stored in the registry. Phase 1 uses QLoRA (gradient descent, slower, well-understood); Phase 1 kill-switch uses Doc-to-LoRA (forward pass, fast, novel). They share the PEFT infrastructure but diverge at the training loop.
- **Evolution operator has no upstream dependencies for Phase 1**: The `evolution-svc` can remain stubbed at 501 without blocking the agent loop or training pipeline. It is a Phase 2+ feature.

---

## MVP Definition

### Launch With (v5.0 — first working implementation)

These are the minimum features required to demonstrate that the full Rune loop closes end-to-end and the hypothesis can be tested.

- [ ] **Hardware validation script** — confirm dual RTX 4090 visible, CUDA compat, PP=2+LoRA confirmed working in Docker before any implementation starts
- [ ] **AdapterRegistry full CRUD** — `store()`, `retrieve_by_id()`, `query_by_task_type()`, `list_all()`; SQLite-backed; integration tests passing
- [ ] **record_trajectory() + format_for_sft()** — model-training lib stubs implemented; trajectory persisted as JSON; formatted as chat messages
- [ ] **generate_node()** — calls vLLM via `VLLMClient.generate()`; returns code string
- [ ] **execute_node()** — subprocess execution with timeout; returns stdout/stderr/exit_code/tests_passed
- [ ] **reflect_node()** — increments attempt_count; appends to trajectory; no LLM call
- [ ] **save_trajectory_node()** — calls `record_trajectory()`; sets outcome
- [ ] **VLLMClient.generate()** — `chat.completions.create()` against local vLLM
- [ ] **VLLMClient.load_adapter() + unload_adapter()** — POST to `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`
- [ ] **build_qlora_config() + apply_lora_adapter()** — PEFT LoraConfig instantiation; NF4 quantization config
- [ ] **QLoRA training pipeline** — end-to-end: trajectory → SFT format → PEFT train → save safetensors → store in registry

### Add After Kill-Switch Validation (v5.x)

These features add the differentiating research value once the baseline loop is confirmed working.

- [ ] **Doc-to-LoRA hypernetwork forward pass** — trigger when kill-switch shows QLoRA adapters improve Pass@1; the hypernetwork replaces gradient descent for speed
- [ ] **run_humaneval_subset() + calculate_pass_at_k()** — evaluation lib implementation for the kill-switch gate
- [ ] **Adapter composition (multi-adapter loading)** — `state["adapter_ids"]` supports loading multiple adapters; generation uses the composition; interference testing needed

### Future Consideration (v6+)

- [ ] **Evolution operator** — fitness-based adapter promotion/pruning/crossover; depends on having a populated adapter library from multiple sessions
- [ ] **Embedding-based automatic adapter selection** — vector similarity search over adapter metadata to auto-populate `state["adapter_ids"]`; requires adapter embedding index
- [ ] **HITL UI beyond basic** — real-time session monitoring, adapter library browser, trajectory viewer

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Hardware validation script | HIGH | MEDIUM | P1 |
| AdapterRegistry CRUD (store + retrieve) | HIGH | LOW | P1 |
| generate_node() | HIGH | MEDIUM | P1 |
| execute_node() (sandbox) | HIGH | HIGH | P1 |
| reflect_node() + save_trajectory_node() | HIGH | LOW | P1 |
| VLLMClient.generate() | HIGH | LOW | P1 |
| VLLMClient.load_adapter() | HIGH | MEDIUM | P1 |
| record_trajectory() + format_for_sft() | HIGH | LOW | P1 |
| build_qlora_config() + apply_lora_adapter() | HIGH | MEDIUM | P1 |
| QLoRA full training pipeline | HIGH | HIGH | P1 |
| run_humaneval_subset() + calculate_pass_at_k() | HIGH | MEDIUM | P2 |
| Doc-to-LoRA hypernetwork | HIGH | VERY HIGH | P2 |
| Adapter composition (multi-adapter) | MEDIUM | HIGH | P2 |
| Evolution operator | MEDIUM | VERY HIGH | P3 |
| Embedding-based adapter retrieval | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Required for end-to-end loop closure and kill-switch gate capability
- P2: Required for hypothesis validation and differentiating research value
- P3: Future phases — do not implement in v5.0

---

## How Each Feature Works in Practice

### 1. vLLM Multi-LoRA Hot-Loading

**Startup:** vLLM process launches via `startup.sh` with `--enable-lora --max-loras 8`. Up to 8 adapters can be hot-resident simultaneously. The server manages an LRU cache: oldest unused adapter is evicted when the 9th is loaded.

**Load API (runtime):** Requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. Call:
```
POST /v1/load_lora_adapter
{"lora_name": "adapter-001", "lora_path": "/adapters/adapter-001/"}
```
The `lora_path` must be the directory containing `adapter_config.json` and weight files. This is what `AdapterRecord.file_path` should point to — the directory, not the `.safetensors` file directly.

**Generate with adapter:** Pass the adapter name as the `model` parameter in the chat completions request:
```
POST /v1/chat/completions
{"model": "adapter-001", "messages": [...]}
```
vLLM routes to the base model + loaded adapter. If `adapter-001` is not loaded, the request fails — `load_adapter()` must be called first.

**Unload API:**
```
POST /v1/unload_lora_adapter
{"lora_name": "adapter-001"}
```

**PP=2 + LoRA compatibility:** Bug #7253 was fixed (PR #7292, merged August 2024) for the `logits_processor` AttributeError. The fix is in all vLLM releases after August 2024. Phase 0 hardware validation must confirm the fix works with the specific vLLM + CUDA + Qwen2.5-Coder-7B + AWQ combination on RTX 4090 hardware. **Do not assume it works — validate first.**

**Configuration note:** `LoraServerConfig` already enforces `tensor_parallel_size != 2` via `__post_init__`. No change needed.

---

### 2. QLoRA Training Pipeline

**Workflow:** trajectory data → SFT format → PEFT QLoRA config → Trainer → save → registry

**Step 1 — Collect trajectory:** `record_trajectory(session_id, steps, outcome)` persists a JSON file at a path like `/trajectories/{session_id}.json`.

**Step 2 — Format for SFT:** `format_for_sft(trajectory)` converts steps to chat format:
```json
[
  {"role": "user", "content": "Write a function that..."},
  {"role": "assistant", "content": "def fib(n): ..."},
  {"role": "user", "content": "Test failed: assert fib(5) == 5. Fix it."},
  {"role": "assistant", "content": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"}
]
```
Each (attempt, correction) pair is a turn. The correction steps are the high-signal training data.

**Step 3 — QLoRA config:**
```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], task_type="CAUSAL_LM", dropout=0.1)
```
`rank=64` is the default in `get_training_config()`; matches the existing `LoraTrainingRequest.rank` field default.

**Step 4 — Train:** HF `Trainer` or `trl.SFTTrainer`; 3 epochs default; learning rate 2e-4. Expected duration: ~30 minutes per session trajectory on a single RTX 4090.

**Step 5 — Save and register:** Save adapter as safetensors directory; compute SHA-256 hash of the directory; call `AdapterRegistry.store()` with the new `AdapterRecord`.

**Hardware constraint:** Training requires the full RTX 4090 (24 GB). Cannot run simultaneously with vLLM inference. Schedule sequentially.

---

### 3. Doc-to-LoRA Hypernetwork

**Architecture (from arXiv:2602.15902):**
1. Frozen base LLM encodes the trajectory document → per-layer token activations
2. Perceiver-based compressor (~309M parameters) maps activations → rank-8 LoRA matrices for each MLP layer
3. Output: a full set of rank-8 LoRA A/B matrices for the target model

**Training objective:** Minimize the distillation gap. Teacher: base LLM with trajectory in its context window. Student: base LLM + generated LoRA adapter, without trajectory in context. Loss: KL divergence between teacher and student output distributions.

**Inference (generation time):** Given a new coding trajectory, run a single forward pass of the hypernetwork. Time: <1 second. No gradient steps. Output: LoRA adapter weights ready to load into vLLM.

**Rune adaptation:** The original Doc-to-LoRA targets factual documents (knowledge internalization). Rune's trajectory format is structurally different — sequential (attempt → error → correction → attempt), causal, and procedural rather than declarative. The hypernetwork must be fine-tuned (or trained from scratch) on coding trajectory data rather than document data. This is the core research question and the kill-switch gate target.

**Implementation path:** Start with the pre-trained Doc-to-LoRA checkpoint from the Sakana AI release (if available publicly). Fine-tune on a corpus of coding trajectories generated by the agent loop. If the checkpoint is not available, train from scratch using the same architecture.

**Kill-switch:** If the hypernetwork-generated adapter does not produce ≥5% Pass@1 improvement on a 20-task HumanEval subset, the Doc-to-LoRA approach is invalidated. Fall back to QLoRA distillation (slower but proven).

---

### 4. Adapter Registry CRUD

**Storage model:** `AdapterRecord` (SQLModel, already defined) maps to a SQLite `adapter_records` table. The registry is a write-once store: no UPDATE operations. Adapters are never modified after creation — only archived (soft-delete via `is_archived=True`).

**Versioning:** `AdapterRecord.version` is an integer incremented by the caller. Version 1 = original. Version 2 = adapter trained from a session that used Version 1 as its base. The lineage is tracked via `session_id` (the session that produced the adapter), not a foreign key to the parent adapter.

**Composition:** The registry does not handle composition directly. Composition happens at inference time via vLLM's multi-LoRA API. The registry stores individual adapters. The agent state `["adapter_ids"]` is a list of adapter IDs to compose.

**Metadata stored:** `task_type` (indexed), `base_model_id`, `rank`, `file_path` (directory), `file_hash` (SHA-256), `source` (`"distillation"` | `"hypernetwork"` | `"manual"`), `pass_rate` (optional, set after evaluation), `fitness_score` (optional, set by evolution operator).

**Implementation note:** `AdapterRegistry` currently takes no constructor arguments. It needs a `db_path: str` or `engine` argument to know where to write. The simplest implementation: `SQLite` path from environment variable `ADAPTER_REGISTRY_DB_PATH`. Default: `~/.rune/registry.db`.

---

### 5. LangGraph Recursive Agent Loop

**Graph structure (already implemented in `graph.py`):**
```
START → generate → execute → reflect → [should_retry]
    → generate (retry)       (attempt_count < max_attempts and not tests_passed)
    → save_trajectory → END  (tests_passed or attempt_count >= max_attempts)
```

**should_retry() is already implemented** (not a stub). The routing logic is complete. The four node functions are stubs.

**generate_node:** Calls `VLLMClient.generate(prompt, model, adapter_id)` where `prompt` is constructed from `state["task_description"]` + `state["test_suite"]`. `adapter_id` is the most recent adapter from `state["adapter_ids"]` (or None if empty). Returns `{"generated_code": str}`.

**execute_node:** Runs `state["generated_code"]` + `state["test_suite"]` in an `asyncio` subprocess. Use `asyncio.create_subprocess_exec("python", "-c", combined_code)` with a 30-second timeout. Parse `exit_code == 0` as `tests_passed = True`. Returns `{"stdout": ..., "stderr": ..., "exit_code": ..., "tests_passed": bool}`.

**reflect_node:** Mechanical in Phase 1 (no LLM reflection call). Increments `attempt_count` by 1. Appends `{"attempt": state["attempt_count"], "code": state["generated_code"], "exit_code": state["exit_code"], "tests_passed": state["tests_passed"]}` to `state["trajectory"]`. Returns `{"attempt_count": N+1, "trajectory": updated_list}`.

**save_trajectory_node:** Calls `record_trajectory(state["session_id"], state["trajectory"], outcome)` where `outcome = "success" if state["tests_passed"] else "exhausted"`. Returns `{"outcome": outcome}`.

**State note:** `RuneState` does not currently have a `session_id` field. One must be added — it is needed by `save_trajectory_node` to call `record_trajectory`. This is a minor schema addition.

---

### 6. Hardware Validation

**What to check:**

| Check | Command / Method | Pass Condition |
|-------|-----------------|----------------|
| CUDA available | `torch.cuda.is_available()` | `True` |
| GPU count | `torch.cuda.device_count()` | `>= 2` |
| GPU 0 name | `torch.cuda.get_device_name(0)` | Contains `"4090"` |
| GPU 1 name | `torch.cuda.get_device_name(1)` | Contains `"4090"` |
| GPU 0 VRAM | `torch.cuda.get_device_properties(0).total_memory` | `>= 24 * 1024**3` bytes |
| GPU 1 VRAM | `torch.cuda.get_device_properties(1).total_memory` | `>= 24 * 1024**3` bytes |
| P2P GPU access | `torch.cuda.can_device_access_peer(0, 1)` | `True` (CXL/PCIe allows P2P) |
| CUDA version | `torch.version.cuda` | Matches vLLM wheel CUDA requirement |
| PyTorch version | `torch.__version__` | Matches vLLM wheel PyTorch requirement |
| vLLM importable | `import vllm` | No ImportError |
| Tensor op (GPU 0) | `torch.ones(1).cuda(0) + torch.ones(1).cuda(0)` | No error |
| Tensor op (GPU 1) | `torch.ones(1).cuda(1) + torch.ones(1).cuda(1)` | No error |
| vLLM PP=2 + LoRA | Start vLLM with `--pipeline-parallel-size 2 --enable-lora` and load a small model | No crash or corrupted output |
| AWQ quantization | Import `autoawq` and load AWQ-quantized weights | No ImportError or load error |

**CXL note:** The Rune hardware uses CXL interconnect between GPUs. P2P memory access via CXL is architecturally supported, but `can_device_access_peer()` may return `False` even with CXL if the CUDA driver does not enumerate the CXL connection as a P2P path. If it returns `False`, PP=2 still works (tensors are copied through CPU memory), but with higher latency. This is acceptable for PP=2.

**What hardware validation does NOT need to check:**
- NVLink (not present, not required for PP=2)
- Tensor parallelism (explicitly excluded — `LoraServerConfig` enforces TP=1)
- Host RAM size (Threadripper 7960X has large RAM pool; not a constraint for 7B model)

---

## Competitor Feature Analysis

| Feature | Standard coding agents (Copilot, Cursor) | Research agents (SWE-agent, Agentless) | Rune approach |
|---------|------------------------------------------|----------------------------------------|--------------|
| Context memory | In-context (token window only) | In-context + retrieval | LoRA weight space (parametric) |
| Inference backend | Cloud API (GPT-4o, Claude) | Cloud API | Local vLLM + LoRA |
| Session persistence | None or embedding store | None | Adapter library in SQLite registry |
| Hardware requirement | Any (cloud) | Any (cloud) | Dual RTX 4090, 48 GB VRAM |
| Adapter generation speed | N/A | N/A | <1s (hypernetwork) or 30 min (QLoRA) |
| Context window independence | No | No | Yes — memory is in weights, not tokens |

---

## Sources

**vLLM LoRA serving (HIGH confidence — official docs):**
- [vLLM LoRA Adapters Documentation](https://docs.vllm.ai/en/stable/features/lora/) — runtime loading API, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, multi-adapter serving
- [vLLM Bug #7253 — LoRA + PP fix](https://github.com/vllm-project/vllm/issues/7253) — confirmed fixed in PR #7292, August 2024
- [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) — PP=2 configuration

**QLoRA training (HIGH confidence — official docs + paper):**
- [QLoRA paper: arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — NF4 quantization, paged optimizers
- [Google Vertex AI LoRA/QLoRA recommendations](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-garden/lora-qlora) — rank, alpha, target modules guidance
- [LoRAFusion EUROSYS '26 speedup paper](https://arxiv.org/html/2510.00206v1) — recent kernel fusion work

**Doc-to-LoRA hypernetwork (HIGH confidence — primary paper):**
- [Doc-to-LoRA: arXiv:2602.15902](https://arxiv.org/abs/2602.15902) — Perceiver architecture, single forward pass, rank-8, <1s generation
- [Sakana AI Doc-to-LoRA announcement](https://pub.sakana.ai/doc-to-lora/) — batched vs iterative mode, chunk composition for long docs
- [SHINE: arXiv:2602.06358](https://arxiv.org/abs/2602.06358) — concurrent hypernetwork-to-LoRA work

**LangGraph agent patterns (HIGH confidence — official docs + existing code):**
- [LangChain Reflection Agents blog](https://blog.langchain.com/reflection-agents/) — generate-reflect loop pattern
- [LangGraph workflows-agents documentation](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — StateGraph, conditional edges
- Direct code inspection: `/Users/noahdolevelixir/Code/rune/services/rune-agent/src/rune_agent/graph.py` — graph topology already correct

**Hardware validation (MEDIUM confidence — PyTorch docs + community):**
- [PyTorch CUDA environment variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html) — official CUDA check methods
- [vLLM GPU installation docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) — CUDA version requirements
- [Dual RTX 4090 vLLM benchmark](https://www.databasemart.com/blog/vllm-gpu-benchmark-dual-rtx4090) — throughput data for 14-16B models

**Codebase inspection (HIGH confidence — direct read):**
- `/Users/noahdolevelixir/Code/rune/services/lora-server/config.py` — `LoraServerConfig`, PP=2 constraint
- `/Users/noahdolevelixir/Code/rune/services/lora-server/startup.sh` — confirmed vLLM flags
- `/Users/noahdolevelixir/Code/rune/libs/adapter-registry/src/adapter_registry/` — registry stubs, models
- `/Users/noahdolevelixir/Code/rune/libs/model-training/src/model_training/` — peft_utils, trajectory, config stubs
- `/Users/noahdolevelixir/Code/rune/services/rune-agent/src/rune_agent/` — graph, nodes, state

---

*Feature research for: Rune v5.0 — first working implementation*
*Researched: 2026-03-05*
*Previous FEATURES.md (scientific article content, v4.0 milestone) superseded by this document*
