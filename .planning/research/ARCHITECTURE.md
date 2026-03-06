# Architecture Research

**Domain:** Local-first coding agent — first working implementation (v5.0)
**Researched:** 2026-03-05
**Confidence:** HIGH — based on direct source inspection of all 11 components; vLLM API verified against official docs

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      External Interface                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  api-service  (FastAPI, :8000)                           │    │
│  │  /adapters  /sessions  /health  /ready                  │    │
│  └────────────────────────┬────────────────────────────────┘    │
├───────────────────────────┼─────────────────────────────────────┤
│                   Core Agent Loop                                │
│  ┌────────────────────────▼────────────────────────────────┐    │
│  │  rune-agent  (LangGraph StateGraph)                      │    │
│  │  generate → execute → reflect → save_trajectory          │    │
│  │  uses: inference lib, model-training lib                 │    │
│  └──────┬─────────────────────────────────┬────────────────┘    │
├─────────┼─────────────────────────────────┼─────────────────────┤
│         │  Inference                       │  Persistence        │
│  ┌──────▼───────────────────┐  ┌──────────▼──────────────────┐  │
│  │  lora-server             │  │  adapter-registry lib        │  │
│  │  vLLM (PP=2, TP=1)       │  │  SQLite via SQLModel         │  │
│  │  Qwen2.5-Coder-7B-AWQ    │  │  adapter_records table       │  │
│  │  port 8000 (vLLM)        │  │  (file_path, metadata, hash) │  │
│  │  port 8001 (health)      │  └──────────┬──────────────────┘  │
│  │  VLLM_ALLOW_RUNTIME_LORA │             │ consumed by          │
│  │  _UPDATING=True          │  ┌──────────▼──────────────────┐  │
│  └──────────────────────────┘  │  api-service storage         │  │
│                                │  training-svc storage        │  │
├───────────────────────────────┐└──────────────────────────────┘  │
│  Training Pipeline            │                                  │
│  ┌────────────────────────────▼────────────────────────────┐    │
│  │  training-svc  (FastAPI, :8002)                          │    │
│  │  POST /train/lora  POST /train/hypernetwork              │    │
│  │  GET /jobs/{job_id}                                      │    │
│  │  uses: model-training lib, adapter-registry lib          │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  evolution-svc  (FastAPI, :8003)                         │    │
│  │  POST /evaluate  POST /evolve  POST /promote  POST /prune│    │
│  │  uses: evaluation lib, adapter-registry lib              │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                      Shared Libraries                            │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ shared   │ │ inference │ │ model-   │ │ evaluation       │   │
│  │ rune_    │ │ adapter_  │ │ training │ │ metrics.py       │   │
│  │ models   │ │ loader    │ │ config   │ │ (HumanEval,      │   │
│  │ CodingS. │ │ completion│ │ peft_    │ │  Pass@k,         │   │
│  │ AdapterR.│ │ (openai   │ │ utils    │ │  fitness)        │   │
│  │ EvolMet. │ │  client)  │ │ trajectory│ │                 │   │
│  └──────────┘ └───────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | Responsibility | Owned Data |
|-----------|---------------|------------|
| `api-service` | External HTTP interface; CRUD proxy over adapter-registry and sessions | None — delegates to adapter-registry lib |
| `rune-agent` | Generate-execute-reflect loop; trajectory accumulation; parametric memory retrieval | RuneState (in-memory, per-run) |
| `lora-server` | vLLM GPU inference; LoRA hot-loading; OpenAI-compatible API | GPU memory (loaded adapters) |
| `training-svc` | Training job dispatch (LoRA distillation, hypernetwork forward pass) | training_svc.db (job records) |
| `evolution-svc` | Adapter fitness evaluation; crossover/mutation; promotion/pruning | evolution_svc.db (evolution records) |
| `adapter-registry` lib | Canonical SQLite store for all adapter metadata | adapter_records table in database.db |
| `inference` lib | vLLM OpenAI client wrapper; adapter load/unload calls | None (stateless) |
| `model-training` lib | QLoRA PEFT config; trajectory formatting for SFT | None (stateless) |
| `evaluation` lib | HumanEval benchmarking; Pass@k; fitness scoring | None (stateless) |
| `shared` lib | Cross-service data contracts (CodingSession, AdapterRef, EvolMetrics) | None |
| `events-py` lib | Event envelope format for future pub/sub integration | None |

---

## Integration Map: New vs. Modified

### 1. adapter-registry lib → api-service

**Current state:** `api-service` declares `adapter-registry` as a uv workspace dependency but its routers return 501 stubs. The `storage.py` creates `database.db` via `create_db_and_tables()` but does not yet import or instantiate `AdapterRegistry`.

**What must change (MODIFY — api-service routers):**
- `routers/adapters.py`: Inject `AdapterRegistry` via FastAPI `Depends`. Implement `list_adapters`, `get_adapter`, `create_adapter` using `registry.list_all()`, `registry.retrieve_by_id()`, `registry.store()`.
- `routers/adapters.py`: Add `POST /adapters` request body schema — currently missing (v4.0 tech debt). Define `AdapterCreateRequest` Pydantic model mirroring `AdapterRecord` fields.
- `dependencies.py`: Add `get_registry()` dependency that instantiates `AdapterRegistry` with the shared SQLite engine.
- `AdapterRegistry.__init__`: Must accept an SQLModel `Session` or engine — the lib currently has no `__init__`, meaning the implementation must add one that accepts a database engine parameter.

**What must change (MODIFY — adapter-registry lib):**
- `registry.py`: Implement all four stubs (`store`, `retrieve_by_id`, `query_by_task_type`, `list_all`) using `sqlmodel.Session`. The `AdapterRecord` SQLModel table is already defined and correct.
- `AdapterRegistry.__init__`: Add `engine` parameter so api-service can pass its shared SQLite engine. Without this, each instantiation creates a separate engine — causing the registry and the service to use different connection pools to the same file.

**Connection pattern:**
```
api-service/dependencies.py
    get_registry() → AdapterRegistry(engine=engine)
                         └── registry.store(record)
                                 └── with Session(engine) as s: s.add(record); s.commit()
```

**No new components needed.**

---

### 2. adapter-registry lib → lora-server

**Current state:** `lora-server` is a Dockerfile-only service (not in uv workspace). It contains its own `vllm_client.py` (`VLLMClient` class) and `health.py`. It has no direct import of `adapter-registry`.

**Integration pattern:** lora-server does NOT import adapter-registry. The coupling is indirect:

```
adapter-registry (SQLite) ─── file_path field ───► /adapters/ volume mount
lora-server startup.sh ─────────────────────────► vLLM loads adapters from /adapters/
POST /v1/load_lora_adapter ──────────────────────► {"lora_name": id, "lora_path": file_path}
```

The `inference` lib's `load_adapter(adapter_id, model_name)` is the bridge:
1. Caller first queries `adapter-registry` for the `AdapterRecord.file_path`.
2. Caller then calls `inference.load_adapter(adapter_id, model_name)` which POSTs to lora-server's `/v1/load_lora_adapter`.
3. The file at `file_path` must be accessible to vLLM — this requires the `/adapters/` volume to be mounted at the same path on the host where vLLM runs.

**What must change (MODIFY — inference lib):**
- `adapter_loader.py`: Implement `load_adapter` to POST `{"lora_name": adapter_id, "lora_path": file_path}` to `{VLLM_BASE_URL}/v1/load_lora_adapter`. The current stub has the wrong signature — it takes `adapter_id` and `model_name` but NOT `file_path`. The implementation must accept `file_path` or look it up.
- `adapter_loader.py`: Implement `unload_adapter` to POST `{"lora_name": adapter_id}` to `{VLLM_BASE_URL}/v1/unload_lora_adapter`.
- `completion.py`: Implement `generate_completion` and `generate_with_adapter` using `get_vllm_client()` and `client.chat.completions.create()`. When `adapter_id` is provided, pass it as `model` parameter in the format `"{base_model}:{adapter_id}"` per vLLM's LoRA serving convention.

**What must change (MODIFY — lora-server startup.sh / Dockerfile):**
- Add `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` environment variable. Without this, the `/v1/load_lora_adapter` endpoint returns 405 or 404.
- The docker-compose.yml already mounts `adapters_data:/adapters` — this volume path must match the `file_path` values stored in adapter-registry records.

**No new components needed.**

---

### 3. model-training lib → training-svc

**Current state:** `training-svc` declares `adapter-registry` and `shared` as workspace deps but NOT `model-training`. This is a v4.0 tech debt item explicitly noted in PROJECT.md.

**What must change (MODIFY — training-svc pyproject.toml):**
- Add `model-training` to `[project].dependencies`.
- Add `model-training = { workspace = true }` to `[tool.uv.sources]`.

**What must change (MODIFY — training-svc routers/training.py):**
- `POST /train/lora`: Implement by calling `model_training.config.get_training_config()`, then `model_training.trajectory.load_trajectory()`, then `model_training.peft_utils.build_qlora_config()`. The actual training loop (trl `SFTTrainer` or equivalent) is NEW code not currently in any lib — it belongs in `training-svc` as a background task, not in the `model-training` lib (which stays stateless).
- `GET /jobs/{job_id}`: Implement job status tracking. Training jobs are long-running (minutes). Use `asyncio.create_task` (same pattern as api-service's `_running_tasks`) to run training in background. Store job state in `training_svc.db`.
- `POST /train/hypernetwork`: Deferred to kill-switch gate phase. Implement as stub 501 for MVP.

**What must change (IMPLEMENT — model-training lib):**
- `trajectory.py`: Implement `record_trajectory` (write to JSON file or SQLite), `load_trajectory` (read back), `format_for_sft` (convert steps to `[{"role": "user", "content": prompt}, {"role": "assistant", "content": code}]` chat format).
- `peft_utils.py`: Implement `build_qlora_config` using `peft.LoraConfig`. Import guard already in place (`if TYPE_CHECKING`). Real import must be at runtime — GPU must be present. Use `BitsAndBytesConfig` for QLoRA quantization.
- `config.py`: Implement `get_training_config` and `validate_config` — pure dict operations, no GPU dependency.

**New component needed — trajectory store:**
Trajectories from `record_trajectory` must persist between agent runs and training jobs. Current architecture has no dedicated trajectory store. Options:
- **RECOMMENDED:** Use a JSON Lines file per session in the `/adapters/` volume (e.g., `/adapters/trajectories/{session_id}.jsonl`). Simple, no new infra, accessible to both rune-agent (writer) and training-svc (reader).
- Alternative: SQLite table in `training_svc.db`. More queryable but adds schema migration.

---

### 4. inference lib → lora-server's vLLM client

**Current state:** There are TWO parallel vLLM client implementations:
- `libs/inference/src/inference/adapter_loader.py` — standalone functions using `AsyncOpenAI`
- `services/lora-server/vllm_client.py` — `VLLMClient` class using `AsyncOpenAI`

Both wrap the same underlying `openai.AsyncOpenAI` client pointed at the vLLM server. This duplication is a problem.

**What the duplication means for implementation:**
- `VLLMClient` in lora-server is used ONLY by lora-server's own health sidecar (currently not used at all — the sidecar only exposes `/health` and `/ready`).
- `inference` lib functions are used by rune-agent and api-service (declared dependency).
- The `VLLMClient.generate()` and `inference.completion.generate_with_adapter()` implement the same operation.

**Resolution (MODIFY — consolidate, not duplicate):**
- Implement `inference` lib functions as the canonical vLLM client. These are used by rune-agent and api-service.
- `VLLMClient` in lora-server: implement its `load_adapter` and `generate` methods as thin wrappers calling the same HTTP endpoints. Do NOT import from the `inference` lib — lora-server is outside the uv workspace and cannot depend on workspace libs.
- The two implementations can have the same logic without being the same code. This is acceptable given the architectural boundary (lora-server is GPU-host-only, inference lib is CPU-importable).

**vLLM API details verified from official docs:**
```
POST /v1/load_lora_adapter
Body: {"lora_name": "<adapter_id>", "lora_path": "<file_path>"}
Env: VLLM_ALLOW_RUNTIME_LORA_UPDATING=True required

POST /v1/unload_lora_adapter
Body: {"lora_name": "<adapter_id>"}

For inference with loaded LoRA:
POST /v1/chat/completions
Body: {"model": "<base_model>:<adapter_id>", "messages": [...]}
```

---

### 5. rune-agent → all libs

**Current state:** rune-agent declares `shared` and `inference` as workspace deps. It does NOT declare `adapter-registry` or `model-training`.

**What must change (MODIFY — rune-agent pyproject.toml):**
- Add `adapter-registry` to implement `save_trajectory_node` (storing the completed trajectory + registering the resulting adapter).
- Add `model-training` to call `record_trajectory()` from `save_trajectory_node`.

**What must change (IMPLEMENT — rune-agent nodes.py):**

`generate_node`:
- Call `inference.completion.generate_with_adapter(prompt, adapter_id, model)` if `state["adapter_ids"]` is non-empty.
- Fall back to `inference.completion.generate_completion(prompt, model)` for base model.
- Adapter loading happens before the loop starts (not inside generate_node). The node assumes adapters are already loaded in vLLM.

`execute_node`:
- Run generated code in subprocess sandbox: `subprocess.run(["python", "-c", code], capture_output=True, timeout=10)`.
- Parse stdout, stderr, exit_code into state.
- `tests_passed` = `exit_code == 0`.
- Test suite execution: append the generated code + test suite as a combined script, execute it.

`reflect_node`:
- Increment `attempt_count`.
- Append current attempt dict to `trajectory`: `{"attempt": n, "code": generated_code, "stdout": stdout, "stderr": stderr, "passed": tests_passed}`.
- No LLM call in reflect for MVP. Reflection is purely mechanical state tracking.

`save_trajectory_node`:
- Call `model_training.trajectory.record_trajectory(session_id, state["trajectory"], outcome)`.
- Set `state["outcome"]` = `"success"` or `"exhausted"`.
- Do NOT register adapter here — adapter registration happens after training, not after session completion.

---

### 6. Doc-to-LoRA Hypernetwork Integration

**Current state:** `training-svc` has `POST /train/hypernetwork` endpoint accepting `HypernetworkTrainingRequest(task_type, trajectory_ids)`. This is a stub. No hypernetwork code exists anywhere in the codebase.

**Integration decision: NEW component within existing structure**

The hypernetwork is a new PyTorch module. It belongs in `model-training` lib, NOT as a new service. Rationale:
- It requires GPU (same constraint as other model-training code).
- It is called by `training-svc` (which already imports model-training).
- A new service would require its own GPU allocation — but both GPUs are already occupied by lora-server (PP=2).
- The hypernetwork forward pass is a pure compute call, not a long-lived server.

**New file to create:** `libs/model-training/src/model_training/hypernetwork.py`

Structure:
```python
class DocToLoraHypernetwork(nn.Module):
    """Cross-attention encoder mapping trajectory token activations to LoRA weight matrices."""

    def __init__(self, rank: int, base_model_dim: int): ...
    def forward(self, trajectory_embeddings: Tensor) -> dict[str, Tensor]: ...

def generate_adapter_from_trajectory(
    trajectory_ids: list[str],
    rank: int = 16,
) -> str:  # returns adapter_id after saving to disk and registry
    ...
```

The `training-svc /train/hypernetwork` endpoint calls `generate_adapter_from_trajectory()` as a background job, saves the result to the `/adapters/` volume, then calls `adapter-registry.store()`.

**Kill-switch gate context:** The hypernetwork is Phase 0 validation — the core hypothesis test. If `generate_adapter_from_trajectory` produces an adapter that achieves 5% Pass@1 improvement over the base model, the project continues. If not, the architecture is invalidated. This should be implemented immediately after the basic infrastructure works, not deferred.

---

## Recommended Project Structure (unchanged from v4.0)

```
libs/
├── adapter-registry/         # SQLite CRUD — implement registry.py
├── inference/                # vLLM client — implement adapter_loader.py, completion.py
├── model-training/           # QLoRA + trajectory — implement all 3 modules + ADD hypernetwork.py
├── evaluation/               # HumanEval + Pass@k — implement for kill-switch gate
├── events-py/                # Already implemented (create_event is real code)
└── shared/                   # Already implemented (rune_models.py is real code)

services/
├── api-service/              # Wire adapter-registry into routers — add request schemas
├── rune-agent/               # Implement 4 node functions — add adapter-registry dep
├── training-svc/             # Wire model-training lib — implement job dispatch
├── evolution-svc/            # Wire evaluation lib — implement fitness pipeline
└── lora-server/              # Add VLLM_ALLOW_RUNTIME_LORA_UPDATING — implement VLLMClient
```

---

## Build Order

Dependencies between components determine the implementation order. This is the constraint graph:

```
[1] adapter-registry lib (CRUD impl)
        ↓
[2] lora-server (VLLM_ALLOW_RUNTIME_LORA_UPDATING + startup validation)
        ↓
[3] inference lib (adapter_loader + completion impl)
        |
        ↓
[4] model-training lib (config + trajectory + peft_utils impl)
        |
        ↓ (parallel)
[5a] rune-agent (generate + execute + reflect + save nodes)
[5b] training-svc (wire model-training, implement /train/lora job dispatch)
        |           |
        ↓           ↓
[6a] api-service (wire adapter-registry into routers + add request schemas)
[6b] evaluation lib (Pass@k + HumanEval for kill-switch gate)
        |
        ↓
[7] model-training hypernetwork.py (DocToLoraHypernetwork)
        ↓
[8] Kill-switch gate validation (Doc-to-LoRA 5% Pass@1 gate)
        ↓
[9] evolution-svc (wire evaluation lib — deferred past kill-switch gate)
```

**Rationale for this order:**

- **Step 1 first:** Everything reads from or writes to the adapter-registry. Without CRUD working, no other integration can be tested end-to-end.
- **Step 2 before Step 3:** The inference lib calls lora-server endpoints. lora-server must be running with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` before adapter loading can be tested. Hardware validation (CUDA, dual GPU, PP=2 startup) happens here.
- **Step 3 before Step 4 and 5a:** rune-agent depends on inference lib for generation. model-training peft_utils ultimately needs a loaded model to train against — but the trajectory recording functions (config, trajectory) are CPU-only and can be done in parallel with Step 3.
- **Steps 5a/5b parallel:** rune-agent and training-svc have no dependency on each other for MVP. Both depend on model-training lib (Step 4).
- **Step 6a late:** api-service CRUD is useful for external visibility but NOT required for the agent loop to work. The agent loop writes directly to adapter-registry lib; api-service is just an HTTP proxy.
- **Step 7 (hypernetwork) after basics:** The hypernetwork cannot be validated until the full pipeline (registry + inference + trajectory recording) is functional. It needs real trajectories as training data.
- **Step 9 last:** evolution-svc depends on evaluation lib, which depends on a working inference pipeline and adapter-registry. Evolution is not part of the MVP loop.

---

## New Data Flows

### Flow 1: Agent Session (new, primary flow)

```
External trigger (task description + test suite)
    ↓
rune-agent.create_graph().invoke(initial_state)
    ↓ [generate_node]
inference.completion.generate_with_adapter(prompt, adapter_id) or generate_completion(prompt)
    → HTTP POST /v1/chat/completions to lora-server:8000
    ← generated_code string
    ↓ [execute_node]
subprocess.run(generated_code + test_suite)
    ← stdout, stderr, exit_code, tests_passed
    ↓ [reflect_node]
state["trajectory"].append({attempt data})
state["attempt_count"] += 1
    ↓ [should_retry edge]
    → if not passed and attempts < max: back to generate_node
    → if passed or exhausted: save_trajectory_node
    ↓ [save_trajectory_node]
model_training.trajectory.record_trajectory(session_id, trajectory, outcome)
    → writes to /adapters/trajectories/{session_id}.jsonl
```

### Flow 2: LoRA Distillation (new, triggered manually or post-session)

```
POST /train/lora → training-svc
    {"task_type": "bug-fix", "rank": 64, "epochs": 3}
    ↓
training-svc loads trajectory via model_training.trajectory.load_trajectory(trajectory_id)
    ↓
model_training.trajectory.format_for_sft(trajectory) → chat format messages
    ↓
model_training.peft_utils.build_qlora_config(rank=64, alpha=128, target_modules=[...])
    ↓
trl.SFTTrainer(model, peft_config, dataset).train()  [NEW code in training-svc, GPU required]
    ↓
adapter_registry.store(AdapterRecord(file_path=saved_path, ...))
    ↓
inference.adapter_loader.load_adapter(adapter_id, model_name)
    → POST /v1/load_lora_adapter to lora-server:8000
```

### Flow 3: Adapter Retrieval for Agent (new, pre-session)

```
Session starts with task_type="bug-fix"
    ↓
adapter_registry.query_by_task_type("bug-fix") → [AdapterRecord, ...]
    ↓
Select highest fitness_score adapter
    ↓
inference.adapter_loader.load_adapter(adapter_id, file_path)
    → POST /v1/load_lora_adapter {"lora_name": adapter_id, "lora_path": file_path}
    ↓
rune-agent.invoke({"adapter_ids": [adapter_id], ...})
```

### Flow 4: Kill-Switch Gate (new, one-time validation)

```
Generate baseline: generate_completion(humaneval_prompt) × 20 tasks
    ↓
evaluation.metrics.calculate_pass_at_k(n_samples, n_correct, k=1) = baseline_score
    ↓
Generate Doc-to-LoRA adapter from coding trajectory
    → training-svc POST /train/hypernetwork
    ↓
Load adapter: inference.adapter_loader.load_adapter(adapter_id, file_path)
    ↓
Generate with adapter: generate_with_adapter(humaneval_prompt, adapter_id) × 20 tasks
    ↓
calculate_pass_at_k() = adapter_score
    ↓
(adapter_score - baseline_score) >= 0.05? → CONTINUE : STOP
```

---

## Modified Existing Data Flows

### api-service adapter CRUD (MODIFIED)

Before: `GET /adapters` → 501 stub
After: `GET /adapters` → `adapter_registry.list_all()` → list of `AdapterRecord` as JSON

The SQLite engine in api-service (`database.db`) and adapter-registry must be the SAME file. Both `api-service/storage.py` and `AdapterRegistry.__init__` must use the same `DATABASE_URL` env var pointing to the same SQLite file. Risk: if training-svc or rune-agent write to adapter-registry using a DIFFERENT engine/path, records will be in separate files.

**Fix:** `AdapterRegistry` must accept an engine parameter, not create its own. Each service that uses it passes its own engine. All services must point `DATABASE_URL` to the SAME SQLite file path if running locally — or use separate files per service with api-service as the read/write proxy for external access.

**RECOMMENDED:** Use a single shared SQLite file at a fixed path (e.g., `/data/rune.db`) mounted as a volume, with `DATABASE_URL` set uniformly across api-service, training-svc, and rune-agent. This avoids the multi-engine problem.

---

## Component Boundaries

| Boundary | Communication | Pattern | Notes |
|----------|---------------|---------|-------|
| rune-agent ↔ lora-server | HTTP (OpenAI API) via inference lib | Request/response | Async, inference lib wraps AsyncOpenAI |
| rune-agent ↔ adapter-registry | In-process (lib import) | Direct function call | Same SQLite file; must share engine |
| rune-agent ↔ model-training | In-process (lib import) | Direct function call | CPU-safe for trajectory.py; GPU required for peft_utils |
| api-service ↔ adapter-registry | In-process (lib import) | FastAPI Depends injection | Same engine via get_registry() |
| training-svc ↔ model-training | In-process (lib import) | Background asyncio task | GPU required for SFTTrainer |
| training-svc ↔ adapter-registry | In-process (lib import) | Direct function call after training | Writes new adapter record |
| training-svc ↔ lora-server | HTTP via inference.adapter_loader | POST /v1/load_lora_adapter | Hot-loads trained adapter immediately |
| evolution-svc ↔ evaluation | In-process (lib import) | Direct function call | Deferred — not MVP |
| evolution-svc ↔ adapter-registry | In-process (lib import) | Direct function call | Reads/updates fitness_score |
| lora-server (vLLM) ↔ /adapters/ volume | Filesystem | File read at load time | Paths must match adapter_registry.file_path |

---

## Architectural Patterns

### Pattern 1: Library-as-Business-Logic, Service-as-HTTP-Shell

**What:** Core logic (CRUD, trajectory formatting, PEFT config) lives in importable Python libs. Services are thin FastAPI wrappers that add HTTP routing, dependency injection, and background task management. No business logic in service files.

**When to use:** Always — all 6 libs are designed this way. Services in this codebase are already structured as HTTP shells.

**Trade-offs:** Libs must be CPU-importable even when they use GPU functionality at runtime. All GPU imports must be deferred behind `if TYPE_CHECKING` guards or inside function bodies (as `model-training/peft_utils.py` already does with `if TYPE_CHECKING: pass`). This is already the established pattern.

**Example:**
```python
# training-svc/routers/training.py — correct pattern
@router.post("/train/lora")
async def train_lora(request: LoraTrainingRequest) -> JSONResponse:
    config = get_training_config(request.task_type, request.rank)  # lib call
    task = asyncio.create_task(_run_training_job(config))          # service responsibility
    _running_tasks.add(task)
    return JSONResponse({"job_id": task_id, "status": "running"})
```

---

### Pattern 2: Single SQLite File with Shared Engine

**What:** All components that access adapter-registry records share a single SQLite file and pass the engine as a parameter to `AdapterRegistry(engine=engine)`. The engine is created once at service startup via `create_engine(DATABASE_URL)`.

**When to use:** For this MVP milestone, where all services run on the same host. SQLite has per-file WAL mode that handles concurrent reads and serialized writes correctly across multiple connections.

**Trade-offs:** SQLite WAL mode (enable with `PRAGMA journal_mode=WAL`) supports concurrent reads + single writer. For MVP throughput (one agent session at a time), this is sufficient. For concurrent sessions, WAL becomes the bottleneck — at which point migrate to PostgreSQL.

**Example:**
```python
# adapter-registry/registry.py
class AdapterRegistry:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def store(self, record: AdapterRecord) -> None:
        with Session(self._engine) as session:
            session.add(record)
            session.commit()
```

---

### Pattern 3: Async Background Tasks for Long-Running Jobs

**What:** Training jobs (minutes) run as `asyncio.Task` objects tracked in a service-level set. Status is polled via `GET /jobs/{job_id}`. Job state is written to the service's local SQLite.

**When to use:** For training-svc only. The pattern is already implemented in api-service for general background tasks (`_running_tasks: Set[asyncio.Task]`).

**Trade-offs:** Background tasks die if the service restarts. For MVP this is acceptable — training jobs are initiated manually and monitored. For production, use a proper task queue (Celery, Huey, or ARQ).

---

### Pattern 4: Volume-Mounted Adapter Files

**What:** LoRA adapter `.safetensors` files are written to a host-mounted volume (`/adapters/`). The path stored in `AdapterRecord.file_path` must be the path as seen by vLLM inside the lora-server container, not the path on the training machine.

**When to use:** Always — this is the only way lora-server can read adapters trained by training-svc without a network file transfer.

**Trade-offs:** Both training-svc and lora-server must mount the same volume at the same path. The docker-compose.yml already defines `adapters_data:/adapters` for lora-server. Training-svc must also mount `adapters_data:/adapters` (currently missing from docker-compose.yml — this is a modification needed).

---

## Anti-Patterns

### Anti-Pattern 1: Multiple AdapterRegistry Instances with Different Engines

**What people do:** Each service creates `AdapterRegistry()` with no arguments, which internally calls `create_engine(DATABASE_URL)`. If `DATABASE_URL` is not set consistently, each service writes to a different SQLite file.

**Why it's wrong:** Adapters stored by training-svc are not visible to api-service. Adapters loaded by rune-agent are not visible to evolution-svc. The registry becomes siloed per service.

**Do this instead:** Pass engine explicitly. Set `DATABASE_URL` to the same path in all service environments. Use a shared volume mount so the SQLite file is physically the same on disk.

---

### Anti-Pattern 2: GPU Imports at Module Level in Lib Code

**What people do:** `from peft import LoraConfig` at the top of `model_training/peft_utils.py`. Importing `peft` imports `torch` which requires CUDA — the test suite on CI (CPU-only) fails to import the module.

**Why it's wrong:** Breaks `uv run pytest` on CPU machines. Breaks api-service startup (which imports model-training lib) on a non-GPU host. Breaks the CI gate.

**Do this instead:** Keep the existing `if TYPE_CHECKING:` pattern. Move real GPU imports inside function bodies:
```python
def build_qlora_config(rank, alpha, target_modules, dropout=0.1):
    from peft import LoraConfig  # deferred — GPU required at call time, not import time
    from transformers import BitsAndBytesConfig
    ...
```

---

### Anti-Pattern 3: Calling load_adapter Without Checking vLLM Max-LoRAs

**What people do:** Load adapters dynamically without tracking how many are currently loaded. vLLM raises an error when `max_loras=8` is exceeded.

**Why it's wrong:** The agent loop fails mid-session when the adapter limit is hit. The error is not handled gracefully.

**Do this instead:** Before loading a new adapter, call `list_loaded_adapters()` and compare against `LoraServerConfig.max_loras`. If at capacity, unload the lowest-fitness adapter first (query adapter-registry for fitness scores).

---

### Anti-Pattern 4: Trajectory Stored Only in RuneState (in memory)

**What people do:** Rely on the `state["trajectory"]` list in LangGraph state as the only record of the session trajectory. `save_trajectory_node` writes it to memory and returns `outcome`.

**Why it's wrong:** If the agent process crashes after generation but before `save_trajectory_node` completes, the trajectory is lost. Trajectories are the training data — losing them wastes GPU compute.

**Do this instead:** Write trajectory steps to the filesystem incrementally in `reflect_node`, not just at the end in `save_trajectory_node`. Append each attempt to `/adapters/trajectories/{session_id}.jsonl` in `reflect_node`. `save_trajectory_node` only needs to finalize the file (write outcome).

---

## Phase-Specific Architectural Warnings

| Build Step | Concern | Required Action |
|------------|---------|-----------------|
| Step 1: adapter-registry impl | `AdapterRecord` is `SQLModel, table=True` — `SQLModel.metadata.create_all(engine)` must be called BEFORE first write | Ensure api-service and training-svc call `create_db_and_tables()` at startup (already done in their lifespan handlers) |
| Step 2: lora-server validation | PP=2 with `enable_lora=True` is the specific safe config. Any change to TP or PP invalidates vLLM bug #21471 safety | Do not change `startup.sh` parallelism config without re-validating on dual 4090 hardware |
| Step 2: lora-server validation | AWQ quantization requires pre-quantized model. `Qwen/Qwen2.5-Coder-7B-Instruct` AWQ variant must be downloaded before startup | Use `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` specifically, or switch quantization to `bitsandbytes` (4-bit) if AWQ variant unavailable |
| Step 3: inference impl | `generate_with_adapter` must pass adapter_id to vLLM as `model="{base}:{adapter_id}"` — NOT as a separate field | Verify the LoRA-per-request format in vLLM docs before implementing |
| Step 4: model-training peft_utils | `BitsAndBytesConfig` requires `bitsandbytes` package, which has CUDA build requirements | Add `bitsandbytes>=0.41.0` to model-training deps; verify it builds on CUDA 12.x |
| Step 5a: rune-agent execute_node | Subprocess sandbox has no filesystem isolation. Agent-generated code can write arbitrary files | Use a timeout (10s), restrict to `/tmp` working directory, consider `seccomp` or Docker exec for production |
| Step 6b: evaluation lib | HumanEval benchmark requires `human_eval` package or `datasets` library and the `openai/HumanEval` dataset | `datasets` is listed in root pyproject.toml dev deps; confirm HumanEval loading works via `datasets.load_dataset("openai/HumanEval")` |
| Step 7: hypernetwork | Training the hypernetwork requires reference adapters as targets. These must be fine-tuned LoRA adapters (not hypernetwork-generated) — a cold-start problem | For MVP: generate a small set of manually fine-tuned LoRA adapters (via /train/lora) as training targets for the hypernetwork meta-learning phase |
| All steps | `training_svc.db` and `database.db` and `evolution_svc.db` are SEPARATE files currently | For the shared adapter-registry use case, all services must point to the same DB file or use api-service as the single write proxy |

---

## Integration Points Summary

### New Files to Create

| File | Type | Purpose |
|------|------|---------|
| `libs/model-training/src/model_training/hypernetwork.py` | NEW | Doc-to-LoRA hypernetwork module |
| `/adapters/trajectories/` | NEW directory | Trajectory JSONL store (on volume) |

### Modified Existing Files

| File | Modification |
|------|-------------|
| `libs/adapter-registry/src/adapter_registry/registry.py` | Implement 4 CRUD stubs; add `__init__(engine)` |
| `libs/inference/src/inference/adapter_loader.py` | Implement `load_adapter`, `unload_adapter`, `list_loaded_adapters` |
| `libs/inference/src/inference/completion.py` | Implement `generate_completion`, `generate_with_adapter`, `batch_generate` |
| `libs/model-training/src/model_training/config.py` | Implement `get_training_config`, `validate_config` |
| `libs/model-training/src/model_training/trajectory.py` | Implement `record_trajectory`, `load_trajectory`, `format_for_sft` |
| `libs/model-training/src/model_training/peft_utils.py` | Implement `build_qlora_config`, `apply_lora_adapter`, `merge_adapter` |
| `services/api-service/src/api_service/routers/adapters.py` | Implement 3 stubs; add `AdapterCreateRequest` schema |
| `services/api-service/src/api_service/dependencies.py` | Add `get_registry()` dependency |
| `services/rune-agent/src/rune_agent/nodes.py` | Implement 4 node stubs |
| `services/rune-agent/pyproject.toml` | Add `adapter-registry`, `model-training` deps |
| `services/training-svc/src/training_svc/routers/training.py` | Implement `/train/lora` and `/jobs/{job_id}`; add background task runner |
| `services/training-svc/pyproject.toml` | Add `model-training` dep |
| `services/lora-server/Dockerfile` | Add `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` env var |
| `services/lora-server/vllm_client.py` | Implement `load_adapter` and `generate` |
| `infra/docker-compose.yml` | Add `adapters_data` volume mount to training-svc; fix port conflicts (api:8000 vs lora:8000) |

### Docker-Compose Port Conflict (Critical)

The current `docker-compose.yml` exposes BOTH `api` service and `lora-server` on `${API_PORT:-8000}:8000`. This is a conflict. They must use different host ports:
- api-service: `8080:8000` (or any non-8000 host port)
- lora-server: `8000:8000` (vLLM must be on 8000 as that's what inference lib defaults to)

---

## Sources

- Direct source inspection: all 11 component source files (HIGH confidence)
- Direct source inspection: `pyproject.toml` dependency declarations for all workspace members (HIGH confidence)
- [vLLM LoRA Adapters — official docs v0.8.1](https://docs.vllm.ai/en/v0.8.1/features/lora.html) — `/v1/load_lora_adapter` endpoint format, `VLLM_ALLOW_RUNTIME_LORA_UPDATING` requirement (HIGH confidence)
- [vLLM stable LoRA docs](https://docs.vllm.ai/en/stable/features/lora/) — dynamic loading API confirmed (HIGH confidence)
- [vLLM issue #21471](https://github.com/vllm-project/vllm/issues/) — TP+LoRA bug on consumer GPUs, PP=2 workaround (referenced in existing codebase config.py)
- `services/lora-server/config.py` — `LoraServerConfig` enforcement of PP=2, TP=1 constraint (HIGH confidence, direct inspection)
- `services/lora-server/startup.sh` — actual vLLM launch flags: `--pipeline-parallel-size 2 --tensor-parallel-size 1 --enable-lora --quantization awq` (HIGH confidence, direct inspection)

---

*Architecture research for: Rune v5.0 — first working implementation, integration of all components*
*Researched: 2026-03-05*
