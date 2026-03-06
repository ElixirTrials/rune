# Pitfalls Research — v5.0 First Implementation

**Domain:** Local coding agent — vLLM LoRA serving, QLoRA training, SQLite registry, LangGraph agent loop, Doc-to-LoRA hypernetwork
**Researched:** 2026-03-05
**Confidence:** HIGH — vLLM pitfalls verified against official docs, GitHub issues, and vLLM forum. SQLite concurrency verified against SQLAlchemy docs. LangGraph pitfalls verified against official error docs. QLoRA pitfalls verified against PEFT/bitsandbytes official sources.

---

## Critical Pitfalls

### Pitfall 1: PP=2 + LoRA — The Architecture Is Fundamentally Incompatible

**What goes wrong:**
The chosen hardware configuration (PP=2, TP=1) to avoid vLLM bug #21471 hits a separate incompatibility: **LoRA with pipeline parallelism (PP > 1) is not working in current vLLM versions**. GitHub issue #7253 documents that when `--pipeline-parallel-size 2` and `--enable-lora` are combined, the LoRA manager crashes because `PPMissingLayer` is not a real language model head. The previous fix (PR #7292) resolved this for one architecture, but regressions appear in newer versions. Issue #30269 shows TP=1/PP=2 breaking again in v0.12.0.

**Why it happens:**
vLLM's LoRA manager was designed around single-GPU or TP-only layouts. Pipeline-parallel models replace some layers with `PPMissingLayer` placeholders. The LoRA manager fails when it attempts to apply adapters to placeholder layers. The fix from 2024 addressed Llama but may not generalize to Qwen2.5.

**How to avoid:**
Before spending any implementation time on the hot-loading integration, verify empirically on the target hardware that `vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --pipeline-parallel-size 2 --enable-lora` starts without error and generates correct output. Run this as the first task of the hardware validation phase. If PP=2 + LoRA is still broken with Qwen2.5, fall back to single-GPU (PP=1, TP=1) even though it only uses 24GB instead of 48GB — correctness is the kill-switch criterion, not throughput.

**Warning signs:**
- Server startup fails with `AttributeError` mentioning `lm_head` or `logits_processor`
- Server starts but generates scrambled/multilingual output (the symptom of bug #21471 variant)
- `health.py` /ready endpoint returns True but completions are incoherent

**Phase to address:**
Phase 0 — Hardware Validation. This must be the first gate before any implementation begins. Do not stub the training pipeline against an assumed working PP=2+LoRA configuration.

---

### Pitfall 2: VLLM_ALLOW_RUNTIME_LORA_UPDATING Is Not Set — Dynamic Loading Silently Fails

**What goes wrong:**
`VLLMClient.load_adapter()` calls `POST /v1/load_lora_adapter` but receives a 404 or "method not allowed" response. The endpoint exists in vLLM but is disabled by default. Dynamic LoRA loading requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` as an environment variable at server startup. Without it, the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints are not registered.

**Why it happens:**
The vLLM docs for the dynamic loading API are on a separate page from the main LoRA documentation. The static loading path (specifying `--lora-modules` at startup) works without this flag, which leads developers to miss it when switching to dynamic loading.

**How to avoid:**
In the `lora-server` Dockerfile and startup command, explicitly set:
```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```
Add a startup smoke test in `health.py` that calls `GET /v1/models` and verifies the server responds before declaring ready, and add a separate test that posts a known-good adapter to `/v1/load_lora_adapter` to confirm dynamic loading is operational.

**Warning signs:**
- 404 response from `/v1/load_lora_adapter`
- vLLM version constraint: versions before 0.6 may not have the endpoint at all even with the flag

**Phase to address:**
Phase 1 — lora-server implementation. Add `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` to the Dockerfile ENV block on day one of lora-server implementation, not as an afterthought.

---

### Pitfall 3: Adapter Format Incompatibility — PEFT Saves Embeddings, vLLM Rejects Them

**What goes wrong:**
`PeftModel.save_pretrained()` saves the LoRA adapter in a directory containing `adapter_config.json` and `adapter_model.safetensors`. However, when the fine-tuning includes vocabulary embeddings or output head weights (which PEFT sometimes saves by default), vLLM's adapter loader rejects the file with an error about unsupported weight keys, because **vLLM does not support LoRA adapters that include `embed_tokens.weight` for quantized models**.

Additionally, the adapter directory must contain exactly the expected PEFT format files — not a merged checkpoint, not a raw tensor file. vLLM's LoRA loader expects:
```
adapter_dir/
  adapter_config.json    ← must specify base_model_name_or_path
  adapter_model.safetensors
```

**Why it happens:**
Some PEFT configurations with `target_modules="all-linear"` inadvertently target embedding layers. The saved file then contains keys like `base_model.model.model.embed_tokens.weight` that pass PEFT validation but fail vLLM's strict loader.

**How to avoid:**
1. Explicitly specify `target_modules` in `build_qlora_config()` — use `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` for Qwen2.5-Coder-7B, not `"all-linear"`.
2. After training, validate the saved adapter keys: load with `safetensors.torch.load_file()` and assert no key contains `embed_tokens`.
3. Verify `adapter_config.json` contains `"base_model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct"` — the lora path passed to `/v1/load_lora_adapter` must match the model name vLLM was started with.

**Warning signs:**
- vLLM logs `ValueError: Unsupported LoRA weight key: base_model.model.model.embed_tokens.weight`
- `adapter_config.json` lists `target_modules: "all-linear"` after training

**Phase to address:**
Phase 1 — model-training lib implementation. Define `target_modules` explicitly in `build_qlora_config()` before any training runs.

---

### Pitfall 4: AWQ Quantization on Base Model + QLoRA Training Are Separate Paths

**What goes wrong:**
The `LoraServerConfig` specifies `quantization: "awq"`, which means vLLM serves an AWQ-quantized base model. But the training pipeline uses QLoRA (bitsandbytes NF4 4-bit quantization during training). These are **two different quantization systems**. The trained LoRA adapter (from QLoRA training with NF4) must be served on top of the AWQ-quantized base model — this combination may not work.

Additionally, for bitsandbytes QLoRA specifically, **hot-swapping adapters at runtime is not supported in vLLM** — the adapter is fixed at startup via `qlora_adapter_name_or_path`. AWQ + dynamic LoRA loading is a separate (and supported) path.

**Why it happens:**
QLoRA training and AWQ inference quantization are both described as "4-bit quantization" in popular writing, creating the impression they are compatible or interchangeable. They are not. QLoRA applies NF4 during training gradient computation; AWQ applies activation-aware weight quantization for inference throughput.

**How to avoid:**
Clarify the two-stage quantization strategy explicitly:
- **Training**: Use QLoRA (bitsandbytes NF4) on the unquantized base model to produce a standard PEFT LoRA adapter. Save the adapter in PEFT format (adapter_config.json + safetensors).
- **Serving**: Serve an AWQ-quantized version of the base model in vLLM. Load the PEFT-format adapter via dynamic loading. Do not use `qlora_adapter_name_or_path` in vLLM startup — use dynamic `/v1/load_lora_adapter` instead.

The adapter produced by QLoRA training is a standard LoRA adapter file, compatible with AWQ-based vLLM serving. The "QLoRA" label refers only to how the adapter was trained, not to its file format.

**Warning signs:**
- Training code uses `bnb_config` with `load_in_4bit=True` but saving includes quantization artifacts in the adapter weights file
- vLLM startup uses `--qlora-adapter-path` flag (which disables dynamic loading)

**Phase to address:**
Phase 0/Phase 1 boundary — establish the training-to-serving adapter pipeline contract before implementing either side.

---

### Pitfall 5: SQLite Concurrency Under FastAPI Async — Deadlocks Without WAL Mode

**What goes wrong:**
The current `training_svc/storage.py` and `evolution_svc/storage.py` create SQLite engines with `check_same_thread=False` but do not enable WAL (Write-Ahead Logging) mode. Under concurrent FastAPI requests (multiple HTTP requests arriving while a write is in progress), SQLite's default journal mode causes requests to receive `sqlite3.OperationalError: database is locked` errors. This is silent during unit tests (single-threaded) but appears immediately under any load.

Additionally, the current pattern creates a module-level engine singleton. When running tests, the same engine may be reused across test functions, causing SQLite state leak between tests.

**Why it happens:**
SQLite's default "DELETE" journal mode only allows one writer at a time and blocks readers during writes. FastAPI's async handlers may issue concurrent requests that each attempt write operations. The `check_same_thread=False` setting prevents thread-safety errors but does not solve concurrency.

**How to avoid:**
Enable WAL mode via a SQLAlchemy connection event immediately after engine creation:

```python
from sqlalchemy import event

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

@event.listens_for(engine, "connect")
def set_wal_mode(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()
```

Apply this pattern to `training_svc/storage.py`, `evolution_svc/storage.py`, and the `adapter-registry` lib's database engine.

**Warning signs:**
- `sqlite3.OperationalError: database is locked` in logs under any concurrent load
- Tests pass when run individually but fail when run in parallel (`pytest -n auto`)
- Database file grows without bound (WAL not checkpointing)

**Phase to address:**
Phase 1 — adapter-registry CRUD implementation. Fix storage.py files across all services before implementing any router logic that writes to the database.

---

### Pitfall 6: LangGraph Recursion Limit Swallows the Real Error

**What goes wrong:**
LangGraph's default recursion limit is 25. The Rune agent loop is: `generate → execute → reflect → [retry or save]`. Each full cycle counts as multiple steps in LangGraph's recursion counter. With `max_attempts=3`, a typical run uses approximately 12 steps. But if a node raises an exception that is caught and silently returns an empty dict (or the conditional edge logic has a bug), the graph may cycle indefinitely until hitting the recursion limit, then raise `GraphRecursionError` with a stack trace that obscures the original bug.

**Why it happens:**
The `should_retry` conditional edge returns `"generate"` if tests didn't pass and attempts remain. If `reflect_node` fails to increment `attempt_count` due to a bug (e.g., returns `{}` instead of `{"attempt_count": state["attempt_count"] + 1}`), the counter never advances and the loop runs until the recursion limit.

**How to avoid:**
1. Set an explicit recursion limit slightly above the expected maximum: `graph.invoke(state, config={"recursion_limit": max_attempts * 4 + 5})`. This surfaces true recursion bugs as errors at a sensible threshold.
2. Assert in `should_retry` that `attempt_count` is strictly monotonically increasing. Add a guard:
   ```python
   if state["attempt_count"] > state["max_attempts"] + 1:
       raise RuntimeError(f"attempt_count={state['attempt_count']} exceeded max_attempts={state['max_attempts']} by more than 1 — reflect_node is not incrementing")
   ```
3. Ensure `reflect_node` always returns a dict with `attempt_count` key, even on exception paths.

**Warning signs:**
- `GraphRecursionError: Recursion limit of 25 reached` in logs
- Agent log shows `attempt_count` staying at 0 across multiple loop iterations
- Tests that mock `reflect_node` pass but integration runs loop forever

**Phase to address:**
Phase 1 — rune-agent recursive loop implementation. Set recursion_limit explicitly in the graph invocation and add the monotonicity assertion before any integration testing.

---

### Pitfall 7: RuneState Missing Required Keys on Initial Invocation

**What goes wrong:**
`RuneState` is a `TypedDict` with all keys required (no `Optional` annotations on most fields). When the graph is invoked with an initial state that is missing keys like `generated_code`, `stdout`, `stderr`, `exit_code`, `tests_passed`, `trajectory`, and `outcome`, the first node that reads those keys raises a `KeyError`. LangGraph does not pre-initialize TypedDict keys to `None`.

**Why it happens:**
The caller constructs an initial state dict with only the input fields (`task_description`, `task_type`, `test_suite`, `adapter_ids`, `attempt_count`, `max_attempts`) and assumes unset fields will be `None` or empty. TypedDict has no default values — missing keys are genuinely absent.

**How to avoid:**
Define a factory function or Pydantic model for the initial state that populates all required fields with defaults:

```python
def make_initial_state(task_description: str, task_type: str, test_suite: str,
                       adapter_ids: list[str] | None = None,
                       max_attempts: int = 3) -> RuneState:
    return RuneState(
        task_description=task_description,
        task_type=task_type,
        test_suite=test_suite,
        adapter_ids=adapter_ids or [],
        attempt_count=0,
        max_attempts=max_attempts,
        generated_code="",
        stdout="",
        stderr="",
        exit_code=-1,
        tests_passed=False,
        trajectory=[],
        outcome=None,
    )
```

**Warning signs:**
- `KeyError: 'generated_code'` on first graph invocation
- Mypy error: `TypedDict "RuneState" has missing keys` at call sites that construct partial dicts

**Phase to address:**
Phase 1 — rune-agent implementation, first day. Write `make_initial_state()` before implementing any node logic.

---

### Pitfall 8: Root conftest.py Fixtures Invisible to Service-Level Tests

**What goes wrong:**
The root `conftest.py` at `/rune/conftest.py` defines factory fixtures (`make_adapter_record`, `make_training_job`, etc.). However, pytest's rootdir detection may not include this file when running tests within a specific service directory. Running `cd services/training-svc && pytest` or `uv run pytest services/training-svc/` may not discover the root conftest because pytest uses the `testpaths` or `rootdir` from the nearest `pyproject.toml`, which may be the service's own `pyproject.toml` — not the repo root.

**Why it happens:**
This is the known tech debt from v4.0: "Root conftest.py rootdir isolation requires factory fixture duplication." The monorepo has per-component `pyproject.toml` files that act as pytest rootdir anchors, preventing pytest from walking up to the repo root to find the root conftest.

**How to avoid:**
Two options, in order of preference:
1. **Duplicate minimally**: Copy only the fixtures actually needed by each service into its local `tests/conftest.py`. Accept the duplication — it is visible and maintainable.
2. **Use pytest plugin imports**: In each service's `conftest.py`, explicitly import from the root conftest: `from conftest import make_adapter_record` — but this requires the root to be on the Python path, which needs a `conftest.py` path hack or `sys.path` manipulation.

Avoid option 2 in general (import hacks in test setup are fragile). Accept the duplication for v5.0.

**Warning signs:**
- `fixture 'make_adapter_record' not found` when running service-level tests
- Tests pass in CI (which runs from repo root) but fail locally (which runs from service directory)

**Phase to address:**
Phase 1 — implementation of any component that needs registry fixtures. Before writing tests that use root fixtures, check that the fixture is available from the service's test runner.

---

### Pitfall 9: POST Stubs Missing Request Body Schemas — 422 on First Real Request

**What goes wrong:**
The `api-service/routers/adapters.py` stub for `POST /adapters` has signature `async def create_adapter() -> JSONResponse` with no request body parameter. When implementation begins and a real request body is expected, adding a Pydantic model parameter changes the route's contract. Any existing test that calls `client.post("/adapters", json=body)` will need to be updated. More critically, if the schema is added mid-implementation, it may be added inconsistently (different field names in the router vs. the registry model vs. the API test).

**Why it happens:**
v4.0 stubs intentionally return 501 and have no body schema because the schema design wasn't finalized. Now that `AdapterRecord` fields are defined in the models, the schema can and must be defined.

**How to avoid:**
Before implementing any router logic, define Pydantic request/response schemas in a `schemas.py` file for each service. Map `AdapterRecord` fields to a `CreateAdapterRequest` schema explicitly:

```python
class CreateAdapterRequest(BaseModel):
    id: str
    version: int
    task_type: str
    base_model_id: str
    rank: int
    file_path: str
    file_hash: str
    file_size_bytes: int
    source: str
    session_id: str
```

Use this schema as the request body type in the router. Ensure the fields match `AdapterRecord` exactly (no snake_case/camelCase mismatch).

**Warning signs:**
- `422 Unprocessable Entity` from FastAPI when calling POST endpoints
- Mypy error: `Argument 1 to "create_adapter" has incompatible type`
- Router function signature uses `request: dict` (untyped) instead of a Pydantic model

**Phase to address:**
Phase 1 — api-service implementation. Write schemas.py for api-service before implementing any router that handles a POST request body.

---

### Pitfall 10: QLoRA bf16 Compute Dtype — Using float16 Causes Silent NaN

**What goes wrong:**
In `build_qlora_config()`, setting `bnb_4bit_compute_dtype=torch.float16` instead of `torch.bfloat16` causes numerical instability during training. NaN losses appear after a few dozen steps. The training does not crash — the loss becomes `nan`, gradients become `nan`, and the resulting adapter weights are all-zero or all-NaN. The adapter saves successfully (no error) but generates garbage output at inference.

**Why it happens:**
NF4 dequantization produces values in a range that overflows float16's exponent during forward passes for larger models. bfloat16 has a wider exponent range (matching float32) and avoids this overflow. The QLoRA paper explicitly recommends bfloat16. Many tutorials and template configs still use float16.

**How to avoid:**
In `model_training/peft_utils.py`, the `build_qlora_config` implementation must use:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NOT "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,  # NOT torch.float16
    bnb_4bit_use_double_quant=True,
)
```

Add a validation check: after the first training step, assert `not torch.isnan(loss)`.

**Warning signs:**
- Loss becomes `nan` after step 10-50
- Training completes but adapter file is 0 bytes or contains NaN weights
- `adapter_model.safetensors` saves but vLLM outputs empty strings

**Phase to address:**
Phase 1 — model-training lib implementation. Set bfloat16 as a hard-coded constant in `build_qlora_config()`, not a parameter default that could be overridden.

---

### Pitfall 11: vLLM max_loras Memory Pre-Allocation OOM at Startup

**What goes wrong:**
With `max_loras: 8` in `LoraServerConfig`, vLLM pre-allocates GPU memory for 8 simultaneous LoRA adapters at server startup. Each adapter slot at rank=64 consumes approximately 3GB of VRAM. On a single RTX 4090 (24GB), pre-allocating 8 adapter slots uses 24GB before any model weights are loaded, causing OOM at startup — not at inference time.

**Why it happens:**
vLLM pre-allocates LoRA memory buffers based on `max_loras` and `max_lora_rank` at initialization. The total LoRA memory budget is `max_loras × model_layers × rank × 2 × dtype_bytes`. For Qwen2.5-Coder-7B at rank=64, this is substantial. The interaction between `max_loras`, `max_lora_rank`, and `gpu_memory_utilization` is not obvious from the documentation.

**How to avoid:**
Reduce `max_loras` to 2-4 for initial testing. Set `max_lora_rank` to the actual maximum rank used (e.g., 64, not 256). Tune `gpu_memory_utilization` down to 0.8 to leave headroom:

```
--max-loras 2
--max-lora-rank 64
--gpu-memory-utilization 0.80
```

Validate the server starts successfully before increasing these values. After PP=2 compatibility is confirmed, profile actual VRAM usage with `nvidia-smi` during startup.

**Warning signs:**
- `CUDA out of memory` at vLLM startup (before any requests)
- `nvidia-smi` shows 100% VRAM before model weights finish loading

**Phase to address:**
Phase 0 — Hardware Validation. Validate the specific `max_loras`/`max_lora_rank` combination that fits in 24GB per GPU before lora-server implementation.

---

### Pitfall 12: Doc-to-LoRA Meta-Training — Weeks on Multiple GPUs, Not Hours

**What goes wrong:**
Doc-to-LoRA hypernetwork meta-training is described in the Sakana AI paper as taking "days to weeks on multiple GPUs." The Rune kill-switch validation gate assumes Doc-to-LoRA can be demonstrated within the v5.0 milestone. If the hypernetwork training is treated as a single training run that completes in hours, the timeline will be missed by an order of magnitude.

**Why it happens:**
The paper uses a 309M parameter Perceiver hypernetwork that must be meta-trained across thousands of (document, adapter) pairs before it can generalize to new documents in a single forward pass. The inference (generating an adapter from a new document) is fast — but reaching that capability requires expensive meta-training upfront.

**How to avoid:**
Two strategies:
1. **Use pre-trained weights**: If Sakana AI releases pre-trained Doc-to-LoRA hypernetwork weights (for Gemma-2-2B), adapt those weights to Qwen2.5-Coder-7B via transfer learning — significantly cheaper than meta-training from scratch.
2. **Scoped meta-training**: For the kill-switch gate, limit the meta-training dataset to coding-specific documents (e.g., 100-200 Python library documentation pages) and run for a fixed compute budget (e.g., 24 hours on dual RTX 4090). Measure whether the adapter quality improves over baseline at that budget.

Do not plan for a full meta-training run in v5.0. The kill-switch gate should be: "does a partially-meta-trained hypernetwork produce adapters that improve Pass@1 vs. baseline?" — not "does a fully-converged hypernetwork achieve the paper's results?"

**Warning signs:**
- Training loss has not decreased after 24 hours (hypernetwork is not learning)
- Memory OOM during hypernetwork forward pass (Perceiver cross-attention is expensive)
- No pre-trained weights available for Qwen2.5-Coder architecture

**Phase to address:**
Phase 1 — Core Hypothesis Validation. Set explicit compute budgets (GPU-hours) before meta-training begins. Define the kill-switch pass/fail criterion in terms of relative improvement, not absolute performance.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Using `check_same_thread=False` without WAL mode | Silences thread error | Database locked errors under any concurrent load | Never — add WAL mode on the same line |
| Hardcoding `sqlite:///training_svc.db` (relative path) | Simple | Breaks when service is run from different working directories (e.g., Docker) | Never — use absolute path or env var with `Path(__file__).parent` |
| Returning `{}` from a failed node instead of raising | Prevents graph crash | Silent data loss; loop counter doesn't advance; triggers recursion limit eventually | Never during implementation; acceptable only in production fallback with explicit logging |
| Skipping adapter key validation after training | Saves 5 minutes | Adapter silently fails when loaded by vLLM; hard to diagnose | Never — run validation immediately after `save_pretrained()` |
| Using global `_graph = None` singleton without thread safety | Avoids repeated compilation | Race condition if graph is initialized from two concurrent requests | Acceptable for v5.0 (single-threaded invocation), must fix before production |
| Using `quantization: "awq"` in config before verifying AWQ+LoRA compatibility | Follows original design | OOM or incorrect output if AWQ base model rejects LoRA adapters | Use non-quantized base model for first integration test, add AWQ afterward |
| Skipping `VLLM_ALLOW_RUNTIME_LORA_UPDATING` validation test | Saves one test | Dynamic loading silently fails; entire hot-loading feature broken | Never — add explicit smoke test on startup |

---

## Integration Gotchas

Common mistakes when connecting the components.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| lora-server → adapter-registry | Calling `POST /v1/load_lora_adapter` with an adapter ID before verifying the file exists at the given path | Registry must store `file_path`; lora-server must verify path is accessible before calling load endpoint |
| training-svc → model-training lib | training-svc router returns 501; model-training lib raises NotImplementedError; wiring them together still requires an async job queue design | Implement a simple synchronous blocking call for v5.0; async job queue is a later optimization |
| rune-agent → lora-server | Agent calls `VLLMClient.generate()` which is a sync function in an async LangGraph node | The `generate_node` must `await` all IO calls; verify `VLLMClient.generate()` is `async def`, not `def` |
| rune-agent → adapter-registry | Agent uses adapter_ids from state to load adapters, but adapters may not be loaded in vLLM yet | Add an explicit `ensure_loaded(adapter_id)` step in `generate_node` before calling generate |
| api-service → adapter-registry lib | api-service creates its own database session; adapter-registry lib creates its own; both may write to different SQLite files | Ensure both use the same DATABASE_URL env var; verify with `os.path.abspath` logging at startup |
| conftest.py root → service tests | Service tests discover root conftest fixtures only when pytest is run from repo root | Either run all tests from repo root, or duplicate fixtures in service-level conftest files |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| vLLM PP=2 single-request latency | Each request must synchronize across 2 GPU stages; latency is higher than PP=1 for small batches | Accept higher single-request latency; PP=2 only benefits throughput at high concurrency | At low concurrency — every request |
| SQLite WAL checkpoint blocking writers | Long-running readers prevent checkpoint; WAL file grows; eventual write stall | Set `PRAGMA wal_autocheckpoint=100` and run checkpoint after training jobs | After ~1000 writes without reads completing |
| LangGraph state growing with each trajectory | `trajectory` list in RuneState accumulates all attempt dicts; large trajectories slow state serialization | Limit trajectory list length or store trajectory externally, reference by ID | After ~50 attempts per session |
| Loading Qwen2.5-Coder-7B without quantization | Full-precision 7B model requires ~14GB per GPU; leaves little VRAM for LoRA and KV cache | Always use AWQ or bitsandbytes loading in vLLM; verify VRAM headroom with `nvidia-smi` | Immediately on a 24GB GPU without quantization |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **PP=2 + LoRA server**: Server starts without error AND generates correct output for a coding prompt. Both conditions must pass — the server can start but produce garbage output (bug #21471 symptom).
- [ ] **Dynamic LoRA loading**: `/v1/load_lora_adapter` returns 200 AND a subsequent completion request using that adapter name returns a non-empty coherent response. Loading succeeding does not mean inference with the adapter works.
- [ ] **SQLite storage**: CRUD operations pass unit tests AND pass under 5 concurrent requests (run `pytest -n 5` or use `httpx.AsyncClient` with multiple concurrent calls). Single-threaded tests always pass.
- [ ] **Training pipeline**: Training runs without NaN loss AND the saved adapter passes format validation (correct keys, no embed_tokens). Training completing does not mean the adapter is valid.
- [ ] **Agent loop**: Loop terminates on success (tests pass) AND terminates when `max_attempts` is reached AND terminates when `generate_node` raises an exception. All three exit paths must be tested.
- [ ] **Fixture discovery**: `uv run pytest libs/adapter-registry` passes AND `uv run pytest services/training-svc` passes, both from the repo root. Fixtures must be discoverable in both cases.
- [ ] **Model-training ↔ training-svc wiring**: `POST /train/lora` creates a TrainingJob in SQLite AND triggers `build_qlora_config` + training execution. The router returning 200 does not mean the lib is wired.
- [ ] **evaluation ↔ rune-agent wiring**: Pass@1 metric can be computed at the end of an agent session. The evaluation lib is currently not wired to any consumer service.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| PP=2 + LoRA broken | MEDIUM | Fall back to PP=1, TP=1 (single GPU); lose 24GB capacity but gain correctness; revisit PP=2 in a later milestone |
| VLLM_ALLOW_RUNTIME_LORA_UPDATING not set | LOW | Add env var to Dockerfile, rebuild container, rerun smoke test |
| Adapter format rejected by vLLM | LOW-MEDIUM | Re-save adapter with explicit `target_modules` (no embeddings); validate keys; reload |
| SQLite database locked errors | LOW | Enable WAL mode via SQLAlchemy event; restart service |
| LangGraph recursion limit hit | LOW | Set explicit recursion_limit in graph.invoke(); add monotonicity assertion in reflect_node |
| NaN loss in QLoRA training | MEDIUM | Switch `bnb_4bit_compute_dtype` to bfloat16; restart training from scratch |
| Meta-training too expensive | HIGH | Acquire pre-trained hypernetwork weights from Sakana AI release; adapt to Qwen2.5 architecture; if unavailable, scope down kill-switch gate to trajectory-based LoRA distillation only (skip hypernetwork for v5.0) |
| Root conftest fixtures invisible | LOW | Duplicate required fixtures in service conftest.py; 30 minutes per service |
| POST endpoint missing schema | LOW | Add Pydantic schema to router; update failing tests; straightforward but requires touching multiple files |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| PP=2 + LoRA incompatibility | Phase 0 — Hardware Validation | `vllm serve --pipeline-parallel-size 2 --enable-lora` starts and generates correct Qwen2.5 output |
| VLLM_ALLOW_RUNTIME_LORA_UPDATING missing | Phase 1 — lora-server | Smoke test: POST to `/v1/load_lora_adapter` returns 200 |
| Adapter format incompatibility | Phase 1 — model-training lib | Saved adapter passes key validation (no embed_tokens) AND loads in vLLM |
| AWQ + QLoRA quantization confusion | Phase 0/1 boundary | Explicitly document training-to-serving adapter pipeline; validate with a test adapter |
| SQLite concurrency without WAL | Phase 1 — adapter-registry | WAL mode set via SQLAlchemy event; concurrent test passes |
| LangGraph recursion limit | Phase 1 — rune-agent | Explicit recursion_limit set; all three exit paths tested |
| RuneState missing keys | Phase 1 — rune-agent, day 1 | `make_initial_state()` function exists; graph.invoke() with initial state succeeds |
| Root conftest fixture visibility | Phase 1 — any component using registry fixtures | `uv run pytest libs/adapter-registry` passes from repo root |
| POST stubs missing schemas | Phase 1 — api-service | `schemas.py` created before any router implementation; no untyped `request: dict` |
| QLoRA float16 NaN loss | Phase 1 — model-training lib | bfloat16 hardcoded in `build_qlora_config`; first training step loss is finite |
| max_loras VRAM OOM at startup | Phase 0 — Hardware Validation | vLLM starts with `max_loras=2`, `max_lora_rank=64` on 24GB GPU |
| Doc-to-LoRA meta-training cost | Phase 1 — Kill-switch gate | Compute budget defined in GPU-hours before training begins; relative improvement metric |

---

## Sources

- [vLLM LoRA Adapters Documentation (latest)](https://docs.vllm.ai/en/latest/features/lora/) — env var requirements, adapter format, max_lora_rank
- [vLLM Bug #21471 — TP+LoRA corrupted output on consumer GPUs](https://github.com/vllm-project/vllm/issues/21471) — closed August 2025; TP=2 with LoRA produces garbage output; PP=2 is the chosen workaround
- [vLLM Bug #7253 — LoRA incompatible with distributed pipeline parallelism](https://github.com/vllm-project/vllm/issues/7253) — CRITICAL: PP+LoRA may still have issues in newer versions
- [vLLM Bug #29049 — OOM when loading LoRA adapters v0.11.1](https://github.com/vllm-project/vllm/issues/29049) — max_loras × max_lora_rank × memory pre-allocation
- [vLLM Bug #30269 — Multi-node TP=1 PP=2 broken in v0.12.0](https://github.com/vllm-project/vllm/issues/30269) — PP regressions between versions
- [vLLM LoRA Resolver Plugins](https://docs.vllm.ai/en/stable/design/lora_resolver_plugins/) — dynamic loading plugin architecture
- [vLLM Forum: LoRA Adapter enabling not working](https://github.com/vllm-project/vllm/discussions/16604) — VLLM_ALLOW_RUNTIME_LORA_UPDATING requirement
- [vLLM Forum: Deploying 4-bit model with LoRA](https://discuss.vllm.ai/t/support-for-deploying-4-bit-fine-tuned-model-with-lora-on-vllm/1186) — AWQ + dynamic LoRA compatibility details
- [HuggingFace 4-bit QLoRA blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — NF4 vs fp4, bfloat16 vs float16 compute dtype
- [PEFT Quantization Documentation](https://huggingface.co/docs/peft/main/en/developer_guides/quantization) — QLoRA training, adapter saving, bitsandbytes config
- [LangGraph GRAPH_RECURSION_LIMIT error docs](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT) — default limit of 25, customization
- [LangGraph INVALID_GRAPH_NODE_RETURN_VALUE docs](https://docs.langchain.com/oss/python/langgraph/errors/INVALID_GRAPH_NODE_RETURN_VALUE) — node must return dict with defined state keys
- [LangGraph infinite loop bug #6731](https://github.com/langchain-ai/langgraph/issues/6731) — agent looping until recursion limit hit
- [SQLite WAL mode documentation](https://sqlite.org/wal.html) — WAL persistence, reader/writer concurrency
- [SQLAlchemy connection event WAL setup — Simon Willison](https://til.simonwillison.net/sqlite/enabling-wal-mode) — PRAGMA journal_mode=WAL via event.listens_for
- [SQLite concurrency pitfalls — JTTI](https://www.jtti.cc/supports/3154.html) — check_same_thread misconception, pool reuse errors
- [FastAPI SQLModel async documentation — Arunansh](https://arunanshub.hashnode.dev/async-database-operations-with-sqlmodel) — async SQLite pattern with aiosqlite
- [Doc-to-LoRA official paper page — Sakana AI](https://pub.sakana.ai/doc-to-lora/) — meta-training cost "days to weeks on multiple GPUs", 309M parameter Perceiver
- [QLoRA multi-GPU issue #96 — artidoro/qlora](https://github.com/artidoro/qlora/issues/96) — multi-GPU training with QLoRA on single process
- [Fine-Tuning Infrastructure guide — Introl Blog](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025) — QLoRA at scale, gradient accumulation, VRAM management
- Direct codebase inspection: `services/lora-server/config.py`, `services/rune-agent/src/rune_agent/graph.py`, `libs/adapter-registry/src/adapter_registry/registry.py`, `services/training-svc/src/training_svc/storage.py`, `conftest.py` — confirms known v4.0 tech debt items

---

*Pitfalls research for: Rune v5.0 — First Implementation milestone*
*Researched: 2026-03-05*
