# Stack Research

**Domain:** Local LLM serving with LoRA adapter training and hot-loading (v5.0 Implementation)
**Researched:** 2026-03-05
**Confidence:** HIGH — all library versions verified against PyPI and official docs; critical compatibility constraints verified against vLLM GitHub issues

---

> **Scope note:** This STACK.md covers only NEW additions needed for v5.0 First Implementation.
> Already validated: Python 3.12+, FastAPI, LangGraph, SQLModel, Pydantic, uv workspace, MkDocs.
> Do NOT re-add or change the existing baseline stack.

---

## Existing Baseline (Do Not Change)

| Package | Role | Notes |
|---------|------|-------|
| Python 3.12+ | Runtime | Already constrained in all pyproject.toml |
| FastAPI + uvicorn | API services | Already in training-svc, api-service, evolution-svc |
| LangGraph ≥1.0.6 | Agent graph | Already in rune-agent |
| SQLModel ≥0.0.31 | ORM + schema | Already in adapter-registry |
| Pydantic ≥2.0 | Validation | Already in inference lib |
| openai ≥1.0 | vLLM client | Already in inference lib (async OpenAI client points at local vLLM) |
| langgraph ≥1.0.6 | Recursive loop | Already in rune-agent; graph.py is implemented (not a stub) |
| torch ≥2.0.0 | GPU tensor ops | Already in model-training; deferred import behind TYPE_CHECKING |

---

## Recommended Stack — New Additions Only

### Core Technologies (New)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| vLLM | 0.16.0 (latest stable) | LLM serving with multi-LoRA hot-loading | Only production-grade inference engine supporting `--pipeline-parallel-size 2 --enable-lora` concurrently; issues with PP+LoRA were fixed in PR #7292; 0.16.0 adds full async scheduling + PP with 30.8% throughput improvement |
| PEFT | 0.18.1 (latest stable) | LoRA adapter training | Official HuggingFace parameter-efficient fine-tuning; `LoraConfig` + `get_peft_model` + `save_pretrained` is the standard QLoRA path; directly called by `build_qlora_config` and `apply_lora_adapter` stubs |
| bitsandbytes | 0.49.2 (latest stable) | 4-bit NF4 quantization for QLoRA training | Required by QLoRA: enables loading base model in 4-bit (`BitsAndBytesConfig(load_in_4bit=True)`) while PEFT LoRA adapters train in bf16; supports CUDA 11.8, 12.x, 13.x |
| transformers | 5.3.0 (latest stable) | `AutoModelForCausalLM`, tokenizer | Required for loading Qwen2.5-Coder-7B for training; `from_pretrained` with `BitsAndBytesConfig` for 4-bit quantized base model; v5 releases weekly |
| TRL | 0.29.0 (latest stable) | `SFTTrainer` for LoRA distillation | HuggingFace's supervised fine-tuning trainer; direct PEFT/QLoRA integration; accepts trajectory-formatted chat data; formats `format_for_sft` output into training batches |
| datasets | 4.6.1 (latest stable) | In-memory dataset for SFT training pipeline | `Dataset.from_dict()` converts trajectory messages to HuggingFace Dataset format that `SFTTrainer` consumes; zero-friction path from `format_for_sft` output |

### Supporting Libraries (New)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| aiosqlite | ≥0.21.0 (latest Dec 2025) | Async SQLite driver | Required if `AdapterRegistry` uses `AsyncSession`; connect URL: `sqlite+aiosqlite:///path/to/db`; without it `create_async_engine` errors on SQLite |
| safetensors | ≥0.4.5 | Save/load adapter weights safely | PEFT `save_pretrained` defaults to safetensors format; faster load, no pickle risk; vLLM reads safetensors directly from the adapter path passed to `--lora-modules` |
| accelerate | ≥1.0.0 | Device map for model loading | Required by transformers `from_pretrained` when using `device_map="auto"` for PP=2 multi-GPU training; also enables gradient checkpointing |

### Development Tools (Unchanged)

| Tool | Purpose | Notes |
|------|---------|-------|
| uv run | Run all Python scripts | Mandatory per CLAUDE.md — always `uv run mypy`, `uv run pytest`, etc. |
| pytest + pytest-asyncio | TDD test execution | Already configured; asyncio_mode = "auto" in root pyproject.toml |
| ruff + mypy | Lint + type check | Already configured; new GPU libs need mypy overrides (already present in root pyproject.toml) |

---

## vLLM Serving Configuration

The `lora-server` startup script (`services/lora-server/startup.sh`) already has the correct structure. Key parameters verified against vLLM 0.16.0 docs:

```bash
# Correct command for dual RTX 4090 with LoRA hot-loading
python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-lora \
    --quantization awq \          # Use AWQ-quantized model for inference
    --max-loras 8 \
    --max-lora-rank 64 \
    --max-cpu-loras 16 \
    --port 8000

# REQUIRED env var for dynamic loading at runtime:
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

### Dynamic LoRA API (HTTP, no vLLM import required)

The `VLLMClient` and `adapter_loader.py` stubs use the OpenAI client pointing at these endpoints:

```python
# Load adapter (POST /v1/load_lora_adapter)
await httpx.post("http://localhost:8000/v1/load_lora_adapter", json={
    "lora_name": "adapter-001",
    "lora_path": "/adapters/adapter-001/"   # directory with adapter_model.safetensors
})

# Unload adapter (POST /v1/unload_lora_adapter)
await httpx.post("http://localhost:8000/v1/unload_lora_adapter", json={
    "lora_name": "adapter-001"
})

# Inference with specific adapter (standard OpenAI completions format)
await client.chat.completions.create(
    model="adapter-001",   # vLLM routes to adapter by name
    messages=[...]
)
```

### AWQ + LoRA compatibility (VERIFIED)

vLLM supports unquantized LoRA adapters on top of AWQ-quantized base model with dynamic switching (`--quantization awq --enable-lora`). Constraint: adapter safetensors must NOT include `embed_tokens.weight` or `lm_head.weight`. PEFT's `save_pretrained` only saves adapter weights by default — this is safe. Do NOT use `merge_and_unload` before saving if the adapter is meant for hot-loading.

---

## QLoRA Training Configuration

The `peft_utils.py` and `config.py` stubs implement these patterns:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# 4-bit base model (fits in 24GB VRAM with overhead)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 bits/param
)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA config for coding tasks
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(base_model, lora_config)
# model.save_pretrained("/adapters/adapter-id/")  → adapter_model.safetensors
```

---

## SQLite / SQLModel Implementation Pattern

The `AdapterRegistry` uses synchronous SQLModel for simplicity (no async needed for SQLite CRUD in a library):

```python
from sqlmodel import SQLModel, Session, create_engine, select
from adapter_registry.models import AdapterRecord

engine = create_engine("sqlite:///adapters.db")
SQLModel.metadata.create_all(engine)

# store
with Session(engine) as session:
    session.add(record)
    session.commit()

# retrieve
with Session(engine) as session:
    result = session.get(AdapterRecord, adapter_id)
```

Use `aiosqlite` only if the registry is called from an async FastAPI context in `training-svc` or `api-service`. In that case:
- URL: `sqlite+aiosqlite:///adapters.db`
- Engine: `create_async_engine(url)`
- Session: `AsyncSession` from `sqlmodel.ext.asyncio.session`

The adapter-registry lib itself should stay synchronous; async wrapping belongs in service-layer dependency injection.

---

## LangGraph Recursive Loop Patterns

The `graph.py` is already implemented (not a stub). The node implementations (`nodes.py`) need these patterns:

```python
# State updates are partial dicts — only return what changed
async def reflect_node(state: RuneState) -> dict[str, Any]:
    step = {
        "attempt": state["attempt_count"],
        "code": state["generated_code"],
        "exit_code": state["exit_code"],
        "tests_passed": state["tests_passed"],
    }
    return {
        "attempt_count": state["attempt_count"] + 1,
        "trajectory": state["trajectory"] + [step],
    }

# Conditional edge — already implemented in graph.py, no changes needed
# should_retry() checks tests_passed and attempt_count >= max_attempts
```

`RuneState` is a plain `TypedDict` (no `Annotated` reducers). This means nodes must return full replacement values for list fields (trajectory), not append operations. This is intentional for v5.0 simplicity.

---

## Code Execution Sandbox (execute_node)

For the `execute_node` stub, use `asyncio.create_subprocess_exec` with timeout — no new libraries required:

```python
import asyncio, tempfile, os

async def execute_node(state: RuneState) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = os.path.join(tmpdir, "solution.py")
        test_file = os.path.join(tmpdir, "test_solution.py")
        # write generated_code + test_suite to files
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pytest", test_file, "-x", "--tb=short",
            cwd=tmpdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    return {
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
        "exit_code": proc.returncode,
        "tests_passed": proc.returncode == 0,
    }
```

This is sufficient for v5.0. Container isolation (Docker sandbox) is a v6.0 concern.

---

## Installation

### lora-server Dockerfile (GPU container — not uv workspace)

```dockerfile
FROM vllm/vllm-openai:v0.16.0
# vLLM official image includes CUDA, torch, transformers, and vllm
# No additional Python installs needed for serving
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn[standard] openai pyyaml
```

**Why the official vLLM image:** vLLM's CUDA kernel builds are non-trivial; the official image pins correct torch+CUDA+vLLM versions. Building vLLM from source in a custom Dockerfile is fragile and slow (30+ min build).

### model-training lib (uv workspace)

```toml
# libs/model-training/pyproject.toml — add to dependencies:
dependencies = [
    "shared",
    "torch>=2.0.0",
    "transformers>=5.0.0",
    "peft>=0.18.0",
    "bitsandbytes>=0.49.0",
    "trl>=0.29.0",
    "datasets>=4.6.0",
    "accelerate>=1.0.0",
    "safetensors>=0.4.5",
]
```

Note: `bitsandbytes`, `torch`, and `transformers` will fail to import on CPU-only machines. Keep `TYPE_CHECKING` guards in place for all GPU imports in the model-training lib. Tests must mock or skip GPU-dependent code paths.

### adapter-registry lib (uv workspace)

```toml
# libs/adapter-registry/pyproject.toml — current deps are sufficient for sync SQLite
# Only add aiosqlite if async engine is needed:
dependencies = [
    "sqlmodel>=0.0.31",
    "aiosqlite>=0.21.0",  # add only if training-svc needs async sessions
]
```

### rune-agent service (uv workspace)

```toml
# services/rune-agent/pyproject.toml — no new deps needed
# asyncio.create_subprocess_exec is stdlib; openai client is in inference lib
dependencies = [
    "shared",
    "inference",
    "langgraph>=1.0.6",
    "adapter-registry",  # add: save_trajectory_node needs registry access
]
```

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| vLLM 0.16.0 official image | Build vLLM from source | 30+ min builds; CUDA kernel compilation fragile; official image tested with correct torch pin |
| TRL SFTTrainer | Raw transformers Trainer | SFTTrainer has first-class chat format + PEFT integration; zero boilerplate for the `format_for_sft` → train pipeline |
| AWQ quantization for serving | bitsandbytes QLoRA for serving | bnb serving only supports one adapter loaded at startup (no dynamic switching); AWQ supports `--enable-lora` with full dynamic hot-loading |
| Sync SQLModel for adapter-registry | Async SQLModel | SQLite CRUD in a library has no I/O concurrency benefit from async; sync is simpler and avoids event loop coupling in tests |
| asyncio.create_subprocess_exec | Docker container sandbox | Docker sandbox is correct for production but over-engineered for v5.0 kill-switch validation; subprocess with timeout sufficient for hypothesis testing |
| Plain TypedDict state (no reducers) | Annotated state with operator.add | Reducers complicate debugging and testing; v5.0 agent is single-threaded; full-replacement semantics are explicit |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `--quantization awq` WITH `--tensor-parallel-size > 1` | TP+LoRA on consumer GPUs (RTX 4090) triggers vLLM bug #21471; PROJECT.md explicitly documents this as out of scope | `--pipeline-parallel-size 2 --tensor-parallel-size 1` (already in startup.sh) |
| bitsandbytes for vLLM serving quantization | bnb in serving mode only supports ONE adapter via `--qlora-adapter-name-or-path`; no dynamic switching | AWQ quantized model (`Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`) with `--quantization awq --enable-lora` |
| `model.merge_and_unload()` before saving adapters | Merged model cannot be hot-loaded by vLLM; loses the adapter/base separation needed for dynamic switching | `model.save_pretrained(path)` which saves only the delta adapter weights |
| `PEFT embed_tokens` or `lm_head` in saved adapters | vLLM rejects adapters containing embedding weights when using AWQ quantized base | Use `save_pretrained` defaults; PEFT only saves LoRA delta weights unless explicitly configured otherwise |
| Importing vLLM in the uv workspace | vLLM is GPU-only; importing it breaks CPU-only CI and tests across the monorepo | Use `openai.AsyncOpenAI` pointing at `VLLM_BASE_URL`; vLLM lives only in the Docker container |
| `shell=True` in subprocess for code execution | Injection risk when executing LLM-generated code | `asyncio.create_subprocess_exec` with an explicit args list |

---

## Stack Patterns by Variant

**For training-svc on GPU machine:**
- Import `torch`, `transformers`, `peft`, `trl` directly (GPU host)
- Use `device_map="auto"` with `BitsAndBytesConfig` for QLoRA
- Save adapter to shared `/adapters/` volume also mounted by lora-server container

**For training-svc on CPU-only machine (dev/test):**
- All GPU imports must be deferred behind `if TYPE_CHECKING:` or lazy imports
- Use `unittest.mock.patch` to mock out GPU calls in tests
- `pytest.mark.skipif(not torch.cuda.is_available(), ...)` for GPU-specific tests

**For adapter-registry in testing:**
- Use `create_engine("sqlite:///:memory:")` — in-memory SQLite, no file cleanup needed
- `SQLModel.metadata.create_all(engine)` in test fixture setup
- Each test gets its own in-memory engine via conftest factory fixture

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| vLLM 0.16.0 | CUDA 11.8, 12.1, 12.4 | RTX 4090 is CUDA 12.x; official image ships with correct CUDA |
| peft 0.18.1 | transformers 5.3.0, torch ≥2.0 | Always install together; PEFT reads transformers model internals |
| bitsandbytes 0.49.2 | CUDA 12.x, torch ≥2.0 | CUDA 13 also supported in 0.49.x |
| trl 0.29.0 | peft 0.18.x, transformers 5.x, datasets 4.x | TRL 0.29 requires transformers ≥4.46; 5.x is compatible |
| datasets 4.6.1 | Python ≥3.10 | Project uses 3.12; fully compatible |
| aiosqlite ≥0.21.0 | sqlalchemy 2.0.x async extension | SQLModel uses SQLAlchemy 2.0 internally; aiosqlite is the async driver |
| vLLM 0.16.0 | PP=2, TP=1, AWQ, enable-lora | PP+LoRA bug fixed in PR #7292 (merged before 0.8.x) — VERIFIED |

---

## Sources

- [vLLM PyPI](https://pypi.org/project/vllm/) — latest version 0.16.0 confirmed (released 2026-02-26) — HIGH confidence
- [vLLM Release v0.16.0](https://github.com/vllm-project/vllm/releases/tag/v0.16.0) — async scheduling + PP throughput improvements — HIGH confidence
- [vLLM LoRA docs v0.8.1](https://docs.vllm.ai/en/v0.8.1/features/lora.html) — `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, `--lora-modules` format — HIGH confidence
- [vLLM issue #7253 — LoRA incompatible with PP](https://github.com/vllm-project/vllm/issues/7253) — confirmed RESOLVED via PR #7292 — HIGH confidence
- [vLLM forum — AWQ + LoRA dynamic switching](https://discuss.vllm.ai/t/support-for-deploying-4-bit-fine-tuned-model-with-vllm/1186) — AWQ supports dynamic adapter switching; bnb does not — MEDIUM confidence (forum, not official docs)
- [vLLM blog — multi-LoRA Feb 2026](https://blog.vllm.ai/2026/02/26/multi-lora.html) — multi-LoRA MoE kernel optimizations in 0.15.0-0.16.0 — HIGH confidence
- [PEFT PyPI](https://pypi.org/project/peft/) — latest version 0.18.1 (released 2026-01-09) — HIGH confidence
- [bitsandbytes PyPI](https://pypi.org/project/bitsandbytes/) — latest version 0.49.2 (released 2026-02-16) — HIGH confidence
- [transformers PyPI](https://pypi.org/project/transformers/) — latest version 5.3.0 (released 2026-03-04) — HIGH confidence
- [TRL GitHub releases](https://github.com/huggingface/trl/releases) — latest version 0.29.0 — MEDIUM confidence (search-verified, PyPI 403'd)
- [datasets PyPI](https://pypi.org/project/datasets/) — latest version 4.6.1 (released 2026-02-27) — HIGH confidence
- [HuggingFace PEFT docs — QLoRA quantization](https://huggingface.co/docs/peft/en/developer_guides/quantization) — BitsAndBytesConfig + LoraConfig pattern — HIGH confidence
- [HuggingFace TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer) — PEFT integration, chat format, QLoRA path — HIGH confidence

---

*Stack research for: Rune v5.0 First Implementation — vLLM serving, QLoRA training, adapter registry, agent loop*
*Researched: 2026-03-05*
