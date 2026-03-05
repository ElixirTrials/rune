# Phase 21: QLoRA Training Pipeline - Research

**Researched:** 2026-03-05
**Domain:** QLoRA fine-tuning (PEFT + TRL + BitsAndBytes), FastAPI background jobs, PEFT safetensors adapter format
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Training data flow:**
- Single-trajectory training per run — one session's successful trajectory produces one adapter
- `format_for_sft()` output (list[dict]) wrapped into a HuggingFace `Dataset` via `Dataset.from_list()`
- SFT format: system/user/assistant chat turns — TRL's SFTTrainer handles tokenization
- No batching across trajectories for v5.0 — each training run uses one trajectory

**QLoRA configuration:**
- bfloat16 compute dtype hardcoded (float16 causes NaN at 7B)
- NF4 quantization via BitsAndBytesConfig (4-bit quantization type)
- Default rank=64, alpha=128 (2x rank), target_modules=["q_proj", "v_proj"]
- Hyperparameters: lr=2e-4, epochs=3, warmup_ratio=0.03, cosine scheduler
- `build_qlora_config()` returns peft LoraConfig, `get_training_config()` returns training hyperparams dict

**Training service design:**
- FastAPI BackgroundTasks for async execution — no external task queue
- In-memory `dict[str, JobStatus]` for job tracking (dataclass with job_id, status, adapter_id, error)
- POST /train/lora: accepts trajectory session_id + optional overrides → returns job_id immediately
- GET /jobs/{job_id}: returns current status (queued/running/completed/failed + adapter_id when done)
- training-svc adds model-training as workspace dependency (TRAIN-07)
- Loses state on restart — acceptable for single-user local MVP

**Adapter output & registration:**
- Trained adapters saved to `~/.rune/adapters/{adapter_id}/` as safetensors
- adapter_id generated as UUID at training start
- Auto-register in AdapterRegistry.store() after successful training with full metadata
- Save adapter_config.json alongside safetensors for PEFT compatibility / vLLM loading

**GPU import strategy (INFRA-04/05):**
- model-training pyproject.toml adds: peft, bitsandbytes, transformers, trl, datasets
- All GPU imports (peft, bitsandbytes, transformers, trl) deferred inside function bodies
- TYPE_CHECKING guards for type annotations only
- CPU-only CI must pass: import model_training succeeds without GPU libs installed

### Claude's Discretion
- Exact SFTTrainer configuration parameters beyond those specified
- Error handling and cleanup on training failure
- Temp directory strategy during training
- Whether to add a `train_qlora()` orchestrator function or keep steps separate

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-03 | User can create a QLoRA config with NF4 quantization and bfloat16 compute dtype via build_qlora_config() | LoraConfig + BitsAndBytesConfig API verified via PEFT docs; NF4+bfloat16 combination is standard |
| TRAIN-04 | User can apply a LoRA adapter to a base model via apply_lora_adapter() | peft.get_peft_model() is the standard call; stub already has correct signature |
| TRAIN-05 | QLoRA training pipeline runs end-to-end: trajectory → SFT format → PEFT train → save safetensors → store in registry | SFTTrainer + peft_config pattern verified; Dataset.from_list + "messages" field works; trainer.save_model() saves adapter only when PEFT is active |
| TRAIN-06 | training-svc exposes POST /train/lora endpoint with async background job tracking | FastAPI BackgroundTasks + module-level dict[str, JobStatus] pattern verified; schemas already exist |
| TRAIN-07 | training-svc pyproject.toml declares model-training as workspace dependency | uv workspace dependency pattern established in existing libs; format: `model-training = { workspace = true }` |
| INFRA-04 | model-training pyproject.toml adds GPU dependencies (peft, bitsandbytes, transformers, trl, datasets) | All packages exist on PyPI; deferred import pattern verified via Phase 19-20 precedent |
| INFRA-05 | All GPU imports deferred inside function bodies (not top-level) for CPU-only CI compatibility | Pattern fully established in Phase 19-20; TYPE_CHECKING guard for annotations only |
</phase_requirements>

---

## Summary

Phase 21 implements the end-to-end QLoRA training pipeline. The key libraries are PEFT (LoraConfig, get_peft_model), BitsAndBytes (BitsAndBytesConfig for 4-bit NF4 quantization), TRL (SFTTrainer + SFTConfig), transformers (AutoModelForCausalLM, AutoTokenizer), and datasets (Dataset.from_list). The work splits across two implementation sites: the model-training library (peft_utils.py, config.py) and the training-svc service (router, job tracking, orchestrator).

The biggest technical pitfall is the vLLM adapter format constraint: if `modules_to_save` is used in LoraConfig (to save embed_tokens or lm_head), vLLM cannot load the resulting adapter. The solution is to NOT use `modules_to_save` — target only q_proj and v_proj, which produces a clean safetensors file with only lora_A and lora_B weight keys. The confirmed locked decision (target_modules=["q_proj", "v_proj"]) already avoids this pitfall.

The SFTTrainer integration is straightforward: pass `peft_config=LoraConfig(...)` and the `messages` field in dataset dicts (matching format_for_sft() output). When peft_config is provided, SFTTrainer automatically handles quantized model loading, adapter attachment, and saves only adapter weights (not the full model) on trainer.save_model().

**Primary recommendation:** Implement `train_qlora()` as a single orchestrator function in a new `model_training/trainer.py` module. This keeps the endpoint handler thin, enables isolated unit testing of the orchestrator, and matches the Phase 19-20 pattern of service-level thin routers delegating to library functions.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| peft | >=0.18.0 | LoraConfig, get_peft_model, PeftModel | HuggingFace PEFT is the standard LoRA library; SFTTrainer integrates natively |
| bitsandbytes | >=0.45.0 | BitsAndBytesConfig for NF4 4-bit quantization | Only library providing QLoRA 4-bit NF4 quantization for PyTorch on NVIDIA GPUs |
| transformers | >=4.47.0 | AutoModelForCausalLM, AutoTokenizer, prepare_model_for_kbit_training | Backbone model loading with quantization_config support |
| trl | >=0.16.0 | SFTTrainer, SFTConfig | HuggingFace supervised fine-tuning trainer; native PEFT peft_config parameter |
| datasets | >=3.0.0 | Dataset.from_list() | HuggingFace datasets; Dataset.from_list wraps list[dict] directly |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch | >=2.0.0 (already in pyproject) | Tensor operations, bfloat16 dtype | Already declared; needed for BitsAndBytesConfig dtype constant |
| hashlib | stdlib | SHA-256 hash of adapter file | For AdapterRecord.file_hash field |
| uuid | stdlib | Generate adapter_id | Standard for unique job/adapter IDs |
| dataclasses | stdlib | JobStatus dataclass for in-memory tracking | Clean typed status object without SQLModel overhead |
| tempfile | stdlib | Temp training output directory | SFTTrainer requires output_dir; temp dir is cleaned up after save_pretrained |
| shutil | stdlib | Copy final adapter files from temp dir | Move trained weights to ~/.rune/adapters/{adapter_id}/ |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FastAPI BackgroundTasks | ARQ/Celery | Celery adds Redis dependency — wrong for single-user local MVP; BackgroundTasks sufficient |
| In-memory dict | Redis/DB job store | Survives restarts but adds infrastructure; acceptable loss per CONTEXT.md decision |
| SFTTrainer | HuggingFace Trainer directly | SFTTrainer wraps Trainer with chat template handling, peft_config integration, and completion-only loss — saves manual setup |
| tempfile.mkdtemp | Fixed output_dir | Temp dir avoids leftover checkpoint directories and cleans up automatically |

**Installation:**
```bash
# Add to model-training pyproject.toml dependencies:
# peft>=0.18.0, bitsandbytes>=0.45.0, transformers>=4.47.0, trl>=0.16.0, datasets>=3.0.0
# Add to training-svc pyproject.toml dependencies:
# model-training (workspace dependency)
uv sync
```

---

## Architecture Patterns

### Recommended Project Structure

New files to create:

```
libs/model-training/src/model_training/
├── config.py           # EXISTING stub — implement get_training_config() + validate_config()
├── peft_utils.py       # EXISTING stub — implement build_qlora_config() + apply_lora_adapter()
├── trainer.py          # NEW — train_qlora() orchestrator function
└── trajectory.py       # EXISTING — format_for_sft() already done

services/training-svc/src/training_svc/
├── routers/
│   └── training.py     # EXISTING stub — implement POST /train/lora + GET /jobs/{job_id}
├── jobs.py             # NEW — JobStatus dataclass + module-level JOB_STORE dict
└── schemas.py          # EXISTING — LoraTrainingRequest needs session_id field added
```

### Pattern 1: GPU Import Deferral (INFRA-04/05)

**What:** All imports from peft, bitsandbytes, transformers, trl are placed inside function bodies — not at module top-level. TYPE_CHECKING guard wraps type annotations only.

**When to use:** Every function in model_training that touches a GPU library.

**Example (from Phase 19-20 established pattern):**
```python
# Source: Phase 19-20 project pattern (model_training/peft_utils.py)
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import LoraConfig  # annotation-only, never executed at runtime


def build_qlora_config(
    rank: int,
    alpha: int,
    target_modules: list[str],
    dropout: float = 0.1,
) -> Any:  # Any at runtime; LoraConfig annotation only used by type-checkers
    """Build a QLoRA LoraConfig with NF4 quantization settings."""
    from peft import LoraConfig                                  # deferred import
    from transformers import BitsAndBytesConfig                  # deferred import
    import torch                                                  # deferred import

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
```

Note: `BitsAndBytesConfig` lives in `transformers`, not `bitsandbytes` directly. The quantization config is passed to `AutoModelForCausalLM.from_pretrained()`, not embedded in LoraConfig.

### Pattern 2: SFTTrainer QLoRA Pipeline

**What:** Load quantized model → attach LoRA via peft_config → train with SFTTrainer → save adapter only.

**When to use:** In `train_qlora()` orchestrator in `model_training/trainer.py`.

**Example:**
```python
# Source: TRL official docs (https://huggingface.co/docs/trl/en/sft_trainer)
# and PEFT docs (https://huggingface.co/docs/peft/package_reference/lora)
def train_qlora(
    session_id: str,
    output_dir: str,
    rank: int = 64,
    alpha: int = 128,
    epochs: int = 3,
    learning_rate: float = 2e-4,
) -> str:
    """Train QLoRA adapter and return path to saved adapter directory."""
    import tempfile
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from model_training.peft_utils import build_qlora_config
    from model_training.trajectory import format_for_sft, load_trajectory

    # 1. Load and format trajectory
    trajectory = load_trajectory(session_id)
    messages = format_for_sft(trajectory)  # returns [] if not successful
    if not messages:
        raise ValueError(f"Trajectory {session_id} is not successful or has no messages")

    # 2. Wrap in HuggingFace Dataset with "messages" field
    dataset = Dataset.from_list([{"messages": messages}])

    # 3. Build quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 4. Load base model with 4-bit quantization
    base_model_id = os.environ.get("RUNE_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # 5. Build LoRA config (no modules_to_save — vLLM compatibility)
    lora_config = build_qlora_config(
        rank=rank,
        alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        dropout=0.1,
    )

    # 6. Train
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_strategy="no",   # no intermediate checkpoints
        logging_steps=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()

    # 7. Save adapter only (SFTTrainer saves only adapter when peft_config is set)
    trainer.save_model(output_dir)
    return output_dir
```

### Pattern 3: FastAPI BackgroundTasks with In-Memory Job Tracking

**What:** Module-level dict stores JobStatus objects; BackgroundTask updates it during training.

**When to use:** POST /train/lora endpoint.

**Example:**
```python
# Source: FastAPI official docs (https://fastapi.tiangolo.com/tutorial/background-tasks/)
# and project pattern (training-svc/routers/training.py)
import uuid
from dataclasses import dataclass, field
from typing import Optional

# Module-level in-memory store — lost on restart (acceptable per CONTEXT.md)
JOB_STORE: dict[str, "JobStatus"] = {}


@dataclass
class JobStatus:
    job_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    adapter_id: Optional[str] = None
    error: Optional[str] = None


@router.post("/train/lora")
async def train_lora(
    request: LoraTrainingRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    job_id = str(uuid.uuid4())
    adapter_id = str(uuid.uuid4())
    JOB_STORE[job_id] = JobStatus(job_id=job_id, status="queued")
    background_tasks.add_task(
        _run_training_job,
        job_id=job_id,
        session_id=request.session_id,
        adapter_id=adapter_id,
        rank=request.rank,
        epochs=request.epochs,
    )
    return JSONResponse(content={"job_id": job_id, "status": "queued"})


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JSONResponse(content={
        "job_id": job.job_id,
        "status": job.status,
        "adapter_id": job.adapter_id,
        "error": job.error,
    })
```

### Pattern 4: AdapterRecord Creation After Training

**What:** Build AdapterRecord from trained adapter directory metadata and call registry.store().

**When to use:** At the end of `_run_training_job()` in training-svc.

**Example:**
```python
# Source: Phase 18 adapter-registry pattern (adapter_registry/models.py)
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from adapter_registry.models import AdapterRecord
from adapter_registry.registry import AdapterRegistry

def _build_adapter_record(
    adapter_id: str,
    adapter_dir: Path,
    session_id: str,
    task_type: str,
    rank: int,
    base_model_id: str,
) -> AdapterRecord:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    file_bytes = safetensors_path.read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    return AdapterRecord(
        id=adapter_id,
        version=1,
        task_type=task_type,
        base_model_id=base_model_id,
        rank=rank,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        file_path=str(adapter_dir),
        file_hash=file_hash,
        file_size_bytes=safetensors_path.stat().st_size,
        source="qlora",
        session_id=session_id,
    )
```

### Anti-Patterns to Avoid

- **Top-level GPU imports:** `import peft` or `import transformers` at module level causes ImportError in CPU-only CI. Always import inside function bodies.
- **`modules_to_save` in LoraConfig:** Adding embed_tokens or lm_head to modules_to_save saves full embedding weights into adapter_model.safetensors. vLLM cannot load adapters that include these keys — raises RuntimeError "Loading lora failed / is unsupported LoRA weight". Do NOT use modules_to_save.
- **Saving the merged model:** Calling `model.merge_and_unload()` then `save_pretrained()` saves the full 7B model (14GB+), not just the adapter (tens of MB). Use `trainer.save_model()` while peft_config is active — this saves only adapter weights.
- **SFTTrainer without `processing_class`:** If tokenizer is not passed, SFTTrainer attempts to load from model name string. When passing a model object (already loaded), explicitly pass `processing_class=tokenizer`.
- **Module-level env var reads in training-svc router:** Follow Phase 19-20 pattern — read env vars inside function bodies for monkeypatch testability.
- **Storing JOB_STORE inside a request handler closure:** The dict must be module-level so all requests share the same store instance. A dict defined inside the handler would be per-request.
- **SFTTrainer dataset without "messages" field:** SFTTrainer auto-detects conversational format from a "messages" key. The format_for_sft() output is list[dict] with role/content keys — must be wrapped as `{"messages": messages_list}` when creating the Dataset.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 4-bit quantization | Custom quantization code | BitsAndBytesConfig | NF4 has specific double-quant, storage dtype, compute dtype interactions — subtle to get right |
| LoRA weight injection | Custom layer replacement | peft.get_peft_model() | Handles freezing base weights, gradient routing, adapter naming conventions |
| Chat tokenization | Custom chat template application | SFTTrainer with "messages" field | Each model has its own chat template (Qwen2.5 has a predefined template) — SFTTrainer applies it correctly via apply_chat_template |
| Adapter-only saving | Custom state_dict filtering | trainer.save_model() (PEFT-aware) | SFTTrainer detects peft_config is set and calls model.save_pretrained() on the PeftModel, which saves only LoRA weights |
| UUID generation | Custom ID scheme | uuid.uuid4() | Standard, collision-free, no external dependencies |
| File integrity | Custom hashing | hashlib.sha256() | stdlib, well-tested |
| Async job dispatch | Thread pool or asyncio.create_task | FastAPI BackgroundTasks | Integrates with FastAPI lifecycle; sufficient for single-user local MVP |

**Key insight:** The PEFT + TRL combination was specifically designed to handle the 15+ edge cases in QLoRA training (gradient checkpointing with quantized layers, adapter-only saving, chat template handling). The entire pipeline is 20-30 lines of orchestration code over library calls — any custom implementation will miss edge cases.

---

## Common Pitfalls

### Pitfall 1: modules_to_save Breaks vLLM Loading
**What goes wrong:** Using `modules_to_save=["embed_tokens", "lm_head"]` in LoraConfig saves the full embedding matrix in adapter_model.safetensors. When vLLM attempts to load the adapter, it fails with RuntimeError because it only knows how to handle lora_A/lora_B weight keys.
**Why it happens:** PEFT includes full module weights (not just deltas) in the adapter file when modules_to_save is set.
**How to avoid:** Do NOT set modules_to_save at all. The locked decision (target_modules=["q_proj", "v_proj"]) already avoids this — vLLM loads cleanly.
**Warning signs:** adapter_model.safetensors is unexpectedly large (>1GB instead of a few MB); vLLM logs "Loading lora failed" or "is unsupported LoRA weight".

### Pitfall 2: SFTTrainer dataset_text_field vs "messages" Field
**What goes wrong:** Passing a dataset with a "text" field containing pre-formatted string instead of a "messages" field. This bypasses chat template application and may produce malformed training examples.
**Why it happens:** SFTTrainer has two dataset mode auto-detection paths: "messages" for conversational (applies chat template) and "text" for pre-tokenized text.
**How to avoid:** Wrap format_for_sft() output as `{"messages": messages_list}` — exactly matching the "conversational" dataset format. SFTTrainer will apply Qwen2.5's built-in chat template automatically.
**Warning signs:** Training loss starts very low (model already knows the format) or very high (format mismatch).

### Pitfall 3: GPU Import at Test-Time in CPU-Only CI
**What goes wrong:** Any `import peft`, `import bitsandbytes`, `import transformers`, or `import trl` at module top-level causes ImportError when the GPU packages are not installed (CPU CI).
**Why it happens:** These packages have binary extensions that may not be installed in CPU test environments.
**How to avoid:** All GPU imports inside function bodies only. TYPE_CHECKING guard for type annotations. `model_training/__init__.py` must NOT import from peft_utils or trainer directly (or at minimum those imports must also be deferred).
**Warning signs:** `import model_training` raises ImportError or ModuleNotFoundError in CI logs.

### Pitfall 4: BackgroundTasks Race Condition on JOB_STORE
**What goes wrong:** The background task starts updating JOB_STORE before the dict entry is created, causing a KeyError or status appearing as "queued" when it's actually failed.
**Why it happens:** BackgroundTasks runs after response is sent; if the dict entry isn't created before add_task(), the task may run before the endpoint sets initial status.
**How to avoid:** Create the JOB_STORE entry with status="queued" BEFORE calling background_tasks.add_task(). The endpoint returns the job_id immediately; the task reads/writes the same entry.
**Warning signs:** GET /jobs/{job_id} returns 404 for a job that was just created.

### Pitfall 5: SFTTrainer output_dir Checkpoint Pollution
**What goes wrong:** SFTTrainer with save_strategy="steps" (default) creates checkpoint-N subdirectories during training. After training, the adapter files may be in a checkpoint subdirectory, not the root output_dir.
**Why it happens:** save_strategy defaults to "steps" and saves checkpoints into output_dir/checkpoint-N/.
**How to avoid:** Set `save_strategy="no"` in SFTConfig to disable intermediate checkpoints, then explicitly call `trainer.save_model(output_dir)` after training completes. This puts adapter_model.safetensors and adapter_config.json directly in output_dir.
**Warning signs:** output_dir contains checkpoint-N/ subdirectories but no adapter_model.safetensors at the root.

### Pitfall 6: LoraTrainingRequest Missing session_id Field
**What goes wrong:** The existing LoraTrainingRequest schema has task_type, adapter_id, rank, epochs — but NOT session_id. The endpoint cannot find the trajectory without a session_id.
**Why it happens:** The stub was created before the training pipeline design was finalized.
**How to avoid:** Add `session_id: str` to LoraTrainingRequest in schemas.py. This is a schema change that requires updating the existing xfail test.
**Warning signs:** POST /train/lora has no way to identify which trajectory to train on.

---

## Code Examples

Verified patterns from official sources:

### BitsAndBytesConfig (4-bit NF4 QLoRA)
```python
# Source: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
# and TRL SFT docs: https://huggingface.co/docs/trl/en/sft_trainer
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 = NormalFloat4, better than int4 for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 avoids NaN at 7B (float16 is unstable)
    bnb_4bit_use_double_quant=True,      # double quantization reduces memory further
)
```

### LoraConfig for CAUSAL_LM (vLLM-compatible)
```python
# Source: https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig
from peft import LoraConfig

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,            # alpha = 2 * rank is standard convention
    target_modules=["q_proj", "v_proj"],  # attention projections only, no modules_to_save
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",    # required for decoder-only models (Qwen2.5, LLaMA, etc.)
    # DO NOT add modules_to_save — breaks vLLM loading
)
```

### Dataset.from_list with "messages" field
```python
# Source: TRL SFT docs dataset formats section
# https://huggingface.co/docs/trl/en/sft_trainer#expected-dataset-type-and-format
from datasets import Dataset

# format_for_sft() returns: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
messages = format_for_sft(trajectory)  # list[dict]
dataset = Dataset.from_list([{"messages": messages}])
# SFTTrainer detects "messages" key → applies Qwen2.5's chat template automatically
```

### SFTTrainer with peft_config (saves adapter only)
```python
# Source: TRL docs PEFT integration section
# https://huggingface.co/docs/trl/en/sft_trainer#train-adapters-with-peft
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_strategy="no",   # disable intermediate checkpoints
    logging_steps=1,
    report_to="none",     # disable wandb/tensorboard in local training
)
trainer = SFTTrainer(
    model=model,                    # pre-loaded quantized model OR model name string
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,        # triggers adapter-only training and saving
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(output_dir)      # saves adapter_model.safetensors + adapter_config.json ONLY
```

### apply_lora_adapter via get_peft_model
```python
# Source: https://huggingface.co/docs/peft/package_reference/lora#peft.LoraModel
from peft import get_peft_model

def apply_lora_adapter(model, config):
    """Wrap base model with LoRA adapter for training."""
    from peft import get_peft_model  # deferred import
    return get_peft_model(model, config)
```

Note: When using SFTTrainer with peft_config, get_peft_model is called internally. apply_lora_adapter is more useful for inference-time adapter loading or when using the PEFT model outside SFTTrainer.

### uv workspace dependency declaration (TRAIN-07)
```toml
# Source: existing workspace pattern (libs/adapter-registry in training-svc/pyproject.toml)
# Add to services/training-svc/pyproject.toml:
[project]
dependencies = [
    "adapter-registry",
    "model-training",   # ADD THIS
    "shared",
    "fastapi>=0.110.0",
    ...
]

[tool.uv.sources]
adapter-registry = { workspace = true }
model-training = { workspace = true }  # ADD THIS
```

### mypy overrides for GPU libraries (already exists in root pyproject.toml)
```toml
# Source: root pyproject.toml - already configured
# peft, datasets, transformers, torch are already in [[tool.mypy.overrides]] with ignore_missing_imports = true
# bitsandbytes and trl may need to be added if not already present
[[tool.mypy.overrides]]
module = [
    "bitsandbytes",
    "bitsandbytes.*",
    "trl",
    "trl.*",
]
ignore_missing_imports = true
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SFTTrainer with dataset_text_field="text" | SFTTrainer with "messages" field (conversational format) | TRL v0.11+ | Chat template auto-applied; correct tokenization for Qwen2.5 |
| Separate peft.prepare_model_for_kbit_training() call | SFTTrainer handles internally when peft_config given | TRL v0.7+ | One less manual step; reduces setup errors |
| trainer.model.save_pretrained() | trainer.save_model() | TRL v0.8+ | Correctly routes to PEFT-aware saving when adapters are active |
| LoraConfig without task_type | LoraConfig(task_type="CAUSAL_LM") | PEFT v0.4+ | task_type required for correct PEFT behavior with CausalLM |

**Deprecated/outdated:**
- `dataset_text_field` parameter: Still works for pre-formatted text strings, but the "messages" field approach is preferred for chat models — applies the model's actual chat template.
- `transformers.Trainer` directly for QLoRA: SFTTrainer wraps Trainer and adds PEFT integration, completion-only loss, and chat template handling — use SFTTrainer.
- `peft.prepare_model_for_kbit_training(model)` before SFTTrainer: SFTTrainer calls this internally when peft_config is provided. Calling it manually is now redundant (though harmless).

---

## Open Questions

1. **Qwen2.5-Coder-7B base model string**
   - What we know: CONTEXT.md specifies Qwen2.5-Coder-7B as the target model
   - What's unclear: Should the default be "Qwen/Qwen2.5-Coder-7B" (base) or "Qwen/Qwen2.5-Coder-7B-Instruct" (instruct-tuned)?
   - Recommendation: Use env var RUNE_BASE_MODEL defaulting to "Qwen/Qwen2.5-Coder-7B-Instruct" since SFT from instruction-tuned models typically performs better for code generation tasks; read inside function body for testability.

2. **SFTConfig bf16 vs bfloat16 in training_args**
   - What we know: SFTConfig inherits from TrainingArguments; bf16=True enables bfloat16 training
   - What's unclear: Whether bf16=True interacts with BitsAndBytesConfig's bnb_4bit_compute_dtype=torch.bfloat16
   - Recommendation: Set both — bf16=True in SFTConfig activates bfloat16 for the LoRA adapter computation; bnb_4bit_compute_dtype=torch.bfloat16 sets the dequantization compute dtype. They work at different levels and do not conflict.

3. **SFTTrainer and single-example dataset**
   - What we know: Single-trajectory training produces a dataset with exactly 1 example
   - What's unclear: Whether SFTTrainer handles a 1-example dataset gracefully (no shuffle issues, no empty eval split)
   - Recommendation: Pass `eval_dataset=None` explicitly (SFTTrainer default) and disable eval with `eval_strategy="no"` in SFTConfig. A 1-example dataset is valid for SFT.

4. **Training output dir cleanup strategy**
   - What we know: CONTEXT.md defers temp directory strategy to Claude's discretion
   - Recommendation: Use `tempfile.mkdtemp()` as the SFTTrainer output_dir during training. After `trainer.save_model()`, copy adapter_model.safetensors and adapter_config.json to `~/.rune/adapters/{adapter_id}/` using `shutil.copy2()`. Then remove the temp dir with `shutil.rmtree()`. This avoids leaving checkpoint debris.

5. **bitsandbytes and trl in root mypy overrides**
   - What we know: Root pyproject.toml already has peft, datasets, transformers, torch in mypy overrides
   - What's unclear: Whether bitsandbytes and trl are already listed
   - Recommendation: Check root pyproject.toml and add `bitsandbytes`, `bitsandbytes.*`, `trl`, `trl.*` to the ignore_missing_imports override block if missing. Current root pyproject.toml only shows peft/torch/transformers/datasets — trl and bitsandbytes likely need to be added.

---

## Sources

### Primary (HIGH confidence)
- TRL official docs https://huggingface.co/docs/trl/en/sft_trainer - SFTTrainer API, dataset formats, peft_config parameter, save_model behavior
- PEFT official docs https://huggingface.co/docs/peft/package_reference/lora - LoraConfig parameters, task_type, modules_to_save behavior
- FastAPI official docs https://fastapi.tiangolo.com/tutorial/background-tasks/ - BackgroundTasks API, in-memory tracking pattern
- Project codebase (Phase 18-20) - established patterns for deferred imports, session-per-method, workspace dependencies, env var reads inside function bodies

### Secondary (MEDIUM confidence)
- vLLM GitHub issue #9280 https://github.com/vllm-project/vllm/issues/9280 - confirmed modules_to_save breaks vLLM LoRA loading
- HuggingFace transformers quantization docs https://huggingface.co/docs/transformers/en/quantization/bitsandbytes - BitsAndBytesConfig parameters
- kaitchup.substack.com "Qwen2.5 QLoRA, LoRA, and Full Fine-tuning" - Qwen2.5 target_modules verification

### Tertiary (LOW confidence)
- WebSearch results on SFTTrainer single-example dataset behavior - needs empirical verification during implementation

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - verified via official HuggingFace docs (TRL, PEFT, transformers)
- Architecture: HIGH - SFTTrainer + peft_config pattern is the documented standard approach; FastAPI BackgroundTasks pattern is established
- Pitfalls: HIGH - modules_to_save/vLLM incompatibility verified via vLLM GitHub issue; import deferral pattern established in Phase 19-20
- Schema change (session_id): HIGH - LoraTrainingRequest clearly missing session_id field based on codebase inspection

**Research date:** 2026-03-05
**Valid until:** 2026-06-05 (stable libraries; TRL/PEFT APIs change slowly)
