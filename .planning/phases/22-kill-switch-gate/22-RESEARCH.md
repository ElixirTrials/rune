# Phase 22: Kill-Switch Gate - Research

**Researched:** 2026-03-05
**Domain:** Perceiver hypernetwork for LoRA weight generation, HumanEval evaluation, Pass@k metrics, kill-switch benchmarking
**Confidence:** HIGH (architecture confirmed from Sakana AI paper + established PEFT/HumanEval docs)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Hypernetwork weight source**
- Train from scratch — no dependency on Sakana AI pre-trained weights
- Minimal bootstrap dataset: 10-50 (trajectory, QLoRA adapter) pairs for proof-of-concept
- Use existing Phase 21 QLoRA pipeline to generate ground-truth adapter weights for training pairs
- Hypernetwork training is an offline step (script/notebook), not exposed as a service endpoint

**Hypernetwork architecture**
- Perceiver-based architecture with small latent array: 32 latents, 256 dim
- Target modules: q_proj and v_proj only (same as Phase 21 QLoRA config)
- Rank-8 output (per success criteria, distinct from QLoRA's rank-64)
- Module lives in model-training lib (libs/model-training/src/model_training/)

**Hypernetwork input encoding**
- Convert trajectory to SFT chat format via format_for_sft() (reuses Phase 20 code)
- Tokenize SFT output using base model tokenizer (Qwen2.5-Coder)
- Perceiver cross-attends over the token embedding sequence

**Training service endpoint**
- POST /train/hypernetwork is for inference (adapter generation), not hypernetwork training
- Takes a trajectory, runs it through the pre-trained hypernetwork, returns generated adapter
- Hypernetwork training itself is offline (separate from the service)

**HumanEval evaluation**
- Fixed 20-task subset — hardcoded task IDs for reproducible comparisons
- Tasks bundled as JSON file in evaluation lib — no network dependency at eval time
- adapter_id parameter is optional in run_humaneval_subset() — None runs baseline (bare model)
- One function handles both baseline and adapter evaluation

**Kill-switch verdict**
- 5% relative improvement threshold: adapter_pass1 >= baseline_pass1 * 1.05
- Output: structured dict with all metrics (baseline_pass1, adapter_pass1, relative_delta, verdict)
- Also prints human-readable summary with PASS/FAIL verdict
- Library function only (run_kill_switch_gate() in evaluation lib) — no service endpoint for v5.0
- On FAIL: report only, no automatic action beyond returning the verdict

### Claude's Discretion
- Code execution strategy for HumanEval (reuse execute_node subprocess sandbox vs dedicated executor)
- Hypernetwork output format (raw weight tensors + manual save vs PEFT model wrapper)
- Exact SFTTrainer configuration for hypernetwork training
- HumanEval task selection criteria (which 20 of 164 tasks to include)
- Error handling for malformed trajectories or generation failures

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DTOL-01 | DocToLoraHypernetwork module exists in model-training lib with Perceiver-based architecture | Perceiver architecture confirmed: latent array nn.Parameter, cross-attention over token embeddings, self-attention processing layers |
| DTOL-02 | Hypernetwork generates rank-8 LoRA adapter weights from coding trajectory in a single forward pass (<1s) | Sakana AI paper confirms rank-8, single forward pass <1s feasible with Perceiver compression |
| DTOL-03 | Generated adapters are compatible with vLLM dynamic LoRA loading (standard PEFT safetensors format) | PEFT checkpoint format fully documented: adapter_model.safetensors + adapter_config.json with specific key naming convention |
| DTOL-04 | training-svc exposes POST /train/hypernetwork endpoint for Doc-to-LoRA training | Stub already exists in training.py returning 501 — needs real implementation calling DocToLoraHypernetwork |
| EVAL-01 | User can run a HumanEval subset benchmark via run_humaneval_subset() | HumanEval task format known: prompt + completion + test + check(entry_point) execution pattern |
| EVAL-02 | User can calculate Pass@k metrics via calculate_pass_at_k() | Unbiased estimator formula known: 1 - C(n-c,k)/C(n,k) from Chen et al. 2021 |
| EVAL-03 | Kill-switch gate compares baseline vs adapter-enhanced Pass@1 (5% improvement threshold) | Verdict logic and output format locked in CONTEXT.md |
</phase_requirements>

---

## Summary

Phase 22 has three distinct technical domains: (1) a Perceiver-based hypernetwork that generates LoRA adapter weights in a single forward pass, (2) a HumanEval subset evaluation pipeline with subprocess-based code execution, and (3) a kill-switch gate comparing baseline vs adapter-enhanced Pass@1.

The Sakana AI Doc-to-LoRA paper (February 2026) confirms the Perceiver architecture is the state-of-the-art approach for this problem — a hypernetwork that cross-attends over token activations to produce rank-8 LoRA weight matrices directly. The user has locked the architecture to 32 latents, 256 dim, targeting q_proj and v_proj with rank-8 output. The critical implementation detail is that generated weights must be serialized in PEFT-compatible format (adapter_model.safetensors + adapter_config.json with `base_model.model.` key prefix) so that vLLM can load them via the existing load_adapter() path.

The HumanEval evaluation pipeline reuses the execute_node subprocess sandbox pattern from Phase 20, adapted to the HumanEval execution pattern: `prompt + completion + test + check(entry_point)` concatenated and run in a subprocess. The Pass@1 unbiased estimator is the special case where n_samples == n_correct / n_samples (pure pass rate when k=1 and n_samples == 1 per problem). The kill-switch gate function is pure library logic — no GPU needed, no service endpoint, just arithmetic on the two Pass@1 scores.

**Primary recommendation:** Build in three clean layers — (1) hypernetwork.py in model-training with deferred GPU imports, (2) evaluation/metrics.py implementations + bundled data/, (3) POST /train/hypernetwork endpoint wiring in training-svc. All three can be tested in CPU CI using the established sys.modules injection pattern.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0.0 (already in model-training) | Perceiver nn.Module, nn.Parameter latent array, cross-attention computation | Required for all neural net work; already declared in pyproject.toml |
| transformers | >=4.47.0 (already in model-training) | AutoTokenizer for Qwen2.5-Coder tokenization of trajectory text | Already declared; used in trainer.py |
| peft | >=0.18.0 (already in model-training) | LoraConfig for adapter_config.json generation; PeftModel patterns for output format | Already declared; used in peft_utils.py |
| safetensors | bundled with transformers/peft | Serialize generated LoRA weight tensors to adapter_model.safetensors | Required for vLLM compatibility |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| math (stdlib) | stdlib | comb() for Pass@k unbiased estimator | calculate_pass_at_k() implementation |
| subprocess (stdlib) | stdlib | HumanEval code execution sandbox | run_humaneval_subset() — reuse existing pattern from execute_node |
| json (stdlib) | stdlib | Load bundled HumanEval subset JSON, write adapter_config.json | Data loading + adapter serialization |
| tempfile (stdlib) | stdlib | Isolated execution directory per HumanEval task | Same pattern as execute_node in nodes.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| subprocess sandbox (reuse) | Docker-isolated executor | Docker isolation is ADV-03 (explicitly out of scope for v5.0); subprocess is sufficient for proof-of-concept |
| manual safetensors write | PEFT PeftModel.save_pretrained() | PEFT save_pretrained requires a full PEFT-wrapped model; raw weight generation then manual JSON+safetensors write is cleaner for hypernetwork output |
| hardcoded 20-task subset | dynamic random sampling | CONTEXT.md locks fixed task IDs — reproducibility over flexibility |

**Installation:**
No new dependencies required. All libraries already declared in model-training pyproject.toml.
For evaluation lib, if safetensors not already transitively available, it will be via the inference lib or transformers.

---

## Architecture Patterns

### Recommended Project Structure
```
libs/model-training/src/model_training/
├── hypernetwork.py          # NEW: DocToLoraHypernetwork Perceiver module
├── trainer.py               # existing: QLoRA pipeline (reused for bootstrap data)
├── peft_utils.py            # existing: build_qlora_config() reused
└── trajectory.py            # existing: format_for_sft() reused for input encoding

libs/evaluation/src/evaluation/
├── metrics.py               # MODIFY: implement run_humaneval_subset(), calculate_pass_at_k(), add run_kill_switch_gate()
└── data/
    └── humaneval_subset.json  # NEW: bundled 20-task HumanEval subset

services/training-svc/src/training_svc/
├── routers/training.py      # MODIFY: implement POST /train/hypernetwork (currently returns 501)
└── schemas.py               # existing: HypernetworkTrainingRequest already defined
```

### Pattern 1: Perceiver Architecture for Weight Generation

**What:** A learned latent array (nn.Parameter) cross-attends over token embeddings to produce a fixed-size output, which is then projected to LoRA weight matrices (lora_A, lora_B) for each target module.

**When to use:** Any scenario requiring variable-length sequence → fixed-shape weight tensor.

**Architecture details from Sakana AI paper and Perceiver literature (HIGH confidence):**
- Latents: `nn.Parameter(torch.randn(num_latents, latent_dim))` — 32 latents, 256 dim as locked
- Cross-attention: latents as Q, token embeddings as K/V (single cross-attention head per paper)
- Self-attention layers: process latents in latent space (multiple transformer blocks)
- Output projection: linear layer maps latent output → weight matrices for each LoRA component

**Perceiver forward pass pattern:**
```python
# Source: lucidrains/perceiver-pytorch + Sakana AI Doc-to-LoRA paper
class DocToLoraHypernetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,     # tokenizer embedding dim (4096 for Qwen2.5-7B)
        num_latents: int = 32,
        latent_dim: int = 256,
        depth: int = 4,     # number of cross+self attention blocks
        heads: int = 8,
        rank: int = 8,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
        num_layers: int = 28,  # Qwen2.5-7B has 28 transformer layers
    ) -> None:
        super().__init__()
        # Learnable latent array — the "compressed memory"
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        # Cross-attention: latents attend to token embeddings
        self.cross_attend = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.input_proj = nn.Linear(input_dim, latent_dim)  # project tokens to latent_dim
        # Self-attention: process latents
        encoder_layer = nn.TransformerEncoderLayer(latent_dim, heads, batch_first=True)
        self.self_attend = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # Output projection: latents → LoRA weight vectors
        # For each target module in each layer: need lora_A (rank x hidden) and lora_B (hidden x rank)
        # hidden_dim for q_proj and v_proj in Qwen2.5-7B-Instruct is 4096
        self.weight_head = nn.Linear(
            latent_dim * num_latents,
            len(target_modules) * num_layers * 2 * rank * 4096  # 2 = A+B, 4096 = hidden
        )

    def forward(self, token_embeddings: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        # token_embeddings: (batch, seq_len, input_dim)
        # Returns: dict mapping PEFT state_dict key -> tensor
        ...
```

### Pattern 2: PEFT-Compatible Adapter Serialization (Manual)

**What:** When a hypernetwork generates raw weight tensors (not via PEFT training), the output must be manually serialized to match PEFT's expected format so vLLM can load it.

**Why manual:** PEFT's `PeftModel.save_pretrained()` requires a fully-initialized PEFT model wrapping a base model — loading a 7B model just to serialize 200KB of adapter weights is wasteful. Manual serialization is cleaner for inference-time weight generation.

**Required files for vLLM compatibility (HIGH confidence — confirmed from PEFT docs and vLLM docs):**

1. `adapter_model.safetensors` — state_dict with PEFT key naming:
   - Pattern: `base_model.model.{layer_path}.lora_A.weight` and `.lora_B.weight`
   - For Qwen2.5-7B: `base_model.model.model.layers.{i}.self_attn.{module}.lora_A.weight`
   - lora_A shape: `(rank, in_features)` — e.g. (8, 4096) for q_proj
   - lora_B shape: `(out_features, rank)` — e.g. (4096, 8) for q_proj

2. `adapter_config.json` — minimum required fields:
```json
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "v_proj"],
  "lora_dropout": 0.0,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "base_model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "inference_mode": true
}
```

**Serialization code pattern:**
```python
# Source: PEFT checkpoint format docs (huggingface.co/docs/peft/en/developer_guides/checkpoint)
import json
from pathlib import Path
from safetensors.torch import save_file

def save_hypernetwork_adapter(
    weights: dict[str, "torch.Tensor"],
    output_dir: str,
    base_model_id: str,
    rank: int = 8,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights in safetensors format
    save_file(weights, str(out / "adapter_model.safetensors"))

    # Write adapter_config.json
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank * 2,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_id,
        "inference_mode": True,
        "modules_to_save": None,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
```

**CRITICAL:** Do NOT include embed_tokens.weight or lm_head.weight in the safetensors — vLLM rejects adapters that include these for quantized models (confirmed from vLLM docs). This is why `modules_to_save` must remain null/None in the config and the weight dict must only contain lora_A/lora_B entries.

### Pattern 3: HumanEval Execution

**What:** Execute LLM-generated code against HumanEval test cases using the official pattern: concatenate prompt + completion + test + `check(entry_point)` call, then exec in a subprocess.

**HumanEval task fields (HIGH confidence — confirmed from Hugging Face dataset viewer):**
| Field | Content |
|-------|---------|
| `task_id` | "HumanEval/0" through "HumanEval/163" |
| `prompt` | Function signature + docstring (Python code prefix) |
| `canonical_solution` | Reference implementation |
| `test` | `def check(candidate): assert ...` |
| `entry_point` | Function name (e.g. "has_close_elements") |

**Execution pattern (HIGH confidence — confirmed from openai/human-eval execution.py):**
```python
# Source: github.com/openai/human-eval/blob/master/human_eval/execution.py
check_program = (
    task["prompt"]
    + completion      # LLM-generated function body
    + "\n"
    + task["test"]    # contains def check(candidate): assert...
    + "\n"
    + f"check({task['entry_point']})"
)
# Execute in subprocess with timeout (reuse execute_node pattern from nodes.py)
```

**Reuse execute_node pattern:** The existing subprocess sandbox in `rune_agent/nodes.py` writes code to a temp file and runs `python script_path`. For HumanEval, the same pattern applies — write `check_program` to a temp file, run it, exit code 0 = passed.

**Key difference from execute_node:** HumanEval wraps generated code in `prompt + test` — do NOT run generated code directly. Always use the full concatenation.

### Pattern 4: Pass@k Unbiased Estimator

**What:** The mathematically correct formula for Pass@k from Chen et al. 2021 — avoids naive estimator bias.

**Formula (HIGH confidence — Chen et al. 2021 HumanEval paper):**
```
Pass@k = 1 - C(n-c, k) / C(n, k)
```
Where n = total samples, c = correct samples, k = attempts allowed.

**For Pass@1 with 1 sample per problem (n=1, k=1):** reduces to `c/n = pass_rate`.

**Python implementation:**
```python
# Source: verified against HuggingFace evaluate/metrics/code_eval implementation
import math

def calculate_pass_at_k(n_samples: int, n_correct: int, k: int = 1) -> float:
    """Unbiased Pass@k estimator (Chen et al. 2021)."""
    if n_correct > n_samples:
        raise ValueError("n_correct cannot exceed n_samples")
    if n_correct == n_samples:
        return 1.0  # all samples correct — trivially passes
    if n_samples - n_correct < k:
        return 1.0  # fewer incorrect than k — guaranteed to find correct
    # Product form avoids large factorial computation (numerically stable)
    # 1 - prod((n-c-i)/(n-i) for i in range(k))
    return 1.0 - math.prod(
        (n_samples - n_correct - i) / (n_samples - i)
        for i in range(k)
    )
```

**For this phase:** We generate 1 sample per HumanEval task (n_samples=1), so Pass@1 is simply the fraction of tasks where exit_code == 0.

### Pattern 5: POST /train/hypernetwork Endpoint

**What:** The existing stub returns 501. Implement it to load a pre-trained hypernetwork, tokenize the trajectory, run a forward pass, save the adapter, and return the adapter_id.

**Key constraint:** The hypernetwork weights file (`.pt` file from offline training) must be loaded at startup or lazily. Since hypernetwork training is offline, the service needs a path to the trained weights file (env var or config).

**Pattern — deferred import + lazy loading (INFRA-05 pattern):**
```python
# Source: established pattern from training.py _run_training_job
@router.post("/train/hypernetwork")
async def train_hypernetwork(
    request: HypernetworkTrainingRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = JobStatus(job_id=job_id, status="queued", adapter_id=str(uuid.uuid4()))
    background_tasks.add_task(_run_hypernetwork_job, job_id, request.trajectory_ids[0])
    return JSONResponse(content={"job_id": job_id, "status": "queued"}, status_code=200)

def _run_hypernetwork_job(job_id: str, trajectory_id: str) -> None:
    JOB_STORE[job_id].status = "running"
    try:
        from model_training.hypernetwork import DocToLoraHypernetwork  # deferred
        # ... load weights, generate adapter, save to disk
        JOB_STORE[job_id].status = "completed"
    except Exception as e:
        JOB_STORE[job_id].status = "failed"
        JOB_STORE[job_id].error = str(e)
```

### Anti-Patterns to Avoid
- **Loading base model in hypernetwork forward pass:** The hypernetwork takes token embeddings, not raw text. Tokenization uses AutoTokenizer (not AutoModelForCausalLM) — no 7B model needed at inference time.
- **Using modules_to_save in LoraConfig for generated adapters:** This would include embed_tokens in the saved artifact, which vLLM rejects for quantized models (confirmed Phase 21-01 decision).
- **Top-level GPU imports in hypernetwork.py:** All torch/transformers imports MUST be inside function bodies (INFRA-05 pattern). Module must be importable in CPU CI.
- **Random 20-task sampling at eval time:** Task IDs must be hardcoded in the bundled JSON file — reproducibility requires fixed subset.
- **Using HumanEval's exec() approach directly:** The original human-eval repo uses in-process exec() with reliability_guard() to disable dangerous ops. Subprocess approach (reusing execute_node pattern) is simpler and sufficient — exit code 0 = pass.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Safetensors serialization | Custom binary format | `safetensors.torch.save_file()` | vLLM expects safetensors specifically; custom formats won't load |
| Pass@k formula | Naive estimator | Product-form unbiased estimator | Naive formula has bias; Chen et al. formula is the standard |
| Attention mechanism | Custom attention | `nn.MultiheadAttention` | PyTorch built-in handles masking, scaling, efficiency correctly |
| Subprocess sandbox | New execution harness | Adapted execute_node pattern | Already tested and working; same timeout + tempdir approach |
| adapter_config.json schema | Guess field names | PEFT checkpoint docs | Wrong field names cause silent load failures in vLLM |

**Key insight:** The PEFT safetensors key naming convention (`base_model.model.` prefix) is the most critical detail. Getting this wrong causes vLLM to silently reject the adapter or raise cryptic errors.

---

## Common Pitfalls

### Pitfall 1: PEFT Key Naming Convention
**What goes wrong:** Hypernetwork saves weights with keys like `layers.0.self_attn.q_proj.lora_A.weight` but PEFT expects `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`.
**Why it happens:** The double `model.model.` prefix comes from PEFT wrapping: PeftModel wraps LoraModel wraps the base model. Qwen2.5-7B has an inner `.model` attribute.
**How to avoid:** Verify the key format by loading a QLoRA adapter saved by Phase 21's train_qlora(), inspecting its state_dict keys, and using those exact key patterns for the hypernetwork output.
**Warning signs:** vLLM raises "adapter not found" or "shape mismatch" errors when loading the adapter.

### Pitfall 2: lora_A / lora_B Shape Convention
**What goes wrong:** lora_A saved as (in_features, rank) but PEFT expects (rank, in_features).
**Why it happens:** LoRA math: output = W*x + B*A*x. A is (rank, in_features), B is (out_features, rank). This is counterintuitive — A has rank as the FIRST dimension.
**How to avoid:** lora_A.weight shape must be (rank, in_features) = (8, 4096). lora_B.weight shape must be (out_features, rank) = (4096, 8).
**Warning signs:** Silent shape mismatch during vLLM loading or incorrect generation output.

### Pitfall 3: HumanEval Execution — prompt + test Concatenation Order
**What goes wrong:** Running only the completion without the prompt, causing NameError (missing imports or helper functions defined in prompt).
**Why it happens:** HumanEval prompts often include `from typing import List` and other necessary imports. The completion is not standalone.
**How to avoid:** Always use: `prompt + completion + "\n" + test + "\n" + f"check({entry_point})"`.
**Warning signs:** Tasks failing with ImportError or NameError when the completion itself looks correct.

### Pitfall 4: GPU Import at Startup in Hypernetwork Module
**What goes wrong:** CPU CI tests fail to import training_svc because hypernetwork.py has top-level `import torch`.
**Why it happens:** INFRA-05 pattern requires all GPU imports deferred inside function bodies.
**How to avoid:** Use TYPE_CHECKING guard for type annotations; defer all `import torch`, `from peft import ...`, etc. inside function bodies (same as trainer.py).
**Warning signs:** `ImportError: No module named 'torch'` during CI test collection.

### Pitfall 5: Perceiver Latent Batch Dimension
**What goes wrong:** Forward pass crashes with shape error because latent array is (num_latents, latent_dim) but attention expects (batch, num_latents, latent_dim).
**Why it happens:** `self.latents` is a 2D parameter; it must be expanded per batch.
**How to avoid:** In forward(): `latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)`.
**Warning signs:** RuntimeError about mismatched dimensions in cross-attention.

### Pitfall 6: calculate_pass_at_k Edge Cases
**What goes wrong:** Division by zero or incorrect result when n_correct == 0 or n_correct == n_samples.
**Why it happens:** Product-form formula with empty range() or all-correct case.
**How to avoid:** Guard: if n_correct == n_samples: return 1.0. If n_samples - n_correct < k: return 1.0 (can always find at least one correct). If n_correct == 0: return 0.0 if k == 1.
**Warning signs:** ZeroDivisionError or Pass@k values outside [0, 1].

### Pitfall 7: HumanEval Timeout — Infinite Loop Tasks
**What goes wrong:** Some HumanEval tasks may generate infinite loops that hang indefinitely.
**Why it happens:** LLM generates incorrect looping logic.
**How to avoid:** Use same timeout mechanism as execute_node (RUNE_EXEC_TIMEOUT env var, default 30s). Treat timeout as task failure (tests_passed = False).

---

## Code Examples

Verified patterns from official sources and established project codebase:

### Perceiver Cross-Attention Pattern
```python
# Source: lucidrains/perceiver-pytorch + established PyTorch patterns
# Latent initialization
self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

# Forward pass — expand latents for batch
batch_size = token_embeddings.shape[0]
latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_latents, latent_dim)

# Project input to latent_dim
kv = self.input_proj(token_embeddings)  # (B, seq_len, latent_dim)

# Cross-attention: latents attend to token embeddings
latents, _ = self.cross_attend(latents, kv, kv)  # latents as Q, kv as K and V

# Self-attention processing in latent space
latents = self.self_attend(latents)  # (B, num_latents, latent_dim)
```

### Phase 21-Compatible sys.modules Injection for CPU CI Tests
```python
# Source: libs/model-training/tests/test_trainer.py (established pattern)
def _inject_fake_hypernetwork_modules() -> None:
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    fake_torch = ModuleType("torch")
    fake_torch.randn = MagicMock()  # type: ignore[attr-defined]
    fake_torch.nn = MagicMock()     # type: ignore[attr-defined]
    fake_torch.no_grad = MagicMock()  # type: ignore[attr-defined]

    sys.modules.setdefault("torch", fake_torch)
    sys.modules.setdefault("safetensors", ModuleType("safetensors"))
    sys.modules.setdefault("safetensors.torch", ModuleType("safetensors.torch"))
```

### HumanEval Task Execution
```python
# Source: derived from openai/human-eval execution.py pattern
import subprocess
import tempfile
from pathlib import Path

def _execute_humaneval_task(
    task: dict,
    completion: str,
    timeout: int = 30,
) -> bool:
    """Returns True if completion passes all test assertions."""
    check_program = (
        task["prompt"]
        + completion
        + "\n"
        + task["test"]
        + "\n"
        + f"check({task['entry_point']})"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "solution.py"
        script.write_text(check_program)
        try:
            proc = subprocess.run(
                ["python", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
```

### PEFT Safetensors Serialization
```python
# Source: PEFT checkpoint format docs (huggingface.co/docs/peft/en/developer_guides/checkpoint)
import json
from pathlib import Path
from safetensors.torch import save_file

def _save_peft_adapter(
    lora_weights: dict[str, "torch.Tensor"],
    output_dir: str,
    base_model_id: str,
    rank: int = 8,
    target_modules: list[str] | None = None,
) -> None:
    """Save raw LoRA weight tensors as a PEFT-compatible adapter directory."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Keys must follow PEFT naming: base_model.model.{path}.lora_{A|B}.weight
    save_file(lora_weights, str(out / "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank * 2,
        "target_modules": target_modules,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_id,
        "inference_mode": True,
        "modules_to_save": None,
        "fan_in_fan_out": False,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
```

### Kill-Switch Gate Function
```python
# Source: derived from CONTEXT.md locked decisions
def run_kill_switch_gate(
    baseline_pass1: float,
    adapter_pass1: float,
    threshold: float = 0.05,
) -> dict[str, object]:
    """Compare baseline vs adapter Pass@1 with 5% relative improvement threshold."""
    relative_delta = (adapter_pass1 - baseline_pass1) / max(baseline_pass1, 1e-9)
    verdict = "PASS" if adapter_pass1 >= baseline_pass1 * (1 + threshold) else "FAIL"
    result = {
        "baseline_pass1": baseline_pass1,
        "adapter_pass1": adapter_pass1,
        "relative_delta": relative_delta,
        "verdict": verdict,
    }
    print(f"Kill-switch gate: {verdict}")
    print(f"  Baseline Pass@1: {baseline_pass1:.3f}")
    print(f"  Adapter Pass@1:  {adapter_pass1:.3f}")
    print(f"  Relative delta:  {relative_delta:+.1%}")
    return result
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-document fine-tuning (gradient descent, minutes) | Hypernetwork forward pass (<1s) | Sakana AI Feb 2026 | Enables real-time adapter generation |
| adapter_model.bin (pickle) | adapter_model.safetensors | PEFT ~0.6+ | Security and speed improvement; vLLM prefers safetensors |
| Pass@1 naive estimator | Unbiased estimator (Chen 2021) | HumanEval paper 2021 | Mathematically correct; standard in benchmarking |

**Deprecated/outdated:**
- `adapter_model.bin`: Still supported by PEFT but safetensors is preferred and what this project uses
- `modules_to_save` in LoraConfig: Including embed_tokens breaks vLLM loading — never use for this project (established Phase 21-01 decision)

---

## Open Questions

1. **Qwen2.5-Coder-7B-Instruct exact layer count and q_proj/v_proj hidden dimensions**
   - What we know: The model is a 7B parameter causal LM with grouped-query attention
   - What's unclear: Exact number of transformer layers, exact q_proj/v_proj in_features and out_features needed to compute lora_A/lora_B shapes, whether GQA means q_proj and v_proj have different shapes
   - Recommendation: Load a Phase 21-produced adapter_model.safetensors from disk and inspect its key shapes before writing the hypernetwork output projection. This is the most reliable way to confirm shapes.

2. **Correct PEFT state_dict key prefix for Qwen2.5-Coder-7B-Instruct**
   - What we know: PEFT adds `base_model.model.` prefix; Qwen2.5 inner attribute is `.model`
   - What's unclear: The exact inner path may be `base_model.model.model.layers.{i}.` or just `base_model.model.layers.{i}.`
   - Recommendation: Again, inspect an existing Phase 21-produced adapter to confirm exact key format.

3. **Hypernetwork input dimension: token IDs vs embeddings**
   - What we know: CONTEXT.md says "tokenize SFT output using base model tokenizer, Perceiver cross-attends over the token embedding sequence"
   - What's unclear: At inference time, does the hypernetwork need the full embedding layer (7B model memory) or just tokenization + a lightweight embedding?
   - Recommendation: Use the tokenizer only (AutoTokenizer, not AutoModelForCausalLM) — the Perceiver's `input_proj` linear layer handles the embedding projection from token IDs to latent space. This avoids loading the 7B base model for adapter generation. The hypernetwork's own embedding layer learns the mapping from token IDs.

4. **RUNE_HYPERNETWORK_WEIGHTS_PATH env var**
   - What we know: Hypernetwork training is offline; POST /train/hypernetwork needs to load pre-trained weights
   - What's unclear: Whether to use a default path or require an explicit env var
   - Recommendation: Read from `RUNE_HYPERNETWORK_WEIGHTS_PATH` env var inside the function body (monkeypatch testability pattern). Default to `~/.rune/hypernetwork.pt`.

---

## Validation Architecture

> `workflow.nyquist_validation` is not set in `.planning/config.json` — skipping this section.

---

## Sources

### Primary (HIGH confidence)
- [PEFT Checkpoint Format Docs](https://huggingface.co/docs/peft/en/developer_guides/checkpoint) — adapter_model.safetensors format, key naming conventions, adapter_config.json schema
- [PEFT LoRA Reference](https://huggingface.co/docs/peft/package_reference/lora) — LoraConfig fields, lora_A/lora_B parameter shapes
- [openai/human-eval execution.py](https://github.com/openai/human-eval/blob/master/human_eval/execution.py) — check_program concatenation pattern, exec_globals approach
- [openai/openai_humaneval dataset](https://huggingface.co/datasets/openai/openai_humaneval) — exact field names: task_id, prompt, canonical_solution, test, entry_point
- [vLLM LoRA Adapters docs](https://docs.vllm.ai/en/stable/features/lora/) — adapter directory format requirements for dynamic loading
- Project codebase: libs/model-training/tests/test_trainer.py — sys.modules injection pattern for CPU CI testing
- Project codebase: services/rune-agent/src/rune_agent/nodes.py — subprocess sandbox execute_node pattern

### Secondary (MEDIUM confidence)
- [Sakana AI Doc-to-LoRA landing page](https://pub.sakana.ai/doc-to-lora/) — confirms Perceiver architecture, rank-8 output, <1s forward pass, teacher-student training
- [lucidrains/perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch) — canonical Perceiver PyTorch implementation patterns (latents as nn.Parameter, expand for batch)
- [leehanchung.github.io Pass@k blog](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/) — unbiased estimator formula and product-form implementation

### Tertiary (LOW confidence — needs validation)
- Qwen2.5-7B exact layer count and q_proj/v_proj hidden dimensions (research finding; must be confirmed by inspecting existing adapter weights from Phase 21)
- PEFT key prefix `base_model.model.model.layers.` for Qwen2.5 (inferred from Qwen architecture; must be confirmed empirically)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in pyproject.toml, confirmed in existing code
- Architecture (Perceiver): HIGH — confirmed from Sakana AI paper, lucidrains implementation, HuggingFace docs
- PEFT format: HIGH — confirmed from official PEFT checkpoint format documentation
- HumanEval execution: HIGH — confirmed from openai/human-eval source code
- Pass@k formula: HIGH — Chen et al. 2021, confirmed from multiple sources
- Qwen2.5-7B exact weight shapes: LOW — needs empirical verification from existing Phase 21 adapter

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (PEFT/vLLM APIs stable; HumanEval dataset unchanging; Perceiver patterns stable)
