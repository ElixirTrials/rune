# Model Pool Refactor — Resident Base Model + Rename sakana_d2l

## Problem

The pipeline calls `run_hypernetwork()` ~10 times per problem (once per phase iteration). Each call:
1. Streams the hypernetwork checkpoint from S3 (~34s, now cached by `_open_checkpoint`)
2. Instantiates HyperLoRA (428M params, ~1.7 GB)
3. Loads the full 9B base model from disk (~2 min for 427 weight shards)
4. Runs one forward pass for activation extraction
5. Deletes everything and frees VRAM

`TransformersProvider` separately loads the same Qwen 9B for inference. Two instances of the same model can't coexist on a 23 GB L4.

Additionally, the module is called `sakana_d2l.py` and functions reference "Sakana" throughout — misleading now that we use our own HPO-trained hypernetwork checkpoint.

## Design

### Module structure (separation of responsibilities)

```
libs/model-training/src/model_training/
├── model_pool.py          NEW  — Resource lifecycle (load, cache, loan out models)
├── adapter_generator.py   NEW  — Stateless adapter creation (text → PEFT adapter)
├── hypernetwork.py        EXPANDED — Checkpoint I/O, weight transfer, flash-attn patching
├── sakana_d2l.py          DELETED
└── ...

libs/inference/src/inference/
└── transformers_provider.py  UPDATED — Borrows base model from pool
```

### `model_pool.py` — Resource lifecycle

Single owner of GPU-resident models. Everything else borrows.

```python
class ModelPool:
    """Process-scoped cache for the base model and hypernetwork.

    The base model is shared between activation extraction (adapter_generator)
    and inference (TransformersProvider). The hypernetwork is loaded once and
    reused across all adapter generation calls.

    Factory method: ModelPool.create(base_model_id, checkpoint_path, device)
    Singleton access: get_pool() / set_pool()
    """

    @classmethod
    def create(cls, base_model_id: str, checkpoint_path: str,
               device: str = "cuda") -> "ModelPool":
        """Factory: create a pool tied to a specific model+checkpoint pair."""

    def base_model(self) -> tuple[Any, Any]:
        """Return (model, tokenizer), loading lazily on first call.
        Model is in eval mode, on self.device, dtype auto-resolved."""

    def hypernetwork(self) -> tuple[Any, Any]:
        """Return (hypernet, hypernet_config), loading lazily on first call."""

    def release(self) -> None:
        """Free all GPU memory. Pool can reload on next access."""

    @property
    def base_model_id(self) -> str: ...
    @property
    def device(self) -> str: ...


# Module-level singleton
_pool: ModelPool | None = None

def get_pool() -> ModelPool:
    """Return the process-wide ModelPool. Raises if not initialized."""

def set_pool(pool: ModelPool) -> None:
    """Set the process-wide ModelPool (called once at pipeline startup)."""
```

DRY: `base_model()` is the single place that calls `AutoModelForCausalLM.from_pretrained()` + `resolve_model_dtype()`. No other module loads a base model.

### `hypernetwork.py` — Checkpoint I/O (expanded)

Absorbs all checkpoint-related functions from `sakana_d2l.py`:

```
Existing:
  _open_checkpoint(path)           — S3-cached checkpoint loading (already done)
  _ensure_local_s3_cache(s3_uri)   — download once, reuse from disk (already done)

Moved from sakana_d2l.py (renamed):
  load_hypernetwork(checkpoint_path, variant, device) → (hypernet, config)
    — was load_sakana_checkpoint. Calls _patch_flash_attention(), _open_checkpoint(),
      instantiates HyperLoRA, loads weights.
  download_checkpoint(variant) → Path
    — downloads pretrained checkpoint from HuggingFace (legacy Sakana variants)
  transfer_aggregator_weights(hypernet, checkpoint_path) → hypernet
    — partial weight transfer for head retraining
  get_aggregator_config(checkpoint_path) → config
    — extract perceiver config from checkpoint
  _assert_transfer_integrity(hypernet, loaded)
    — validates load_state_dict result
  _patch_flash_attention()
    — flash-attn compatibility patching for ctx_to_lora
```

### `adapter_generator.py` — Stateless adapter creation

Replaces the public API of `sakana_d2l.py`. Supports two modes:
- **Pool mode** (fast): borrows resident models from a ModelPool
- **Standalone mode** (backwards compat): loads/unloads per call, like the old code

```
generate_adapter(text, output_dir, pool=None, *, checkpoint_path=None,
                 base_model_name=None, variant="gemma_demo",
                 device="cpu", max_length=512, scaling_factor=0.16) → str
    — was generate_adapter_from_sakana.
    — When pool is provided: uses pool.hypernetwork() and pool.base_model().
    — When pool is None: falls back to standalone loading (load_hypernetwork +
      extract_activations), then frees VRAM. This preserves existing behaviour
      for scripts that don't use run_phased_pipeline.

_save_adapter(lora_dict, output_dir, base_model_name, hc, scaling_factor)
    — was _save_sakana_adapter. Converts HyperLoRA output to PEFT format.

extract_activations(text, base_model_name, layer_indices, device, max_length)
    — KEPT for backwards compat. Standalone scripts (experiment_harness,
      run_optimization, e2e_test) that don't have a pool still call this.
      Loads model, extracts activations, frees model — same as before.
```

### `TransformersProvider` — Uses ModelPool

Updated to borrow the base model from the pool instead of loading its own copy.

```python
class TransformersProvider(InferenceProvider):
    def __init__(self, model_name="", device="cpu", torch_dtype="auto",
                 pool: ModelPool | None = None):
        self._pool = pool  # if provided, borrows model from pool

    def _load_model_if_needed(self):
        if self._model is not None:
            return
        if self._pool is not None:
            self._model, self._tokenizer = self._pool.base_model()
        else:
            # fallback: load standalone (for tests, CLI tools)
            ...
```

### `rune_runner.py` — Pipeline startup

```python
async def run_phased_pipeline(...):
    # Initialize pool once at pipeline start
    pool = ModelPool.create(base_model_id, checkpoint_path, device)
    set_pool(pool)

    # TransformersProvider borrows from pool
    # (factory.py get_provider needs pool kwarg or reads from get_pool())

    # run_hypernetwork uses pool
    ...
```

### `run_hypernetwork()` update — backwards compatible

```python
def run_hypernetwork(trajectory_text, output_dir, base_model_id=...,
                     checkpoint_path=None, device="cpu",
                     scaling_factor=0.16, pool=None):
    from model_training.adapter_generator import generate_adapter
    if pool is not None:
        return generate_adapter(text=trajectory_text, output_dir=output_dir,
                                pool=pool, scaling_factor=scaling_factor)
    # Standalone fallback — existing scripts (Gemma-2B, etc.) still work
    return generate_adapter(text=trajectory_text, output_dir=output_dir,
                            checkpoint_path=checkpoint_path,
                            base_model_name=base_model_id,
                            device=device, scaling_factor=scaling_factor)
```

Keeps the old params for backwards compat. `run_phased_pipeline` passes `pool=pool`; standalone scripts pass `checkpoint_path` + `base_model_id` as before.

## Rename mapping

| Old | New |
|-----|-----|
| `model_training.sakana_d2l` (module) | `model_training.adapter_generator` + `model_training.hypernetwork` |
| `generate_adapter_from_sakana()` | `generate_adapter()` |
| `load_sakana_checkpoint()` | `load_hypernetwork()` |
| `_save_sakana_adapter()` | `_save_adapter()` |
| `extract_activations()` | DELETED (use `pool.base_model()` + `extract_activations_with_model()`) |
| `download_checkpoint()` | stays, moves to `hypernetwork.py` |
| `transfer_aggregator_weights()` | stays, moves to `hypernetwork.py` |
| `get_aggregator_config()` | stays, moves to `hypernetwork.py` |
| `_patch_flash_attention()` | stays, moves to `hypernetwork.py` |
| `_assert_transfer_integrity()` | stays, moves to `hypernetwork.py` |

All docstrings updated to remove "Sakana" references. Comments that explain the two checkpoint formats (from-scratch vs legacy Sakana) keep a brief mention for historical context.

## Import site updates (26 sites across 18 files)

Every `from model_training.sakana_d2l import ...` must be updated to import from the new module. The full list:

**libs/model-training/src/:**
- `d2l_train.py` (2 sites) — imports `generate_adapter_from_sakana`, `_save_sakana_adapter`, `load_sakana_checkpoint`
- `round2_train.py` (1 site) — imports `load_sakana_checkpoint`, `_save_sakana_adapter`

**libs/model-training/tests/:**
- `test_d2l_weight_transfer.py` (6 sites) — imports `transfer_aggregator_weights`, `_assert_transfer_integrity`, `get_aggregator_config`
- `test_hypernet_from_scratch.py` (2 sites) — imports `load_sakana_checkpoint`, `_patch_flash_attention`

**scripts/:**
- `rune_runner.py` (1 site) — `generate_adapter_from_sakana`
- `compare_output.py` (1) — `generate_adapter_from_sakana`
- `experiment_harness.py` (1) — `generate_adapter_from_sakana`, `load_sakana_checkpoint`, `extract_activations`
- `e2e_test.py` (2) — `generate_adapter_from_sakana`, `download_checkpoint`, others
- `e2e_benchmark.py` (1) — `download_checkpoint`
- `benchmark_challenging.py` (1) — `download_checkpoint`
- `optimization/run_optimization.py` (1) — `generate_adapter_from_sakana`, `load_sakana_checkpoint`, `extract_activations`
- `optimization/run_training_hpo.py` (0 direct, but uses `_patch_flash_attention`)
- `train_hypernet_hpo.py` (1) — `_patch_flash_attention`
- `precompute_teacher_logits.py` (1) — `_patch_flash_attention`
- `paper/run_all_conditions.py` (1) — `_save_sakana_adapter`, `load_sakana_checkpoint`
- `paper/run_gate3.py` (1) — `_save_sakana_adapter`, `load_sakana_checkpoint`
- `_diag/test_ce_only.py` (1) — `_patch_flash_attention`
- `_diag/test_grad_norms.py` (1) — `_patch_flash_attention`

## Task breakdown

### Task 1: Create `model_pool.py`

Create `libs/model-training/src/model_training/model_pool.py`.

Implements:
- `ModelPool` class with `create()` factory, `base_model()`, `hypernetwork()`, `release()`, properties
- Module-level `get_pool()` / `set_pool()` singleton access
- `base_model()` loads lazily: `AutoModelForCausalLM.from_pretrained()` + `AutoTokenizer` + `resolve_model_dtype()`. Sets pad_token. Moves to device. Eval mode. Cached after first call.
- `hypernetwork()` loads lazily via `load_hypernetwork()` from `hypernetwork.py` (Task 2 dependency — for now, import `load_sakana_checkpoint` from `sakana_d2l`; Task 2 will fix the import).
- All GPU imports deferred inside method bodies (INFRA-05 pattern).

Tests: `libs/model-training/tests/test_model_pool.py`
- `test_create_pool` — factory returns ModelPool with correct properties
- `test_base_model_lazy_loading` — base_model() calls from_pretrained only on first call (monkeypatch)
- `test_hypernetwork_lazy_loading` — hypernetwork() calls load function only on first call (monkeypatch)
- `test_get_set_pool_singleton` — set_pool/get_pool round-trip
- `test_get_pool_uninitialized_raises` — get_pool() raises RuntimeError before set_pool()
- `test_release_clears_cache` — after release(), next base_model() call reloads

Gate: `uv run pytest libs/model-training/tests/test_model_pool.py -x && uv run ruff check libs/model-training/src/model_training/model_pool.py && uv run mypy libs/model-training/src/model_training/model_pool.py`

### Task 2: Expand `hypernetwork.py` with functions from `sakana_d2l.py`

Move these functions from `sakana_d2l.py` into `hypernetwork.py`:
- `_patch_flash_attention()` — as-is
- `download_checkpoint()` — as-is, update docstring ("Download pretrained checkpoint from HuggingFace")
- `load_sakana_checkpoint()` → rename to `load_hypernetwork()`, update docstring
- `transfer_aggregator_weights()` — update docstring (remove "Sakana" references)
- `get_aggregator_config()` — update docstring
- `_assert_transfer_integrity()` — update docstring

Also move module-level constants from `sakana_d2l.py`:
- `HF_REPO_ID`, `DEFAULT_VARIANT`, `LOCAL_CACHE_DIR`
- `_flash_attention_patched` global flag

Leave `sakana_d2l.py` intact for now — it will be updated in Task 4 to import from the new locations, and deleted in Task 6.

Update `__init__.py` comment to reference `model_training.hypernetwork` (line 21).

Tests: `libs/model-training/tests/test_hypernet_from_scratch.py` — update imports from `sakana_d2l` to `hypernetwork`. `test_d2l_weight_transfer.py` — update all 6 import sites. Both test files must pass unchanged in behavior.

Gate: `uv run pytest libs/model-training/tests/test_hypernet_from_scratch.py libs/model-training/tests/test_d2l_weight_transfer.py -x && uv run ruff check libs/model-training/src/model_training/hypernetwork.py && uv run mypy libs/model-training/src/model_training/hypernetwork.py`

### Task 3: Create `adapter_generator.py`

Create `libs/model-training/src/model_training/adapter_generator.py`.

Implements:
- `generate_adapter(text, output_dir, pool, scaling_factor=0.16) → str`
  - Gets `(hypernet, hc)` from `pool.hypernetwork()`
  - Gets `(model, tokenizer)` from `pool.base_model()`
  - Calls `extract_activations_with_model()` from `d2l_probe` directly (no wrapper)
  - Generates LoRA weights via `hypernet.generate_weights()`
  - Calls `combine_lora` for bias handling
  - Calls `_save_adapter()` for PEFT output
  - Does NOT delete/free the model (pool owns it)

- `_save_adapter(lora_dict, output_dir, base_model_name, hc, scaling_factor)`
  - Exact same logic as `_save_sakana_adapter`, renamed. Updated docstring.

All GPU imports deferred (INFRA-05). Import `extract_activations_with_model` from `d2l_probe`.

Tests: `libs/model-training/tests/test_adapter_generator.py`
- Test `_save_adapter` produces correct PEFT files (safetensors + config JSON)
- Test `generate_adapter` calls pool methods and produces output dir (monkeypatch pool, hypernet, model)

Gate: `uv run pytest libs/model-training/tests/test_adapter_generator.py -x && uv run ruff check libs/model-training/src/model_training/adapter_generator.py && uv run mypy libs/model-training/src/model_training/adapter_generator.py`

### Task 4: Update `rune_runner.py` to use ModelPool

Update `scripts/rune_runner.py`. **Key constraint: backwards compat.** The existing `run_hypernetwork()` signature must still work for standalone callers (Gemma-2B, Sakana checkpoint, e2e tests). Only `run_phased_pipeline` passes the pool.

1. `run_phased_pipeline()`:
   - Create `ModelPool.create(base_model_id, checkpoint_path, device)` at pipeline start
   - Call `set_pool(pool)` so TransformersProvider can find it
   - Pass `pool=pool` to `run_hypernetwork()` (in addition to existing params)
   - Call `pool.release()` in a finally block at pipeline end

2. `run_hypernetwork()`:
   - ADD optional `pool=None` parameter (keep ALL existing params)
   - When `pool is not None`: call `generate_adapter(text, output_dir, pool=pool, scaling_factor=...)`
   - When `pool is None`: call `generate_adapter(text, output_dir, checkpoint_path=..., base_model_name=..., device=..., scaling_factor=...)` — standalone fallback
   - Keep the `gc.collect()` / `cuda.empty_cache()` preamble in the `pool is None` branch only

3. Update all ~10 `run_hypernetwork(...)` call sites inside `run_phased_pipeline` to pass `pool=pool` as an additional kwarg (existing params stay).

4. CLI main at bottom of file — create pool if checkpoint provided, pass to pipeline.

Gate: `uv run ruff check scripts/rune_runner.py && uv run mypy scripts/rune_runner.py`

### Task 5: Update `TransformersProvider` to use ModelPool

Update `libs/inference/src/inference/transformers_provider.py`:

1. Add optional `pool` parameter to `__init__`:
   ```python
   def __init__(self, model_name="", device="cpu", torch_dtype="auto",
                pool: Any = None):
       self._pool = pool
   ```

2. Update `_load_model_if_needed()`:
   - If `self._pool is not None`: borrow model and tokenizer from `pool.base_model()`
   - Else: load standalone (existing code, for backwards compat in tests/CLI)

3. Update `libs/inference/src/inference/factory.py`:
   - In `get_provider()` for `ptype == "transformers"`: try to get pool from `get_pool()` (import from `model_training.model_pool`), pass to `TransformersProvider(pool=pool)`. Catch RuntimeError (pool not initialized) and fall back to standalone.

Gate: `uv run ruff check libs/inference/src/inference/transformers_provider.py libs/inference/src/inference/factory.py && uv run mypy libs/inference/src/inference/`

### Task 6: Update all remaining import sites + delete `sakana_d2l.py`

Update all remaining files that import from `sakana_d2l`:

**Pattern A — imports that move to `hypernetwork`:**
- `_patch_flash_attention` → `from model_training.hypernetwork import _patch_flash_attention`
- `load_sakana_checkpoint` → `from model_training.hypernetwork import load_hypernetwork`
- `download_checkpoint` → `from model_training.hypernetwork import download_checkpoint`
- `transfer_aggregator_weights` → `from model_training.hypernetwork import transfer_aggregator_weights`
- `get_aggregator_config` → `from model_training.hypernetwork import get_aggregator_config`
- `_assert_transfer_integrity` → `from model_training.hypernetwork import _assert_transfer_integrity`

**Pattern B — imports that move to `adapter_generator`:**
- `generate_adapter_from_sakana` → `from model_training.adapter_generator import generate_adapter`
- `_save_sakana_adapter` → `from model_training.adapter_generator import _save_adapter`

**Pattern C — `extract_activations` (deleted):**
- Callers that used `extract_activations(text, model_name, ...)` need updating:
  - `experiment_harness.py`, `run_optimization.py` — these are standalone scripts that don't use the pipeline. They should create their own `ModelPool` or use `extract_activations_with_model` from `d2l_probe` directly with a manually loaded model.

**Files to update (18 files):**
- `libs/model-training/src/model_training/d2l_train.py` (2 sites)
- `libs/model-training/src/model_training/round2_train.py` (1 site)
- `libs/model-training/src/model_training/__init__.py` (comment update)
- `libs/model-training/tests/test_d2l_weight_transfer.py` (6 sites — done in Task 2)
- `libs/model-training/tests/test_hypernet_from_scratch.py` (2 sites — done in Task 2)
- `scripts/compare_output.py`
- `scripts/experiment_harness.py`
- `scripts/e2e_test.py` (2 sites)
- `scripts/e2e_benchmark.py`
- `scripts/benchmark_challenging.py`
- `scripts/optimization/run_optimization.py`
- `scripts/optimization/run_training_hpo.py`
- `scripts/train_hypernet_hpo.py`
- `scripts/precompute_teacher_logits.py`
- `scripts/paper/run_all_conditions.py`
- `scripts/paper/run_gate3.py`
- `scripts/_diag/test_ce_only.py`
- `scripts/_diag/test_grad_norms.py`
- `scripts/e2e_training_smoke.py`

After all imports updated: delete `libs/model-training/src/model_training/sakana_d2l.py`.

Gate: `uv run ruff check libs/ scripts/ && uv run mypy libs/ scripts/ && uv run pytest libs/model-training/tests/ -x -q`

### Task 7: Update CLAUDE.md and documentation references

Update references to `sakana_d2l` in:
- `CLAUDE.md` — Important Files section, Architecture section, Adapter Research section, Template Editing section
- `docs/` — any doc referencing `sakana_d2l`
- `scripts/e2e_test.py` docstring (lines 4-6 mention `sakana_d2l`)

Gate: `grep -r "sakana_d2l" . --include="*.md" --include="*.py" | grep -v __pycache__ | grep -v ".pyc"` should return zero results.

### Task 8: Full test suite verification

Run the complete gate:
```bash
uv run ruff check libs/ scripts/ services/
uv run mypy libs/ scripts/ services/
uv run pytest -x -q
```

All 1002+ tests must pass. Zero ruff errors. Zero mypy errors.
No remaining references to `sakana_d2l` anywhere in the codebase.

## VRAM budget verification

After this refactor, during a pipeline run:
- One Qwen 9B in fp16: ~18 GB
- One hypernetwork in fp16: ~0.9 GB
- Activations + KV cache: ~3 GB headroom
- Total: fits L4 (23 GB)

The base model stays resident for the entire pipeline run. No more load/unload cycles.

## Backwards compatibility: scripts that don't use the pipeline

Several standalone scripts (`experiment_harness.py`, `run_optimization.py`, `e2e_test.py`, `compare_output.py`, `benchmark_challenging.py`) call `generate_adapter_from_sakana` directly without going through `run_phased_pipeline`. **These must keep working without changes to their calling pattern.**

`generate_adapter()` supports standalone mode (pool=None) — these scripts just update the import path and function name, passing the same kwargs as before. No pool creation needed.

The existing Gemma-2B + Sakana hypernetwork flow is preserved: standalone scripts call `generate_adapter(text, output_dir, checkpoint_path=..., base_model_name=..., device=...)` which loads/unloads per call exactly like the old code.
