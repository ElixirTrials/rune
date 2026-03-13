# Phase 27: Partial Weight Transfer - Research

**Researched:** 2026-03-13
**Domain:** PyTorch state_dict partial loading, parameter freezing, HyperLoRA aggregator config extraction
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Weight Transfer Source**
- Source checkpoint: `qwen_4b_d2l/checkpoint-20000/pytorch_model.bin` from HuggingFace `SakanaAI/HyperLoRA`
- Existing `load_sakana_checkpoint()` in `sakana_d2l.py` already handles download and loading for this variant
- Checkpoint contains keys prefixed with `aggregator.`, `head.`, and `extra_` — exact namespaces need verification by printing `checkpoint.keys()` before implementing transfer logic
- STATE.md blocker: "qwen_4b_d2l checkpoint key namespaces inferred from code — print checkpoint.keys() before implementing partial weight transfer"

**Aggregator Freezing**
- All `aggregator.*` parameters get `requires_grad=False` after loading
- Verify by iterating `model.named_parameters()` — every param starting with `aggregator.` must have `requires_grad == False`
- Aggregator contains the pretrained Perceiver weights that encode document-to-LoRA mapping — these transfer directly

**Head Re-initialization**
- `head.*` parameters are NOT loaded from checkpoint — they must appear in `missing_keys` from `load_state_dict(strict=False)`
- Head output dimensions must match Qwen3-Coder-Next's 12 attention layers (from probe cache / `build_qwen3_hypernet_config()`)
- Re-initialization uses PyTorch default init (not custom) — training (Phase 29) will learn the new head

**Assertion Strategy**
- Post-load assertion explicitly fails if any expected aggregator key is missing from the loaded state dict
- Assertion checks `missing_keys` contains only `head.*` keys (aggregator fully loaded)
- Assertion checks `unexpected_keys` is empty or contains only known-irrelevant keys

**aggregator_config Population**
- `d2l_config.py` currently sets `aggregator_config=None` with comment "Phase 27 sets real value via get_aggregator_config()"
- Phase 27 implements `get_aggregator_config()` that inspects the loaded checkpoint's aggregator structure and returns config
- This completes the HypernetConfig — all fields populated after Phase 27

### Claude's Discretion
- Exact function signature for the weight transfer function (arguments, return type)
- How to structure the assertion checks (single function vs multiple)
- Whether to create a new module file or add to existing `sakana_d2l.py`
- Test mock strategy for faking checkpoint loading without GPU
- Error messages for assertion failures

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| XFER-01 | Partial weight transfer from `qwen_4b_d2l` checkpoint freezes `aggregator.*` parameters (`requires_grad=False`) | PyTorch `load_state_dict(strict=False)` + `p.requires_grad_(False)` loop is the correct pattern; verified against existing code in `sakana_d2l.py` lines 211-229 |
| XFER-02 | Head (`head.*`) is re-initialized with correct output dimensions for Qwen3-Coder-Next's 12 attention layers | `head.*` keys absent from the partial state dict cause them to land in `missing_keys` after `strict=False` load, leaving them at PyTorch default init; asserting `missing_keys` contains only `head.*` confirms correct shape for the 12-layer target |
</phase_requirements>

---

## Summary

Phase 27 implements `transfer_aggregator_weights()` — a function that loads the `qwen_4b_d2l` Sakana checkpoint, copies only `aggregator.*` parameters into a fresh `HyperLoRA` built for Qwen3-Coder-Next, freezes them, and verifies the result via assertions on `missing_keys` / `unexpected_keys`. The `head.*` parameters are intentionally excluded from transfer: they land in `missing_keys`, receive PyTorch default initialization, and will be trained in Phase 29.

The existing `load_sakana_checkpoint()` function in `sakana_d2l.py` already contains the `strict=False` loading pattern (lines 211-229) and the `aggregator.`/`head.`/`extra_` key filtering logic. Phase 27 extends this with: (a) explicit post-load assertions that fail hard on any missing aggregator key, (b) a `requires_grad_(False)` loop for all `aggregator.*` parameters, and (c) `get_aggregator_config()` that extracts the aggregator's structural config from the loaded checkpoint so `d2l_config.py` can populate the `aggregator_config=None` placeholder.

The STATE.md blocker is real and must be addressed first in the implementation plan: print `checkpoint.keys()` against the actual `qwen_4b_d2l` checkpoint to confirm the exact key prefixes before writing any transfer logic. The existing code infers `aggregator.` / `head.` / `extra_` prefixes but does not verify them.

**Primary recommendation:** Add `transfer_aggregator_weights()` and `get_aggregator_config()` to `sakana_d2l.py` (same file, no new module needed); keep the assertion logic in a private helper `_assert_transfer_integrity()`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | installed | `load_state_dict()`, `requires_grad_()`, tensor ops | GPU dep; deferred import per INFRA-05 |
| ctx_to_lora.modeling.hypernet | (Sakana pkg) | `HyperLoRA`, `HypernetConfig` — the model being loaded into | Already used in `sakana_d2l.py` |
| model_training.d2l_config | in-repo | `build_qwen3_hypernet_config()` — creates HyperLoRA with correct Qwen3 head shape | Phase 25/26 established function |
| model_training.d2l_probe | in-repo | `load_probe_cache()` — provides real feature_sizes for head dimension verification | Phase 26 established function |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| logging | stdlib | Module-level logger for transfer progress | All public functions |
| pathlib.Path | stdlib | Checkpoint path handling | Already used in module |

### No New Dependencies Required
Phase 27 uses libraries already installed and in-repo. No `uv add` needed.

---

## Architecture Patterns

### Recommended File Layout

```
libs/model-training/src/model_training/
├── sakana_d2l.py        # ADD: transfer_aggregator_weights(), get_aggregator_config(),
│                        #      _assert_transfer_integrity()
├── d2l_config.py        # UPDATE: build_qwen3_hypernet_config() accepts aggregator_config
│                        #         OR document that caller must set it after calling
│                        #         get_aggregator_config()
└── __init__.py          # (no change needed — GPU deps not exported at top level)

libs/model-training/tests/
└── test_d2l_weight_transfer.py   # NEW: tests for transfer, freezing, assertions
```

Adding to `sakana_d2l.py` is preferred over a new module because:
- All HyperLoRA checkpoint logic already lives there
- `download_checkpoint()` and `load_sakana_checkpoint()` are the natural callees
- No new import surface to maintain

### Pattern 1: Partial State Dict Transfer (strict=False)

**What:** Build a filtered state dict containing only `aggregator.*` keys, then call `load_state_dict(strict=False)`. Inspect the returned `_IncompatibleKeys` namedtuple.

**When to use:** Any time you want to transfer a subset of a checkpoint into a larger or differently-sized model.

**Implementation:**
```python
# Source: extends existing sakana_d2l.py lines 211-229 pattern
import torch  # noqa: PLC0415

def transfer_aggregator_weights(
    hypernet: Any,           # HyperLoRA instance built for Qwen3-Coder-Next
    checkpoint_path: str | Path,
) -> Any:                    # returns the same hypernet (mutated in-place)
    sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Filter: aggregator keys that also exist in the target model
    aggregator_sd = {
        k: v for k, v in sd.items()
        if k.startswith("aggregator.") and k in hypernet.state_dict()
    }

    loaded = hypernet.load_state_dict(aggregator_sd, strict=False)

    # Assertion: aggregator must be fully loaded
    _assert_transfer_integrity(hypernet, loaded)

    # Freeze aggregator parameters
    for name, param in hypernet.named_parameters():
        if name.startswith("aggregator."):
            param.requires_grad_(False)

    return hypernet
```

**Key insight:** `strict=False` returns `_IncompatibleKeys(missing_keys=[...], unexpected_keys=[...])`. `missing_keys` = keys in the model not present in the supplied state dict (i.e., `head.*` — expected). `unexpected_keys` = keys in the state dict not present in the model (should be empty for aggregator-only transfer).

### Pattern 2: Post-Load Integrity Assertion

**What:** After `strict=False` loading, assert that: (1) no aggregator key is missing, (2) all missing keys are `head.*`, (3) no unexpected keys exist.

**When to use:** Always after partial weight transfer — fail hard rather than silently using randomly-initialized aggregator weights.

```python
# Source: derived from CONTEXT.md assertion strategy
def _assert_transfer_integrity(
    hypernet: Any,
    loaded: Any,   # _IncompatibleKeys namedtuple from load_state_dict()
) -> None:
    # 1. All head.* parameters must be in missing_keys (not loaded from checkpoint)
    for key in loaded.missing_keys:
        if not key.startswith("head."):
            msg = (
                f"Unexpected missing key after aggregator transfer: '{key}'. "
                "Expected only head.* keys to be missing."
            )
            raise AssertionError(msg)

    # 2. No unexpected keys (aggregator keys not recognized by model)
    if loaded.unexpected_keys:
        msg = (
            f"Unexpected keys after aggregator transfer: {loaded.unexpected_keys}. "
            "These keys were in the checkpoint but not in the model."
        )
        raise AssertionError(msg)

    # 3. Every aggregator.* key in the model must have been loaded
    model_aggregator_keys = {
        k for k in hypernet.state_dict() if k.startswith("aggregator.")
    }
    missing_aggregator = {k for k in loaded.missing_keys if k.startswith("aggregator.")}
    if missing_aggregator:
        msg = (
            f"Aggregator keys missing after transfer: {missing_aggregator}. "
            "The checkpoint may use different key prefixes — print checkpoint.keys() to verify."
        )
        raise AssertionError(msg)

    logger.info(
        "Transfer integrity OK: %d aggregator keys loaded, %d head keys re-initialized",
        len(model_aggregator_keys),
        len(loaded.missing_keys),
    )
```

### Pattern 3: requires_grad Freezing and Verification

**What:** After loading, iterate `named_parameters()` and call `requires_grad_(False)` on every `aggregator.*` param. Verify afterward.

```python
# Source: standard PyTorch pattern — verified via existing project code
# Freeze
for name, param in hypernet.named_parameters():
    if name.startswith("aggregator."):
        param.requires_grad_(False)

# Verify (in test or assertion)
for name, param in hypernet.named_parameters():
    if name.startswith("aggregator."):
        assert not param.requires_grad, f"Expected frozen: {name}"
    elif name.startswith("head."):
        assert param.requires_grad, f"Expected trainable: {name}"
```

**Critical:** Call `param.requires_grad_(False)` NOT `param.requires_grad = False`. The `_()` in-place method is the PyTorch-idiomatic form and correctly propagates through the optimizer.

### Pattern 4: get_aggregator_config()

**What:** Inspect the loaded checkpoint's `hypernet_config` (a `HypernetConfig` stored in the `.bin` file under key `"hypernet_config"`) to extract the aggregator sub-config for use in `build_qwen3_hypernet_config()`.

```python
# Source: sakana_d2l.py line 199 — checkpoint stores sd["hypernet_config"]
def get_aggregator_config(checkpoint_path: str | Path) -> Any:
    import torch  # noqa: PLC0415
    sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    hc = sd["hypernet_config"]
    # HypernetConfig has aggregator_config field from the source checkpoint
    return hc.aggregator_config
```

**Why:** The `qwen_4b_d2l` checkpoint's `HypernetConfig` carries the Perceiver's structural config in `aggregator_config`. By reading it from the source checkpoint, Phase 27 avoids hardcoding Perceiver hyperparameters and ensures the target `HypernetConfig` is built with matching aggregator structure.

### Pattern 5: Test Mock Strategy for Checkpoint Loading

**What:** The `torch.load()` call inside `transfer_aggregator_weights()` cannot run in CI without a real `.bin` file. Use `monkeypatch` (pytest) to replace `torch.load` with a function returning a fake checkpoint dict.

```python
# Source: project test pattern (sys.modules injection + monkeypatch)
import torch
import torch.nn as nn

class _FakeAggregatorModel(nn.Module):
    """Minimal HyperLoRA mock with aggregator.* and head.* parameters."""
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = nn.Linear(4, 4)   # aggregator.weight, aggregator.bias
        self.head = nn.Linear(4, 12)         # head.weight, head.bias

def _make_fake_checkpoint(model: nn.Module) -> dict:
    """Build a fake checkpoint dict matching real structure."""
    sd = model.state_dict()
    # Only include aggregator keys in the checkpoint
    agg_sd = {k: v for k, v in sd.items() if k.startswith("aggregator.")}
    return {
        "hypernet_config": None,  # or a MagicMock
        **agg_sd,
    }
```

The monkeypatch replaces `torch.load` at the call site, or the test calls `transfer_aggregator_weights()` directly with a pre-built state dict via a helper that exposes the inner logic.

**Alternative (preferred):** Extract the state dict building into a helper that accepts either a path OR a pre-built dict, making it testable without filesystem/GPU.

### Anti-Patterns to Avoid

- **`strict=True` for partial transfer:** Will raise `RuntimeError` because `head.*` keys exist in the model but not the checkpoint. Always use `strict=False` for partial transfer.
- **Iterating `model.parameters()` to freeze (not `named_parameters()`):** Cannot filter by name. Use `named_parameters()`.
- **Setting `param.requires_grad = False` as attribute assignment:** Works in PyTorch but `requires_grad_(False)` is idiomatic and avoids leaf tensor confusion.
- **Not asserting on `missing_keys`:** If aggregator keys are renamed between checkpoint versions, the transfer silently uses random init for all weights. The assertion catches this.
- **Loading `head.*` from the old checkpoint:** The old head was sized for Qwen-4B's layers. Loading it would give wrong output dimensions for Qwen3-Coder-Next's 12 layers. The key filtering (`startswith("aggregator.")`) prevents this.
- **Calling `hypernet.eval()` before freezing:** `.eval()` does not freeze parameters — it only affects BatchNorm/Dropout behavior. Freezing requires `requires_grad_(False)`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Partial state dict loading | Custom weight-copying loop | `load_state_dict(strict=False)` | Returns `missing_keys`/`unexpected_keys` for free; handles tensor shape mismatch with clear error |
| Parameter freezing | Custom optimizer param filter | `param.requires_grad_(False)` | PyTorch native; optimizer automatically skips requires_grad=False params |
| Key namespace discovery | Guessing from docs | `torch.load()` + print checkpoint keys | CONTEXT.md explicitly flags this as a blocker — do it in Wave 0 |
| Checkpoint download | Custom HF download | `download_checkpoint(variant="qwen_4b_d2l")` already in sakana_d2l.py | Already implemented and cached |

**Key insight:** `load_state_dict(strict=False)` is the entire mechanism. The phase is really about wrapping it with the right assertions and freeze loop — not building novel loading logic.

---

## Common Pitfalls

### Pitfall 1: Unknown Checkpoint Key Namespaces

**What goes wrong:** The code in `sakana_d2l.py` lines 212-215 filters for `aggregator.`, `head.`, and `extra_` prefixes, but this was inferred from the code, not verified against the actual `qwen_4b_d2l` checkpoint. If the real checkpoint uses different prefixes (e.g., `perceiver.` instead of `aggregator.`), the transfer loads zero weights and the assertion fails.

**Why it happens:** `qwen_4b_d2l` and `gemma_demo` may use different key conventions. The existing code paths primarily exercised `gemma_demo`.

**How to avoid:** Wave 0 task: load the checkpoint with `torch.load()` and print/log the key prefixes. Implement the transfer only after confirming the actual prefix. The `_assert_transfer_integrity()` function will catch any remaining mismatch.

**Warning signs:** `_assert_transfer_integrity` raises `AssertionError` about missing aggregator keys. The transfer loaded 0 weight tensors.

### Pitfall 2: Head Shape Mismatch Between Source and Target

**What goes wrong:** If any `head.*` key from the old checkpoint accidentally gets included in the partial state dict (e.g., a prefix filter bug), `load_state_dict` will raise a shape mismatch error because old head was sized for Qwen-4B's layers, not Qwen3-Coder-Next's 12 layers.

**Why it happens:** The checkpoint filter must use `startswith("aggregator.")` only, not a broader prefix.

**How to avoid:** The filter `k.startswith("aggregator.")` is precise. Verify in `_assert_transfer_integrity` that no `head.*` keys appear in `unexpected_keys` (they should be absent from the supplied state dict entirely, not rejected as unexpected).

**Warning signs:** `load_state_dict` raises `RuntimeError: size mismatch for head.*` — this means `head.*` keys were accidentally included in the partial state dict.

### Pitfall 3: requires_grad_(False) Not Applied to All Aggregator Parameters

**What goes wrong:** If some aggregator parameters are in submodules of `hypernet.aggregator`, they appear in `named_parameters()` as `aggregator.resampler.layers.0.weight`, etc. A check like `if name == "aggregator"` misses these. The `startswith("aggregator.")` check is required.

**Why it happens:** Deep module hierarchies — `aggregator` is itself a multi-layer Perceiver with nested parameters.

**How to avoid:** Use `name.startswith("aggregator.")` in the freeze loop. After freezing, verify with a second iteration asserting `not param.requires_grad` for all aggregator params.

**Warning signs:** `optimizer.param_groups` still includes aggregator params after freezing. Phase 29 training updates aggregator weights (should be frozen).

### Pitfall 4: get_aggregator_config Returns None

**What goes wrong:** If the `qwen_4b_d2l` checkpoint's `HypernetConfig` was built without an explicit `aggregator_config` (set to `None`), `get_aggregator_config()` returns `None`. The `d2l_config.py` `aggregator_config=None` placeholder would remain, causing `HyperLoRA` construction to fail in Phase 28/29.

**Why it happens:** Older Sakana checkpoints may predatethe `aggregator_config` field being added to `HypernetConfig`.

**How to avoid:** In the Wave 0 verification step, print `sd["hypernet_config"]` to confirm `aggregator_config` is populated. If it's `None`, raise an explicit error in `get_aggregator_config()` rather than returning `None` silently. Phase 28 depends on this being non-None.

**Warning signs:** `get_aggregator_config()` returns `None`. `build_qwen3_hypernet_config()` called with `aggregator_config=None` hits the existing placeholder path.

### Pitfall 5: torch.load weights_only=False Required

**What goes wrong:** Using `torch.load(..., weights_only=True)` raises an error because the checkpoint contains `HypernetConfig` (a Python dataclass / object), not just tensors. PyTorch's `weights_only=True` mode restricts loading to tensor-safe objects only.

**Why it happens:** The checkpoint stores arbitrary Python objects (`hypernet_config`, `base_model_name_or_path`) alongside tensors.

**How to avoid:** Use `weights_only=False` as in the existing `load_sakana_checkpoint()` (line 197). This is a security trade-off (trusted checkpoint from HuggingFace). Document the reason with a comment.

**Warning signs:** `_pickle.UnpicklingError` or `WeightsUnpickler` error on `torch.load`.

---

## Code Examples

### Verified: Existing Checkpoint Loading Pattern (sakana_d2l.py lines 211-229)

```python
# Source: sakana_d2l.py lines 211-229 (existing — read 2026-03-13)
hypernet_keys = [
    k for k in sd.keys() if k.startswith(("aggregator.", "head.", "extra_"))
]
hypernet_sd = {k: sd[k] for k in hypernet_keys if k in hypernet.state_dict()}

if hypernet_sd:
    loaded = hypernet.load_state_dict(hypernet_sd, strict=False)
    logger.info("Loaded %d perceiver weight tensors", len(hypernet_sd))
    if loaded.missing_keys:
        logger.info("Missing keys: %d", len(loaded.missing_keys))
```

Phase 27 narrows the filter to `aggregator.` only (not `head.` or `extra_`) and adds assertions.

### Verified: requires_grad_(False) Pattern

```python
# Source: standard PyTorch API — confirmed in pytorch docs (no version change)
# Freeze all aggregator parameters
for name, param in hypernet.named_parameters():
    if name.startswith("aggregator."):
        param.requires_grad_(False)

# Verification loop (use in test or assertion)
frozen_count = 0
trainable_count = 0
for name, param in hypernet.named_parameters():
    if name.startswith("aggregator."):
        assert not param.requires_grad, f"Should be frozen: {name}"
        frozen_count += 1
    elif name.startswith("head."):
        assert param.requires_grad, f"Should be trainable: {name}"
        trainable_count += 1
logger.info("Frozen: %d, trainable: %d", frozen_count, trainable_count)
```

### Verified: _IncompatibleKeys Structure

```python
# Source: PyTorch load_state_dict documentation / source
# load_state_dict(strict=False) returns _IncompatibleKeys namedtuple:
#   .missing_keys   — keys in model state_dict but NOT in supplied dict
#   .unexpected_keys — keys in supplied dict but NOT in model state_dict
loaded = hypernet.load_state_dict(aggregator_sd, strict=False)
print(loaded.missing_keys)    # ["head.weight", "head.bias", ...]
print(loaded.unexpected_keys) # [] (should be empty for correct transfer)
```

### Wave 0 Verification Script (run before implementing)

```python
# uv run python -c "..."  — verify checkpoint key namespaces
import torch
from pathlib import Path

ckpt = Path.home() / ".cache/rune/sakana_d2l/qwen_4b_d2l/pytorch_model.bin"
sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
prefixes = sorted({k.split(".")[0] for k in sd if isinstance(sd[k], torch.Tensor)})
print("Top-level tensor key prefixes:", prefixes)
print("hypernet_config:", sd.get("hypernet_config"))
```

Expected output: `['aggregator', 'head']` (possibly `'extra'`). Any deviation from this changes the filter logic in `transfer_aggregator_weights()`.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `load_sakana_checkpoint()` loads aggregator + head from matching checkpoint | `transfer_aggregator_weights()` loads aggregator only, re-initializes head for new target | Phase 27 | Enables reuse of Perceiver across different target model architectures |
| `aggregator_config=None` placeholder in `build_qwen3_hypernet_config()` | `get_aggregator_config()` populates from loaded checkpoint | Phase 27 | HypernetConfig fully specified; Phase 28/29 can build and train |
| `load_state_dict` with no post-load verification | `_assert_transfer_integrity()` asserts key completeness | Phase 27 | Fails hard on checkpoint/model mismatches rather than silently using random init |

---

## Open Questions

1. **Exact checkpoint key prefixes for `qwen_4b_d2l`**
   - What we know: Existing `sakana_d2l.py` code filters for `aggregator.`, `head.`, `extra_` prefixes. This was inferred, not verified.
   - What's unclear: Whether `qwen_4b_d2l` uses exactly these prefixes or variations (e.g., `perceiver.`, `aggregator_model.`).
   - Recommendation: First task in Wave 0 is to run the verification script above. Do not implement the transfer function until the actual prefixes are confirmed.

2. **Whether `hypernet_config.aggregator_config` is non-None in `qwen_4b_d2l`**
   - What we know: `HypernetConfig` has an `aggregator_config` field; `build_qwen3_hypernet_config()` currently sets it to `None`.
   - What's unclear: Whether the saved `qwen_4b_d2l` checkpoint's `HypernetConfig` has `aggregator_config` populated.
   - Recommendation: Print `sd["hypernet_config"]` in Wave 0 verification. If `None`, the `get_aggregator_config()` function must either extract the config by inspecting the aggregator module structure (derive from loaded weights) or document that it returns `None` and Phase 28 must handle that case.

3. **Whether `HyperLoRA` built with `build_qwen3_hypernet_config()` has `aggregator.*` keys at all**
   - What we know: `build_qwen3_hypernet_config()` constructs a `HypernetConfig` with `aggregator_config=None`. `HyperLoRA(hc)` instantiation may behave differently than `HyperLoRA` from the source checkpoint.
   - What's unclear: Whether the Qwen3-targeted `HyperLoRA` has the same `aggregator.*` module structure as the `qwen_4b_d2l` one, or if `aggregator_config=None` causes the aggregator to be absent entirely.
   - Recommendation: Wave 0 task includes printing `list(HyperLoRA(build_qwen3_hypernet_config()).state_dict().keys())` to confirm aggregator modules are present before attempting transfer.

---

## Sources

### Primary (HIGH confidence)
- `/Users/noahdolevelixir/Code/rune/libs/model-training/src/model_training/sakana_d2l.py` (read 2026-03-13) — existing `load_sakana_checkpoint()` implementation with `strict=False` pattern and `aggregator.`/`head.`/`extra_` filtering at lines 211-229
- `/Users/noahdolevelixir/Code/rune/libs/model-training/src/model_training/d2l_config.py` (read 2026-03-13) — `build_qwen3_hypernet_config()` `aggregator_config=None` placeholder at line 142-143
- `.planning/phases/27-partial-weight-transfer/27-CONTEXT.md` (read 2026-03-13) — all locked decisions
- `.planning/STATE.md` (read 2026-03-13) — "qwen_4b_d2l checkpoint key namespaces inferred from code" blocker
- PyTorch `load_state_dict(strict=False)` API — `_IncompatibleKeys(missing_keys, unexpected_keys)` return type is stable and documented

### Secondary (MEDIUM confidence)
- `/Users/noahdolevelixir/Code/rune/libs/model-training/tests/test_d2l_probe.py` (read 2026-03-13) — established `sys.modules` injection + tiny `nn.Module` mock pattern for CI tests without GPU deps
- `/Users/noahdolevelixir/Code/rune/libs/model-training/tests/test_hypernetwork.py` (read 2026-03-13) — `_inject_fake_hn_modules()` pattern for `torch`/`safetensors` mocking

### Tertiary (LOW confidence)
- `qwen_4b_d2l` checkpoint key prefixes — INFERRED from existing code; not verified against actual `.bin` file. Confirmed as a STATE.md blocker requiring Wave 0 verification.
- `hypernet_config.aggregator_config` population in `qwen_4b_d2l` checkpoint — not verified; depends on Sakana checkpoint version.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project; no new deps
- Architecture (partial load + freeze pattern): HIGH — `load_state_dict(strict=False)` is stable PyTorch API; verified pattern exists in codebase already
- Architecture (assertion logic): HIGH — derived directly from CONTEXT.md locked decisions
- Checkpoint key namespaces: LOW — inferred; STATE.md explicitly flags this as blocker requiring Wave 0 verification
- Test mock strategy: HIGH — established sys.modules + monkeypatch patterns in existing test files
- get_aggregator_config: MEDIUM — approach is correct; whether checkpoint has non-None aggregator_config is unverified

**Research date:** 2026-03-13
**Valid until:** 2026-06-13 (stable PyTorch API; checkpoint is static artifact)
