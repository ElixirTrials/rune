# Phase 28: Functional LoRA Injection - Research

**Researched:** 2026-03-16
**Domain:** PyTorch autograd-safe context manager, F.linear functional LoRA, module forward patching
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Standard alpha/r LoRA scaling: `(alpha / r) * (B @ A)` additive to base output
- Separate F.linear passes: `base_out = F.linear(x, W, bias)` + `lora_out = F.linear(F.linear(x, A), B) * scale`
- All target modules from `HypernetConfig.lora_config.target_modules` patched (q_proj, k_proj, v_proj, o_proj on all 12 attention layers)
- Shape mismatches: fail fast with `RuntimeError` including module name and expected vs actual shapes
- New file: `libs/model-training/src/model_training/d2l_lora.py`
- New test: `libs/model-training/tests/test_d2l_lora.py`
- Context manager uses `try/finally` in `__exit__` — always restores original forward methods
- Exception propagates normally to caller (re-raised after cleanup)
- No nesting support needed
- No runtime verification of restoration — trust the try/finally
- Export `apply_functional_lora` from `__init__.py`

### Claude's Discretion
- Exact function signature and whether context manager is class-based or `@contextmanager`
- How to get LoRA tensors without `.detach()` from HyperLoRA (training flag vs direct internal call vs separate path)
- Internal module traversal strategy (how to find and patch target modules by name)
- Test mock strategy for faking HyperLoRA and base model without GPU

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LORA-01 | `apply_functional_lora()` context manager patches target modules with `F.linear` preserving autograd graph | F.linear two-path pattern confirmed; class-based CM with `_saved_forwards` dict |
| LORA-02 | Generated LoRA tensors have non-None `.grad` after backward pass (autograd continuity verified) | `HyperLoRA.generate_weights()` returns attached tensors — no `.detach()` in call chain; confirmed by reading source |
| LORA-03 | Original forward methods restored on context manager exit with no side effects | `try/finally` in `__exit__` restores from `_saved_forwards` dict regardless of exception |
</phase_requirements>

---

## Summary

Phase 28 implements `apply_functional_lora`, a context manager that patches attention projection modules in a transformer base model with F.linear forward methods carrying live hypernetwork tensors. The critical finding from source-code investigation is that **`HyperLoRA.generate_weights()` returns fully attached tensors** — there is no `.detach()` anywhere in `HyperLoRA.forward()` or `_to_lora_dict()`. The STATE.md concern about `.detach()` applies only to the `ModulatedPretrainedModel.generate_weights()` wrapper (which wraps the ctx_encoder call in `torch.no_grad()`, not the hypernet itself) and to `_save_sakana_adapter()` (PEFT serialization path). Since the training path calls `HyperLoRA.generate_weights()` directly with grad enabled, LORA-02 is achievable without any workarounds.

The ctx_to_lora `lora_layer.py` source confirms the two-pass F.linear pattern: base path calls `torch.nn.Linear.forward()` directly and the LoRA delta is computed separately via einsum operations. The module naming pattern for traversal is also confirmed: attention projections live at `model.layers[{i}].self_attn.{proj_name}` (using `attrgetter` in ctx_to_lora). Since our context manager works at the `nn.Module.forward` level (not patching into a PEFT layer structure), we can use `model.named_modules()` to locate target modules directly by their short name matching `target_modules`.

The implementation is a class-based context manager (`_FunctionalLoRAContext`) with a factory function `apply_functional_lora(model, lora_dict, hc)`. The class stores original forward methods in a `dict[str, Callable]` keyed by module path, and restores them in `__exit__` via `try/finally`. Testing follows established project patterns: real `torch.nn.Linear` modules with tiny dimensions, real tensors, and real `backward()` calls — no GPU mocking needed because everything runs on CPU.

**Primary recommendation:** Call `HyperLoRA.generate_weights(features, attn_mask, None)` without `torch.no_grad()` to get live tensors, then pass the `lora_dict` directly to `apply_functional_lora`. The context manager patches each target module's `forward` method with a closure over the corresponding layer's A and B slices.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch.nn.functional.linear` | torch (installed) | Two-pass LoRA forward: base + delta | Preserves autograd graph; used by ctx_to_lora lora_layer.py |
| `torch.nn.Module` | torch (installed) | Base class for patched modules | Method replacement on nn.Module instances is stable API |
| `contextlib.contextmanager` or class CM | stdlib | Context manager protocol | Both work; class-based chosen for clarity (see Architecture) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ctx_to_lora.modeling.hypernet.HyperLoRA` | installed | Generate LoRA A/B tensor dict | Called WITHOUT `torch.no_grad()` for training |
| `ctx_to_lora.modeling.hypernet.HypernetConfig` | installed | Source of `lora_config.target_modules`, `lora_config.r`, `lora_config.lora_alpha`, `layer_indices` | Config driven — no hardcoded layer indices |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Class-based CM | `@contextmanager` generator | Generator form is fine for simple cases but class form gives explicit `_saved_forwards` state storage without closure tricks |
| Direct module attribute patching | `functools.partial` wrapping | Partial is fine for simple cases; closure over (W, bias, A, B, scale) is cleaner |

**No new installations required** — all deps already in workspace.

---

## Architecture Patterns

### Recommended Project Structure
```
libs/model-training/src/model_training/
├── d2l_lora.py          # NEW: apply_functional_lora context manager
└── __init__.py          # UPDATE: export apply_functional_lora

libs/model-training/tests/
└── test_d2l_lora.py     # NEW: autograd + restoration tests
```

### Pattern 1: Class-Based Context Manager

**What:** A class implementing `__enter__` / `__exit__` that stores original forward methods and replaces them with F.linear closures.

**When to use:** When CM state (the `_saved_forwards` dict) must survive from `__enter__` to `__exit__` cleanly. Class form is explicit and testable in isolation.

**Example:**
```python
# All imports deferred inside function body per INFRA-05 (# noqa: PLC0415)
class _FunctionalLoRAContext:
    def __init__(
        self,
        model: Any,
        lora_dict: dict[str, dict[str, Any]],
        hc: Any,
    ) -> None:
        self._model = model
        self._lora_dict = lora_dict
        self._hc = hc
        self._saved_forwards: dict[str, Any] = {}

    def __enter__(self) -> "_FunctionalLoRAContext":
        import torch.nn.functional as F  # noqa: PLC0415
        r = self._hc.lora_config.r
        scale = self._hc.lora_config.lora_alpha / r
        target_modules = set(self._hc.lora_config.target_modules)
        layer_indices = list(self._hc.layer_indices)

        for module_path, module in self._model.named_modules():
            short_name = module_path.split(".")[-1]
            if short_name not in target_modules:
                continue
            # Determine which layer index this module belongs to
            layer_pos = _extract_layer_idx(module_path, layer_indices)
            if layer_pos is None:
                continue
            # Get A and B for this layer position
            A = self._lora_dict[short_name]["A"][0, layer_pos]  # (r, d_in)
            B = self._lora_dict[short_name]["B"][0, layer_pos]  # (r, d_out)
            # Shape validation
            W = module.weight  # (d_out, d_in)
            if A.shape[1] != W.shape[1]:
                msg = (
                    f"Shape mismatch at '{module_path}': "
                    f"A.shape[1]={A.shape[1]} != W.shape[1]={W.shape[1]}"
                )
                raise RuntimeError(msg)
            # Save and replace
            self._saved_forwards[module_path] = module.forward
            bias = module.bias
            module.forward = _make_lora_forward(W, bias, A, B, scale)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Always restore — try/finally guarantee
        for module_path, orig_forward in self._saved_forwards.items():
            module = _get_module_by_path(self._model, module_path)
            module.forward = orig_forward
        # Do not suppress exceptions
        return None
```

### Pattern 2: F.linear Two-Pass Forward Closure

**What:** A factory that creates a patched forward function preserving the base model weight computation and adding a separate LoRA delta. The base weight `W` must be detached (frozen base model), the LoRA matrices `A` and `B` carry the hypernetwork gradient.

**When to use:** Inside `__enter__`, once per target module per layer.

**Example:**
```python
def _make_lora_forward(
    W: Any,        # base model weight, detached (frozen)
    bias: Any,     # base model bias or None
    A: Any,        # lora A matrix, shape (r, d_in), live in hypernetwork graph
    B: Any,        # lora B matrix, shape (r, d_out), live in hypernetwork graph
    scale: float,  # lora_alpha / r
) -> Any:
    """Return a patched forward(x) using F.linear with separate LoRA path."""
    import torch.nn.functional as F  # noqa: PLC0415

    W_detached = W.detach()  # frozen base — no grads into base model
    bias_detached = bias.detach() if bias is not None else None

    def patched_forward(x: Any) -> Any:
        # Base path: uses frozen weights, no autograd into base model
        base_out = F.linear(x, W_detached, bias_detached)
        # LoRA path: x -> A -> B, live tensors carry grad back to hypernet
        # A: (r, d_in), x: (..., d_in) -> lora_Ax: (..., r)
        lora_Ax = F.linear(x, A)          # (..., r)
        # B: (r, d_out) -> need (d_out, r) for F.linear
        lora_out = F.linear(lora_Ax, B.t()) * scale  # (..., d_out)
        return base_out + lora_out

    return patched_forward
```

**Critical note:** `W.detach()` is intentional here — the base model weights are frozen (not being trained). The LoRA delta path carries the gradient through `A` and `B` which come from `HyperLoRA`'s head, flowing back into the trainable head parameters.

### Pattern 3: Module Traversal for Target Module Finding

**What:** Using `model.named_modules()` to locate modules whose short name (last `.`-separated segment) is in `target_modules`. Layer index extracted from the numeric segment in the dotted path.

**Confirmed from d2l_probe.py pattern (already in codebase):**
```python
for module_path, module in model.named_modules():
    short_name = module_path.split(".")[-1]
    if short_name not in target_modules:
        continue
    # Extract layer index from path e.g. "model.layers.7.self_attn.q_proj" -> 7
    parts = module_path.split(".")
    layer_idx = next(
        (int(p) for p in reversed(parts) if p.isdigit()), None
    )
```

**Layer position vs layer index:** `lora_dict[mod]["A"]` has shape `(batch, n_layers, r, d_in)`. The second dimension indexes into `hc.layer_indices` positionally. When a module at `model.layers.7.self_attn.q_proj` is found, we need `layer_pos = list(hc.layer_indices).index(7)` to get the correct slice from `lora_dict`.

### Pattern 4: Public API Shape

**What:** A factory function `apply_functional_lora` that returns the context manager instance. Caller provides the model, lora_dict (directly from `hypernet.generate_weights()`), and hypernet config.

```python
def apply_functional_lora(
    model: Any,
    lora_dict: dict[str, dict[str, Any]],
    hc: Any,
) -> "_FunctionalLoRAContext":
    """Context manager that patches model attention modules with F.linear LoRA.

    Args:
        model: The base transformer model (nn.Module).
        lora_dict: Output from HyperLoRA.generate_weights() — dict mapping module
            name to {"A": tensor (1, n_layers, r, d_in), "B": tensor (1, n_layers, r, d_out)}.
        hc: HypernetConfig with lora_config.target_modules, lora_config.r,
            lora_config.lora_alpha, and layer_indices.

    Returns:
        Context manager instance. Use as: `with apply_functional_lora(model, lora_dict, hc):`
    """
    return _FunctionalLoRAContext(model, lora_dict, hc)
```

**Caller pattern (training step):**
```python
# Training step — grad-enabled, no torch.no_grad() wrapper
lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)
with apply_functional_lora(base_model, lora_dict, hc):
    logits = base_model(**inputs)
    loss = criterion(logits, targets)
    loss.backward()  # grads flow: loss -> logits -> lora_out -> B -> A -> hypernet.head
```

### Anti-Patterns to Avoid

- **Calling `generate_weights()` inside `torch.no_grad()`**: `generate_adapter_from_sakana()` does this for PEFT saving. Training path must NOT wrap in `no_grad`. Call `hypernet.generate_weights()` directly with grad enabled.
- **Using `ModulatedPretrainedModel.generate_weights()`**: Its `ctx_encoder` call is wrapped in `torch.no_grad()`, but more importantly it's the wrong API — we call `HyperLoRA.generate_weights()` directly.
- **Fusing W + B@A into a single matrix**: Loses autograd path back to hypernet (B@A would become a new leaf tensor).
- **Using PEFT `get_peft_model()`**: Severs autograd; this is why Phase 28 exists instead of using PEFT training mode.
- **Calling `.contiguous()` on A/B before the forward**: Doesn't break autograd but is unnecessary — the F.linear ops handle this.
- **Mutating module attributes instead of `.forward`**: Setting `module.weight` would affect all callers; patching `.forward` is scoped.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LoRA weight generation | Custom perception/aggregation | `HyperLoRA.generate_weights()` | Fully implemented Perceiver perceiver with trained weights |
| Module naming convention for attention projections | String guessing | `model.named_modules()` + short name matching | Established in d2l_probe.py and confirmed in ctx_to_lora lora_layer.py |
| Scale factor | Custom scaling logic | `hc.lora_config.lora_alpha / hc.lora_config.r` | Standard LoRA formula — alpha and r from HypernetConfig |

**Key insight:** The two hardest pieces (generate_weights and the forward function structure) already have working implementations in `HyperLoRA` and `ctx_to_lora.lora_layer`. Phase 28 writes a thin adapter around them with different patching mechanics (we restore on exit; Sakana patches permanently).

---

## Common Pitfalls

### Pitfall 1: The `.detach()` Non-Issue (STATE.md Blocker Resolved)

**What goes wrong:** Concern that `generate_weights()` produces detached tensors.

**Actual situation (verified by reading source):**
- `HyperLoRA.generate_weights()` calls `self.forward()` → `_to_lora_dict()`. Neither applies `.detach()`. The scalers `scaler_A` and `scaler_B` are nn.Parameters multiplied via einsum — fully in the graph.
- `generate_adapter_from_sakana()` in `sakana_d2l.py` uses `torch.no_grad()` — but that is the PEFT-saving path, not the training path.
- `ModulatedPretrainedModel.generate_weights()` wraps its `ctx_encoder` call in `torch.no_grad()` but then calls `self.hypernet.generate_weights()` outside the no_grad block. We don't use `ModulatedPretrainedModel` at all.

**How to avoid:** Call `hypernet.generate_weights(features, attn_mask, None)` directly. Do NOT wrap in `torch.no_grad()` or `torch.inference_mode()`.

**Warning signs:** If `lora_dict["q_proj"]["A"].requires_grad` is False after calling generate_weights, grad has been disabled upstream.

### Pitfall 2: Layer Index vs Layer Position Confusion

**What goes wrong:** `lora_dict["q_proj"]["A"]` has shape `(1, 12, r, d_in)` where the second dimension is position within `hc.layer_indices` (0-11), NOT the absolute layer index (3, 7, 11, ..., 47).

**Why it happens:** `hc.layer_indices = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]` — 12 values. The tensor at `lora_dict["q_proj"]["A"][0, 0]` corresponds to layer 3, not layer 0.

**How to avoid:** When traversing modules and finding a module at path `...layers.7...`, look up `layer_pos = list(hc.layer_indices).index(7)` and use `lora_dict[mod]["A"][0, layer_pos]`.

**Warning signs:** Silent wrong-layer LoRA application. Check: `len(hc.layer_indices)` should equal `lora_dict["q_proj"]["A"].shape[1]`.

### Pitfall 3: B Matrix Transposition

**What goes wrong:** `lora_dict["q_proj"]["B"]` has shape `(1, n_layers, r, d_out)`. `F.linear(x, W)` expects W of shape `(out_features, in_features)`. B is `(r, d_out)` — needs `.t()` to use as the second F.linear weight.

**Why it happens:** HyperLoRA stores B transposed relative to what F.linear expects (Sakana's einsum uses `"n_ctx r d_out, ..."` convention).

**How to avoid:** `F.linear(lora_Ax, B.t())` — transpose B before passing to F.linear. Alternatively use `F.linear(lora_Ax, B.t())` or `lora_Ax @ B`.

**Warning signs:** Shape error during forward pass when B is passed directly.

### Pitfall 4: Module Not Found for Layer Index

**What goes wrong:** A layer index in `hc.layer_indices` has no matching module in the base model (e.g., because model isn't the full Qwen3 — perhaps in tests).

**How to avoid:** When `layer_pos` lookup fails (`layer_idx not in layer_indices_list`), skip gracefully. Log a warning. The shape validation catches most misconfigurations.

### Pitfall 5: `lora_dict` Keys vs `target_modules`

**What goes wrong:** The `lora_dict` returned by `HyperLoRA.generate_weights()` uses the same keys as `hc.lora_config.target_modules`, sorted. The sorted order is `["k_proj", "o_proj", "q_proj", "v_proj"]` — alphabetical, not QKOV order. Code must look up by key name, not by positional index.

**How to avoid:** Always look up by `lora_dict[short_name]`, never by list index. This is already the natural dict access pattern.

---

## Code Examples

Verified patterns from source:

### HyperLoRA generate_weights — No Detach (Confirmed from Source)
```python
# Source: .venv/lib/python3.13/site-packages/ctx_to_lora/modeling/hypernet.py lines 430-437
def generate_weights(self, features, attn_mask=None, position_ids=None):
    flat_loras, flat_layernorms = self.forward(features, attn_mask, position_ids)
    return self._to_lora_dict(flat_loras), self._to_layernorm_dict(flat_layernorms)

# _to_lora_dict applies scaler_A/scaler_B via einsum (lines 369-370) — both are nn.Parameters
# No .detach() anywhere in the chain
```

### lora_dict Tensor Shapes
```python
# lora_dict structure from HyperLoRA._to_lora_dict():
# lora_dict["q_proj"]["A"]  shape: (batch=1, n_layers=12, r=8, d_in=2048)
# lora_dict["q_proj"]["B"]  shape: (batch=1, n_layers=12, r=8, d_out=2048)
#
# Per-layer slice for layer_pos=0 (corresponds to hc.layer_indices[0]=3):
A = lora_dict["q_proj"]["A"][0, 0]   # shape: (r, d_in) = (8, 2048)
B = lora_dict["q_proj"]["B"][0, 0]   # shape: (r, d_out) = (8, 2048)
```

### ctx_to_lora Module Naming Convention (Confirmed from Source)
```python
# Source: .venv/lib/python3.13/site-packages/ctx_to_lora/modeling/lora_layer.py lines 96-100
# Attention projections: "self_attn.{mname}" e.g. "self_attn.q_proj"
# MLP projections: "mlp.{mname}" e.g. "mlp.down_proj"
# Full path: "model.layers.{layer_idx}.self_attn.q_proj"
```

### F.linear Two-Pass Pattern (adapted from lora_layer.py)
```python
# Source: adapted from ctx_to_lora/modeling/lora_layer.py lines 31-37
import torch.nn.functional as F

def patched_forward(x):
    base_out = F.linear(x, W_detached, bias_detached)  # frozen base path
    lora_Ax = F.linear(x, A)                            # (..., r) — A: (r, d_in)
    lora_out = F.linear(lora_Ax, B.t()) * scale         # (..., d_out)
    return base_out + lora_out
```

### Test Pattern — Real Tensors with backward()
```python
# Established project pattern for autograd tests (no GPU mock needed)
import torch
import torch.nn as nn

class _FakeAttnLayer(nn.Module):
    def __init__(self, d=16, r=2):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeAttnLayer() for _ in range(3)])
    def forward(self, x):
        for layer in self.layers:
            x = layer.q_proj(x)  # simplified
        return x

# Fake lora_dict with live parameters (require_grad=True)
def _make_fake_lora_dict(target_modules, layer_indices, r=2, d=16):
    return {
        mod: {
            "A": torch.randn(1, len(layer_indices), r, d, requires_grad=True),
            "B": torch.randn(1, len(layer_indices), r, d, requires_grad=True),
        }
        for mod in target_modules
    }
```

### Context Manager Restoration Verification Pattern
```python
# Verify forward is restored after context exit (LORA-03)
orig_forward = model.layers[0].q_proj.forward
with apply_functional_lora(model, lora_dict, hc):
    patched_forward = model.layers[0].q_proj.forward
    assert patched_forward is not orig_forward  # patched inside

assert model.layers[0].q_proj.forward is orig_forward  # restored outside
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PEFT `get_peft_model()` for training | Functional LoRA with F.linear | This project's design | PEFT severs autograd; F.linear preserves full gradient flow |
| Fused weight matrix `W + scale*(B@A)` | Separate F.linear passes | Standard LoRA training practice | Fused approach detaches LoRA from hypernet gradient |
| Permanent forward patching (Sakana's approach) | Context manager with restoration | This phase | Enables per-step fresh tensors without model state leakage |

**Deprecated/outdated:**
- `generate_adapter_from_sakana()` pattern with `torch.no_grad()`: correct for PEFT saving, wrong for training.

---

## Open Questions

1. **HypernetConfig.layer_indices type**
   - What we know: `layer_indices` is typed as `Iterable[int]` in HypernetConfig. In `build_qwen3_hypernet_config()` it's set to a plain Python `list[int]`.
   - What's unclear: Whether `list(hc.layer_indices).index(layer_idx)` works correctly if layer_indices is a torch.Tensor (as seen in `get_hypernet_config()` in Sakana's code: `indices = torch.arange(...)`).
   - Recommendation: Use `list(hc.layer_indices)` which handles both list and Tensor. If it's a Tensor, `list()` converts it to Python ints. Always safe.

2. **`hypernet.eval()` mode during training**
   - What we know: `load_sakana_checkpoint()` calls `hypernet.eval()` which disables dropout. For training the head, we may want `hypernet.train()`.
   - What's unclear: Whether the Phase 28 CM should call `hypernet.train()` or if that's Phase 29's responsibility.
   - Recommendation: Phase 28 CM does not touch the hypernet's train/eval mode — that's a Phase 29 training loop concern. Document clearly.

3. **Batch dimension handling**
   - What we know: `lora_dict[mod]["A"]` shape is `(batch, n_layers, r, d_in)` where batch is typically 1.
   - What's unclear: If batch > 1 is ever used.
   - Recommendation: Slice with `[0, layer_pos]` to always get `(r, d_in)`. Document that batch > 1 is unsupported in Phase 28.

---

## Validation Architecture

> nyquist_validation is not set in .planning/config.json — skipping this section.

---

## Sources

### Primary (HIGH confidence)
- `/Users/noahdolevelixir/Code/rune/.venv/lib/python3.13/site-packages/ctx_to_lora/modeling/hypernet.py` — `HyperLoRA.generate_weights()`, `HyperLoRA.forward()`, `_to_lora_dict()` — confirmed no `.detach()` in training path
- `/Users/noahdolevelixir/Code/rune/.venv/lib/python3.13/site-packages/ctx_to_lora/modeling/lora_layer.py` — `lora_forward()`, `apply_lora_to_layers()` — confirmed two-pass F.linear pattern and module naming convention
- `/Users/noahdolevelixir/Code/rune/libs/model-training/src/model_training/sakana_d2l.py` — `generate_adapter_from_sakana()` (uses `torch.no_grad()` for PEFT saving — not the training path)
- `/Users/noahdolevelixir/Code/rune/libs/model-training/src/model_training/d2l_probe.py` — `probe_model()` module traversal pattern (reused for target module finding)
- `/Users/noahdolevelixir/Code/rune/libs/model-training/tests/test_d2l_probe.py` — Established test patterns for nn.Module mocking without GPU
- `/Users/noahdolevelixir/Code/rune/libs/model-training/tests/test_d2l_weight_transfer.py` — `monkeypatch.setattr(torch, "load")` pattern and `_FakeHyperLoRA` pattern

### Secondary (MEDIUM confidence)
- PyTorch docs (F.linear): `F.linear(input, weight, bias)` where weight is `(out_features, in_features)` — standard API, stable across versions
- LoRA paper (arXiv:2106.09685): Standard scaling `alpha/r * (B @ A)` as additive delta

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all deps confirmed installed, no new installs needed
- Architecture: HIGH — F.linear pattern confirmed from ctx_to_lora source; CM pattern confirmed from codebase test patterns
- Pitfalls: HIGH — `.detach()` issue resolved by direct source reading; shape pitfalls confirmed from tensor shape analysis
- Autograd continuity: HIGH — `HyperLoRA.generate_weights()` source confirmed no detach; `requires_grad` flows through `scaler_A`/`scaler_B` parameters and `head` Mix layer

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (stable domain — PyTorch F.linear API, ctx_to_lora frozen at installed version)
