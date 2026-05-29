# Fix PEFT LoRA Scaling to Match ctx_to_lora Training

## Problem

ctx_to_lora's training forward pass applies `scaling = lora_alpha` (raw, e.g. 16).
PEFT's standard forward applies `scaling = lora_alpha / rank` (e.g. 16/8 = 2.0).
Factor-of-`rank` mismatch means hypernetwork-generated weights operate at 1/8th their intended magnitude.

The forward-pass math is identical (`base + dropout(x) @ A.T @ B.T * scaling`), only the constant differs.

## Fix

Set `lora_alpha = alpha * rank` in the PEFT `LoraConfig`, making PEFT compute `(alpha * rank) / rank = alpha`, matching training exactly.

## Changes

### wrapper.py:90

```python
# Before
lora_config = LoraConfig(r=rank, lora_alpha=alpha, ...)

# After
lora_config = LoraConfig(r=rank, lora_alpha=alpha * rank, ...)
```

### config.py:32 — adapter_scaling default

```python
# Before: 0.075 (compensated for wrong PEFT scaling)
adapter_scaling: float = 0.075

# After: 1.0 (= training-equivalent; <1 = damped, >1 = amplified)
adapter_scaling: float = 1.0
```

### bench.yaml — all sweep/default values

| Field | Before | After |
|-------|--------|-------|
| `adapter_scaling` | 3.0 | 1.0 |
| `diag_scaling_sweep` | `[0.075, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]` | `[0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]` |
| `probe_scaling_sweep` | `[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]` | `[0.1, 0.25, 0.5, 0.75, 1.0, 1.5]` |
| `hpo.adapter_scaling.low` | 1.5 | 0.1 |
| `hpo.adapter_scaling.high` | 10.0 | 2.0 |

### graph.py:114 — fallback value

Already `1.0` — now correct by coincidence. No change needed.

## Semantic shift

`adapter_scaling` meaning changes:
- **Before**: raw multiplier on B weights, absorbed a hidden 1/rank factor from PEFT mismatch
- **After**: true scaling relative to training. 1.0 = as-trained. 0.5 = half strength. 2.0 = double.

## What stays the same

- B-only scaling pattern (`"lora_B" in k`) in graph.py and all benchmark scripts — unchanged
- ctx_to_lora dependency — still used for `HyperLoRA` weight generation and perceiver architecture
- PEFT dependency — still provides model wrapping, weight management, forward pass
- `_to_peft_state_dict` conversion — unchanged
- `hotswap_adapter` via `set_peft_model_state_dict` — unchanged

## Testing

- Unit tests: update `test_wrapper.py` if any assert on the LoraConfig alpha value
- Previous HPO/sweep results are invalid under new scaling — must re-run after fix
