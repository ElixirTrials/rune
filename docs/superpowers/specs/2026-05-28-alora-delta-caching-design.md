# aLoRA-Style Delta KV Caching for Zero-Prefill Continuation

Follow-up to `2026-05-28-prefill-continue-design.md`. Eliminates the per-round prefill cost by caching base-model KV entries and applying adapter deltas at attention time.

## Problem

The prefill + continue approach (Approach A) works correctly but recomputes the KV cache from scratch each continuation round. For round N, the prefill covers ~(prompt + N*512) tokens. While this is fast with FlashAttention (<5% of round time at current scale), it becomes the dominant cost if:

- Token budgets increase (max_tokens > 2048)
- Code tasks produce very long outputs (1000+ lines)
- The model is scaled up (larger models have more expensive prefill)

## Core Idea

Separate the KV cache into **base** (adapter-independent) and **delta** (adapter-specific) components. Cache the base component across continuation rounds. Recompute only the delta when the adapter changes.

Based on the [aLoRA paper](https://arxiv.org/abs/2512.17910) (Dec 2025), adapted for single-GPU transformers (not vLLM).

## Architecture

### Standard LoRA attention (current)

```
K = (W_k + B_k @ A_k) @ X     # K depends on adapter → cached K is invalid after swap
V = (W_v + B_v @ A_v) @ X     # same for V
```

KV cache entries are computed with adapter weights baked in. Swapping the adapter invalidates the entire cache.

### Delta-separated attention (proposed)

```
K_base = W_k @ X               # base model only → stable across adapter swaps
K_delta = (B_k @ A_k) @ X      # adapter contribution → recomputed per swap
K = K_base + K_delta            # combined at attention time
```

`K_base` is computed once and cached. On adapter swap, only `K_delta` needs recomputation.

### Memory constraint

On a 24GB GPU with ~18.8GB used by model weights + hypernetwork:
- Full KV cache for 4096 tokens: ~2.7 GB (won't fit alongside model)
- K_delta is low-rank: `B_k @ A_k` has rank r (typically 8-64). Delta cache is r/d_head × smaller than full cache.

**Solution: CPU-offloaded base KV cache.**

```
GPU memory:
  - Model weights: ~18 GB
  - K_delta, V_delta (current adapter, low-rank): ~100-300 MB
  - Generation KV cache (growing): standard

CPU memory:
  - K_base, V_base (all past tokens): ~2.7 GB
  - Streamed to GPU block-by-block during attention
```

### Implementation approach

**Option 1: PEFT hook-based (minimal changes)**

Register forward hooks on the LoRA-targeted attention layers:

```python
def _pre_attention_hook(module, args):
    # Before attention, split K/V into base + delta
    # Store base on CPU, apply delta from current adapter
    ...

def _post_adapter_swap_hook():
    # After adapter swap, recompute deltas for all cached positions
    # K_delta_new = (B_k_new @ A_k_new) @ X_cached
    ...
```

Requires caching input activations X at each adapter-targeted layer. Memory cost: num_layers × seq_len × hidden_dim × 2 bytes. For Qwen3.5-9B (40 layers, 3584 hidden, 4096 tokens): ~1.1 GB on CPU.

**Option 2: Custom attention module (cleaner, more invasive)**

Subclass the Qwen attention module to separate base and delta KV computation:

```python
class DeltaCachingAttention(Qwen2Attention):
    def forward(self, hidden_states, ...):
        # Compute Q with full adapter (Q changes don't need caching)
        Q = self.q_proj(hidden_states)  # includes LoRA

        # For new tokens: compute K_base, V_base, K_delta, V_delta separately
        K_base = self.k_proj.base_layer(hidden_states)
        K_delta = self.k_proj.lora_B(self.k_proj.lora_A(hidden_states))
        K = K_base + K_delta

        # Cache K_base on CPU, K_delta on GPU
        self.base_kv_cache.append(K_base.cpu())
        ...
```

### Delta recomputation cost

On adapter swap, recompute `K_delta = (B_k_new @ A_k_new) @ X` for all cached positions. **This requires the input activations X to be stored** — either on CPU or GPU. Without stored activations, the delta cannot be recomputed and the approach degrades to a full prefill (which is just Approach A). Input activation caching is the enabling prerequisite, not an optimization.

With stored activations:

- This is a low-rank matmul: (d_model → r → d_head) applied to each cached position
- For r=16, d_model=3584, d_head=128, 4096 positions: ~0.5 GFLOP per layer
- Across 40 layers: ~20 GFLOP total
- On A10 (31 TFLOPS bf16): <1ms

The delta recomputation is negligible. The dominant cost is CPU→GPU transfer of the base cache + input activations.

### CPU→GPU transfer cost

PCIe Gen4 x16: ~25 GB/s. For 2.7 GB base KV cache: ~108ms per round.

This is worse than the prefill approach (~200-400ms) if the sequence is short, but better for long sequences because the transfer cost is constant while prefill cost grows linearly.

**Crossover point**: approximately 3000-4000 tokens. Below that, prefill is faster. Above that, delta caching wins.

### Hybrid strategy

```python
if accumulated_tokens < DELTA_CACHE_THRESHOLD:
    # Short sequence: prefill is cheaper
    use_prefill_continue()
else:
    # Long sequence: delta caching is cheaper
    use_delta_kv_cache()
```

## Files Changed

| File | Change |
|------|--------|
| `model/delta_cache.py` (new) | `DeltaKVCache` class managing base (CPU) + delta (GPU) cache separation |
| `model/inference.py` | `generate_continuation()` gains optional `delta_cache` parameter |
| `model/adapter.py` | `hotswap_adapter()` returns the old/new weight diff for delta recomputation |
| `engine/graph.py` | Continuation loop passes `delta_cache` to `generate_continuation()` when available |
| `config.py` | `delta_cache_threshold: int = 3000` — token count above which delta caching activates |

## What Does NOT Change

- Approach A (prefill + continue) remains the default for short sequences
- The continuation loop structure from Approach A is unchanged
- Hypernetwork, policy, templates, state — all unchanged

## Open Questions

1. **Input activation caching vs. delta caching**: Storing X (input activations) allows recomputing any delta. Storing K_delta directly is smaller but must be recomputed entirely on adapter swap. Which is more practical?

2. **PCIe bandwidth on target hardware**: The 108ms transfer estimate assumes PCIe Gen4. On cloud instances with slower interconnect, this could be worse. Need benchmarks.

3. **Interaction with `_try_completion`**: Layer 1 (JSON-level continuation) also needs KV cache. Should it use the delta cache too, or is it short enough to just prefill?

4. **Quantized base cache**: Storing K_base in int8 halves the transfer cost to ~54ms. Does the quantization error affect generation quality?

## Risks

**Risk: CPU→GPU transfer latency exceeds prefill cost for typical sequences.**
Mitigation: Hybrid strategy with a threshold. Start with prefill-only (Approach A); add delta caching only when benchmarks confirm it helps.

**Risk: Hook-based approach adds per-token overhead during generation.**
Mitigation: Hooks are only active during continuation rounds, not normal generation. The overhead per token is one low-rank matmul — negligible.

**Risk: PEFT internals change across versions, breaking hooks.**
Mitigation: Pin PEFT version. The custom attention module approach (Option 2) is more robust but more invasive.
