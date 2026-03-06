---
phase: 22-kill-switch-gate
plan: 01
subsystem: model-training
tags: [hypernetwork, perceiver, lora, peft, tdd, cpu-ci]
dependency_graph:
  requires: []
  provides: [DocToLoraHypernetwork, save_hypernetwork_adapter]
  affects: [model-training]
tech_stack:
  added: [safetensors]
  patterns: [deferred-gpu-import, tdd, lazy-class-proxy]
key_files:
  created:
    - libs/model-training/src/model_training/hypernetwork.py
    - libs/model-training/tests/test_hypernetwork.py
  modified:
    - libs/model-training/src/model_training/__init__.py
decisions:
  - "Lazy proxy pattern for DocToLoraHypernetwork: _LazyHypernetworkProxy builds real nn.Module subclass on first instantiation — keeps module importable without torch at class-definition time (INFRA-05)"
  - "save_hypernetwork_adapter: modules_to_save=None — vLLM rejects embed_tokens/lm_head in adapter artifacts (consistent with Phase 21-01 decision)"
  - "Test small hidden_dim/num_layers: all torch tests use hidden_dim=32, num_layers=1-2 to keep CPU CI fast and avoid linear layer OOM"
metrics:
  duration: "~18 min"
  completed_date: "2026-03-05"
  tasks: 1
  files: 3
requirements: [DTOL-01, DTOL-02, DTOL-03]
---

# Phase 22 Plan 01: DocToLoraHypernetwork Summary

**One-liner:** Perceiver-based hypernetwork generating rank-8 LoRA adapters from token IDs in a single forward pass via lazy nn.Module proxy pattern for CPU-only importability.

## What Was Built

`DocToLoraHypernetwork` in `libs/model-training/src/model_training/hypernetwork.py` — a Perceiver architecture that cross-attends over token embeddings with learned latents to produce PEFT-compatible LoRA weight matrices for `q_proj` and `v_proj` across all transformer layers.

Key components:
- **Lazy proxy pattern**: `DocToLoraHypernetwork` is a `_LazyHypernetworkProxy` at module level; the real `nn.Module` subclass is built and cached on first instantiation. This keeps the module importable without `torch` installed (INFRA-05 compliance).
- **Perceiver architecture**: learned token embedding (not the 7B base model's), 32 latent vectors (256 dim), cross-attention (latents attend over token embeddings), self-attention stack (depth=4, heads=8), linear weight head projecting to all LoRA parameters.
- **forward() output**: PEFT state_dict keys following `base_model.model.model.layers.{i}.self_attn.{module}.lora_{A|B}.weight` with lora_A shape (rank, hidden_dim) and lora_B shape (hidden_dim, rank). Returns first batch element.
- **save_hypernetwork_adapter()**: writes `adapter_model.safetensors` via deferred `safetensors.torch.save_file` import and `adapter_config.json` with correct PEFT fields. `modules_to_save=None` prevents vLLM rejection.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Failing tests for DocToLoraHypernetwork | 373f210 | test_hypernetwork.py (290 lines) |
| GREEN | Implementation + __init__.py update | 4f91ad3 | hypernetwork.py (268 lines), __init__.py, test fixes |

## Verification Results

- `uv run python -c "from model_training.hypernetwork import DocToLoraHypernetwork"` — PASSED (CPU-only, no torch)
- `uv run pytest libs/model-training/tests/test_hypernetwork.py -x -v` — 7/7 PASSED
- `uv run ruff check libs/model-training/src/model_training/hypernetwork.py` — PASSED
- `uv run mypy libs/model-training/src/model_training/hypernetwork.py` — PASSED (no issues)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test construction OOM for full-size model**
- **Found during:** GREEN phase (first test run)
- **Issue:** `test_hypernetwork_latents_shape` created `DocToLoraHypernetwork(input_dim=32000)` with default `hidden_dim=4096` and 28 layers, making `weight_head` a `8192 x 3,670,016` linear layer — hung indefinitely on CPU.
- **Fix:** Tests that exercise forward/construction use `hidden_dim=32, num_layers=1-2` to stay fast on CPU. Latents shape test keeps `num_latents=32, latent_dim=256` (the defaults being tested) but uses small `hidden_dim`.
- **Files modified:** `libs/model-training/tests/test_hypernetwork.py`
- **Commit:** 4f91ad3

**2. [Rule 1 - Bug] Removed unused `torch.nn as nn` TYPE_CHECKING import**
- **Found during:** GREEN phase (ruff check)
- **Issue:** `import torch.nn as nn` under `TYPE_CHECKING` was unused (the forward() type annotation uses `torch.Tensor` directly).
- **Fix:** Removed `torch.nn as nn` from TYPE_CHECKING block.
- **Files modified:** `libs/model-training/src/model_training/hypernetwork.py`
- **Commit:** 4f91ad3

## Self-Check: PASSED

All files verified present. All commits verified in git log.
- FOUND: libs/model-training/src/model_training/hypernetwork.py
- FOUND: libs/model-training/tests/test_hypernetwork.py
- FOUND: .planning/phases/22-kill-switch-gate/22-01-SUMMARY.md
- FOUND commit: 373f210 (RED phase)
- FOUND commit: 4f91ad3 (GREEN phase)
