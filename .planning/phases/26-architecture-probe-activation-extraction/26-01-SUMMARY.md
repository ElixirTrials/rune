---
phase: 26-architecture-probe-activation-extraction
plan: "01"
subsystem: model-training
tags: [probe, architecture, activation-extraction, cache, d2l]
dependency_graph:
  requires: []
  provides:
    - d2l_probe.probe_model
    - d2l_probe.extract_activations_with_model
    - d2l_probe.load_probe_cache
    - d2l_probe.save_probe_cache
  affects:
    - d2l_config.build_qwen3_hypernet_config (feature_sizes now from probe cache)
    - sakana_d2l.extract_activations (delegates to extract_activations_with_model)
tech_stack:
  added: []
  patterns:
    - probe via model.named_modules() with child-name set intersection
    - SHA-256 16-char hex filename for probe JSON cache
    - output_hidden_states=True at call time (not model load time)
    - sys.modules injection pattern for CPU test isolation
key_files:
  created:
    - libs/model-training/src/model_training/d2l_probe.py
    - libs/model-training/tests/test_d2l_probe.py
  modified:
    - libs/model-training/src/model_training/d2l_config.py
    - libs/model-training/src/model_training/sakana_d2l.py
    - libs/model-training/src/model_training/__init__.py
decisions:
  - probe_model uses child-name set intersection (ATTN_PROJECTIONS.issubset) rather than
    string matching on module class name — more robust across model variants
  - output_hidden_states=True passed at call time not model init — avoids breaking
    AutoModelForCausalLM.from_pretrained signature in sakana_d2l
  - SHA-256 16-char hex cache filename avoids path-unsafe characters in model names
  - dict comprehension avoided (ruff C420) by using dict.fromkeys with typed local variable
  - feature_sizes fallback uses `hidden_size or 2048` to satisfy mypy (hidden_size is int|None in Qwen3NextConfig)
metrics:
  duration_seconds: 392
  completed_date: "2026-03-13"
  tasks_completed: 2
  files_created: 2
  files_modified: 3
requirements-completed: [ARCH-01, ARCH-02]
---

# Phase 26 Plan 01: Architecture Probe and Activation Extraction Summary

**One-liner:** Model-agnostic architecture probe via named_modules() with SHA-256 JSON cache and extract_activations_with_model() accepting pre-loaded model+tokenizer.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create d2l_probe.py with probe, cache, extraction | 594e49e | d2l_probe.py, test_d2l_probe.py |
| 2 | Wire probe into d2l_config, sakana_d2l, __init__ | 1207004 | d2l_config.py, sakana_d2l.py, __init__.py |

## What Was Built

### d2l_probe.py (new)

Four public functions:

- `probe_model(model)` — iterates `model.named_modules()`, finds layers where `ATTN_PROJECTIONS.issubset(child_names)`, extracts layer index from last numeric path segment, captures q/k/v/o_proj weight dimensions as `{"in": int, "out": int}`.
- `save_probe_cache(model_name, probe_result)` — adds metadata (model_name, SHA-256 hash, ISO-8601 UTC timestamp), writes to `~/.cache/rune/probes/{hash16}.json`.
- `load_probe_cache(model_name)` — returns `None` on miss (never raises), reads and returns JSON dict.
- `extract_activations_with_model(text, model, tokenizer, layer_indices, model_name, max_length)` — passes `output_hidden_states=True` at call time, stacks `hidden_states[i]` for each index (no +1 offset), auto-loads layer_indices from probe cache when `None`.

### d2l_config.py (modified)

`build_qwen3_hypernet_config()` now loads probe cache for `QWEN3_NEXT_CANONICAL_NAME` and uses real `feature_sizes` when available. Falls back to `hidden_size` placeholder when no cache (CI-safe). Existing test `test_build_qwen3_hypernet_config_returns_twelve_layer_indices` continues to pass via the fallback path.

### sakana_d2l.py (modified)

`extract_activations()` is now a backward-compatible wrapper: loads model without `output_hidden_states=True` in `from_pretrained`, then delegates to `extract_activations_with_model()`. Function signature unchanged.

### __init__.py (modified)

Added probe module to commented export documentation block. No actual imports added (GPU deps stay deferred per INFRA-05).

## Test Results

8 new tests in `test_d2l_probe.py`, all passing on CPU:
- `test_probe_model_finds_attention_layers` — [3] from 3xDeltaNet + 1xAttn model
- `test_probe_model_skips_deltanet_layers` — [1, 3] from mixed model
- `test_probe_model_captures_feature_sizes` — correct q_proj/v_proj in/out dims
- `test_save_and_load_probe_cache` — JSON round-trip with metadata fields
- `test_load_probe_cache_returns_none_on_miss` — None returned, no raise
- `test_extract_activations_with_model_returns_correct_shape` — (1, N, seq, hidden)
- `test_extract_activations_with_model_auto_detects_from_cache` — layer_indices from cache
- `test_extract_activations_with_model_raises_without_cache` — RuntimeError with "layer_indices"

Total suite: 60 passed, 1 xfailed, 1 xpassed (no regressions).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ruff E501 in d2l_probe.py module docstring**
- **Found during:** Task 2 verification
- **Issue:** First line of module docstring exceeded 88 chars
- **Fix:** Wrapped docstring to two lines
- **Files modified:** d2l_probe.py
- **Commit:** 1207004

**2. [Rule 1 - Bug] ruff C420 in d2l_config.py fallback feature_sizes**
- **Found during:** Task 2 verification
- **Issue:** Dict comprehension `{mod: hidden for mod in target_modules}` flagged as C420
- **Fix:** Used `dict.fromkeys(target_modules, hidden)` with typed local variable `_placeholder: dict[str, int]`
- **Files modified:** d2l_config.py
- **Commit:** 1207004

**3. [Rule 1 - Bug] mypy error — `cfg.hidden_size` typed as `int | None`**
- **Found during:** Task 2 mypy check
- **Issue:** `dict.fromkeys(target_modules, cfg.hidden_size)` produced `dict[str, int | None]` incompatible with `tuple[dict[str, int], dict[str, int]]`
- **Fix:** `hidden: int = cfg.hidden_size or 2048` with explicit type annotation
- **Files modified:** d2l_config.py
- **Commit:** 1207004

**4. [Rule 1 - Bug] Pre-existing long comment line in __init__.py**
- **Found during:** Task 2 ruff check
- **Issue:** `#   from model_training.d2l_config import get_d2l_qwen3_config, build_qwen3_hypernet_config` was 91 chars
- **Fix:** Split to multi-line comment import style
- **Files modified:** __init__.py
- **Commit:** 1207004

**5. [Rule 1 - Bug] ruff format reformatted sakana_d2l.py chained method call**
- **Found during:** Task 2 ruff format check
- **Issue:** `.to(device).eval()` chain reformatted by ruff; also triggered mypy `arg-type` error
- **Fix:** Reverted to separate `model.to(device)` / `model.eval()` statements matching original style
- **Files modified:** sakana_d2l.py
- **Commit:** 1207004

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| d2l_probe.py exists | FOUND |
| test_d2l_probe.py exists | FOUND |
| d2l_config.py exists | FOUND |
| sakana_d2l.py exists | FOUND |
| commit 594e49e exists | FOUND |
| commit 1207004 exists | FOUND |
| test_d2l_probe.py min_lines 80 | 319 lines (PASS) |
