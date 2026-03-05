---
phase: 21-qlora-training-pipeline
plan: 01
subsystem: model-training
tags: [qlora, peft, lora, bitsandbytes, trl, transformers, datasets, training, adapter-registry]

# Dependency graph
requires:
  - phase: 18-adapter-registry
    provides: AdapterRegistry.store() and AdapterRecord model for registering trained adapters
  - phase: 20-agent-loop
    provides: trajectory.py load_trajectory/format_for_sft used by train_qlora orchestrator
provides:
  - build_qlora_config() returning peft.LoraConfig with CAUSAL_LM and NF4 quantization settings
  - apply_lora_adapter() wrapping base model with LoRA via get_peft_model
  - get_training_config() returning complete hyperparameter dict with locked values
  - validate_config() with range checks for rank/epochs/learning_rate
  - train_qlora() orchestrating trajectory load -> SFT dataset -> QLoRA training -> save adapter
  - train_and_register() wrapping train_qlora with AdapterRegistry.store() and file hash/size metadata
  - GPU deps declared in model-training pyproject.toml
  - mypy overrides for bitsandbytes and trl in root pyproject.toml
affects: [21-qlora-training-pipeline, training-svc]

# Tech tracking
tech-stack:
  added: [peft>=0.18.0, bitsandbytes>=0.45.0, trl>=0.16.0, datasets>=3.0.0, transformers>=4.47.0]
  patterns:
    - "Deferred GPU imports: all peft/transformers/trl/torch imports inside function bodies for CPU-only importability (INFRA-05)"
    - "sys.modules injection for testing deferred imports without real GPU packages"
    - "xfail(strict=False) for GPU-dependent tests that cannot run in CPU CI"

key-files:
  created:
    - libs/model-training/src/model_training/trainer.py
    - libs/model-training/tests/test_trainer.py
  modified:
    - libs/model-training/src/model_training/peft_utils.py
    - libs/model-training/src/model_training/config.py
    - libs/model-training/src/model_training/__init__.py
    - libs/model-training/pyproject.toml
    - pyproject.toml
    - libs/model-training/tests/test_peft_utils.py
    - libs/model-training/tests/test_config.py

key-decisions:
  - "bfloat16 compute dtype set in BitsAndBytesConfig, NOT in LoraConfig — LoraConfig is purely about LoRA layer structure"
  - "LoraConfig has no modules_to_save — would break vLLM adapter loading"
  - "sys.modules injection pattern chosen for mocking deferred GPU imports (unittest.mock.patch('peft.get_peft_model') fails without the module installed)"
  - "train_and_register reads RUNE_ADAPTER_DIR/RUNE_BASE_MODEL/RUNE_DATABASE_URL inside function body for monkeypatch testability"

patterns-established:
  - "Deferred GPU imports: from <gpu_lib> import X inside function body, never at module level"
  - "sys.modules injection: inject fake ModuleType before calling function that defers imports, restore after"
  - "GPU-heavy tests: @pytest.mark.xfail(strict=False) to allow both CI pass and GPU-enabled pass"

requirements-completed: [TRAIN-03, TRAIN-04, TRAIN-05, INFRA-04, INFRA-05]

# Metrics
duration: 25min
completed: 2026-03-05
---

# Phase 21 Plan 01: QLoRA Training Pipeline Implementation Summary

**QLoRA training pipeline with NF4 quantization, SFTTrainer orchestration, and AdapterRegistry integration — all GPU imports deferred for CPU-only importability**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-05T20:00:00Z
- **Completed:** 2026-03-05T20:25:00Z
- **Tasks:** 2 completed
- **Files modified:** 9

## Accomplishments

- Implemented `build_qlora_config()` and `apply_lora_adapter()` with deferred peft imports
- Implemented `get_training_config()` with all locked hyperparameters and `validate_config()` with range checks
- Created `train_qlora()` orchestrator: load_trajectory -> format_for_sft -> Dataset.from_list -> BitsAndBytesConfig -> SFTTrainer -> save_model
- Created `train_and_register()` wrapping full pipeline with file hash/size computation and AdapterRegistry.store()
- All 5 GPU dep packages declared in model-training pyproject.toml; bitsandbytes/trl mypy overrides added
- 23 tests pass, 2 xfail (expected GPU-only tests); CPU-only importability confirmed for all 4 public modules

## Task Commits

Each task was committed atomically:

1. **Task 1: peft_utils, config, GPU deps, mypy overrides, tests** - `8a60a22` (feat)
2. **Task 2: train_qlora orchestrator, __init__.py, trainer tests** - `2e3a1e2` (feat)

**Plan metadata:** *(pending final commit)*

## Files Created/Modified

- `libs/model-training/src/model_training/peft_utils.py` - build_qlora_config and apply_lora_adapter with deferred peft imports
- `libs/model-training/src/model_training/config.py` - get_training_config and validate_config implementations
- `libs/model-training/src/model_training/trainer.py` - NEW: train_qlora and train_and_register orchestrators
- `libs/model-training/src/model_training/__init__.py` - Updated comment documenting GPU deferred import pattern
- `libs/model-training/pyproject.toml` - Added peft, bitsandbytes, transformers, trl, datasets dependencies
- `pyproject.toml` - Added bitsandbytes and trl mypy override blocks
- `libs/model-training/tests/test_peft_utils.py` - Green-phase tests with sys.modules injection
- `libs/model-training/tests/test_config.py` - Green-phase tests covering all validation paths
- `libs/model-training/tests/test_trainer.py` - NEW: importability, FileNotFoundError, ValueError tests

## Decisions Made

- **bfloat16 in BitsAndBytesConfig not LoraConfig**: bfloat16 compute dtype is about the quantized forward pass, not the LoRA layer geometry. Setting it in LoraConfig is wrong.
- **No modules_to_save in LoraConfig**: Would cause adapter to include embed_tokens/lm_head in the saved PEFT artifact, breaking vLLM adapter loading which expects only q_proj/v_proj weights.
- **sys.modules injection for GPU mock tests**: `unittest.mock.patch("peft.get_peft_model")` fails when `peft` is not installed. Injecting a `ModuleType` fake into `sys.modules` before calling the deferred-import function is the correct CPU-CI pattern.
- **RUNE_* env vars read inside function bodies**: Consistent with Phase 19/20 factory.py and trajectory.py patterns — enables monkeypatch.setenv() in tests.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_apply_lora_adapter_calls_get_peft_model using sys.modules injection**
- **Found during:** Task 1 (test_peft_utils.py)
- **Issue:** `unittest.mock.patch("peft.get_peft_model")` raises `ModuleNotFoundError: No module named 'peft'` in CPU CI because `patch` attempts to resolve the module before substituting it
- **Fix:** Replaced patch() with direct `sys.modules["peft"] = fake_peft` injection before calling `apply_lora_adapter`, with cleanup in finally block
- **Files modified:** `libs/model-training/tests/test_peft_utils.py`
- **Verification:** Test passes without peft installed
- **Committed in:** `8a60a22` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed ruff I001/E501 errors in trainer.py**
- **Found during:** Task 2 (ruff check)
- **Issue:** Import ordering (stdlib `torch` after third-party `datasets`/`transformers`) and 98-char comment line violated ruff rules
- **Fix:** Reordered deferred imports to stdlib-first (`torch`), then third-party (`datasets`, `transformers`, `trl`), then first-party. Shortened comment line.
- **Files modified:** `libs/model-training/src/model_training/trainer.py`
- **Verification:** `uv run ruff check` passes with "All checks passed!"
- **Committed in:** `2e3a1e2` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both fixes necessary for test correctness and linting compliance. No scope creep.

## Issues Encountered

None beyond the auto-fixed deviations above.

## User Setup Required

None - no external service configuration required. GPU hardware validation is a separate user concern (deferred per Phase 18-19 decisions).

## Next Phase Readiness

- model-training library is complete: peft_utils, config, trainer all implemented
- CPU-only importability confirmed — training-svc can import these without GPU in dev
- Ready for Phase 21-02 (training-svc endpoint that calls train_and_register via HTTP)
- GPU validation remains deferred to hardware-specific testing by user

---
*Phase: 21-qlora-training-pipeline*
*Completed: 2026-03-05*
