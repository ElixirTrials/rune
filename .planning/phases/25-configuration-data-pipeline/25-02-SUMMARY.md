---
phase: 25
plan: 02
subsystem: data-pipeline
tags: [data, humaneval, augmentation, trajectory, mining, stubs]
dependency_graph:
  requires: [25-01]
  provides: [generate_trajectory_dataset, augment_trajectories, d2l_mining stubs]
  affects: [phase-27-training-loop]
tech_stack:
  added: []
  patterns:
    - sys.modules injection for GPU/network dep mocking in tests
    - deferred imports (PLC0415) for optional heavy dependencies
    - asyncio.run + asyncio.gather for async LLM calls in sync context
key_files:
  modified:
    - libs/model-training/src/model_training/d2l_data.py
    - libs/model-training/tests/test_d2l_data.py
  created:
    - libs/model-training/src/model_training/d2l_mining.py
decisions:
  - sys.modules injection over unittest.mock.patch for deferred imports of GPU/network deps
  - augmentation_prompts as local list (not module-level constant) to satisfy ruff N806
  - type: ignore not needed for deferred OllamaProvider import when mypy runs CI-style (both libs included)
metrics:
  duration_seconds: 421
  completed_date: "2026-03-13T21:35:50Z"
  tasks_completed: 2
  files_modified: 3
  tests_added: 5
  tests_total: 52
requirements-completed: [DATA-04, DATA-05]
---

# Phase 25 Plan 02: Trajectory Generation and Augmentation Summary

HumanEval-backed trajectory dataset generation and LLM augmentation pipeline added to d2l_data.py, plus NotImplementedError stubs for GitHub mining.

## What Was Built

### Task 1: generate_trajectory_dataset + augment_trajectories (TDD)

Added two new functions to `libs/model-training/src/model_training/d2l_data.py`:

**`generate_trajectory_dataset(source, max_tasks)`**
- Deferred import of `datasets.load_dataset` (no module-level import)
- Loads `openai_humaneval` split="test" with `trust_remote_code=True`
- Produces records: `task_id` from dataset, `activation_text` = prompt only, `teacher_text` = prompt + canonical solution
- 164 HumanEval tasks available; `max_tasks` parameter for CI-fast slicing

**`augment_trajectories(trajectories, n_variants, model, ollama_base_url)`**
- Deferred imports of `asyncio` and `inference.ollama_provider.OllamaProvider`
- Three augmentation strategies: paraphrase, reorder/drop steps, rename variables
- Internal `_augment_one` / `_augment_all` async pattern with `asyncio.gather`
- CRITICAL: augmented records inherit source `task_id` — preserves split integrity

**Tests added** (5 new, all passing):
- `test_generate_trajectory_dataset_returns_records_with_required_fields`
- `test_generate_trajectory_dataset_task_id_starts_with_humaneval`
- `test_augment_trajectories_returns_n_variants_per_input`
- `test_augmented_records_inherit_source_task_id`
- `test_augmented_and_original_records_zero_task_id_leakage`

### Task 2: d2l_mining.py Stubs

Created `libs/model-training/src/model_training/d2l_mining.py`:

**`mine_pr_diff_chains(repo, max_prs, github_token)`**
- Documents: task_id=`pr_{repo}_{pr_number}`, steps=commit diffs, outcome=merged/closed
- Raises `NotImplementedError("Run on L4 VM with GITHUB_TOKEN")`

**`mine_issue_commit_chains(repo, max_issues, github_token)`**
- Documents: task_id=`issue_{repo}_{issue_number}`, steps=linked commits, outcome=closed/open
- Raises `NotImplementedError("Run on L4 VM with GITHUB_TOKEN")`

## Test Results

- Total tests: 52 passed, 1 xfailed, 1 xpassed
- New trajectory/augmentation tests: 10/10 pass (5 original d2l_data + 5 new)
- ruff check + ruff format: zero errors on all new/modified files
- mypy (CI-style, both libs): zero new errors (2 pre-existing errors in sakana_d2l.py unrelated to this plan)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Replaced `unittest.mock.patch` with `sys.modules` injection for deferred imports**
- **Found during:** Task 1 GREEN phase
- **Issue:** `patch("inference.ollama_provider.OllamaProvider", ...)` triggered an actual import of the inference module which fails in test environments without the full dependency chain. Similarly, `patch("datasets.load_dataset", ...)` failed because `datasets` wasn't in sys.modules yet.
- **Fix:** Added `_inject_fake_datasets_module()` and `_inject_fake_inference_modules()` helpers that create stub modules in `sys.modules` before the functions under test perform their deferred imports. This is the established project pattern (see MEMORY.md: "Test injection pattern: `sys.modules["package"] = ModuleType("package")` for faking GPU deps").
- **Files modified:** `libs/model-training/tests/test_d2l_data.py`
- **Commit:** 8d98179

**2. [Rule 1 - Bug] Renamed `_AUGMENTATION_PROMPTS` to `augmentation_prompts`**
- **Found during:** Task 1 verification (ruff check)
- **Issue:** ruff N806 flagged a SCREAMING_SNAKE_CASE variable inside a function body
- **Fix:** Renamed the local list to lowercase `augmentation_prompts`
- **Files modified:** `libs/model-training/src/model_training/d2l_data.py`
- **Commit:** 8d98179

## Commits

| Hash | Message |
|------|---------|
| 56ebcd3 | test(25-02): add failing tests for trajectory generation and augmentation |
| 8d98179 | feat(25-02): add generate_trajectory_dataset and augment_trajectories to d2l_data.py |
| 1caa928 | feat(25-02): create d2l_mining.py stubs for GitHub trajectory extraction |

## Self-Check: PASSED

- FOUND: libs/model-training/src/model_training/d2l_data.py
- FOUND: libs/model-training/src/model_training/d2l_mining.py
- FOUND: libs/model-training/tests/test_d2l_data.py
- FOUND commit: 56ebcd3 (test RED phase)
- FOUND commit: 8d98179 (feat GREEN phase - d2l_data.py)
- FOUND commit: 1caa928 (feat - d2l_mining.py stubs)
