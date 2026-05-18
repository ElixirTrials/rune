# Benchmark HPO: Optuna over Rune Pipeline Parameters

## Summary

Optimize Rune pipeline hyperparameters (scaling_factor, temperature, repetition_penalty, max retries) on failed MBPP problems using Optuna, then re-run the full benchmark with optimized params. Uses the real `run_phased_pipeline()` from `rune_runner.py` — no reimplementation.

## Problem Split

127 MBPP problems failed all 3 attempts in the initial run. Split 70/30 seed-deterministic:
- **Tuning set (~89 problems):** Optuna trials sample from these
- **Validation set (~38 problems):** Held out, evaluated once after HPO with best params

The MBPP description bug is now fixed (was reading `"text"` instead of `"prompt"` from sanitized split), so all problems get proper natural language descriptions.

## Search Space

| Parameter | Range | Type | Current | Mechanism |
|-----------|-------|------|---------|-----------|
| `scaling_factor` | [0.02, 0.50] | log-uniform | 0.16 | Passed to `run_hypernetwork()` |
| `temperature` | [0.1, 0.7] | uniform | 0.25 | `RUNE_TEMPERATURE` env var |
| `repetition_penalty` | [1.0, 1.3] | uniform | 1.04 | `RUNE_REPETITION_PENALTY` env var |
| `max_phase_iterations` | [2, 6] | int | 5 | `RUNE_MAX_PHASE_ITERATIONS` env var |

Template style is not in the search space — the pipeline uses the Jinja2 templates (code.j2, code_retry.j2, diagnose.j2, etc.) which are the real templates.

## Architecture

### New script: `scripts/optimization/run_benchmark_hpo.py`

```
1. Load MBPP problems, filter to failed IDs, split tuning/validation
2. Create Optuna study (TPE sampler, SQLite persistence)
3. Per trial:
   a. Suggest hyperparameters
   b. Set env vars (RUNE_TEMPERATURE, etc.)
   c. Sample N problems from tuning set
   d. For each problem:
      - Call run_phased_pipeline(project_prompt=problem.prompt, ...)
      - Extract accumulated_code from result
      - Run MBPP test assertions in sandbox
      - Log per-problem verdict to MLflow
   e. Return pass rate as objective
4. After HPO: evaluate best params on validation set
5. Save best params to PipelineConfig
```

### Condition (v) in `run_all_conditions.py`

Delete `run_condition_rune_iterative()` (~300 lines). Replace with a function that calls `run_phased_pipeline()` per benchmark problem and scores with the benchmark adapter's `score()` method.

### MLflow Logging

**Experiment name:** `benchmark-hpo-mbpp`

#### Study-level run (parent)

| Key | Type | Example |
|-----|------|---------|
| `study_name` | param | `mbpp-hpo-20260518` |
| `sampler` | param | `TPE` |
| `n_trials` | param | `30` |
| `problems_per_trial` | param | `8` |
| `tuning_set_size` | param | `89` |
| `validation_set_size` | param | `38` |
| `seed` | param | `42` |
| `best_trial_number` | metric | `17` |
| `best_pass_rate` | metric | `0.625` |
| `validation_pass_rate` | metric | `0.553` |
| `tuning_vs_validation_gap` | metric | `0.072` |
| `total_wall_time_s` | metric | total HPO duration |

Best params logged as `best/scaling_factor`, `best/temperature`, etc.

#### Per-trial run (nested under parent)

| Key | Type | Example |
|-----|------|---------|
| `trial_number` | param | `7` |
| `scaling_factor` | param | `0.12` |
| `temperature` | param | `0.35` |
| `repetition_penalty` | param | `1.08` |
| `max_phase_iterations` | param | `4` |
| `pass_rate` | metric | `0.625` (5/8) |
| `n_passed` | metric | `5` |
| `n_problems` | metric | `8` |
| `wall_time_s` | metric | trial duration |
| `mean_attempts_used` | metric | avg code attempts across problems |
| `diagnose_fire_rate` | metric | fraction of problems where diagnose phase activated |

#### Per-problem metrics (logged as step metrics within trial run)

For each problem in the trial, log with step=problem_index:

| Key | Type | Description |
|-----|------|-------------|
| `problem/{problem_id}/passed` | metric | 1 or 0 |
| `problem/{problem_id}/code_attempts` | metric | number of code generation attempts |
| `problem/{problem_id}/diagnose_fired` | metric | 1 if diagnose phase activated |
| `problem/{problem_id}/n_subtasks` | metric | number of subtasks from decompose |
| `problem/{problem_id}/wall_time_s` | metric | per-problem pipeline duration |
| `problem/{problem_id}/accumulated_code_len` | metric | len(accumulated_code) |
| `problem/{problem_id}/error` | param | first 500 chars of error if failed |

#### Phase-level detail (from `run_phased_pipeline()` return dict)

The pipeline returns `phases` (list of phase dicts) and `evolution` stats. Extract and log:

| Key | Type | Description |
|-----|------|-------------|
| `problem/{id}/phase_decompose_ok` | metric | 1 if decompose succeeded |
| `problem/{id}/phase_plan_ok` | metric | 1 if plan succeeded |
| `problem/{id}/phase_code_attempts` | metric | total code attempts across subtasks |
| `problem/{id}/phase_integrate_ok` | metric | 1 if integrate succeeded |
| `problem/{id}/evolution_sweeps` | metric | number of evolution sweeps run |
| `problem/{id}/adapters_generated` | metric | total adapters produced |

#### Artifacts

- `best_params.json` — best hyperparameters dict
- `study.db` — full Optuna SQLite database
- `validation_results.json` — per-problem pass/fail on validation set
- `trial_summary.csv` — all trials with params + pass rates

## What Gets Deleted

- `run_condition_rune_iterative()` in `run_all_conditions.py` (~lines 426-734)
- All supporting closures: `_generate_text()`, `_generate_adapter()`, `_extract_error_summary()`, `_safe_id()`
- The monkey-patch of `HyperLoRA.forward` in that function (the real pipeline handles this in `rune_runner.py`)
- The `--rune-one-shot` CLI flag and one-shot codepath

## What Stays

- `pregenerate_rune_adapters()` — still needed for gates (one-shot pregeneration)
- `run_optimization.py` — existing optimizer for the custom task pool
- `rune_runner.py` — the real pipeline, untouched
- MBPP description fix in `mbpp.py`

** MAKE SURE NOT TO DUPLICATE EXISTING CODE ** 
** MAKE SURE NOT TO RE-INVENT THE WHEEL **
** KISS, TDD, YAGNI, Clean Code **
** PASS RUFF, MYPY, PYTEST TO BE DONE **
