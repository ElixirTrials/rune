# HPO Trace & Pipeline Debug — 2026-05-19

## Problem Statement

User reported MLflow traces only appear at the end of the HPO benchmark run, not during. Investigation revealed the trace issue is a symptom of a deeper problem: the pipeline itself is crashing on nearly every problem.

## What We Found

### 1. Traces ARE reaching the server mid-run

Queried the MLflow tracking server (`http://localhost:5000`) directly via REST API. The currently running trial-030 has a live trace with 49 spans visible in the DB while still executing. Traces from finished trials of the previous HPO have timestamps matching their execution order (spread across the full 12-minute run), not bunched at the end.

### 2. The "only at end" perception is caused by mass failures

49/50 sampled traces from the previous completed HPO failed with `[Errno 32] Broken pipe`. Each failed pipeline run completed in ~1.2 seconds with only 1 span (the root span — the pipeline crashed before LangChain/LangGraph ran). Because every problem failed almost instantly, 240+ traces appeared within seconds of each other near the run's end, creating the appearance of a batch dump.

### 3. `flush_trace_async_logging()` is a no-op

The code calls `mlflow.flush_trace_async_logging()` after each problem (line 370), but `mlflow.config.enable_async_logging()` is never called. Per MLflow 3.12 docs, async logging must be explicitly enabled. Without it, traces are logged synchronously (which is actually fine — they reach the server immediately). The flush call is harmless but does nothing.

### 4. Broken pipe is the real blocker

Every pipeline run hits `[Errno 32] Broken pipe` almost immediately. The HPO's `except Exception` at `run_benchmark_hpo.py:339` catches this and records a failed verdict with 0 code attempts — the pipeline never reaches decompose/plan/code phases. This is why:
- All traces have `spans=1` (root only, no child spans from LangGraph)
- All `execution_time_ms` values are ~1200ms
- Pass rate is near zero

## What We Built

### `scripts/optimization/run_single_mbpp.py`

A minimal single-problem runner that calls the same `run_phased_pipeline()` as the HPO but with **no exception handling** — errors propagate with full tracebacks.

```bash
uv run python scripts/optimization/run_single_mbpp.py \
    --hypernet-checkpoint s3://elixirtrials-949678234935-eu-west-2-artifacts/checkpoints/hypernet_hpo/checkpoint.pt \
    --problem-id mbpp/429 \
    2>&1 | tee /tmp/rune_single.log
```

Options: `--problem-id` (specific problem or random), `--base-model` (default Qwen/Qwen3.5-9B), `--device` (default cuda).

## Next Steps

1. **Run `run_single_mbpp.py`** on a GPU node and capture the full Broken pipe traceback
2. **Fix the root cause** of the Broken pipe (likely inference provider connection — vLLM/transformers backend dying or not started)
3. **Verify** a single problem completes end-to-end before restarting HPO
4. Optionally add `mlflow.config.enable_async_logging()` before line 798 in `run_benchmark_hpo.py` for proper async trace flushing (low priority — sync logging works fine)

## Key Files

| File | Role |
|------|------|
| `scripts/optimization/run_benchmark_hpo.py` | HPO runner (Optuna + MLflow) |
| `scripts/optimization/run_single_mbpp.py` | New single-problem debug runner |
| `scripts/rune_runner.py` | Core pipeline (`run_phased_pipeline`) |
| `libs/inference/src/inference/transformers_provider.py` | Inference backend (likely where Broken pipe originates) |
| `libs/model-training/src/model_training/model_pool.py` | ModelPool (base model + hypernet resident in GPU memory) |

## MLflow Environment

- MLflow 3.12.0, tracking server at `http://localhost:5000`
- Experiment: `benchmark-hpo-mbpp` (id=28)
- Current running HPO: parent run `3fb18d0b1e8f` with trial-030 active
- Previous completed HPO: parent run `9cbeb3a33bd7` (30 trials, mostly failed)
