# training-svc

FastAPI service for LoRA fine-tuning and hypernetwork training jobs.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/train/lora` | Submit a LoRA fine-tuning job |
| POST | `/train/hypernetwork` | Submit a hypernetwork training job |
| GET | `/jobs/{job_id}` | Check job status |

## Job Model

Jobs are tracked in-memory via `JOB_STORE` with states: `pending` → `running` → `completed` / `failed`.

Request schemas:
- `LoraTrainingRequest` — session_id, adapter_id, task_type, rank, epochs, learning_rate
- `HypernetworkTrainingRequest` — hypernetwork training parameters

## Running

```bash
uv run uvicorn training_svc.main:app --reload
```

## Architecture

Training jobs run as background tasks via FastAPI's `BackgroundTasks`. GPU-dependent imports (`model_training.trainer`) are deferred inside job functions per INFRA-05 pattern, keeping the service importable in CPU-only environments.

Adapter IDs are validated against path-traversal characters before use in filesystem operations.
