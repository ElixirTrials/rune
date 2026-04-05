# Monorepo Service Mapping

## Overview

Rune is built within an existing monorepo that provides shared infrastructure (event system, API scaffolding, model training utilities, data pipeline). This document maps each new Rune component to its position in the monorepo, identifies which existing services it extends or runs alongside, and names the specific integration points.

For the component build order and dependency chain, see [Build Order](../appendices/build-order.md).

---

## Service Mapping

### New Services

| Rune Service | Path | Extends / Runs Alongside | Integration Points |
|-------------|------|--------------------------|-------------------|
| `rune-agent` | `services/rune-agent/` | **Implemented** — LangGraph state graph (generate → execute → reflect) with 5-phase pipeline (decompose → plan → code → integrate → diagnose/repair). Not a REST API; runs as a LangGraph workflow. | Consumes `libs/adapter-registry` for adapter selection; uses `libs/inference` providers for generation; uses `libs/events-py` for event publishing; manages sandbox containers via `libs/shared` |
| `training-svc` | `services/training-svc/` | **Implemented** — FastAPI extending `libs/model-training`. POST `/train/lora`, POST `/train/hypernetwork`, GET `/jobs/{id}` all work. | Consumes PEFT utilities and hypernetwork from `model-training`; reads adapter corpus from `adapter-registry`; coordinates GPU via vLLM sleep/wake REST calls managed by `scripts/swarm_workers.py` |
| `evolution-svc` | `services/evolution-svc/` | **Stubs** — all 4 endpoints return 501; real logic lives in `scripts/swarm_evolution.py` | Reads adapter metadata from `adapter-registry`; evaluates adapter fitness using held-out test sets; writes promotion/pruning events via `libs/events-py` |
| `api-service` | `services/api-service/` | **Stubs** — all 6 domain endpoints return 501; only health/ready endpoints work | Intended REST interface for adapter registry queries and session state |

### New Libraries

| Rune Library | Path | Extends / New | Consumers |
|-------------|------|---------------|-----------|
| `adapter-registry` | `libs/adapter-registry/` | New (implemented) | `rune-agent`, `training-svc`, `evolution-svc`, `api-service` |

### Extended Existing Components

| Component | Path | What Changes |
|-----------|------|-------------|
| `model-training` | `libs/model-training/` | Hypernetwork (DocToLoraHypernetwork), D2L training pipeline (d2l_train, d2l_data, d2l_probe, d2l_config, d2l_lora, d2l_prep, d2l_mining), TIES/DARE merging (merging.py), QLoRA trainer, PEFT utilities, Sakana D2L integration |
| `api-service` | `services/api-service/` | REST routes defined for `/adapters` (registry CRUD), `/sessions` (agent session state); domain endpoints are stubs returning 501, only health/ready work |
| `inference` | `libs/inference/` | Provider-agnostic interface (InferenceProvider ABC) with TransformersProvider, LlamaCppProvider, OllamaProvider, VLLMProvider backends and factory for configuration-based selection |
| `shared` | `libs/shared/` | Hardware probe, sandbox (SubprocessBackend), checkpoint DB, template loader, Rune data models (CodingSession, SwarmConfig, PipelinePhase), storage utils |
| `evaluation` | `libs/evaluation/` | OOD benchmark, fitness scoring, Pass@k metrics, generalization delta |

---

## Integration Point Details

### adapter-registry (dependency root)

Every Rune component depends on the adapter registry. It provides two interfaces:

| Interface | Protocol | Consumers |
|-----------|----------|-----------|
| Python API (`adapter_registry.registry.AdapterRegistry`) | Direct import (in-process) | `rune-agent`, `training-svc`, `evolution-svc` |
| REST API (via `api-service`) | HTTP | External tools, UI, monitoring |

The registry owns the SQLite database and the filesystem adapter store. Key exceptions: `AdapterAlreadyExistsError`, `AdapterNotFoundError`. See [Adapter Storage](adapter-storage.md) for schema and path conventions.

### GPU Coordination (vLLM sleep/wake)

GPU coordination uses vLLM sleep/wake REST calls managed by `scripts/swarm_workers.py`. When a training job needs the GPU, the worker puts vLLM to sleep, runs QLoRA in a subprocess, then wakes vLLM:

```mermaid
flowchart LR
    TrainReq([Training Request]) --> Sleep[POST /sleep to vLLM]
    Sleep --> Train[QLoRA in subprocess]
    Train --> Wake[POST /wake_up to vLLM]
    Wake --> Resume[Inference resumes]
```

See [GPU Strategy](multi-gpu-strategy.md) for the full GPU coordination protocol.

### rune-agent <-> inference providers

The agent uses the `InferenceProvider` interface from `libs/inference/` for generation. The provider is selected via configuration-based factory, supporting multiple backends:

| Field | Value |
|-------|-------|
| Interface | `InferenceProvider` ABC (`libs/inference/`) |
| Backends | `TransformersProvider`, `LlamaCppProvider`, `OllamaProvider`, `VLLMProvider` |
| Selection | Factory-based, driven by configuration |
| Concurrency | Single-tenant (one agent session at a time in v1) |

### evolution-svc <-> adapter-registry (lifecycle)

The evolution service reads adapter metadata, evaluates fitness on held-out tests, and writes lifecycle events. Note: evolution logic primarily lives in `scripts/swarm_evolution.py`, not in the service endpoints (which are stubs).

| Operation | Description |
|-----------|-------------|
| Evaluate | Run held-out tests against adapter, compute pass rate |
| Promote | Move high-fitness task adapter to domain level |
| Prune | Mark low-fitness adapters as archived (not deleted — write-once) |
| Merge | Combine overlapping adapters into a new composite adapter |

### scripts/ (fat orchestrator)

The `scripts/` directory is the primary execution layer, collapsing the microservice architecture into single-process orchestration:

| Script | Role |
|--------|------|
| `rune_runner.py` | 5-phase pipeline: decompose → plan → code → integrate → diagnose/repair |
| `swarm.py` | Multi-agent orchestrator: agents + training pool + evolution + watchdog |
| `swarm_workers.py` | Training pool manager: QLoRA in subprocess, vLLM sleep/wake |
| `swarm_evolution.py` | Evolution worker: TIES/DARE merge, pruning, lineage tracking |
| `e2e_test.py` | End-to-end test exercising full pipeline |
| `e2e_benchmark.py` | End-to-end benchmark with Gemma |
| `e2e_inference_smoke.py` | Inference smoke test |
| `e2e_training_smoke.py` | Training smoke test |
| `benchmark_challenging.py` | 3-task end-to-end benchmark |
| `compare_output.py` | Output comparison tool |
| `mine_github.py` | GitHub training data mining |
| `demo_project.py` / `demo_run.py` | Demo scripts |
| `bootstrap.py` | Path setup for scripts importing from libs/ |
| `eval/run_benchmarks.py` | Coding benchmark evaluation (HumanEval+, MBPP+, BigCodeBench) |
| `eval/generate_completions.py` | Completion generation for benchmark evaluation |
| `eval/config.py` | Benchmark evaluation configuration |
| `optimization/run_optimization.py` | Bayesian parameter optimization (Optuna TPE) |
| `optimization/scoring.py` | Fitness scoring for optimization trials |
| `optimization/task_pool.py` | Diverse task sampling for optimization |
| `optimization/template_library.py` | Prompt/trajectory style variants |
| `experiment_harness.py` | Isolated adapter/prompt experiments (~15s/trial) |

---

## Existing Services Not Modified

These existing monorepo services are not modified by Rune and continue operating independently:

| Service / Library | Role | Rune Relationship |
|------------------|------|-------------------|
| `libs/shared` | Extended: hardware.py, checkpoint_db.py, sandbox.py, template_loader.py, rune_models.py, storage_utils.py | Consumed by scripts, services, and other libs |
| `libs/evaluation` | Extended: ood_benchmark.py, metrics.py | Used by `evolution-svc` and `scripts/swarm_evolution.py` for fitness evaluation |

---

## Monorepo Layout (Post-Rune)

```
rune/
  scripts/                  # Fat orchestrator layer
    rune_runner.py          # 5-phase pipeline (decompose → plan → code → integrate → diagnose/repair)
    swarm.py                # Multi-agent orchestrator
    swarm_workers.py        # Training pool, GPU coordination
    swarm_evolution.py      # TIES/DARE merge, pruning
    e2e_test.py             # End-to-end test
    e2e_benchmark.py        # End-to-end benchmark with Gemma
    e2e_inference_smoke.py  # Inference smoke test
    e2e_training_smoke.py   # Training smoke test
    mine_github.py          # GitHub training data mining
    compare_output.py       # Output comparison tool
    eval/                   # Coding benchmark evaluation
      run_benchmarks.py     # HumanEval+, MBPP+, BigCodeBench runner
      generate_completions.py
      config.py
    optimization/           # Bayesian parameter optimization (Optuna)
      run_optimization.py   # Overnight TPE search
      scoring.py            # Fitness scoring
      task_pool.py          # Diverse task sampling
      template_library.py   # Prompt/trajectory variants
    experiment_harness.py   # Isolated adapter experiments
  services/
    api-service/            # REST API (stubs — domain endpoints return 501)
    rune-agent/             # LangGraph workflow: generate → execute → reflect
    training-svc/           # LoRA and hypernetwork training jobs (FastAPI, implemented)
    evolution-svc/          # Adapter lifecycle endpoints (FastAPI, stubs — logic in scripts/)
  libs/
    adapter-registry/       # SQLite + filesystem adapter store
    model-training/         # Hypernetwork, D2L pipeline, TIES/DARE, trainer
    inference/              # Provider-agnostic: Transformers, llama.cpp, Ollama, vLLM
    shared/                 # Hardware, sandbox, templates, models, checkpoint DB
    evaluation/             # OOD benchmark, Pass@k, fitness scoring
    events-py/              # Event envelope and helpers
  docs/                     # MkDocs documentation
```
