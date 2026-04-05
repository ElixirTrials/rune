# Components Overview

This page provides an overview of all microservices and shared libraries in this monorepo.

## Available Components

| Component | Description | Status | Documentation |
| :--- | :--- | :--- | :--- |
| **scripts/** | Fat orchestrator layer: 5-phase pipeline (`rune_runner.py`), swarm orchestrator (`swarm.py`), training pool, evolution worker. | Implemented (primary execution path) | -- |
| **adapter-registry** | SQLite + filesystem store for LoRA adapter metadata with write-once enforcement, fitness queries, and lineage tracking. | Implemented | [API Reference](adapter-registry/api/index.md) |
| **api-service** | FastAPI orchestrator providing HTTP endpoints for adapter management, session tracking, and service coordination. | Stubs (domain endpoints return 501; health checks work) | [API Reference](api-service/api/index.md) |
| **evaluation** | Adapter benchmarking with OOD testing, Pass@k metrics, fitness scoring, and generalization delta computation. | Implemented | [API Reference](evaluation/api/index.md) |
| **events-py** | Shared event envelope shapes (created/updated/deleted) and helpers used by Python services. | Implemented | [API Reference](events-py/api/index.md) |
| **evolution-svc** | Adapter evaluation, evolution, promotion, and pruning service. REST stubs; evolution logic in `scripts/swarm_evolution.py`. | Stubs (endpoints return 501; logic in `scripts/swarm_evolution.py`) | [API Reference](evolution-svc/api/index.md) |
| **inference** | Provider-agnostic inference interface with TransformersProvider, LlamaCppProvider, OllamaProvider, and VLLMProvider backends. | Implemented | [API Reference](inference/api/index.md) |
| **model-training** | DocToLoraHypernetwork, D2L training pipeline, TIES/DARE adapter merging, QLoRA fine-tuning, and PEFT utilities. | Implemented | [API Reference](model-training/api/index.md) |
| **rune-agent** | LangGraph state graph implementing the recursive generate → execute → reflect coding loop with single-iteration mode for pipeline integration. | Implemented (LangGraph workflow) | [API Reference](rune-agent/api/index.md) |
| **shared** | Hardware probe, sandbox (SubprocessBackend), checkpoint DB, Jinja2 template loader, Rune data models, and storage utilities. | Implemented | [API Reference](shared/api/index.md) |
| **training-svc** | FastAPI service for LoRA fine-tuning and hypernetwork training jobs (POST /train/lora, POST /train/hypernetwork, GET /jobs/{id}). | Implemented (all endpoints functional) | [API Reference](training-svc/api/index.md) |
