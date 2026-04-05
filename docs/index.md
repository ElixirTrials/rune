# Rune Documentation

Rune is a local-first coding agent that uses LoRA weight space as episodic memory. It implements a 5-phase template-driven pipeline (decompose → plan → code → integrate → diagnose/repair), parallel swarm orchestration, a Doc-to-LoRA hypernetwork, TIES/DARE adapter merging, and a flat adapter registry with lineage tracking.

## Core Subsystems

- **Pipeline** — 5-phase coding pipeline with 18 Jinja2 templates, per-phase iteration, DAG-ordered code execution, and two-step diagnose/repair. Entry: `scripts/rune_runner.py`
- **Adapter Registry** — SQLite + filesystem store for LoRA adapters with write-once enforcement, fitness queries, and lineage tracking. Entry: `libs/adapter-registry/`
- **Hypernetwork** — Perceiver-based Doc-to-LoRA hypernetwork generating rank-8 LoRA adapters in a single forward pass. Entry: `libs/model-training/`

## Documentation

- [Architecture: 5-Phase Pipeline](architecture/recursive-loop.md) — Pipeline phases, swarm execution, template system
- [Architecture: Adapter Storage](architecture/adapter-storage.md) — Registry schema, write-once policy, querying
- [Architecture: Monorepo Mapping](architecture/monorepo-mapping.md) — Component layout, integration points
- [Architecture: GPU Strategy](architecture/multi-gpu-strategy.md) — Multi-GPU coordination, pipeline parallelism
- [Swarm Architecture](swarm-architecture.md) — Fat orchestrator, training pool, evolution worker
- [Implementation Plan](implementation-plan.md) — Original phased build plan with current status annotations
- [Components Overview](components-overview.md) — All services and libraries
- [Risk Matrix](appendices/risk-matrix.md) — Primary research risks with mitigations
- [Build Order](appendices/build-order.md) — Component dependency chain
