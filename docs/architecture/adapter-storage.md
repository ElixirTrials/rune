# Adapter Storage Strategy

## Overview

Every LoRA adapter produced by Rune — whether by direct fine-tuning or hypernetwork inference — is stored as a versioned `.safetensors` file on the local filesystem with metadata in SQLite. This document specifies the filesystem path convention, the SQLite schema, and the write-once enforcement policy that prevents catastrophic forgetting.

For the component that implements this storage, see [Monorepo Mapping](monorepo-mapping.md) (`libs/adapter-registry`). For how adapters are produced, see [Recursive Loop](recursive-loop.md).

---

## Filesystem Path Convention

Adapters are stored under a configurable root directory (default: `~/.rune/adapters/`). The structure is flat — each adapter gets a directory named by its ID.

```
~/.rune/adapters/
  {adapter_id}/
    adapter.safetensors
    adapter_config.json
```

### Path Components

| Component | Format | Example |
|-----------|--------|---------|
| Task type | Kebab-case task category | `bug-fix`, `feature-impl`, `refactor` |
| Adapter ID | UUID v4 | `a1b2c3d4-e5f6-7890-abcd-ef1234567890` |
| Version | Monotonically increasing integer | `v1`, `v2`, `v3` |

### File Contents

| File | Format | Purpose |
|------|--------|---------|
| `adapter.safetensors` | safetensors | LoRA weight matrices (A and B) for all targeted layers |
| `adapter_config.json` | JSON | PEFT configuration: rank, alpha, target modules, base model ID |

The `adapter_config.json` follows the standard PEFT format, ensuring adapters are loadable by any PEFT-compatible tool without Rune-specific code.

---

## SQLite Schema

The adapter registry maintains a SQLite database at `~/.rune/adapter_registry.db`. This database is the source of truth for adapter metadata — the filesystem stores weights, SQLite stores everything else.

### AdapterRecord (table: adapter_records)

The schema is defined as a SQLModel class (`AdapterRecord`). Fields with `(indexed)` have `Field(index=True)` in the model definition.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID string |
| `version` | INTEGER NOT NULL | Lineage tracking |
| `task_type` | TEXT NOT NULL | e.g. 'bug-fix' (indexed) |
| `base_model_id` | TEXT NOT NULL | e.g. 'Qwen/Qwen2.5-Coder-7B' |
| `rank` | INTEGER NOT NULL | LoRA rank |
| `created_at` | TEXT NOT NULL | ISO 8601 |
| `file_path` | TEXT NOT NULL | Path to .safetensors |
| `file_hash` | TEXT NOT NULL | SHA-256 |
| `file_size_bytes` | INTEGER NOT NULL | |
| `pass_rate` | REAL | 0.0-1.0, NULL if unevaluated |
| `fitness_score` | REAL | Composite fitness, NULL if unevaluated |
| `source` | TEXT NOT NULL | 'distillation', 'evolution', 'manual' |
| `session_id` | TEXT NOT NULL | Session that produced this |
| `is_archived` | INTEGER DEFAULT 0 | Soft delete |
| `parent_ids` | TEXT | JSON list of parent adapter IDs |
| `generation` | INTEGER DEFAULT 0 | Evolutionary generation |
| `training_task_hash` | TEXT | Deduplication key (indexed) |
| `agent_id` | TEXT | Swarm agent identifier |

Indexing is handled by SQLModel via `Field(index=True)` on the model class — no separate index creation is needed.

---

## Write-Once Enforcement Policy

Adapter immutability is a correctness requirement, not a convention. The write-once policy prevents catastrophic forgetting — an existing adapter must never be overwritten by a new one. This is enforced at two levels.

### Storage API Level

The `adapter_registry` library exposes a `store()` method that is the only code path for writing adapters. This method enforces:

1. **ID uniqueness**: If `id` already exists in SQLite, the write is rejected with `AdapterAlreadyExistsError`.
2. **Path uniqueness**: If `file_path` already exists on the filesystem, the write is rejected.
3. **No update API**: There is no overwrite or update-weights method. The API surface makes overwrites impossible by omission.

### Lifecycle Operations

| Operation | Allowed | Mechanism |
|-----------|---------|-----------|
| Create new adapter | Yes | `store()` — new ID, new path |
| Read adapter metadata | Yes | `retrieve_by_id()`, `query_by_task_type()`, `query_best_for_task()`, `list_all()` |
| Read adapter weights | Yes | Load `.safetensors` from `file_path` |
| Overwrite existing adapter | No | Rejected by storage API with `AdapterAlreadyExistsError` |
| Delete adapter file | No | Not exposed in API |
| Archive adapter (soft delete) | Yes | `archive()` sets `is_archived = 1` |
| Update fitness score | Yes | `update_fitness()` — metadata-only, weights unchanged |

Metadata fields (`pass_rate`, `fitness_score`, `is_archived`) are mutable because they describe the adapter's evaluation state, not its weights. The weight file and its hash are immutable after creation.

### New Versions, Not Overwrites

When the evolution operator produces a new adapter from one or more parents, it creates a new adapter entry with `parent_ids` (a JSON list) pointing to the sources. The original adapters remain unchanged. The version field tracks lineage — version 2 of an adapter is a successor to version 1, but version 1 still exists and is still queryable.

---

## Querying Without GPU

A design requirement is that adapter metadata must be queryable without loading weights into GPU memory. The SQLite database stores all metadata needed for adapter selection, fitness ranking, and lifecycle management. The only operation that requires GPU memory is loading `adapter.safetensors` into the inference provider for inference.

This separation means the adapter registry can power dashboards, CLI tools, and batch evaluation scripts on CPU-only machines while the GPU machines handle serving and training.

---

## Storage Budget

At LoRA rank 64 targeting all attention layers of Qwen2.5-Coder-7B, a single adapter is approximately 50-200 MB depending on target modules. At the upper bound:

| Adapters | Storage |
|----------|---------|
| 100 | ~20 GB |
| 500 | ~100 GB |
| 1000 | ~200 GB |

This is well within local SSD capacity. The SQLite database remains small (< 10 MB) even at 1000 adapters — metadata is text and numbers, not weights.
