# Adapter Storage Strategy

## Overview

Every LoRA adapter produced by Rune — whether by direct fine-tuning or hypernetwork inference — is stored as a versioned `.safetensors` file on the local filesystem with metadata in SQLite. This document specifies the filesystem path convention, the SQLite schema, and the write-once enforcement policy that prevents catastrophic forgetting.

For the component that implements this storage, see [Monorepo Mapping](monorepo-mapping.md) (`libs/adapter-registry`). For how adapters are produced, see [Recursive Loop](recursive-loop.md).

---

## Filesystem Path Convention

Adapters are stored under a configurable root directory (default: `~/.rune/adapters/`). The path encodes the adapter's position in the three-level hierarchy and ensures uniqueness via version identifiers.

```
~/.rune/adapters/
  project/
    {project_id}/
      v{version}/
        adapter.safetensors
        adapter_config.json
  domain/
    {domain_slug}/
      v{version}/
        adapter.safetensors
        adapter_config.json
  task/
    {task_type}/
      {adapter_id}/
        adapter.safetensors
        adapter_config.json
```

### Path Components

| Component | Format | Example |
|-----------|--------|---------|
| Hierarchy level | `project`, `domain`, `task` | `task/` |
| Project ID | Slugified project name | `rune-core` |
| Domain slug | Kebab-case domain descriptor | `python-async`, `fastapi-patterns` |
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

### adapters table

```sql
CREATE TABLE adapters (
    id              TEXT PRIMARY KEY,   -- UUID v4
    version         INTEGER NOT NULL,   -- Monotonically increasing per lineage
    task_type       TEXT NOT NULL,       -- e.g. 'bug-fix', 'feature-impl'
    hierarchy_level TEXT NOT NULL,       -- 'project', 'domain', or 'task'
    domain          TEXT,               -- Domain slug (NULL for task-level)
    project_id      TEXT,               -- Project ID (NULL for task/domain-level)
    base_model_id   TEXT NOT NULL,       -- e.g. 'Qwen/Qwen2.5-Coder-7B-Instruct'
    rank            INTEGER NOT NULL,    -- LoRA rank (e.g. 64)
    created_at      TEXT NOT NULL,       -- ISO 8601 timestamp
    file_path       TEXT NOT NULL,       -- Relative path from adapter root
    file_hash       TEXT NOT NULL,       -- SHA-256 of adapter.safetensors
    file_size_bytes INTEGER NOT NULL,    -- Size of adapter.safetensors
    pass_rate       REAL,               -- 0.0-1.0, NULL if not yet evaluated
    fitness_score   REAL,               -- Composite fitness (pass_rate + generalization)
    source          TEXT NOT NULL,       -- 'fine-tune' or 'hypernetwork'
    trajectory_id   TEXT,               -- FK to trajectories table
    session_id      TEXT NOT NULL,       -- Session that produced this adapter
    is_archived     INTEGER DEFAULT 0,  -- 1 if pruned by evolution-svc
    parent_id       TEXT,               -- ID of adapter this was promoted from
    UNIQUE(file_path)
);
```

### trajectories table

```sql
CREATE TABLE trajectories (
    id              TEXT PRIMARY KEY,   -- UUID v4
    task_id         TEXT NOT NULL,       -- External task identifier
    task_type       TEXT NOT NULL,       -- Matches adapter task_type
    attempt_count   INTEGER NOT NULL,    -- Number of generate-execute-reflect cycles
    outcome         TEXT NOT NULL,       -- 'success' or 'exhausted'
    created_at      TEXT NOT NULL,       -- ISO 8601 timestamp
    trajectory_json TEXT NOT NULL        -- Full trajectory record (see Recursive Loop doc)
);
```

### Indexes

```sql
CREATE INDEX idx_adapters_task_type ON adapters(task_type);
CREATE INDEX idx_adapters_hierarchy ON adapters(hierarchy_level);
CREATE INDEX idx_adapters_fitness ON adapters(fitness_score DESC);
CREATE INDEX idx_adapters_session ON adapters(session_id);
CREATE INDEX idx_adapters_created ON adapters(created_at);
CREATE INDEX idx_adapters_archived ON adapters(is_archived);
```

---

## Write-Once Enforcement Policy

Adapter immutability is a correctness requirement, not a convention. The write-once policy prevents catastrophic forgetting — an existing adapter must never be overwritten by a new one. This is enforced at two levels.

### Storage API Level

The `adapter_registry` library exposes a `store_adapter()` function that is the only code path for writing adapters. This function enforces:

1. **ID uniqueness**: If `id` already exists in SQLite, the write is rejected with an error (not silently ignored).
2. **Path uniqueness**: If `file_path` already exists on the filesystem, the write is rejected.
3. **No update API**: There is no `update_adapter()` or `overwrite_adapter()` function. The API surface makes overwrites impossible by omission.

### Lifecycle Operations

| Operation | Allowed | Mechanism |
|-----------|---------|-----------|
| Create new adapter | Yes | `store_adapter()` — new ID, new path |
| Read adapter metadata | Yes | `get_adapter()`, `query_adapters()` |
| Read adapter weights | Yes | Load `.safetensors` from `file_path` |
| Overwrite existing adapter | No | Rejected by storage API |
| Delete adapter file | No | Not exposed in API |
| Archive adapter (soft delete) | Yes | `archive_adapter()` sets `is_archived = 1` |
| Update fitness score | Yes | `update_fitness()` — metadata-only, weights unchanged |
| Update pass rate | Yes | `update_pass_rate()` — metadata-only, weights unchanged |

Metadata fields (`pass_rate`, `fitness_score`, `is_archived`) are mutable because they describe the adapter's evaluation state, not its weights. The weight file and its hash are immutable after creation.

### New Versions, Not Overwrites

When the evolution operator promotes a task-level adapter to domain level, it creates a new adapter entry at the domain level with a `parent_id` pointing to the source. The original task-level adapter remains unchanged. The version field tracks lineage — `v2` of a domain adapter is a successor to `v1`, but `v1` still exists and is still queryable.

---

## Querying Without GPU

A design requirement is that adapter metadata must be queryable without loading weights into GPU memory. The SQLite database stores all metadata needed for adapter selection, fitness ranking, and lifecycle management. The only operation that requires GPU memory is loading `adapter.safetensors` into the vLLM serving process for inference.

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
