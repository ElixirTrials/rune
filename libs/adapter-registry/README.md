# adapter-registry

SQLite + filesystem store for LoRA adapter metadata with write-once enforcement.

## Key Classes

- **`AdapterRecord`** (`models.py`) — SQLModel table model tracking adapter metadata: `id`, `version`, `task_type`, `base_model_id`, `rank`, `created_at`, `file_path`, `file_hash`, `file_size_bytes`, `pass_rate`, `fitness_score`, `source`, `session_id`, `is_archived`, `parent_ids`, `generation`, `training_task_hash`, `agent_id`
- **`AdapterRegistry`** (`registry.py`) — CRUD operations backed by SQLite via SQLModel

## Key Methods

| Method | Description |
|--------|-------------|
| `store(record)` | Store a new adapter (raises `AdapterAlreadyExistsError` on duplicate) |
| `retrieve_by_id(id)` | Get adapter by ID (raises `AdapterNotFoundError`) |
| `query_by_task_type(type)` | All adapters matching a task type |
| `query_best_for_task(type, top_k)` | Top-k by fitness score |
| `list_all()` | All non-archived adapters |
| `archive(id)` | Soft-delete (set `is_archived=True`) |
| `update_fitness(id, pass_rate, fitness_score)` | Update evaluation metrics |
| `is_task_solved(task_hash, threshold)` | Check if task already solved |
| `get_lineage(id)` | Walk parent_ids chain |
| `query_unevaluated(task_type)` | Adapters with no pass_rate |
| `get_task_types()` | All distinct task types |

## Exceptions

- `AdapterAlreadyExistsError` — Duplicate adapter ID on `store()`
- `AdapterNotFoundError` — Missing adapter on `retrieve_by_id()` or `archive()`

## Write-Once Policy

Weight files (`.safetensors`) and their hashes are immutable after creation. Metadata fields (`pass_rate`, `fitness_score`, `is_archived`) are mutable. No `update()` or `overwrite()` method exists.

## Usage

```python
from sqlalchemy import create_engine
from adapter_registry import AdapterRegistry, AdapterRecord

engine = create_engine("sqlite:///adapters.db")
registry = AdapterRegistry(engine=engine)

record = AdapterRecord(
    id="adapter-001", version=1, task_type="bug-fix",
    base_model_id="Qwen/Qwen2.5-Coder-7B", rank=16,
    created_at="2026-01-01T00:00:00Z",
    file_path="/adapters/adapter-001.safetensors",
    file_hash="abc123", file_size_bytes=1024,
    source="distillation", session_id="sess-001",
)
registry.store(record)
```
