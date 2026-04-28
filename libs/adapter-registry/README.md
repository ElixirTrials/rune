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

## Adapter ID Conventions

| Adapter class | ID pattern | Set by |
|---------------|------------|--------|
| Standard (QLoRA / round-1 hypernet output) | `<uuid>` | Training pipeline |
| Oracle adapters | `oracle_<bin_key>` | `libs/corpus-producer/src/corpus_producer/trainer_bridge.py` |
| Round-2 hypernetwork adapters | `round2_<uuid[:8]>` | `libs/model-training/src/model_training/round2_train.py::register_round2_adapter` |

`bin_key` is `<phase>_<benchmark>` (e.g., `code_humaneval`, `plan_mbpp`) or `diagnose_pooled` — 25 bins total across 4 pipeline phases × 6 benchmarks plus one pooled diagnose bin.

### Reserved `task_type` values

| Value | Meaning |
|-------|---------|
| `round2_hypernet` | Adapter produced by the round-2 oracle-teacher distillation loop |
| `oracle_<bin_key>` patterns | Per-bin oracle adapters trained for round-2 teaching |

### Lineage semantics

- `generation`: round-2 adapters set `generation=2`. Standard round-1 adapters are `generation=1` (or unset). Evolution-produced merges follow their own generational counter.
- `parent_ids`: for round-2 adapters, `parent_ids = json.dumps(sorted(oracle_ids))` — the sorted list of teacher oracle IDs used during training. For merged adapters, this stores the source adapter IDs. `get_lineage(id)` walks this chain.

## Usage

```python
from sqlalchemy import create_engine
from adapter_registry import AdapterRegistry, AdapterRecord

engine = create_engine("sqlite:///adapters.db")
registry = AdapterRegistry(engine=engine)

record = AdapterRecord(
    id="adapter-001", version=1, task_type="bug-fix",
    base_model_id="Qwen/Qwen3.5-9B", rank=16,
    created_at="2026-01-01T00:00:00Z",
    file_path="/adapters/adapter-001.safetensors",
    file_hash="abc123", file_size_bytes=1024,
    source="distillation", session_id="sess-001",
)
registry.store(record)
```
