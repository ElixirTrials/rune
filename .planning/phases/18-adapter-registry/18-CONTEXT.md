# Phase 18: Adapter Registry - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the 4 CRUD methods in AdapterRegistry (store, retrieve_by_id, query_by_task_type, list_all) with SQLite persistence via SQLModel. This is the foundation hub — every other v5.0 component reads from or writes to the registry. The AdapterRecord model, exception classes, method signatures, and TDD tests already exist as wireframes.

</domain>

<decisions>
## Implementation Decisions

### Database Lifecycle
- DATABASE_URL env var for service discovery — standard pattern, each service reads it, creates engine, passes to AdapterRegistry(engine=engine)
- Default path: `~/.rune/registry.db` (user home, persists across projects)
- Auto-create tables on init — `SQLModel.metadata.create_all(engine)` in constructor, idempotent
- Docker services share DB via volume mount — single `/data/rune.db` volume all containers access

### Concurrency Model
- Synchronous SQLModel in the lib — standard `Session`, no async. FastAPI services wrap with `run_in_executor()` if needed
- WAL mode via SQLAlchemy connect event — `PRAGMA journal_mode=WAL` fires on every new connection automatically
- Internal session lifecycle — each method creates and closes its own `Session` via `with Session(engine) as session:`

### Store Behavior
- No file_path validation in store() — it's a metadata operation, file may be on a different host/container. Trust the caller.
- Duplicate ID check raises AdapterAlreadyExistsError (already in the docstring)
- Store is write-once — no UPDATE operations. Adapters are immutable after creation.

### Test Strategy
- In-memory SQLite (`sqlite:///:memory:`) for unit tests — fast, isolated, no cleanup
- Factory fixture `make_adapter_record()` already exists in conftest.py — use it
- Evolve existing TDD red-phase tests to green by implementing the stubs
- Add integration tests: concurrent writes, WAL mode verification, query filtering

### Claude's Discretion
- Exact Session configuration details
- Whether to add `__repr__` to AdapterRecord for debugging
- Index strategy beyond the existing task_type index
- Error message formatting in exceptions

</decisions>

<specifics>
## Specific Ideas

- Constructor signature: `AdapterRegistry(engine: Engine)` — engine is required, not optional
- WAL mode set via SQLAlchemy event_listen on "connect" — `connection.execute(text("PRAGMA journal_mode=WAL"))`
- `list_all()` filters `WHERE is_archived = False` — archived records are soft-deleted
- `query_by_task_type()` uses the existing index on `task_type` field

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AdapterRecord` model (models.py): Fully defined with 14 fields, SQLModel table, ready for create_all()
- `AdapterAlreadyExistsError`, `AdapterNotFoundError` (exceptions.py): Complete exception hierarchy
- `make_adapter_record` fixture (conftest.py): Factory with sensible defaults for testing
- Root conftest.py has duplicate fixture for component isolation

### Established Patterns
- Relative imports within component (`from adapter_registry.models import AdapterRecord`)
- Full type hints throughout (mypy strict mode)
- Google-style docstrings with Args, Returns, Raises, Examples sections
- NotImplementedError stubs with descriptive messages

### Integration Points
- `api-service/routers/adapters.py` — will call AdapterRegistry for CRUD endpoints
- `rune-agent/nodes.py` — `save_trajectory_node` will call `store()` after recording trajectory
- `training-svc/routers/training.py` — will call `store()` after QLoRA training produces an adapter
- `lora-server/vllm_client.py` — will call `retrieve_by_id()` to get adapter file_path for loading

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 18-adapter-registry*
*Context gathered: 2026-03-05*
