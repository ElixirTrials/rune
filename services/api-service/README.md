# API Service

## Purpose
The API Service is the central orchestrator for the application. It provides HTTP endpoints for the frontend, manages database persistence, and triggers background agent tasks.

## Wiring Up New Endpoints

1.  **Create a Router**: Create a new module in `src/api_service/routers/` (e.g., `tasks.py`).
    ```python
    from fastapi import APIRouter, Depends
    from api_service.dependencies import get_db

    router = APIRouter()

    @router.post("/tasks")
    async def create_task(data: TaskCreate, db = Depends(get_db)):
        ...
    ```
2.  **Register Router**: Import and include the router in `src/api_service/main.py`.
    ```python
    from api_service.routers import tasks
    app.include_router(tasks.router, prefix="/api/v1", tags=["tasks"])
    ```

## Database Access

- We use **SQLModel** (Pydantic + SQLAlchemy).
- Define models in `src/api_service/models.py` or `libs/shared/src/shared/models.py` if shared.
- Use `alembic` for migrations:
    ```bash
    uv run alembic revision --autogenerate -m "Add task table"
    uv run alembic upgrade head
    ```

## calling Agents

Do not import agent code directly if possible. Instead, use the factories in `inference` or clean interfaces in agent packages.

```python
from agent_a_service.graph import create_graph

async def run_agent_task(task_id: str):
    graph = create_graph()
    await graph.ainvoke(...)
```

## Best Practices

- **Dependency Injection**: Always use `Depends()` for DB sessions and services.
- **Background Tasks**: Use `FastAPI.BackgroundTasks` or a proper queue (e.g. Celery/Arq) for long-running agent jobs.
- **Type Safety**: Use Pydantic models for all Request/Response bodies.
