"""Pytest configuration for services/api-service.

Provides database fixtures and a FastAPI TestClient with the test database
injected via dependency override. Also re-exports factory fixtures from root
conftest.py (duplicated here due to pytest rootdir isolation from local
pyproject.toml).
"""
from typing import Any, AsyncGenerator, Callable, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="function")
def db_engine():
    """Create an in-memory SQLite database engine for testing.

    Each test gets a fresh database.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """Create a database session for testing.

    Automatically rolls back changes after each test.
    """
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = session_local()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def test_client(db_session) -> Generator[TestClient, None, None]:
    """Create a FastAPI test client with a test database session."""
    from api_service.dependencies import get_db
    from api_service.main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def async_client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing async endpoints."""
    from api_service.dependencies import get_db
    from api_service.main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    yield
    try:
        from shared.lazy_cache import _clear_all_singletons

        _clear_all_singletons()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Factory fixtures — duplicated from root conftest.py because pytest rootdir
# isolation (services/api-service/pyproject.toml) prevents root discovery.
# ---------------------------------------------------------------------------


@pytest.fixture
def make_adapter_record() -> Callable[..., Any]:
    """Factory fixture for AdapterRecord domain objects."""
    from adapter_registry.models import AdapterRecord

    def _factory(**kwargs: Any) -> AdapterRecord:
        defaults: dict[str, Any] = {
            "id": "test-adapter-001",
            "version": 1,
            "task_type": "bug-fix",
            "base_model_id": "Qwen/Qwen2.5-Coder-7B",
            "rank": 16,
            "created_at": "2026-01-01T00:00:00Z",
            "file_path": "/adapters/test-adapter-001.safetensors",
            "file_hash": "abc123def456",
            "file_size_bytes": 1024,
            "pass_rate": None,
            "fitness_score": None,
            "source": "distillation",
            "session_id": "test-session-001",
            "is_archived": False,
        }
        return AdapterRecord(**{**defaults, **kwargs})

    return _factory


@pytest.fixture
def make_coding_session() -> Callable[..., Any]:
    """Factory fixture for CodingSession domain objects."""
    from shared.rune_models import CodingSession

    def _factory(**kwargs: Any) -> CodingSession:
        defaults: dict[str, Any] = {
            "session_id": "test-session-001",
            "task_description": "Fix the off-by-one error in list slicing",
            "task_type": "bug-fix",
            "adapter_refs": [],
            "attempt_count": 0,
            "outcome": None,
        }
        return CodingSession(**{**defaults, **kwargs})

    return _factory
