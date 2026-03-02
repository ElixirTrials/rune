"""Pytest Configuration and Fixtures.

This file contains pytest configuration and shared fixtures for testing.
Fixtures defined here are automatically available to all tests in this
directory and subdirectories.

Best Practices:
- Define reusable fixtures here
- Use fixtures for setup/teardown
- Use pytest markers for test categorization
- Keep fixtures focused and composable
"""

from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


# Example: Database fixtures
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
    # Create all tables
    # Base.metadata.create_all(bind=engine)

    try:
        yield engine
    finally:
        # Close all connections and dispose of the engine
        engine.dispose()
        # Force garbage collection to clean up connections
        import gc

        gc.collect()


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


# Example: FastAPI test client fixtures
@pytest.fixture(scope="function")
def test_client(db_session) -> Generator[TestClient, None, None]:
    """Create a FastAPI test client with a test database session."""
    from api_service.dependencies import get_db
    from api_service.main import app

    # Override database dependency
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client

    # Clean up overrides
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


# Example: Mock data fixtures
@pytest.fixture
def sample_user_data():
    """Provide sample user data for testing."""
    return {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True,
    }


@pytest.fixture
def sample_users_list():
    """Provide a list of sample users for testing."""
    return [
        {"id": 1, "username": "user1", "email": "user1@example.com"},
        {"id": 2, "username": "user2", "email": "user2@example.com"},
        {"id": 3, "username": "user3", "email": "user3@example.com"},
    ]


# Example: Environment variable fixtures
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MODEL_BACKEND", "local")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


# Example: Mock external services
@pytest.fixture
def mock_external_api(requests_mock):
    """Mock external API calls using requests-mock.

    Usage: pip install requests-mock.
    """
    requests_mock.get(
        "https://api.example.com/users/1", json={"id": 1, "name": "Test User"}
    )
    return requests_mock


# Example: Async mock for aiohttp
@pytest.fixture
def mock_aiohttp_session(aioresponses):
    """Mock aiohttp HTTP calls.

    Usage: Already included in dependencies (aioresponses).
    """
    aioresponses.get(
        "https://api.example.com/data", payload={"status": "success", "data": [1, 2, 3]}
    )
    return aioresponses


# Example: Cleanup fixture
@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests.

    autouse=True means this runs automatically for every test.
    """
    yield
    # Cleanup after test - reset lazy_singleton cached instances
    try:
        from shared.lazy_cache import _clear_all_singletons

        _clear_all_singletons()
    except ImportError:
        # shared module not available in this test context
        pass


# Example: Parametrized fixture
@pytest.fixture(params=["vertex", "local"])
def model_backend(request):
    """Parametrized fixture to test with different model backends.

    Tests using this fixture will run once for each parameter.
    """
    return request.param
