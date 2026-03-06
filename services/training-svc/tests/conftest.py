"""Pytest configuration for services/training-svc."""

from collections.abc import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel


@pytest.fixture(scope="function")
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    with Session(db_engine) as session:
        yield session


@pytest.fixture(scope="function")
def test_client(db_engine, db_session) -> Generator[TestClient, None, None]:
    from training_svc.dependencies import get_db
    from training_svc.main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with patch("training_svc.storage.engine", db_engine):
        with TestClient(app) as client:
            yield client
    app.dependency_overrides.clear()
