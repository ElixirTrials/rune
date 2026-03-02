"""Dependency injection for FastAPI endpoints."""

from collections.abc import Generator

from sqlmodel import Session

from api_service.storage import engine


def get_db() -> Generator[Session, None, None]:
    """Provide a database session for dependency injection.

    Yields:
        Database session that automatically commits or rolls back.
    """
    with Session(engine) as session:
        yield session
