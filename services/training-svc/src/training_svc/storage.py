"""Database storage configuration for training service."""

from shared.storage_utils import create_service_engine
from sqlmodel import SQLModel

engine = create_service_engine("sqlite:///training_svc.db")


def create_db_and_tables() -> None:
    """Create all database tables."""
    SQLModel.metadata.create_all(engine)
