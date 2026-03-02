"""Shared data models for the application."""

from typing import Any, Dict

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Entity(SQLModel, table=True):
    """Generic entity processed by agents."""

    id: str = Field(primary_key=True)
    entity_type: str = Field(index=True)
    data: Dict[str, Any] = Field(sa_column=Column(JSON))
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class Task(SQLModel, table=True):
    """Generic task flowing through the pipeline."""

    id: str = Field(primary_key=True)
    status: str = Field(index=True)
    input_data: Dict[str, Any] = Field(sa_column=Column(JSON))
    output_data: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
