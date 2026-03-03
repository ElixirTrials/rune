"""SQLModel tables for evolution job tracking."""

from typing import Optional

from sqlmodel import Field, SQLModel


class EvolutionJob(SQLModel, table=True):
    """Persistent record of an evolution job."""

    __tablename__ = "evolution_jobs"

    id: str = Field(primary_key=True)
    status: str = Field(index=True)
    task_type: str
    created_at: str
    adapter_id: Optional[str] = Field(default=None)
