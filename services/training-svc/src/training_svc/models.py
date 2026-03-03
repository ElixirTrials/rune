"""SQLModel tables for training job tracking."""

from typing import Optional

from sqlmodel import Field, SQLModel


class TrainingJob(SQLModel, table=True):
    """Persistent record of a training job."""

    __tablename__ = "training_jobs"

    id: str = Field(primary_key=True)
    status: str = Field(index=True)
    task_type: str
    created_at: str
    adapter_id: Optional[str] = Field(default=None)
