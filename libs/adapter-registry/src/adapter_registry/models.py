"""SQLModel table model for LoRA adapter metadata records."""

from typing import Optional

from sqlmodel import Field, SQLModel


class AdapterRecord(SQLModel, table=True):
    """A stored LoRA adapter metadata record.

    Tracks metadata for a single LoRA adapter including its lineage,
    storage location, and evaluation metrics. Backed by SQLite via SQLModel.

    Attributes:
        id: Unique adapter identifier (UUID string).
        version: Adapter version number for lineage tracking.
        task_type: Task category this adapter was trained on (e.g. 'bug-fix').
        base_model_id: Identifier of the base model this adapter was trained from.
        rank: LoRA rank used during training.
        created_at: ISO 8601 timestamp of adapter creation.
        file_path: Filesystem path to the adapter weights file.
        file_hash: SHA-256 hash of the adapter weights file for integrity checks.
        file_size_bytes: Size of the adapter weights file in bytes.
        pass_rate: Pass rate on benchmark tasks (0.0 to 1.0), if evaluated.
        fitness_score: Overall evolutionary fitness score, if evaluated.
        source: How the adapter was created ('distillation', 'evolution', 'manual').
        session_id: ID of the coding session that produced this adapter.
        is_archived: Whether this adapter has been archived (soft delete).
        parent_ids: JSON-encoded list of parent adapter IDs for lineage tracking.
        generation: Evolutionary generation number (0 for initial adapters).
        training_task_hash: Deduplication key for the training task.
        agent_id: Identifier of the swarm agent that produced this adapter.

    Example:
        >>> record = AdapterRecord(
        ...     id="adapter-001", version=1, task_type="bug-fix",
        ...     base_model_id="Qwen/Qwen2.5-Coder-7B", rank=16,
        ...     created_at="2026-01-01T00:00:00Z",
        ...     file_path="/adapters/adapter-001.safetensors",
        ...     file_hash="abc123", file_size_bytes=1024,
        ...     source="distillation", session_id="sess-001",
        ... )
        >>> record.task_type
        'bug-fix'
        >>> record.is_archived
        False
    """

    __tablename__ = "adapter_records"

    id: str = Field(primary_key=True)
    version: int
    task_type: str = Field(index=True)
    base_model_id: str
    rank: int
    created_at: str
    file_path: str
    file_hash: str
    file_size_bytes: int
    pass_rate: Optional[float] = Field(default=None)
    fitness_score: Optional[float] = Field(default=None)
    source: str
    session_id: str
    is_archived: bool = Field(default=False)
    parent_ids: Optional[str] = Field(default=None)
    generation: int = Field(default=0)
    training_task_hash: Optional[str] = Field(default=None, index=True)
    agent_id: Optional[str] = Field(default=None)
