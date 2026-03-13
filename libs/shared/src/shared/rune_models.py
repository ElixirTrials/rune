"""Rune-specific Pydantic models for cross-service data contracts.

Defines the shared data shapes for coding sessions, adapter references,
and evolutionary fitness metrics used across all Rune services.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class TaskStatus(str, Enum):
    """Canonical status strings used across services and swarm checkpoints.

    Inherits from ``str`` so values are JSON-serialisable and can be compared
    directly to string literals in existing code (e.g. ``record.status == "running"``).

    Example:
        >>> TaskStatus.RUNNING
        <TaskStatus.RUNNING: 'running'>
        >>> TaskStatus.RUNNING == "running"
        True
        >>> "running" == TaskStatus.RUNNING
        True
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SUCCESS = "success"
    EXHAUSTED = "exhausted"


class AdapterRef(BaseModel):
    """Reference to a stored LoRA adapter.

    Used within CodingSession to track which adapters were loaded
    during an agent coding session.

    Attributes:
        adapter_id: UUID of the adapter in the registry.
        task_type: Task category this adapter was trained on (e.g. 'bug-fix').
        fitness_score: Evolutionary fitness score, if evaluated.

    Example:
        >>> ref = AdapterRef(adapter_id="abc-123", task_type="bug-fix")
        >>> ref.adapter_id
        'abc-123'
        >>> ref.fitness_score is None
        True
    """

    adapter_id: str
    task_type: str
    fitness_score: Optional[float] = None


class CodingSession(BaseModel):
    """A complete agent coding session record.

    Tracks the full lifecycle of a Rune agent session including which
    adapters were used, how many generate-execute-reflect cycles occurred,
    and the final outcome.

    Attributes:
        session_id: Unique session identifier.
        task_description: Human-readable description of the coding task.
        task_type: Task category (e.g. 'bug-fix', 'feature-impl').
        adapter_refs: List of adapters loaded during this session.
        attempt_count: Number of generate-execute-reflect cycles completed.
        outcome: Final session result ('success', 'exhausted', or None if in progress).

    Example:
        >>> session = CodingSession(
        ...     session_id="sess-001",
        ...     task_description="Fix import error",
        ...     task_type="bug-fix",
        ... )
        >>> session.adapter_refs
        []
        >>> session.attempt_count
        0
        >>> session.outcome is None
        True
    """

    session_id: str
    task_description: str
    task_type: str
    adapter_refs: list[AdapterRef] = []
    attempt_count: int = 0
    outcome: Optional[str] = None


class EvolMetrics(BaseModel):
    """Evolution evaluation metrics for an adapter.

    Captures the performance and fitness measurements used by the
    evolution service to decide which adapters to keep, mutate, or retire.

    Attributes:
        adapter_id: UUID of the adapter being evaluated.
        pass_rate: Pass rate on benchmark tasks (0.0 to 1.0).
        fitness_score: Overall evolutionary fitness score.
        generalization_delta: Difference between in-distribution and OOD performance.

    Example:
        >>> metrics = EvolMetrics(
        ...     adapter_id="adapter-001",
        ...     pass_rate=0.85,
        ...     fitness_score=0.9,
        ... )
        >>> metrics.generalization_delta is None
        True
    """

    adapter_id: str
    pass_rate: float
    fitness_score: float
    generalization_delta: Optional[float] = None


class SwarmConfig(BaseModel):
    """Configuration for a swarm execution run.

    Attributes:
        db_url: SQLite database URL for the adapter registry.
        task_source: Path to task definitions file or inline task list.
        population_size: Number of concurrent swarm agents.
        max_generations: Maximum evolutionary generations.
        evolution_interval: Seconds between evolution sweeps.
        sandbox_backend: Execution backend ('subprocess' or 'nsjail').
        base_model_id: HuggingFace model identifier for inference.
        hypernetwork_checkpoint: Path to pretrained hypernetwork checkpoint.
    """

    db_url: str = "sqlite:///rune_swarm.db"
    task_source: str = "tasks.json"
    population_size: int = 8
    max_generations: int = 10
    evolution_interval: int = 7200
    sandbox_backend: str = "subprocess"
    base_model_id: str = "Qwen/Qwen2.5-Coder-7B"
    hypernetwork_checkpoint: str | None = None


class SwarmCheckpoint(BaseModel):
    """Status record for a single swarm task execution.

    Attributes:
        run_id: Unique identifier for the swarm run.
        task_hash: Hash of the task being executed.
        agent_id: Identifier of the agent executing the task.
        status: Current status (pending, running, completed, failed).
        outcome: Result description when completed.
        started_at: ISO 8601 timestamp when execution began.
        completed_at: ISO 8601 timestamp when execution finished.
    """

    run_id: str
    task_hash: str
    agent_id: str
    status: str = "pending"
    outcome: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
