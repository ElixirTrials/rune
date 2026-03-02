"""Rune-specific Pydantic models for cross-service data contracts.

Defines the shared data shapes for coding sessions, adapter references,
and evolutionary fitness metrics used across all Rune services.
"""

from typing import Optional

from pydantic import BaseModel


class AdapterRef(BaseModel):
    """Reference to a stored LoRA adapter.

    Used within CodingSession to track which adapters were loaded
    during an agent coding session.

    Attributes:
        adapter_id: UUID of the adapter in the registry.
        task_type: Task category this adapter was trained on (e.g. 'bug-fix').
        fitness_score: Evolutionary fitness score, if evaluated.
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
    """

    adapter_id: str
    pass_rate: float
    fitness_score: float
    generalization_delta: Optional[float] = None
