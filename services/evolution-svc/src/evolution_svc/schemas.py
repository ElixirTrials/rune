"""Pydantic request/response schemas for evolution service."""

from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    """Request to evaluate an adapter's performance."""

    adapter_id: str
    task_type: str


class EvaluationResponse(BaseModel):
    """Response containing adapter evaluation metrics."""

    adapter_id: str
    pass_rate: float
    fitness_score: float
    generalization_delta: float


class EvolveRequest(BaseModel):
    """Request to evolve adapters via crossover/mutation."""

    adapter_ids: list[str]
    task_type: str


class PromoteRequest(BaseModel):
    """Request to promote an adapter to a higher tier."""

    adapter_id: str
    target_level: str


class PruneRequest(BaseModel):
    """Request to prune an underperforming adapter."""

    adapter_id: str
