"""Pydantic request/response schemas for training service."""

from pydantic import BaseModel


class LoraTrainingRequest(BaseModel):
    """Request to train a LoRA adapter."""

    session_id: str  # required — identifies trajectory to train on
    task_type: str = "code-gen"
    adapter_id: str | None = None
    rank: int = 64
    epochs: int = 3
    learning_rate: float = 2e-4  # optional override


class HypernetworkTrainingRequest(BaseModel):
    """Request to train via hypernetwork forward pass."""

    task_type: str
    trajectory_ids: list[str]


class JobStatusResponse(BaseModel):
    """Response for training job status."""

    job_id: str
    status: str
    adapter_id: str | None = None
    error: str | None = None  # error message on failure
