"""Pydantic request/response schemas for training service."""

from pydantic import BaseModel


class LoraTrainingRequest(BaseModel):
    """Request to train a LoRA adapter."""

    task_type: str
    adapter_id: str | None = None
    rank: int = 64
    epochs: int = 3


class HypernetworkTrainingRequest(BaseModel):
    """Request to train via hypernetwork forward pass."""

    task_type: str
    trajectory_ids: list[str]


class JobStatusResponse(BaseModel):
    """Response for training job status."""

    job_id: str
    status: str
    adapter_id: str | None = None
