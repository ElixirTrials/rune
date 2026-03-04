"""Training router with stub 501 endpoints."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from training_svc.schemas import (
    HypernetworkTrainingRequest,
    LoraTrainingRequest,
)

router = APIRouter(tags=["training"])


@router.post("/train/lora")
async def train_lora(request: LoraTrainingRequest) -> JSONResponse:
    """Train a LoRA adapter.

    Args:
        request: LoRA training parameters including task_type, optional adapter_id,
            rank, and epochs.

    Returns:
        JSONResponse with training job information including job_id and status.

    Raises:
        HTTPException: 501 while not yet implemented.

    Example:
        >>> body = {"task_type": "code-gen", "rank": 64, "epochs": 3}
        >>> response = client.post("/train/lora", json=body)
        >>> response.status_code
        200
        >>> 'job_id' in response.json()
        True
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.post("/train/hypernetwork")
async def train_hypernetwork(request: HypernetworkTrainingRequest) -> JSONResponse:
    """Train via hypernetwork forward pass.

    Args:
        request: Hypernetwork training parameters including task_type and
            trajectory_ids.

    Returns:
        JSONResponse with training job information including job_id and status.

    Raises:
        HTTPException: 501 while not yet implemented.

    Example:
        >>> body = {"task_type": "gen", "trajectory_ids": ["t-1"]}
        >>> response = client.post("/train/hypernetwork", json=body)
        >>> response.status_code
        200
        >>> 'job_id' in response.json()
        True
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    """Get training job status.

    Args:
        job_id: Unique identifier for the training job.

    Returns:
        JSONResponse with job status information including job_id, status,
        and optional adapter_id when training completes.

    Raises:
        HTTPException: 501 while not yet implemented.

    Example:
        >>> response = client.get("/jobs/job-123")
        >>> response.status_code
        200
        >>> 'status' in response.json()
        True
        >>> 'job_id' in response.json()
        True
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})
