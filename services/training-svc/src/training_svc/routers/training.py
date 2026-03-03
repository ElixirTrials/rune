"""Training router with stub 501 endpoints."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from training_svc.schemas import (
    HypernetworkTrainingRequest,
    JobStatusResponse,
    LoraTrainingRequest,
)

router = APIRouter(tags=["training"])


@router.post("/train/lora")
async def train_lora(request: LoraTrainingRequest) -> JSONResponse:
    """Train a LoRA adapter. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.post("/train/hypernetwork")
async def train_hypernetwork(request: HypernetworkTrainingRequest) -> JSONResponse:
    """Train via hypernetwork forward pass. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    """Get training job status. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})
