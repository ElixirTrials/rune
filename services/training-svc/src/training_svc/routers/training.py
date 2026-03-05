"""Training router — POST /train/lora, POST /train/hypernetwork stub, GET /jobs/{id}."""

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from training_svc.jobs import JOB_STORE, JobStatus
from training_svc.schemas import (
    HypernetworkTrainingRequest,
    LoraTrainingRequest,
)

router = APIRouter(tags=["training"])


def _run_training_job(
    job_id: str,
    session_id: str,
    adapter_id: str,
    task_type: str,
    rank: int,
    epochs: int,
    learning_rate: float,
) -> None:
    """Background worker: runs train_and_register and updates JOB_STORE.

    Import is deferred inside the function body per INFRA-05 (GPU deps
    should not be imported at service startup in non-GPU environments).
    """
    JOB_STORE[job_id].status = "running"
    try:
        from model_training.trainer import train_and_register  # noqa: PLC0415

        train_and_register(
            session_id=session_id,
            adapter_id=adapter_id,
            task_type=task_type,
            rank=rank,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        JOB_STORE[job_id].status = "completed"
    except Exception as e:  # noqa: BLE001
        JOB_STORE[job_id].status = "failed"
        JOB_STORE[job_id].error = str(e)


@router.post("/train/lora")
async def train_lora(
    request: LoraTrainingRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Dispatch a QLoRA training job as a background task.

    Args:
        request: LoRA training parameters — session_id is required.
        background_tasks: FastAPI background task runner.

    Returns:
        JSONResponse with job_id and status="queued".

    Example:
        >>> body = {"session_id": "s-1", "task_type": "code-gen", "epochs": 3}
        >>> response = client.post("/train/lora", json=body)
        >>> response.status_code
        200
        >>> response.json()["status"]
        'queued'
    """
    job_id = str(uuid.uuid4())
    adapter_id = request.adapter_id or str(uuid.uuid4())

    JOB_STORE[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        adapter_id=adapter_id,
    )

    background_tasks.add_task(
        _run_training_job,
        job_id,
        request.session_id,
        adapter_id,
        request.task_type,
        request.rank,
        request.epochs,
        request.learning_rate,
    )

    return JSONResponse(content={"job_id": job_id, "status": "queued"}, status_code=200)


@router.post("/train/hypernetwork")
async def train_hypernetwork(request: HypernetworkTrainingRequest) -> JSONResponse:
    """Train via hypernetwork forward pass (Phase 22 — not yet implemented).

    Args:
        request: Hypernetwork training parameters including task_type and
            trajectory_ids.

    Returns:
        JSONResponse with 501 Not Implemented.

    Raises:
        HTTPException: 501 while not yet implemented.

    Example:
        >>> body = {"task_type": "gen", "trajectory_ids": ["t-1"]}
        >>> response = client.post("/train/hypernetwork", json=body)
        >>> response.status_code
        501
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    """Get training job status.

    Args:
        job_id: Unique identifier for the training job.

    Returns:
        JSONResponse with job_id, status, adapter_id, and optional error.

    Raises:
        HTTPException: 404 if job_id not found.

    Example:
        >>> response = client.get("/jobs/job-123")
        >>> response.status_code
        404
    """
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JSONResponse(
        content={
            "job_id": job.job_id,
            "status": job.status,
            "adapter_id": job.adapter_id,
            "error": job.error,
        },
        status_code=200,
    )
