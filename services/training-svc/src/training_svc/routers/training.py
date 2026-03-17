"""Training router — POST /train/lora, POST /train/hypernetwork, GET /jobs/{id}."""

import logging
import os
import traceback
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from training_svc.jobs import _JOB_STORE_LOCK, JOB_STORE, JobStatus
from training_svc.schemas import (
    HypernetworkTrainingRequest,
    LoraTrainingRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["training"])

_PATH_UNSAFE_CHARS = frozenset({"/", "\\", ".", "\x00"})


def _validate_adapter_id(adapter_id: str) -> None:
    r"""Raise HTTPException(422) if adapter_id contains path-traversal characters.

    Args:
        adapter_id: The adapter identifier to validate.

    Raises:
        HTTPException: 422 if adapter_id contains ``/``, ``\\``, ``.``, or null.
    """
    if any(c in adapter_id for c in _PATH_UNSAFE_CHARS) or ".." in adapter_id:
        raise HTTPException(
            status_code=422,
            detail=(
                f"adapter_id '{adapter_id}' contains unsafe characters. "
                "Must not contain '/', '\\', '..', or null bytes."
            ),
        )


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
    with _JOB_STORE_LOCK:
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
        with _JOB_STORE_LOCK:
            JOB_STORE[job_id].status = "completed"
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("Training job %s failed", job_id)
        with _JOB_STORE_LOCK:
            JOB_STORE[job_id].status = "failed"
            JOB_STORE[job_id].error = f"{e}\n\n{tb}"


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
    _validate_adapter_id(adapter_id)

    with _JOB_STORE_LOCK:
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


def _run_hypernetwork_job(
    job_id: str, adapter_id: str, trajectory_id: str, task_type: str
) -> None:
    """Background worker: runs DocToLoraHypernetwork forward pass and saves adapter.

    Import is deferred inside the function body per INFRA-05 (GPU deps
    should not be imported at service startup in non-GPU environments).

    Args:
        job_id: Unique job identifier used to update JOB_STORE.
        adapter_id: Pre-assigned adapter identifier (assigned at request time).
        trajectory_id: Trajectory ID to load and tokenize.
        task_type: Task type string (stored for context, not used in forward pass).
    """
    with _JOB_STORE_LOCK:
        JOB_STORE[job_id].status = "running"
    try:
        import torch  # noqa: PLC0415
        from model_training.hypernetwork import (  # noqa: PLC0415
            DocToLoraHypernetwork,
            save_hypernetwork_adapter,
        )
        from model_training.trajectory import (  # noqa: PLC0415
            format_for_sft,
            load_trajectory,
        )
        from transformers import AutoTokenizer  # noqa: PLC0415

        # Load and tokenize trajectory
        trajectory = load_trajectory(trajectory_id)
        messages = format_for_sft(trajectory)
        # Concatenate messages into a single text for tokenization
        text = " ".join(m["content"] for m in messages)

        # Resolve env vars inside function body for testability
        base_model_id = os.environ.get(
            "RUNE_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
        )
        hypernetwork_weights_path = os.environ.get(
            "RUNE_HYPERNETWORK_WEIGHTS_PATH",
            str(Path.home() / ".rune" / "hypernetwork.pt"),
        )
        adapter_base = os.environ.get("RUNE_ADAPTER_DIR")

        # Tokenize using base model tokenizer (lightweight, no 7B model loading)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        token_ids = tokenizer.encode(text, return_tensors="pt")

        # Load hypernetwork and generate adapter
        hypernetwork = DocToLoraHypernetwork(
            input_dim=tokenizer.vocab_size,
        )
        hypernetwork.load_state_dict(
            torch.load(hypernetwork_weights_path, map_location="cpu")
        )
        hypernetwork.eval()

        with torch.no_grad():
            weights = hypernetwork(token_ids)

        # Save adapter
        if adapter_base:
            adapter_dir = str(Path(adapter_base) / adapter_id)
        else:
            adapter_dir = str(Path.home() / ".rune" / "adapters" / adapter_id)

        save_hypernetwork_adapter(
            weights=weights,
            output_dir=adapter_dir,
            base_model_id=base_model_id,
        )

        # Register adapter in registry (reuse service engine for DB consistency)
        import hashlib  # noqa: PLC0415
        from datetime import datetime, timezone  # noqa: PLC0415

        from adapter_registry.models import AdapterRecord  # noqa: PLC0415
        from adapter_registry.registry import AdapterRegistry  # noqa: PLC0415

        from training_svc.storage import engine as svc_engine  # noqa: PLC0415

        safetensors_path = Path(adapter_dir) / "adapter_model.safetensors"
        file_hash = hashlib.sha256(safetensors_path.read_bytes()).hexdigest()
        file_size_bytes = safetensors_path.stat().st_size

        registry = AdapterRegistry(engine=svc_engine)
        record = AdapterRecord(
            id=adapter_id,
            version=1,
            task_type=task_type,
            base_model_id=base_model_id,
            rank=8,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            file_path=adapter_dir,
            file_hash=file_hash,
            file_size_bytes=file_size_bytes,
            source="hypernetwork",
            session_id=trajectory_id,
        )
        registry.store(record)

        with _JOB_STORE_LOCK:
            JOB_STORE[job_id].status = "completed"
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("Hypernetwork job %s failed", job_id)
        with _JOB_STORE_LOCK:
            JOB_STORE[job_id].status = "failed"
            JOB_STORE[job_id].error = f"{e}\n\n{tb}"


@router.post("/train/hypernetwork")
async def train_hypernetwork(
    request: HypernetworkTrainingRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Dispatch a hypernetwork adapter generation job as a background task.

    Accepts a trajectory, runs it through the pre-trained hypernetwork in a
    single forward pass, saves the adapter in PEFT format, and returns a
    job_id for status polling via GET /jobs/{job_id}.

    Args:
        request: Hypernetwork training parameters including task_type and
            trajectory_ids (uses first trajectory_id).
        background_tasks: FastAPI background task runner.

    Returns:
        JSONResponse with job_id and status="queued".

    Raises:
        HTTPException: 422 if trajectory_ids is empty.

    Example:
        >>> body = {"task_type": "gen", "trajectory_ids": ["t-1"]}
        >>> response = client.post("/train/hypernetwork", json=body)
        >>> response.status_code
        200
        >>> response.json()["status"]
        'queued'
    """
    if not request.trajectory_ids:
        raise HTTPException(
            status_code=422,
            detail="trajectory_ids must not be empty.",
        )

    job_id = str(uuid.uuid4())
    # Assign adapter_id at creation time so GET /jobs/{id} returns it immediately
    adapter_id = str(uuid.uuid4())

    with _JOB_STORE_LOCK:
        JOB_STORE[job_id] = JobStatus(
            job_id=job_id,
            status="queued",
            adapter_id=adapter_id,
        )

    background_tasks.add_task(
        _run_hypernetwork_job,
        job_id,
        adapter_id,
        request.trajectory_ids[0],
        request.task_type,
    )

    return JSONResponse(content={"job_id": job_id, "status": "queued"}, status_code=200)


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
    with _JOB_STORE_LOCK:
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
