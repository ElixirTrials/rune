"""Training pool management and memory watchdog for swarm execution.

Provides async workers for managing concurrent training jobs with GPU
time-sharing (sleep/wake for single-GPU mode) and memory pressure monitoring.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.hardware import HardwareBudget

logger = logging.getLogger(__name__)


@dataclass
class TrainingRequest:
    """A request to train a LoRA adapter.

    Attributes:
        session_id: Unique session identifier.
        task_type: Task category for training.
    """

    session_id: str
    task_type: str


async def _drain_queue(
    queue: asyncio.Queue[TrainingRequest],
    max_batch: int,
    timeout: float = 1.0,
) -> list[TrainingRequest]:
    """Drain up to max_batch items from an async queue.

    Args:
        queue: Async queue to drain.
        max_batch: Maximum items to collect.
        timeout: Seconds to wait for the first item.

    Returns:
        List of collected TrainingRequest items.
    """
    batch: list[TrainingRequest] = []
    try:
        first = await asyncio.wait_for(queue.get(), timeout=timeout)
        batch.append(first)
    except (asyncio.TimeoutError, TimeoutError):
        return batch

    while len(batch) < max_batch:
        try:
            item = queue.get_nowait()
            batch.append(item)
        except asyncio.QueueEmpty:
            break
    return batch


def _train_in_subprocess(
    session_id: str,
    task_type: str,
    adapter_dir: str,
    db_url: str,
) -> dict[str, Any]:
    """Top-level picklable function for training in a subprocess.

    Imports GPU modules inside function body (INFRA-05 pattern).

    Args:
        session_id: Unique session identifier.
        task_type: Task category.
        adapter_dir: Directory to save adapter weights.
        db_url: Database URL for the adapter registry.

    Returns:
        Dict with adapter_id, session_id, task_type, adapter_dir.
    """
    import uuid

    adapter_id = str(uuid.uuid4())

    try:
        from model_training.trainer import train_qlora

        train_qlora(
            session_id=session_id,
            task_type=task_type,
            output_dir=adapter_dir,
        )
    except ImportError:
        logger.warning(
            "model_training.trainer not available; skipping actual training "
            "for session %s",
            session_id,
        )

    return {
        "adapter_id": adapter_id,
        "session_id": session_id,
        "task_type": task_type,
        "adapter_dir": adapter_dir,
    }


async def _sleep_vllm(base_url: str) -> None:
    """Tell vLLM to release GPU memory for training.

    Args:
        base_url: vLLM server base URL.
    """
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            await client.post(f"{base_url}/sleep", timeout=30)
    except Exception:
        logger.warning("Failed to sleep vLLM at %s", base_url)


async def _wake_vllm(base_url: str) -> None:
    """Tell vLLM to reclaim GPU memory after training.

    Args:
        base_url: vLLM server base URL.
    """
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            await client.post(f"{base_url}/wake_up", timeout=30)
    except Exception:
        logger.warning("Failed to wake vLLM at %s", base_url)


async def training_pool_manager(
    training_queue: asyncio.Queue[TrainingRequest],
    budget: "HardwareBudget",
    db_url: str = "sqlite:///rune_swarm.db",
    adapter_base_dir: str = "/tmp/rune_adapters",
    vllm_base_url: str = "http://localhost:8100/v1",
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Async manager that processes training requests from a queue.

    Uses ProcessPoolExecutor for CPU-bound training. In single-GPU mode,
    coordinates with vLLM via sleep/wake to time-share the GPU.

    Args:
        training_queue: Queue of TrainingRequest items.
        budget: Hardware budget for concurrency decisions.
        db_url: Database URL for the adapter registry.
        adapter_base_dir: Base directory for saving adapter weights.
        vllm_base_url: vLLM server URL for sleep/wake coordination.
        shutdown_event: Event to signal graceful shutdown.
    """
    loop = asyncio.get_event_loop()
    max_workers = max(1, budget.training_slots)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        while True:
            if shutdown_event and shutdown_event.is_set():
                break

            batch = await _drain_queue(
                training_queue, max_batch=max_workers, timeout=2.0
            )
            if not batch:
                continue

            # Single-GPU: sleep vLLM before training
            if budget.single_gpu_mode and budget.training_slots > 0:
                await _sleep_vllm(vllm_base_url)

            futures = []
            for req in batch:
                adapter_dir = f"{adapter_base_dir}/{req.session_id}"
                future = loop.run_in_executor(
                    pool,
                    _train_in_subprocess,
                    req.session_id,
                    req.task_type,
                    adapter_dir,
                    db_url,
                )
                futures.append(future)

            results = await asyncio.gather(*futures, return_exceptions=True)

            # Single-GPU: wake vLLM after training
            if budget.single_gpu_mode and budget.training_slots > 0:
                await _wake_vllm(vllm_base_url)

            for result in results:
                if isinstance(result, Exception):
                    logger.error("Training failed: %s", result)
                else:
                    logger.info("Training completed: %s", result)


async def memory_watchdog(
    budget: "HardwareBudget",
    agents: list[asyncio.Task[Any]],
    check_interval: float = 30.0,
    threshold: float = 0.85,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Monitor memory pressure and cancel youngest agent if too high.

    Args:
        budget: Hardware budget (unused but kept for future VRAM checks).
        agents: Mutable list of running agent tasks.
        check_interval: Seconds between memory checks.
        threshold: Memory usage fraction triggering cancellation.
        shutdown_event: Event to signal graceful shutdown.
    """
    import psutil

    while True:
        if shutdown_event and shutdown_event.is_set():
            break

        mem = psutil.virtual_memory()
        usage = mem.percent / 100.0

        if usage > threshold and agents:
            youngest = agents[-1]
            logger.warning(
                "Memory pressure %.1f%% > %.1f%% threshold, cancelling youngest agent",
                usage * 100,
                threshold * 100,
            )
            youngest.cancel()
            agents.remove(youngest)

        await asyncio.sleep(check_interval)
