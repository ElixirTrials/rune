"""Tests for scripts/swarm_workers.py — training pool and memory watchdog."""

import asyncio

# Import from scripts path
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from shared.hardware import HardwareBudget

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from swarm_workers import (
    TrainingRequest,
    _drain_queue,
    _train_in_subprocess,
    memory_watchdog,
)


async def test_drain_queue_collects_batch() -> None:
    queue: asyncio.Queue[TrainingRequest] = asyncio.Queue()
    for i in range(5):
        await queue.put(TrainingRequest(session_id=f"s-{i}", task_type="bug-fix"))
    batch = await _drain_queue(queue, max_batch=3, timeout=0.1)
    assert len(batch) == 3
    assert batch[0].session_id == "s-0"


async def test_drain_queue_empty_returns_empty() -> None:
    queue: asyncio.Queue[TrainingRequest] = asyncio.Queue()
    batch = await _drain_queue(queue, max_batch=3, timeout=0.1)
    assert batch == []


def test_train_in_subprocess_returns_result() -> None:
    mock_trainer = MagicMock()
    mock_trainer.train_qlora = MagicMock()
    with patch.dict(
        "sys.modules",
        {"model_training.trainer": mock_trainer},
    ):
        result = _train_in_subprocess(
            session_id="test-sess",
            task_type="bug-fix",
            adapter_dir="/tmp/test-adapter",
            db_url="sqlite:///:memory:",
        )
    assert result["session_id"] == "test-sess"
    assert result["task_type"] == "bug-fix"
    assert "adapter_id" in result


async def test_memory_watchdog_cancels_on_high_pressure() -> None:
    budget = HardwareBudget(
        max_agents=4,
        max_concurrent_loras=8,
        training_slots=1,
        vram_per_gpu_mb=24576,
        single_gpu_mode=True,
    )

    mock_task = MagicMock(spec=asyncio.Task)
    agents: list[asyncio.Task] = [mock_task]
    shutdown = asyncio.Event()

    mock_mem = MagicMock()
    mock_mem.percent = 90.0  # Above 85% threshold

    async def _run_watchdog() -> None:
        # Run one iteration then shutdown
        with patch("psutil.virtual_memory", return_value=mock_mem):
            # Set shutdown after a short delay
            async def _shutdown_soon() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            asyncio.create_task(_shutdown_soon())
            await memory_watchdog(
                budget, agents, check_interval=0.01, shutdown_event=shutdown
            )

    await _run_watchdog()
    mock_task.cancel.assert_called_once()


async def test_memory_watchdog_no_action_when_fine() -> None:
    budget = HardwareBudget(
        max_agents=4,
        max_concurrent_loras=8,
        training_slots=1,
        vram_per_gpu_mb=24576,
        single_gpu_mode=True,
    )

    mock_task = MagicMock(spec=asyncio.Task)
    agents: list[asyncio.Task] = [mock_task]
    shutdown = asyncio.Event()

    mock_mem = MagicMock()
    mock_mem.percent = 50.0  # Below threshold

    async def _run_watchdog() -> None:
        with patch("psutil.virtual_memory", return_value=mock_mem):

            async def _shutdown_soon() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            asyncio.create_task(_shutdown_soon())
            await memory_watchdog(
                budget, agents, check_interval=0.01, shutdown_event=shutdown
            )

    await _run_watchdog()
    mock_task.cancel.assert_not_called()
