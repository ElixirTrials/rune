"""Integration tests for the swarm orchestrator."""

import asyncio
import sys
from pathlib import Path

from shared.checkpoint_db import SwarmCheckpointDB
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from swarm import agent_supervisor, load_task_pool
from swarm_workers import TrainingRequest

SAMPLE_TASKS = [
    {
        "task_id": "test-task-1",
        "prompt": "def add(a, b):\n",
        "canonical_solution": "    return a + b\n",
        "test": "assert add(1, 2) == 3\n",
    },
    {
        "task_id": "test-task-2",
        "prompt": "def mul(a, b):\n",
        "canonical_solution": "    return a * b\n",
        "test": "assert mul(2, 3) == 6\n",
    },
]


def _make_checkpoint_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return SwarmCheckpointDB(engine)


async def test_swarm_dry_run() -> None:
    checkpoint_db = _make_checkpoint_db()
    queue: asyncio.Queue[TrainingRequest] = asyncio.Queue()
    result = await agent_supervisor(
        agent_id="agent-0",
        tasks=SAMPLE_TASKS,
        checkpoint_db=checkpoint_db,
        training_queue=queue,
        dry_run=True,
    )
    assert result["completed"] == 2
    assert result["failed"] == 0
    assert result["skipped"] == 0
    assert checkpoint_db.is_completed("test-task-1")
    assert checkpoint_db.is_completed("test-task-2")


async def test_agent_supervisor_retries_on_failure() -> None:
    checkpoint_db = _make_checkpoint_db()
    queue: asyncio.Queue[TrainingRequest] = asyncio.Queue()
    bad_tasks = [
        {"task_id": "bad-1", "prompt": "", "test": "assert False"},
        {"task_id": "good-1", "prompt": "x = 1\n", "test": "assert x == 1\n"},
    ]
    result = await agent_supervisor(
        agent_id="agent-0",
        tasks=bad_tasks,
        checkpoint_db=checkpoint_db,
        training_queue=queue,
        dry_run=False,
    )
    assert result["failed"] >= 1
    assert result["completed"] >= 1


async def test_agent_supervisor_retires_after_max_failures() -> None:
    checkpoint_db = _make_checkpoint_db()
    queue: asyncio.Queue[TrainingRequest] = asyncio.Queue()
    bad_tasks = [
        {"task_id": f"bad-{i}", "prompt": "", "test": "assert False"} for i in range(10)
    ]
    result = await agent_supervisor(
        agent_id="agent-0",
        tasks=bad_tasks,
        checkpoint_db=checkpoint_db,
        training_queue=queue,
        dry_run=False,
        max_failures=3,
    )
    assert result["failed"] == 3
    assert result["completed"] == 0


async def test_swarm_checkpoint_recovery() -> None:
    checkpoint_db = _make_checkpoint_db()
    checkpoint_db.mark_running("test-task-1", "old-agent")
    checkpoint_db.mark_completed("test-task-1", "already-done")

    queue: asyncio.Queue[TrainingRequest] = asyncio.Queue()
    result = await agent_supervisor(
        agent_id="agent-0",
        tasks=SAMPLE_TASKS,
        checkpoint_db=checkpoint_db,
        training_queue=queue,
        dry_run=True,
    )
    assert result["skipped"] == 1
    assert result["completed"] == 1


def test_load_task_pool_missing_file() -> None:
    tasks = load_task_pool("/nonexistent/path.json")
    assert tasks == []
