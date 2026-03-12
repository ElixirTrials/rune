"""Fat orchestrator for parallel swarm execution.

Collapses Rune's microservice architecture into a single-process
orchestrator for local hardware. Coordinates agent supervisors,
training pool, evolution worker, and memory watchdog via asyncio.

Usage:
    uv run scripts/swarm.py [--config scripts/swarm_config.yaml] [--agents N] [--hours H] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

# Add scripts dir to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

from shared.checkpoint_db import SwarmCheckpointDB
from shared.hardware import HardwareProbe
from shared.rune_models import SwarmConfig
from shared.sandbox import get_sandbox_backend
from swarm_evolution import evolution_worker
from swarm_workers import TrainingRequest, memory_watchdog, training_pool_manager

logger = logging.getLogger(__name__)


def load_task_pool(task_source: str) -> list[dict[str, Any]]:
    """Load tasks from a JSON file or return empty list.

    Args:
        task_source: Path to a JSON file containing task definitions.

    Returns:
        List of task dicts with at minimum 'task_id' and 'description'.
    """
    path = Path(task_source)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    logger.warning("Task source %s not found, using empty task pool", task_source)
    return []


async def agent_supervisor(
    agent_id: str,
    tasks: list[dict[str, Any]],
    checkpoint_db: SwarmCheckpointDB,
    training_queue: asyncio.Queue[TrainingRequest],
    dry_run: bool = False,
    max_failures: int = 5,
    shutdown_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Tier-2 supervisor for a single swarm agent.

    Processes tasks from the pool, checking checkpoints for dedup,
    running code via sandbox, and queuing training requests.

    Args:
        agent_id: Unique identifier for this agent.
        tasks: List of task dicts to process.
        checkpoint_db: Checkpoint DB for dedup and recovery.
        training_queue: Queue for submitting training requests.
        dry_run: If True, skip actual execution.
        max_failures: Consecutive failures before retiring.
        shutdown_event: Event to signal graceful shutdown.

    Returns:
        Summary dict with completed/failed/skipped counts.
    """
    completed = 0
    failed = 0
    skipped = 0
    consecutive_failures = 0
    sandbox = get_sandbox_backend()

    for task in tasks:
        if shutdown_event and shutdown_event.is_set():
            break

        task_hash = task.get("task_id", str(uuid.uuid4()))

        if checkpoint_db.is_completed(task_hash):
            skipped += 1
            continue

        checkpoint_db.mark_running(task_hash, agent_id)

        if dry_run:
            checkpoint_db.mark_completed(task_hash, "dry-run")
            completed += 1
            consecutive_failures = 0
            continue

        try:
            code = task.get("prompt", "") + task.get("canonical_solution", "")
            test = task.get("test", "")
            script = code + "\n" + test
            result = sandbox.run(script, timeout=30)

            if result.exit_code == 0 and not result.timed_out:
                checkpoint_db.mark_completed(task_hash, "success")
                completed += 1
                consecutive_failures = 0
                await training_queue.put(
                    TrainingRequest(session_id=task_hash, task_type="swarm")
                )
            else:
                checkpoint_db.mark_failed(task_hash, agent_id)
                failed += 1
                consecutive_failures += 1
        except Exception:
            logger.exception("Agent %s failed on task %s", agent_id, task_hash)
            checkpoint_db.mark_failed(task_hash, agent_id)
            failed += 1
            consecutive_failures += 1

        if consecutive_failures >= max_failures:
            logger.warning(
                "Agent %s retiring after %d consecutive failures",
                agent_id,
                max_failures,
            )
            break

        # Backoff on failure
        if consecutive_failures > 0:
            await asyncio.sleep(min(2**consecutive_failures, 30))

    return {
        "agent_id": agent_id,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }


async def run_swarm(config: SwarmConfig, dry_run: bool = False) -> None:
    """Tier-1 orchestrator — runs the full swarm with asyncio.TaskGroup.

    Args:
        config: Swarm configuration.
        dry_run: If True, skip actual execution and training.
    """
    from adapter_registry import AdapterRegistry
    from sqlmodel import create_engine

    logger.info("Starting swarm with config: %s", config.model_dump())

    # Detect hardware
    probe = HardwareProbe.detect()
    budget = probe.compute_budget()
    logger.info("Hardware: %s, Budget: %s", probe, budget)

    # Initialize registry and checkpoint DB
    registry_engine = create_engine(config.db_url)
    registry = AdapterRegistry(engine=registry_engine)
    checkpoint_engine = create_engine("sqlite:///swarm_checkpoints.db")
    checkpoint_db = SwarmCheckpointDB(checkpoint_engine)

    # Load tasks
    tasks = load_task_pool(config.task_source)
    if not tasks:
        logger.warning("No tasks to process")
        return

    # Shared state
    training_queue: asyncio.Queue[TrainingRequest] = asyncio.Queue()
    shutdown_event = asyncio.Event()
    num_agents = min(config.population_size, budget.max_agents)

    # Partition tasks across agents
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(num_agents)]
    for i, task in enumerate(tasks):
        partitions[i % num_agents].append(task)

    async with asyncio.TaskGroup() as tg:
        # Agent supervisors
        agent_tasks = []
        for i in range(num_agents):
            t = tg.create_task(
                agent_supervisor(
                    agent_id=f"agent-{i}",
                    tasks=partitions[i],
                    checkpoint_db=checkpoint_db,
                    training_queue=training_queue,
                    dry_run=dry_run,
                    shutdown_event=shutdown_event,
                )
            )
            agent_tasks.append(t)

        # Training pool (background)
        tg.create_task(
            training_pool_manager(
                training_queue=training_queue,
                budget=budget,
                db_url=config.db_url,
                shutdown_event=shutdown_event,
            )
        )

        # Evolution worker (background)
        tg.create_task(
            evolution_worker(
                registry=registry,
                interval_seconds=config.evolution_interval,
                shutdown_event=shutdown_event,
            )
        )

        # Memory watchdog (background)
        tg.create_task(
            memory_watchdog(
                budget=budget,
                agents=agent_tasks,
                shutdown_event=shutdown_event,
            )
        )

        # Wait for all agents to finish, then shutdown background tasks
        done = await asyncio.gather(*agent_tasks, return_exceptions=True)
        shutdown_event.set()

        for result in done:
            if isinstance(result, Exception):
                logger.error("Agent failed: %s", result)
            else:
                logger.info("Agent result: %s", result)


def main() -> None:
    """CLI entry point for the swarm orchestrator."""
    parser = argparse.ArgumentParser(description="Rune Swarm Orchestrator")
    parser.add_argument(
        "--config",
        default="scripts/swarm_config.yaml",
        help="Path to swarm config YAML",
    )
    parser.add_argument("--agents", type=int, help="Override population_size")
    parser.add_argument("--hours", type=float, help="Max runtime in hours")
    parser.add_argument("--dry-run", action="store_true", help="Skip execution")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config_path = Path(args.config)
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text())
        config = SwarmConfig(**raw)
    else:
        config = SwarmConfig()

    if args.agents:
        config.population_size = args.agents

    asyncio.run(run_swarm(config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
