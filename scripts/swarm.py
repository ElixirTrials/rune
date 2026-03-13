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
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# Add scripts dir to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

from shared.checkpoint_db import SwarmCheckpointDB
from shared.hardware import HardwareProbe
from shared.rune_models import SwarmConfig
from shared.sandbox import get_sandbox_backend
from swarm_evolution import evolution_worker
from swarm_workers import TrainingRequest, memory_watchdog, training_pool_manager

if TYPE_CHECKING:
    from adapter_registry import AdapterRegistry

logger = logging.getLogger(__name__)

_MAX_BACKOFF_SECONDS = 30


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


def _register_iteration_adapter(
    iter_result: dict[str, Any],
    task_hash: str,
    agent_id: str,
    registry: "AdapterRegistry",
    base_model_id: str,
) -> None:
    """Register the final adapter from an iteration loop in the registry.

    Examines the iteration results to find the last adapter produced,
    registers it with a fitness score derived from test pass rates.

    Args:
        iter_result: Return value from run_project().
        task_hash: Stable hash for the task.
        agent_id: ID of the agent that produced the adapter.
        registry: Adapter registry for storing metadata.
        base_model_id: Base model the adapter was generated for.
    """
    from adapter_registry.models import AdapterRecord

    adapter_dir = iter_result.get("adapter_dir", "")
    iterations = iter_result.get("iterations", [])
    if not iterations or not adapter_dir:
        return

    # Find the last adapter directory on disk
    adapter_base = Path(adapter_dir)
    if not adapter_base.exists():
        return

    adapter_subdirs = sorted(adapter_base.iterdir())
    if not adapter_subdirs:
        return

    # Use the last adapter produced
    last_adapter = adapter_subdirs[-1]
    safetensors = last_adapter / "adapter_model.safetensors"
    if not safetensors.exists():
        return

    # Compute fitness from iteration results
    total_iters = len(iterations)
    passing_iters = sum(1 for it in iterations if it.get("tests_passed", False))
    pass_rate = passing_iters / total_iters if total_iters > 0 else 0.0

    from evaluation.metrics import evaluate_fitness

    fitness = evaluate_fitness(
        adapter_id=f"{task_hash}-{agent_id}",
        pass_rate=pass_rate,
        diversity_score=0.5,
    )

    now = datetime.now(timezone.utc).isoformat()
    file_size = safetensors.stat().st_size

    record = AdapterRecord(
        id=f"{task_hash}-{agent_id}",
        version=1,
        task_type="iteration",
        base_model_id=base_model_id,
        rank=8,
        created_at=now,
        file_path=str(last_adapter),
        file_hash="",
        file_size_bytes=file_size,
        pass_rate=pass_rate,
        fitness_score=fitness,
        source="iteration",
        session_id=iter_result.get("session_id", ""),
        agent_id=agent_id,
        generation=0,
    )

    try:
        registry.store(record)
        logger.info(
            "Registered adapter %s (fitness=%.3f, pass_rate=%.2f)",
            record.id,
            fitness,
            pass_rate,
        )
    except Exception:
        logger.exception("Failed to register adapter %s", record.id)


async def agent_supervisor(
    agent_id: str,
    tasks: list[dict[str, Any]],
    checkpoint_db: SwarmCheckpointDB,
    training_queue: asyncio.Queue[TrainingRequest],
    registry: "AdapterRegistry",
    base_model_id: str = "Qwen/Qwen2.5-Coder-7B",
    device: str = "cpu",
    dry_run: bool = False,
    max_failures: int = 5,
    shutdown_event: asyncio.Event | None = None,
    use_iteration_loop: bool = False,
    hypernetwork_checkpoint: str | None = None,
    max_iterations: int = 5,
) -> dict[str, Any]:
    """Tier-2 supervisor for a single swarm agent.

    Processes tasks from the pool, checking checkpoints for dedup,
    running code via sandbox, and queuing training requests.

    When use_iteration_loop=True, uses rune_runner's hypernetwork iteration
    loop instead of the standard sandbox approach. Each agent runs its own
    adapter chain per task, and registers the resulting adapter in the registry.

    Args:
        agent_id: Unique identifier for this agent.
        tasks: List of task dicts to process.
        checkpoint_db: Checkpoint DB for dedup and recovery.
        training_queue: Queue for submitting training requests.
        registry: Adapter registry for storing adapter metadata.
        base_model_id: HuggingFace model ID for adapter config.
        device: Device for hypernetwork computation.
        dry_run: If True, skip actual execution.
        max_failures: Consecutive failures before retiring.
        shutdown_event: Event to signal graceful shutdown.
        use_iteration_loop: Use hypernetwork iteration loop per task.
        hypernetwork_checkpoint: Path to pretrained hypernetwork .pt file.
        max_iterations: Max iterations for iteration loop mode.

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

        # Use task_id if provided; otherwise derive a stable hash from the task
        # content so retries do not create a new identity for the same work.
        task_hash = task.get(
            "task_id",
            hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:16],
        )

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
            if use_iteration_loop:
                # Hypernetwork iteration loop — each agent gets its own
                # adapter chain for the task
                from rune_runner import run_project as run_iteration_project

                project_prompt = task.get("task_description", task.get("prompt", ""))
                iter_result = await run_iteration_project(
                    project_prompt=project_prompt,
                    max_iterations=max_iterations,
                    checkpoint_path=hypernetwork_checkpoint,
                    base_model_id=base_model_id,
                    device=device,
                )

                # Register the adapter in the registry for evolution
                _register_iteration_adapter(
                    iter_result=iter_result,
                    task_hash=task_hash,
                    agent_id=agent_id,
                    registry=registry,
                    base_model_id=base_model_id,
                )

                if iter_result["final_tests_passed"]:
                    checkpoint_db.mark_completed(task_hash, "success")
                    completed += 1
                    consecutive_failures = 0
                    await training_queue.put(
                        TrainingRequest(session_id=task_hash, task_type="iteration")
                    )
                else:
                    checkpoint_db.mark_failed(task_hash, agent_id)
                    failed += 1
                    consecutive_failures += 1
            else:
                # Standard sandbox execution
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
        await asyncio.sleep(min(2**consecutive_failures, _MAX_BACKOFF_SECONDS))

    return {
        "agent_id": agent_id,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }


async def run_swarm(config: SwarmConfig, dry_run: bool = False) -> dict[str, Any]:
    """Tier-1 orchestrator — runs the full swarm with asyncio.TaskGroup.

    Args:
        config: Swarm configuration.
        dry_run: If True, skip actual execution and training.

    Returns:
        Summary dict with agent results and evolution sweep results.
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
        return {"agents": [], "evolution": None}

    # Shared state
    training_queue: asyncio.Queue[TrainingRequest] = asyncio.Queue()
    shutdown_event = asyncio.Event()
    num_agents = min(config.population_size, budget.max_agents)

    # Partition tasks across agents (round-robin)
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(num_agents)]
    for i, task in enumerate(tasks):
        partitions[i % num_agents].append(task)

    use_iteration = bool(config.hypernetwork_checkpoint)

    agent_results: list[Any] = []

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
                    registry=registry,
                    base_model_id=config.base_model_id,
                    device=config.device,
                    dry_run=dry_run,
                    shutdown_event=shutdown_event,
                    use_iteration_loop=use_iteration,
                    hypernetwork_checkpoint=config.hypernetwork_checkpoint,
                    max_iterations=config.max_iterations,
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
                agent_results.append({"error": str(result)})
            else:
                logger.info("Agent result: %s", result)
                agent_results.append(result)

    # Final evolution sweep after all agents complete
    from swarm_evolution import evolution_sweep

    evolution_result = evolution_sweep(registry)
    logger.info("Final evolution sweep: %s", evolution_result)

    return {
        "agents": agent_results,
        "evolution": evolution_result,
        "registry": registry,
    }


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
