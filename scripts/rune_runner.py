"""Rune Runner: Outer iteration loop with hypernetwork state machine.

The hypernetwork carries the ENTIRE project lifecycle state through weight-space.
The prompt is the launch trigger and stays constant. Each iteration's output
(plan, code, test results) gets encoded into an adapter via hypernetwork, and
the NEXT iteration loads that adapter.

Iteration 0 (bootstrap):
  methodology doc → H() → methodology_adapter

Iteration 1 (planning):
  project prompt + methodology_adapter → plan output → H() → plan_adapter

Iteration 2+ (coding/fixing):
  project prompt + latest_adapter → code → execute → H() → next_adapter

Usage:
    uv run scripts/rune_runner.py --project "Build a Python statistics library"
    uv run scripts/rune_runner.py --project "..." --max-iterations 10
    uv run scripts/rune_runner.py --project "..." --model-path /path/to/model.gguf
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()  # noqa: E402

from shared.hardware import get_best_device  # noqa: E402

logger = logging.getLogger(__name__)

# Base model ID for adapter config
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
METHODOLOGY_DOC_PATH = Path(__file__).parent.parent / "docs" / "rune-methodology.md"
ADAPTER_BASE_DIR = Path.home() / ".rune" / "adapters"


def _load_methodology() -> str:
    """Load the methodology document for bootstrap iteration."""
    if METHODOLOGY_DOC_PATH.exists():
        return METHODOLOGY_DOC_PATH.read_text()
    logger.warning(
        "Methodology doc not found at %s, using inline fallback",
        METHODOLOGY_DOC_PATH,
    )
    return (
        "Phase 1: Decompose the project into tasks with dependencies.\n"
        "Phase 2: For each task, generate code with a test suite.\n"
        "Phase 3: Execute tests, validate, iterate on failures.\n"
        "Phase 4: Integrate components, run the full test suite.\n"
        "Output code as ```python blocks. Output plans as numbered lists.\n"
    )


def _build_trajectory_text(state: dict[str, Any], iteration: int) -> str:
    """Build trajectory text from iteration state for hypernetwork encoding.

    Args:
        state: The RuneState dict after iteration execution.
        iteration: Current iteration number.

    Returns:
        Trajectory text combining code, stdout, stderr, and test results.
    """
    parts = [f"=== Iteration {iteration} ==="]

    if state.get("generated_code"):
        parts.append(f"Code:\n{state['generated_code']}")
    if state.get("stdout"):
        parts.append(f"stdout:\n{state['stdout']}")
    if state.get("stderr"):
        parts.append(f"stderr:\n{state['stderr']}")
    parts.append(f"tests_passed: {state.get('tests_passed', False)}")
    parts.append(f"exit_code: {state.get('exit_code', -1)}")

    return "\n\n".join(parts)


async def run_iteration(
    graph: Any,
    project_prompt: str,
    adapter_id: str | None,
    session_id: str,
    iteration: int,
) -> dict[str, Any]:
    """Run one iteration of the agent loop.

    Args:
        graph: Compiled single-iteration LangGraph.
        project_prompt: The constant project definition prompt.
        adapter_id: Current adapter to load, or None for base model.
        session_id: Session ID for trajectory tracking.
        iteration: Current iteration number.

    Returns:
        Final state dict from the graph execution.
    """
    initial_state = {
        "task_description": project_prompt,
        "task_type": "project",
        "test_suite": "",
        "adapter_ids": [adapter_id] if adapter_id else [],
        "session_id": f"{session_id}-iter{iteration}",
        "attempt_count": 0,
        "max_attempts": 1,  # Single attempt per iteration
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": False,
        "trajectory": [],
        "outcome": None,
    }

    return await graph.ainvoke(initial_state)


def run_hypernetwork(
    trajectory_text: str,
    output_dir: str,
    base_model_id: str,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> str | None:
    """Run the hypernetwork to produce a new adapter from trajectory.

    If no checkpoint is available, logs a warning and returns None
    (the system falls back to base model inference).

    Args:
        trajectory_text: Accumulated trajectory to encode.
        output_dir: Where to save the adapter.
        base_model_id: Base model identifier.
        checkpoint_path: Path to pretrained hypernetwork checkpoint.
        device: Device for computation ('cpu', 'mps', 'cuda').

    Returns:
        Path to saved adapter directory, or None if hypernetwork unavailable.
    """
    if not checkpoint_path or not Path(checkpoint_path).exists():
        logger.warning(
            "No hypernetwork checkpoint at %s — skipping adapter generation",
            checkpoint_path,
        )
        return None

    try:
        from model_training.hypernetwork import (
            generate_adapter,
            load_pretrained,
        )

        hypernetwork = load_pretrained(checkpoint_path, device=device)
        return generate_adapter(
            hypernetwork=hypernetwork,
            trajectory_text=trajectory_text,
            output_dir=output_dir,
            base_model_id=base_model_id,
            device=device,
        )
    except Exception:
        logger.exception("Hypernetwork adapter generation failed")
        return None


async def run_project(
    project_prompt: str,
    max_iterations: int = 10,
    checkpoint_path: str | None = None,
    base_model_id: str = DEFAULT_BASE_MODEL,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the full project lifecycle as an iteration state machine.

    Args:
        project_prompt: Natural language project definition.
        max_iterations: Maximum number of iterations.
        checkpoint_path: Path to pretrained hypernetwork checkpoint.
        base_model_id: HuggingFace model ID of the base model.
        device: Device for hypernetwork computation.

    Returns:
        Summary dict with iteration results.
    """
    from rune_agent.graph import create_single_iteration_graph

    graph = create_single_iteration_graph()
    session_id = f"rune-{uuid.uuid4().hex[:8]}"
    adapter_dir = ADAPTER_BASE_DIR / session_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Accumulated trajectory across all iterations
    accumulated_trajectory = ""
    current_adapter_id: str | None = None
    current_adapter_path: str | None = None
    iteration_results: list[dict[str, Any]] = []

    # --- Iteration 0: Bootstrap (methodology → adapter) ---
    logger.info("=== Iteration 0: Bootstrap ===")
    methodology = _load_methodology()
    accumulated_trajectory += f"=== Bootstrap ===\n{methodology}\n\n"

    bootstrap_adapter_dir = str(adapter_dir / "iter0_methodology")
    adapter_path = run_hypernetwork(
        trajectory_text=accumulated_trajectory,
        output_dir=bootstrap_adapter_dir,
        base_model_id=base_model_id,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    if adapter_path:
        current_adapter_id = "methodology"
        current_adapter_path = adapter_path
        # Register adapter with the provider
        from inference import get_provider

        provider = get_provider()
        await provider.load_adapter(current_adapter_id, current_adapter_path)
        logger.info("Bootstrap adapter loaded: %s", current_adapter_id)
    else:
        logger.info("No hypernetwork checkpoint — running with base model only")

    # --- Iterations 1..N: Planning then coding ---
    for iteration in range(1, max_iterations + 1):
        logger.info(
            "=== Iteration %d/%d (adapter=%s) ===",
            iteration,
            max_iterations,
            current_adapter_id or "none",
        )

        # Run one iteration with constant project prompt
        state = await run_iteration(
            graph=graph,
            project_prompt=project_prompt,
            adapter_id=current_adapter_id,
            session_id=session_id,
            iteration=iteration,
        )

        # Record result
        result = {
            "iteration": iteration,
            "adapter_id": current_adapter_id,
            "tests_passed": state.get("tests_passed", False),
            "exit_code": state.get("exit_code", -1),
            "generated_code": state.get("generated_code", ""),
        }
        iteration_results.append(result)

        # Build trajectory from this iteration's output
        iter_trajectory = _build_trajectory_text(state, iteration)
        accumulated_trajectory += iter_trajectory + "\n\n"

        # Check if project is complete (tests pass)
        if state.get("tests_passed"):
            logger.info("Project completed at iteration %d!", iteration)
            break

        # Run hypernetwork on accumulated trajectory → new adapter
        iter_adapter_dir = str(adapter_dir / f"iter{iteration}")
        new_adapter_path = run_hypernetwork(
            trajectory_text=accumulated_trajectory,
            output_dir=iter_adapter_dir,
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        if new_adapter_path:
            new_adapter_id = f"progress_v{iteration}"
            from inference import get_provider

            provider = get_provider()
            # Unload previous adapter if any
            if current_adapter_id:
                try:
                    await provider.unload_adapter(current_adapter_id)
                except Exception:
                    pass
            # Load new adapter
            await provider.load_adapter(new_adapter_id, new_adapter_path)
            current_adapter_id = new_adapter_id
            current_adapter_path = new_adapter_path
            logger.info("New adapter loaded: %s", current_adapter_id)

    return {
        "session_id": session_id,
        "total_iterations": len(iteration_results),
        "project_prompt": project_prompt,
        "final_tests_passed": iteration_results[-1]["tests_passed"]
        if iteration_results
        else False,
        "iterations": iteration_results,
        "adapter_dir": str(adapter_dir),
    }


def main() -> None:
    """CLI entry point for the Rune iteration runner."""
    parser = argparse.ArgumentParser(
        description="Rune E2E Runner — Hypernetwork iteration state machine"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project definition (natural language prompt)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of iterations (default: 10)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to pretrained hypernetwork checkpoint (.pt)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to GGUF model (sets LLAMACPP_MODEL_PATH)",
    )
    parser.add_argument(
        "--base-model-id",
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace model ID for adapter config",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for hypernetwork computation (cpu, mps, cuda)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Configure provider based on args
    if args.model_path:
        os.environ["INFERENCE_PROVIDER"] = "llamacpp"
        os.environ["LLAMACPP_MODEL_PATH"] = args.model_path
    elif "INFERENCE_PROVIDER" not in os.environ:
        os.environ["INFERENCE_PROVIDER"] = "ollama"

    result = asyncio.run(
        run_project(
            project_prompt=args.project,
            max_iterations=args.max_iterations,
            checkpoint_path=args.checkpoint,
            base_model_id=args.base_model_id,
            device=args.device if args.device != "auto" else get_best_device(),
        )
    )

    logger.info("=== Run Complete ===")
    logger.info("Session: %s", result["session_id"])
    logger.info("Iterations: %d", result["total_iterations"])
    logger.info("Final tests passed: %s", result["final_tests_passed"])
    logger.info("Adapters saved to: %s", result["adapter_dir"])


if __name__ == "__main__":
    main()
