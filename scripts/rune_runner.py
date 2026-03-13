"""Rune Runner: Multi-phase swarm pipeline with Jinja2 template-driven adapters.

4-phase sequential pipeline where Phases 2 and 3 run as parallel swarms:

Phase 1: DECOMPOSE (single agent)
  trajectory = render("decompose.j2", project=project_spec)
  base model + decompose_adapter → model outputs subtask list

Phase 2: PLAN (swarm — N agents in parallel)
  for each subtask:
    trajectory = render("plan.j2", project=project_spec, subtask=subtask)
    base model + plan_adapter → architecture plan

Phase 3: CODE (swarm — N agents in parallel, with retries via H())
  for each subtask:
    trajectory = render("code.j2", subtask=subtask, plan=plan, skeleton=prior_code)
    on retry: render("code_retry.j2", ..., errors=error_summary, history=hist)
    base model + code_adapter → working code

Phase 4: INTEGRATE (single agent)
  trajectory = render("integrate.j2", project=project_spec, skeletons=all_code)
  base model + integrate_adapter → final integrated codebase

All instructions and state flow through adapter weights, not prompts.
Phase instructions are stored as Jinja2 templates.

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
import re
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()  # noqa: E402

from shared.hardware import get_best_device  # noqa: E402
from shared.template_loader import render_prompt, render_trajectory  # noqa: E402

logger = logging.getLogger(__name__)

# Base model ID for adapter config
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
ADAPTER_BASE_DIR = Path.home() / ".rune" / "adapters"


# ---------------------------------------------------------------------------
# Helpers — kept from original (used by template variable preparation)
# ---------------------------------------------------------------------------


def _extract_error_summary(stderr: str) -> str:
    """Extract the most actionable error info from raw stderr.

    Pulls out the final exception line and the most relevant traceback
    context rather than dumping the entire stderr.
    """
    if not stderr:
        return ""
    lines = stderr.strip().splitlines()

    error_lines = [
        ln
        for ln in lines
        if "Error" in ln or "Exception" in ln or "assert" in ln.lower()
    ]
    final_error = error_lines[-1].strip() if error_lines else lines[-1].strip()

    failed_tests = [
        ln.strip()
        for ln in lines
        if ln.strip().startswith("FAIL:") or ln.strip().startswith("ERROR:")
    ]

    parts = []
    if failed_tests:
        parts.append("Failed: " + "; ".join(failed_tests[:5]))
    parts.append(final_error)
    return "\n".join(parts)


def _extract_code_skeleton(code: str) -> str:
    """Compress generated code into a structural skeleton.

    Keeps class/function signatures, docstrings, and import lines.
    Drops implementation bodies to save tokens for the perceiver.
    """
    if not code:
        return ""
    skeleton_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith(("import ", "from "))
            or stripped.startswith(("class ", "def "))
            or stripped.startswith('"""')
            or stripped.startswith("@")
            or stripped.startswith(("raise ", "return "))
            or "self." in stripped
            and "=" in stripped
            and stripped.startswith("self.")
        ):
            skeleton_lines.append(line)
    return "\n".join(skeleton_lines[:40])


def _count_test_results(stdout: str, stderr: str) -> tuple[int, int]:
    """Parse unittest output to count passed/failed tests."""
    total = 0
    failed = 0

    ran_match = re.search(r"Ran (\d+) test", stderr or "")
    if ran_match:
        total = int(ran_match.group(1))

    fail_match = re.search(r"failures=(\d+)", stderr or "")
    err_match = re.search(r"errors=(\d+)", stderr or "")
    if fail_match:
        failed += int(fail_match.group(1))
    if err_match:
        failed += int(err_match.group(1))

    passed = max(0, total - failed)
    return passed, total


def _extract_failed_tests(stderr: str) -> str:
    """Extract names of failing tests from unittest stderr."""
    if not stderr:
        return ""
    lines = stderr.strip().splitlines()
    failed = [
        ln.strip()
        for ln in lines
        if ln.strip().startswith("FAIL:") or ln.strip().startswith("ERROR:")
    ]
    return "; ".join(failed[:5])


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------


def _parse_subtask_list(model_output: str) -> list[dict[str, str]]:
    """Parse Phase 1 output into [{name, description}, ...].

    Expects numbered list format: ``1. name — description``
    Also accepts ``1. name - description`` or ``1. name: description``.
    """
    subtasks: list[dict[str, str]] = []
    for line in model_output.splitlines():
        line = line.strip()
        # Match: "1. name — description" or "1. name - description"
        match = re.match(r"^\d+\.\s*(.+?)\s*(?:—|-|:)\s*(.+)$", line)
        if match:
            subtasks.append(
                {
                    "name": match.group(1).strip(),
                    "description": match.group(2).strip(),
                }
            )
    if not subtasks:
        # Fallback: treat entire output as a single subtask
        subtasks.append(
            {
                "name": "implementation",
                "description": model_output[:200].strip(),
            }
        )
    return subtasks


def _parse_plan(model_output: str) -> str:
    """Extract plan text from Phase 2 output."""
    return model_output.strip()


# ---------------------------------------------------------------------------
# Core iteration runner
# ---------------------------------------------------------------------------


async def run_iteration(
    graph: Any,
    project_prompt: str,
    adapter_id: str | None,
    session_id: str,
    iteration: int,
    phase: str | None = None,
) -> dict[str, Any]:
    """Run one iteration of the agent loop.

    Args:
        graph: Compiled single-iteration LangGraph.
        project_prompt: The constant project definition prompt.
        adapter_id: Current adapter to load, or None for base model.
        session_id: Session ID for trajectory tracking.
        iteration: Current iteration number.
        phase: Pipeline phase for template-driven prompts.

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
        "max_attempts": 1,
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": False,
        "trajectory": [],
        "phase": phase,
        "outcome": None,
    }

    return await graph.ainvoke(initial_state)


# ---------------------------------------------------------------------------
# Hypernetwork adapter generation
# ---------------------------------------------------------------------------


def _is_sakana_checkpoint(checkpoint_path: str) -> bool:
    """Detect whether a checkpoint is a Sakana HyperLoRA checkpoint."""
    import torch  # noqa: PLC0415

    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hc = sd.get("hypernet_config")
    return hc is not None and not isinstance(hc, dict)


def run_hypernetwork(
    trajectory_text: str,
    output_dir: str,
    base_model_id: str,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> str | None:
    """Run the hypernetwork to produce a new adapter from trajectory.

    Supports two checkpoint formats:
      - Sakana Doc-to-LoRA (ctx_to_lora HyperLoRA perceiver)
      - Rune DocToLoraHypernetwork (token-embedding perceiver)

    If no checkpoint is available, logs a warning and returns None
    (the system falls back to base model inference).

    Args:
        trajectory_text: Structured trajectory to encode.
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
        if _is_sakana_checkpoint(checkpoint_path):
            from model_training.sakana_d2l import generate_adapter_from_sakana

            return generate_adapter_from_sakana(
                text=trajectory_text,
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                base_model_name=base_model_id,
                device=device,
            )
        else:
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


# ---------------------------------------------------------------------------
# Adapter loading helper
# ---------------------------------------------------------------------------


async def _load_adapter(
    adapter_id: str,
    adapter_path: str,
    current_adapter_id: str | None,
) -> str:
    """Load an adapter into the inference provider, unloading the previous one."""
    from inference import get_provider

    provider = get_provider()
    if current_adapter_id:
        try:
            await provider.unload_adapter(current_adapter_id)
        except Exception:
            pass
    await provider.load_adapter(adapter_id, adapter_path)
    return adapter_id


# ---------------------------------------------------------------------------
# Multi-phase pipeline
# ---------------------------------------------------------------------------


async def run_phased_pipeline(
    project_prompt: str,
    max_iterations: int = 10,
    checkpoint_path: str | None = None,
    base_model_id: str = DEFAULT_BASE_MODEL,
    device: str = "cpu",
    population_size: int = 2,
    max_retries_per_subtask: int = 3,
) -> dict[str, Any]:
    """Run the 4-phase pipeline with template-driven adapters.

    Phase 1: DECOMPOSE — single agent decomposes project into subtasks
    Phase 2: PLAN — parallel swarm plans each subtask
    Phase 3: CODE — parallel swarm codes each subtask (with retries via H())
    Phase 4: INTEGRATE — single agent integrates all code

    Args:
        project_prompt: Natural language project definition.
        max_iterations: Maximum total iterations across all phases.
        checkpoint_path: Path to pretrained hypernetwork checkpoint.
        base_model_id: HuggingFace model ID of the base model.
        device: Device for hypernetwork computation.
        population_size: Number of parallel agents for swarm phases.
        max_retries_per_subtask: Max retry attempts per subtask in Phase 3.

    Returns:
        Summary dict with phase results, adapters, and final code.
    """
    from rune_agent.graph import create_single_iteration_graph

    graph = create_single_iteration_graph()
    session_id = f"rune-{uuid.uuid4().hex[:8]}"
    adapter_dir = ADAPTER_BASE_DIR / session_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    phase_results: dict[str, Any] = {}
    registered_adapters: list[dict[str, str]] = []
    iteration_counter = 0

    # ---------------------------------------------------------------
    # Phase 1: DECOMPOSE (single agent)
    # ---------------------------------------------------------------
    logger.info("=== Phase 1: DECOMPOSE ===")
    decompose_trajectory = render_trajectory("decompose", project=project_prompt)
    logger.info("Decompose trajectory: %d chars", len(decompose_trajectory))

    decompose_adapter_dir = str(adapter_dir / "phase1_decompose")
    decompose_adapter_path = run_hypernetwork(
        trajectory_text=decompose_trajectory,
        output_dir=decompose_adapter_dir,
        base_model_id=base_model_id,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    current_adapter_id: str | None = None
    if decompose_adapter_path:
        current_adapter_id = await _load_adapter(
            "phase1-decompose", decompose_adapter_path, None
        )
        registered_adapters.append(
            {
                "adapter_id": current_adapter_id,
                "task_type": "phase1-decompose",
                "phase": "decompose",
            }
        )

    # Run decompose iteration with phase-specific prompt
    decompose_prompt = render_prompt("decompose", task_description=project_prompt)
    iteration_counter += 1
    state = await run_iteration(
        graph=graph,
        project_prompt=decompose_prompt,
        adapter_id=current_adapter_id,
        session_id=session_id,
        iteration=iteration_counter,
        phase="decompose",
    )

    subtasks = _parse_subtask_list(state.get("generated_code", ""))
    logger.info(
        "Decomposed into %d subtasks: %s",
        len(subtasks),
        [s["name"] for s in subtasks],
    )
    phase_results["decompose"] = {
        "subtasks": subtasks,
        "adapter_id": current_adapter_id,
    }

    # ---------------------------------------------------------------
    # Phase 2: PLAN (parallel swarm — one agent per subtask)
    # ---------------------------------------------------------------
    logger.info("=== Phase 2: PLAN ===")
    plans: dict[str, str] = {}

    async def _plan_subtask(idx: int, subtask: dict[str, str]) -> tuple[str, str]:
        """Plan a single subtask — runs as a parallel coroutine."""
        traj = render_trajectory(
            "plan",
            subtask=subtask,
            subtask_index=idx + 1,
            total_subtasks=len(subtasks),
            project=project_prompt,
        )
        plan_adapter_dir = str(adapter_dir / f"phase2_plan_{subtask['name']}")
        plan_adapter_path = run_hypernetwork(
            trajectory_text=traj,
            output_dir=plan_adapter_dir,
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        plan_adapter_id: str | None = None
        if plan_adapter_path:
            plan_adapter_id = f"phase2-plan-{subtask['name']}"
            await _load_adapter(plan_adapter_id, plan_adapter_path, None)
            registered_adapters.append(
                {
                    "adapter_id": plan_adapter_id,
                    "task_type": f"phase2-plan-{subtask['name']}",
                    "phase": "plan",
                }
            )

        plan_prompt = render_prompt("plan", task_description=subtask["description"])
        plan_state = await run_iteration(
            graph=graph,
            project_prompt=plan_prompt,
            adapter_id=plan_adapter_id,
            session_id=session_id,
            iteration=iteration_counter + idx + 1,
            phase="plan",
        )
        return subtask["name"], _parse_plan(plan_state.get("generated_code", ""))

    # Run planning in parallel
    plan_tasks = [_plan_subtask(i, st) for i, st in enumerate(subtasks)]
    plan_results = await asyncio.gather(*plan_tasks)
    for name, plan_text in plan_results:
        plans[name] = plan_text
    iteration_counter += len(subtasks)

    logger.info("Plans generated for %d subtasks", len(plans))
    phase_results["plan"] = {"plans": {k: v[:200] for k, v in plans.items()}}

    # ---------------------------------------------------------------
    # Phase 3: CODE (parallel swarm, with retries via H())
    # ---------------------------------------------------------------
    logger.info("=== Phase 3: CODE ===")
    code_outputs: dict[str, str] = {}

    async def _code_subtask(idx: int, subtask: dict[str, str]) -> tuple[str, str]:
        """Code a single subtask with retry loop — runs as a parallel coroutine."""
        plan = plans.get(subtask["name"], "")
        existing_code = ""
        last_state: dict[str, Any] | None = None

        for attempt in range(max_retries_per_subtask):
            if attempt == 0:
                traj = render_trajectory(
                    "code",
                    subtask=subtask,
                    subtask_index=idx + 1,
                    total_subtasks=len(subtasks),
                    plan=plan,
                    existing_code=existing_code,
                )
            else:
                # Retry: build trajectory with error context
                assert last_state is not None  # guaranteed by attempt > 0
                passed, total = _count_test_results(
                    last_state.get("stdout", ""),
                    last_state.get("stderr", ""),
                )
                error_summary = _extract_error_summary(last_state.get("stderr", ""))
                failed_tests = _extract_failed_tests(last_state.get("stderr", ""))
                tests_passed = last_state.get("tests_passed", False)

                # Build fix guidance
                if total == 0:
                    fix_guidance = (
                        "Code failed to execute. Check syntax, imports, indentation."
                    )
                elif passed == 0:
                    fix_guidance = "No tests pass. Focus on basic structure first."
                else:
                    fix_guidance = (
                        f"{total - passed} test(s) failing. "
                        "Fix without breaking passing tests."
                    )

                traj = render_trajectory(
                    "code_retry",
                    subtask=subtask,
                    attempt=attempt + 1,
                    max_retries=max_retries_per_subtask,
                    plan=plan,
                    existing_code=existing_code,
                    passed=passed,
                    total=total,
                    tests_passed=tests_passed,
                    error_summary=error_summary,
                    failed_tests=failed_tests,
                    fix_guidance=fix_guidance,
                    history=None,
                )

            code_adapter_dir = str(
                adapter_dir / f"phase3_code_{subtask['name']}_v{attempt}"
            )
            code_adapter_path = run_hypernetwork(
                trajectory_text=traj,
                output_dir=code_adapter_dir,
                base_model_id=base_model_id,
                checkpoint_path=checkpoint_path,
                device=device,
            )

            code_adapter_id: str | None = None
            if code_adapter_path:
                code_adapter_id = f"phase3-code-{subtask['name']}-v{attempt}"
                await _load_adapter(code_adapter_id, code_adapter_path, None)
                registered_adapters.append(
                    {
                        "adapter_id": code_adapter_id,
                        "task_type": f"phase3-code-{subtask['name']}",
                        "phase": "code",
                    }
                )

            code_prompt = render_prompt("code", task_description=subtask["description"])
            code_state = await run_iteration(
                graph=graph,
                project_prompt=code_prompt,
                adapter_id=code_adapter_id,
                session_id=session_id,
                iteration=iteration_counter
                + idx * max_retries_per_subtask
                + attempt
                + 1,
                phase="code",
            )

            last_state = code_state
            existing_code = code_state.get("generated_code", "")

            if code_state.get("tests_passed"):
                logger.info(
                    "  Subtask '%s' PASSED on attempt %d",
                    subtask["name"],
                    attempt + 1,
                )
                return subtask["name"], existing_code

            logger.info(
                "  Subtask '%s' FAILED attempt %d — retrying via H()",
                subtask["name"],
                attempt + 1,
            )

        # Exhausted retries — return last attempt's code
        logger.warning(
            "Subtask '%s' failed after %d attempts",
            subtask["name"],
            max_retries_per_subtask,
        )
        return subtask["name"], existing_code

    # Run coding in parallel
    code_tasks = [_code_subtask(i, st) for i, st in enumerate(subtasks)]
    code_results = await asyncio.gather(*code_tasks)
    for name, code_text in code_results:
        code_outputs[name] = code_text
    iteration_counter += len(subtasks) * max_retries_per_subtask

    logger.info("Code generated for %d subtasks", len(code_outputs))
    phase_results["code"] = {
        "outputs": {k: v[:200] for k, v in code_outputs.items()},
    }

    # ---------------------------------------------------------------
    # Phase 4: INTEGRATE (single agent)
    # ---------------------------------------------------------------
    logger.info("=== Phase 4: INTEGRATE ===")
    skeletons = {
        name: _extract_code_skeleton(code) for name, code in code_outputs.items()
    }
    integrate_trajectory = render_trajectory(
        "integrate",
        project=project_prompt,
        subtask_count=len(subtasks),
        skeletons=skeletons,
    )

    integrate_adapter_dir = str(adapter_dir / "phase4_integrate")
    integrate_adapter_path = run_hypernetwork(
        trajectory_text=integrate_trajectory,
        output_dir=integrate_adapter_dir,
        base_model_id=base_model_id,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    integrate_adapter_id: str | None = None
    if integrate_adapter_path:
        integrate_adapter_id = await _load_adapter(
            "phase4-integrate", integrate_adapter_path, current_adapter_id
        )
        registered_adapters.append(
            {
                "adapter_id": "phase4-integrate",
                "task_type": "phase4-integrate",
                "phase": "integrate",
            }
        )

    integrate_prompt = render_prompt("integrate", task_description=project_prompt)
    iteration_counter += 1
    integrate_state = await run_iteration(
        graph=graph,
        project_prompt=integrate_prompt,
        adapter_id=integrate_adapter_id,
        session_id=session_id,
        iteration=iteration_counter,
        phase="integrate",
    )

    final_code = integrate_state.get("generated_code", "")
    final_passed = integrate_state.get("tests_passed", False)

    phase_results["integrate"] = {
        "adapter_id": integrate_adapter_id,
        "tests_passed": final_passed,
    }

    logger.info("=== Pipeline Complete ===")
    logger.info("Final tests passed: %s", final_passed)

    return {
        "session_id": session_id,
        "total_iterations": iteration_counter,
        "project_prompt": project_prompt,
        "final_tests_passed": final_passed,
        "phases": phase_results,
        "adapter_dir": str(adapter_dir),
        "subtasks": [s["name"] for s in subtasks],
        "adapters": registered_adapters,
        "accumulated_code": final_code,
    }


# ---------------------------------------------------------------------------
# Legacy entry point — delegates to phased pipeline when checkpoint present
# ---------------------------------------------------------------------------


async def run_project(
    project_prompt: str,
    max_iterations: int = 10,
    checkpoint_path: str | None = None,
    base_model_id: str = DEFAULT_BASE_MODEL,
    device: str = "cpu",
    max_retries_per_subtask: int = 3,
) -> dict[str, Any]:
    """Run the full project lifecycle.

    Delegates to ``run_phased_pipeline()`` which uses the 4-phase
    template-driven approach with swarm parallelism.

    Args:
        project_prompt: Natural language project definition.
        max_iterations: Maximum total iterations across all phases.
        checkpoint_path: Path to pretrained hypernetwork checkpoint.
        base_model_id: HuggingFace model ID of the base model.
        device: Device for hypernetwork computation.
        max_retries_per_subtask: Max attempts per subtask (errors go through H()).

    Returns:
        Summary dict with phase results.
    """
    return await run_phased_pipeline(
        project_prompt=project_prompt,
        max_iterations=max_iterations,
        checkpoint_path=checkpoint_path,
        base_model_id=base_model_id,
        device=device,
        max_retries_per_subtask=max_retries_per_subtask,
    )


def main() -> None:
    """CLI entry point for the Rune iteration runner."""
    parser = argparse.ArgumentParser(
        description="Rune E2E Runner — Multi-phase pipeline with template-driven adapters"
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
    parser.add_argument(
        "--population-size",
        type=int,
        default=2,
        help="Number of parallel agents for swarm phases (default: 2)",
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
        run_phased_pipeline(
            project_prompt=args.project,
            max_iterations=args.max_iterations,
            checkpoint_path=args.checkpoint,
            base_model_id=args.base_model_id,
            device=args.device if args.device != "auto" else get_best_device(),
            population_size=args.population_size,
        )
    )

    logger.info("=== Run Complete ===")
    logger.info("Session: %s", result["session_id"])
    logger.info("Iterations: %d", result["total_iterations"])
    logger.info("Final tests passed: %s", result["final_tests_passed"])
    logger.info("Adapters saved to: %s", result["adapter_dir"])
    logger.info("Adapters registered: %d", len(result["adapters"]))


if __name__ == "__main__":
    main()
