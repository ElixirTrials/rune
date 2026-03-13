"""Rune Runner: Outer iteration loop with hypernetwork state machine.

The hypernetwork carries the ENTIRE project lifecycle state through weight-space.
The prompt is the launch trigger and stays constant. Each iteration's output
(plan, code, test results) gets encoded into a structured trajectory document,
which the hypernetwork converts into LoRA adapter weights for the next iteration.

The trajectory is NOT accumulated raw — it is compressed into a structured
summary at each step to fit within the perceiver's 512-token window while
maximizing signal density.

Iteration 0 (bootstrap):
  methodology doc + project spec → H() → methodology_adapter

Iteration 1 (planning):
  project spec + methodology → generate architecture plan
  Plan output is decomposed into subtasks for subsequent iterations.

Iteration 2+ (chunked implementation):
  Each iteration tackles one subtask from the plan.
  Structured trajectory carries: subtask spec, prior code skeleton,
  error diagnosis from previous attempt, guidance for next step.
  Errors from failed iterations feed back through the hypernetwork,
  NOT through the prompt (prompt stays constant).

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


# ---------------------------------------------------------------------------
# Trajectory construction — structured, dense, fits 512 tokens
# ---------------------------------------------------------------------------


def _extract_error_summary(stderr: str) -> str:
    """Extract the most actionable error info from raw stderr.

    Pulls out the final exception line and the most relevant traceback
    context rather than dumping the entire stderr.
    """
    if not stderr:
        return ""
    lines = stderr.strip().splitlines()

    # Find the last exception/error line
    error_lines = [
        l for l in lines if "Error" in l or "Exception" in l or "assert" in l.lower()
    ]
    final_error = error_lines[-1].strip() if error_lines else lines[-1].strip()

    # Find failed test names (unittest output)
    failed_tests = [
        l.strip()
        for l in lines
        if l.strip().startswith("FAIL:") or l.strip().startswith("ERROR:")
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
            or "self." in stripped and "=" in stripped and stripped.startswith("self.")
        ):
            skeleton_lines.append(line)
    return "\n".join(skeleton_lines[:40])


def _count_test_results(stdout: str, stderr: str) -> tuple[int, int]:
    """Parse unittest output to count passed/failed tests."""
    total = 0
    failed = 0

    # unittest prints "Ran N tests" and "FAILED (failures=X, errors=Y)"
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


def _build_trajectory_text(
    project_prompt: str,
    state: dict[str, Any],
    iteration: int,
    iteration_history: list[dict[str, Any]],
) -> str:
    """Build a structured, dense trajectory for hypernetwork encoding.

    Instead of accumulating raw output, this builds a single focused
    document that fits within 512 tokens and contains maximum signal:

    1. Project specification (what to build)
    2. Current progress summary (what works, what doesn't)
    3. Error diagnosis (specific failures to fix)
    4. Code skeleton (structural outline, not full code)
    5. Guidance for next iteration

    Args:
        project_prompt: The original project specification.
        state: Current iteration's execution state.
        iteration: Current iteration number.
        iteration_history: List of prior iteration result dicts.

    Returns:
        Structured trajectory text optimized for 512-token perceiver.
    """
    parts: list[str] = []

    # --- Section 1: Project spec (always included, anchors the task) ---
    # Truncate to keep it dense — first 300 chars covers the key requirements
    spec_summary = project_prompt[:500]
    parts.append(f"PROJECT: {spec_summary}")

    # --- Section 2: Progress summary ---
    passed, total = _count_test_results(
        state.get("stdout", ""), state.get("stderr", "")
    )
    tests_passed = state.get("tests_passed", False)

    progress_lines = [f"ITERATION: {iteration}"]
    if total > 0:
        progress_lines.append(f"TESTS: {passed}/{total} passed")
    else:
        progress_lines.append(
            "TESTS: no test output (code may have failed to parse or run)"
        )
    progress_lines.append(f"STATUS: {'PASSING' if tests_passed else 'FAILING'}")

    # Track improvement across iterations
    if iteration_history:
        prev_results = []
        for h in iteration_history:
            p, t = _count_test_results(h.get("stdout", ""), h.get("stderr", ""))
            prev_results.append(f"iter{h['iteration']}:{p}/{t}")
        progress_lines.append(f"HISTORY: {', '.join(prev_results)}")

    parts.append("\n".join(progress_lines))

    # --- Section 3: Error diagnosis (only if failing) ---
    if not tests_passed:
        error_summary = _extract_error_summary(state.get("stderr", ""))
        if error_summary:
            parts.append(f"ERRORS:\n{error_summary}")

    # --- Section 4: Code skeleton (structure only, saves tokens) ---
    code = state.get("generated_code", "")
    skeleton = _extract_code_skeleton(code)
    if skeleton:
        parts.append(f"CODE STRUCTURE:\n{skeleton}")

    # --- Section 5: Guidance for next iteration ---
    if not tests_passed:
        guidance_lines = ["NEXT STEPS:"]
        if total == 0:
            guidance_lines.append(
                "- Code failed to execute. Check imports, syntax, and indentation."
            )
            guidance_lines.append(
                "- Ensure all referenced names are defined before use."
            )
        elif passed == 0:
            guidance_lines.append(
                "- No tests passing. Focus on getting basic structure correct first."
            )
            guidance_lines.append("- Implement one feature at a time, test it, move on.")
        elif passed < total:
            guidance_lines.append(
                f"- {passed}/{total} tests pass. Fix the {total - passed} failing "
                f"test(s) without breaking existing ones."
            )
            guidance_lines.append("- Read the error messages carefully and fix root cause.")
        parts.append("\n".join(guidance_lines))

    return "\n\n".join(parts)


def _build_bootstrap_trajectory(project_prompt: str, methodology: str) -> str:
    """Build the bootstrap trajectory that combines methodology + project spec.

    This gives the hypernetwork both the coding standards AND the specific
    project context, so the first adapter is already project-aware.
    """
    return (
        f"PROJECT: {project_prompt[:500]}\n\n"
        f"METHODOLOGY:\n{methodology}\n\n"
        "PHASE: bootstrap — apply methodology to plan and implement this project. "
        "Use dataclasses for data models, type annotations on all signatures, "
        "unittest.TestCase for tests, and clean layered architecture."
    )


def _decompose_into_subtasks(project_prompt: str) -> list[dict[str, str]]:
    """Decompose a large project prompt into sequential subtasks.

    Each subtask is small enough that a 2B model can generate it within
    the token budget. Subtasks build on each other — each one produces
    a testable unit that subsequent subtasks can import/use.

    The decomposition follows the methodology's phased approach:
      1. Data model / core types
      2. Core logic / business rules
      3. Interface layer (CLI, API, etc.)
      4. Integration tests

    Args:
        project_prompt: Full project specification.

    Returns:
        List of subtask dicts with 'name', 'prompt', and 'test_focus' keys.
    """
    subtasks: list[dict[str, str]] = []

    # Heuristic decomposition based on common project patterns.
    # The prompt is analyzed for structural keywords to identify layers.
    prompt_lower = project_prompt.lower()

    # Detect if this is a multi-layer application
    has_data_layer = any(
        kw in prompt_lower
        for kw in ["data layer", "database", "sqlite", "store", "model", "schema"]
    )
    has_domain_layer = any(
        kw in prompt_lower
        for kw in ["domain", "business rule", "service", "validation", "logic"]
    )
    has_interface_layer = any(
        kw in prompt_lower
        for kw in ["cli", "api", "interface", "argparse", "endpoint", "command"]
    )
    has_tests = any(
        kw in prompt_lower
        for kw in ["test suite", "test", "unittest", "pytest"]
    )

    if has_data_layer:
        subtasks.append({
            "name": "data_layer",
            "prompt": (
                f"From this project spec, implement ONLY the data layer:\n\n"
                f"{project_prompt}\n\n"
                "Focus on: data classes/models, storage class, CRUD operations. "
                "Include basic unit tests for the data layer only. "
                "Use dataclasses, type annotations, and unittest.TestCase."
            ),
            "test_focus": "data model CRUD and queries",
        })

    if has_domain_layer:
        subtasks.append({
            "name": "domain_layer",
            "prompt": (
                f"From this project spec, implement ONLY the domain/business logic layer:\n\n"
                f"{project_prompt}\n\n"
                "Assume the data layer already exists with the store/model classes. "
                "Focus on: service class, validation rules, state transitions, "
                "business logic. Include unit tests for domain rules only. "
                "Use type annotations and unittest.TestCase."
            ),
            "test_focus": "validation rules and business logic",
        })

    if has_interface_layer:
        subtasks.append({
            "name": "interface_layer",
            "prompt": (
                f"From this project spec, implement ONLY the interface layer:\n\n"
                f"{project_prompt}\n\n"
                "Assume data and domain layers already exist. "
                "Focus on: CLI/API layer, argument parsing, output formatting. "
                "Include unit tests for the interface layer only. "
                "Use type annotations and unittest.TestCase."
            ),
            "test_focus": "interface parsing and output",
        })

    if has_tests:
        subtasks.append({
            "name": "integration",
            "prompt": (
                f"From this project spec, write a comprehensive integration test suite:\n\n"
                f"{project_prompt}\n\n"
                "Assume all layers (data, domain, interface) are implemented. "
                "Write integration tests that exercise cross-layer workflows, "
                "edge cases, and error paths. Use unittest.TestCase."
            ),
            "test_focus": "end-to-end integration and edge cases",
        })

    # Fallback: if no structure detected, create a single-step decomposition
    if not subtasks:
        subtasks.append({
            "name": "implementation",
            "prompt": project_prompt,
            "test_focus": "all functionality",
        })

    return subtasks


def _build_subtask_trajectory(
    subtask: dict[str, str],
    subtask_index: int,
    total_subtasks: int,
    prior_code_skeleton: str,
    state: dict[str, Any] | None,
    iteration_history: list[dict[str, Any]],
) -> str:
    """Build trajectory for a specific subtask iteration.

    The trajectory tells the hypernetwork what subtask we're on, what code
    already exists (skeleton only), and what errors to fix.

    Args:
        subtask: Current subtask dict with name, prompt, test_focus.
        subtask_index: 0-based index of current subtask.
        total_subtasks: Total number of subtasks.
        prior_code_skeleton: Skeleton of all code generated so far.
        state: Execution state from last attempt (None for first attempt).
        iteration_history: Prior iteration results.

    Returns:
        Structured trajectory text for the hypernetwork.
    """
    parts: list[str] = []

    # Subtask context
    parts.append(
        f"SUBTASK: {subtask_index + 1}/{total_subtasks} — {subtask['name']}\n"
        f"FOCUS: {subtask['test_focus']}"
    )

    # Prior code skeleton (what already exists)
    if prior_code_skeleton:
        parts.append(f"EXISTING CODE:\n{prior_code_skeleton[:300]}")

    # Current state (if retrying)
    if state:
        passed, total = _count_test_results(
            state.get("stdout", ""), state.get("stderr", "")
        )
        tests_passed = state.get("tests_passed", False)
        parts.append(
            f"TESTS: {passed}/{total} {'PASSING' if tests_passed else 'FAILING'}"
        )

        if not tests_passed:
            error_summary = _extract_error_summary(state.get("stderr", ""))
            if error_summary:
                parts.append(f"ERRORS:\n{error_summary}")

            # Guidance based on failure type
            if total == 0:
                parts.append(
                    "FIX: Code failed to execute. Check syntax, imports, indentation."
                )
            elif passed == 0:
                parts.append(
                    "FIX: No tests pass. Focus on basic structure first."
                )
            else:
                parts.append(
                    f"FIX: {total - passed} test(s) failing. Fix without breaking passing tests."
                )

    # History across subtasks
    if iteration_history:
        hist_parts = []
        for h in iteration_history[-3:]:  # Last 3 only to save tokens
            p, t = _count_test_results(h.get("stdout", ""), h.get("stderr", ""))
            hist_parts.append(f"iter{h['iteration']}:{p}/{t}")
        parts.append(f"HISTORY: {', '.join(hist_parts)}")

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


def _is_sakana_checkpoint(checkpoint_path: str) -> bool:
    """Detect whether a checkpoint is a Sakana HyperLoRA checkpoint.

    Sakana checkpoints contain a ``hypernet_config`` key that is a
    ``ctx_to_lora`` ``HypernetConfig`` object (not a plain dict).
    """
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


async def run_project(
    project_prompt: str,
    max_iterations: int = 10,
    checkpoint_path: str | None = None,
    base_model_id: str = DEFAULT_BASE_MODEL,
    device: str = "cpu",
    max_retries_per_subtask: int = 3,
) -> dict[str, Any]:
    """Run the full project lifecycle with task decomposition and chunked generation.

    The project is decomposed into subtasks (data layer, domain, interface,
    integration tests). Each subtask gets multiple retry attempts. Between
    attempts, errors flow through the hypernetwork (not the prompt) so the
    model's weights are conditioned on what went wrong.

    Flow:
      1. Bootstrap: methodology + project spec → H() → methodology adapter
      2. Decompose project into subtasks
      3. For each subtask:
         a. Build subtask-specific trajectory → H() → subtask adapter
         b. Run generation with constant prompt (subtask prompt)
         c. If tests fail, feed errors through H() for next attempt
         d. Up to max_retries_per_subtask attempts per subtask
      4. Accumulate code across subtasks

    Args:
        project_prompt: Natural language project definition.
        max_iterations: Maximum total iterations across all subtasks.
        checkpoint_path: Path to pretrained hypernetwork checkpoint.
        base_model_id: HuggingFace model ID of the base model.
        device: Device for hypernetwork computation.
        max_retries_per_subtask: Max attempts per subtask (errors go through H()).

    Returns:
        Summary dict with iteration results.
    """
    from rune_agent.graph import create_single_iteration_graph

    graph = create_single_iteration_graph()
    session_id = f"rune-{uuid.uuid4().hex[:8]}"
    adapter_dir = ADAPTER_BASE_DIR / session_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    current_adapter_id: str | None = None
    current_adapter_path: str | None = None
    iteration_results: list[dict[str, Any]] = []
    accumulated_code: list[str] = []  # Code from completed subtasks
    iteration_counter = 0

    # --- Phase 0: Bootstrap (methodology + project spec → adapter) ---
    logger.info("=== Phase 0: Bootstrap ===")
    methodology = _load_methodology()
    bootstrap_trajectory = _build_bootstrap_trajectory(project_prompt, methodology)
    logger.info("Bootstrap trajectory: %d chars", len(bootstrap_trajectory))

    bootstrap_adapter_dir = str(adapter_dir / "iter0_methodology")
    adapter_path = run_hypernetwork(
        trajectory_text=bootstrap_trajectory,
        output_dir=bootstrap_adapter_dir,
        base_model_id=base_model_id,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    if adapter_path:
        current_adapter_id = "methodology"
        current_adapter_path = adapter_path
        from inference import get_provider

        provider = get_provider()
        await provider.load_adapter(current_adapter_id, current_adapter_path)
        logger.info("Bootstrap adapter loaded: %s", current_adapter_id)
    else:
        logger.info("No hypernetwork checkpoint — running with base model only")

    # --- Phase 1: Decompose project into subtasks ---
    subtasks = _decompose_into_subtasks(project_prompt)
    logger.info(
        "Decomposed into %d subtasks: %s",
        len(subtasks),
        [s["name"] for s in subtasks],
    )

    # --- Phase 2: Chunked implementation (one subtask per chunk) ---
    for subtask_idx, subtask in enumerate(subtasks):
        if iteration_counter >= max_iterations:
            logger.warning("Hit max iterations (%d), stopping", max_iterations)
            break

        logger.info(
            "=== Subtask %d/%d: %s ===",
            subtask_idx + 1,
            len(subtasks),
            subtask["name"],
        )

        # Build the code skeleton from all prior subtasks
        prior_skeleton = _extract_code_skeleton("\n\n".join(accumulated_code))

        subtask_passed = False
        last_state: dict[str, Any] | None = None

        for attempt in range(max_retries_per_subtask):
            iteration_counter += 1
            if iteration_counter > max_iterations:
                break

            logger.info(
                "  Attempt %d/%d for subtask '%s' (adapter=%s)",
                attempt + 1,
                max_retries_per_subtask,
                subtask["name"],
                current_adapter_id or "none",
            )

            # Build trajectory for this subtask + attempt → H() → new adapter
            subtask_trajectory = _build_subtask_trajectory(
                subtask=subtask,
                subtask_index=subtask_idx,
                total_subtasks=len(subtasks),
                prior_code_skeleton=prior_skeleton,
                state=last_state,
                iteration_history=iteration_results,
            )
            logger.info(
                "  Subtask trajectory: %d chars", len(subtask_trajectory)
            )

            # Generate adapter from subtask trajectory
            iter_adapter_dir = str(
                adapter_dir / f"subtask{subtask_idx}_attempt{attempt}"
            )
            new_adapter_path = run_hypernetwork(
                trajectory_text=subtask_trajectory,
                output_dir=iter_adapter_dir,
                base_model_id=base_model_id,
                checkpoint_path=checkpoint_path,
                device=device,
            )

            if new_adapter_path:
                new_adapter_id = f"subtask{subtask_idx}_v{attempt}"
                from inference import get_provider

                provider = get_provider()
                if current_adapter_id:
                    try:
                        await provider.unload_adapter(current_adapter_id)
                    except Exception:
                        pass
                await provider.load_adapter(new_adapter_id, new_adapter_path)
                current_adapter_id = new_adapter_id
                current_adapter_path = new_adapter_path
                logger.info("  Adapter loaded: %s", current_adapter_id)

            # Run one iteration — prompt is CONSTANT (subtask prompt),
            # all error context flows through the adapter weights
            state = await run_iteration(
                graph=graph,
                project_prompt=subtask["prompt"],
                adapter_id=current_adapter_id,
                session_id=session_id,
                iteration=iteration_counter,
            )

            result = {
                "iteration": iteration_counter,
                "subtask": subtask["name"],
                "attempt": attempt + 1,
                "adapter_id": current_adapter_id,
                "tests_passed": state.get("tests_passed", False),
                "exit_code": state.get("exit_code", -1),
                "generated_code": state.get("generated_code", ""),
                "stdout": state.get("stdout", ""),
                "stderr": state.get("stderr", ""),
            }
            iteration_results.append(result)
            last_state = state

            if state.get("tests_passed"):
                logger.info(
                    "  Subtask '%s' PASSED on attempt %d",
                    subtask["name"],
                    attempt + 1,
                )
                subtask_passed = True
                # Accumulate the passing code for subsequent subtasks
                accumulated_code.append(state.get("generated_code", ""))
                break

            logger.info(
                "  Subtask '%s' FAILED attempt %d — errors will flow through H()",
                subtask["name"],
                attempt + 1,
            )

        if not subtask_passed:
            logger.warning(
                "Subtask '%s' failed after %d attempts, continuing to next",
                subtask["name"],
                max_retries_per_subtask,
            )
            # Still accumulate the last attempt's code (partial progress)
            if last_state and last_state.get("generated_code"):
                accumulated_code.append(last_state["generated_code"])

    # Determine final outcome
    final_passed = False
    if iteration_results:
        # Check if the last subtask passed
        final_passed = iteration_results[-1].get("tests_passed", False)

    return {
        "session_id": session_id,
        "total_iterations": len(iteration_results),
        "project_prompt": project_prompt,
        "final_tests_passed": final_passed,
        "iterations": iteration_results,
        "adapter_dir": str(adapter_dir),
        "subtasks": [s["name"] for s in subtasks],
        "accumulated_code": "\n\n".join(accumulated_code),
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
