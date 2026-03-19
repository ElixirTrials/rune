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
from shared.sandbox import count_test_results, extract_failed_tests  # noqa: E402
from shared.template_loader import render_trajectory  # noqa: E402

logger = logging.getLogger(__name__)

# Base model ID for adapter config
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
ADAPTER_BASE_DIR = Path.home() / ".rune" / "adapters"

# ---------------------------------------------------------------------------
# Phase iteration config — env var driven
# ---------------------------------------------------------------------------
#
# Global default:  RUNE_MAX_PHASE_ITERATIONS  (applies to all phases)
# Per-phase:       RUNE_MAX_ITERATIONS_DECOMPOSE
#                  RUNE_MAX_ITERATIONS_PLAN
#                  RUNE_MAX_ITERATIONS_CODE
#                  RUNE_MAX_ITERATIONS_INTEGRATE
#
# Per-phase overrides take precedence over the global default.
# CLI --max-phase-iterations overrides the global env var.
# Hardcoded fallback is 5.

_FALLBACK_MAX_ITERATIONS = 5


def _get_phase_iterations(phase: str, cli_override: int | None = None) -> int:
    """Resolve max iterations for a phase from env vars and CLI args.

    Resolution order (first non-None wins):
      1. RUNE_MAX_ITERATIONS_{PHASE}   (per-phase env var)
      2. cli_override                   (--max-phase-iterations)
      3. RUNE_MAX_PHASE_ITERATIONS      (global env var)
      4. _FALLBACK_MAX_ITERATIONS       (hardcoded 5)
    """
    per_phase = os.environ.get(f"RUNE_MAX_ITERATIONS_{phase.upper()}")
    if per_phase is not None:
        return int(per_phase)
    if cli_override is not None:
        return cli_override
    global_env = os.environ.get("RUNE_MAX_PHASE_ITERATIONS")
    if global_env is not None:
        return int(global_env)
    return _FALLBACK_MAX_ITERATIONS


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
    return "\n".join(skeleton_lines[:80])


# _count_test_results and _extract_failed_tests are now imported from
# shared.sandbox as count_test_results and extract_failed_tests.
# Keep thin wrappers for backward compatibility with internal callers.
_count_test_results = count_test_results
_extract_failed_tests = extract_failed_tests


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------


def _parse_subtask_list(model_output: str) -> list[dict[str, str]]:
    """Parse Phase 1 output into [{name, description}, ...].

    Expects numbered list format: ``1. name — description``
    Also accepts ``1. name - description``, ``1. name: description``,
    ``1. **name:** description`` (markdown bold), or ``1. description sentence``
    (no separator).
    """
    subtasks: list[dict[str, str]] = []
    for line in model_output.splitlines():
        line = line.strip()
        # Strip markdown bold markers
        line = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", line)
        # Primary: "1. name — description" or "1. name - desc" or "1. name: desc"
        match = re.match(r"^\d+\.\s*(.+?)\s*(?:—|-|:)\s*(.+)$", line)
        if match:
            subtasks.append(
                {
                    "name": match.group(1).strip(),
                    "description": match.group(2).strip(),
                }
            )
            continue
        # Fallback: "1. description sentence" (no separator)
        match = re.match(r"^\d+\.\s*(.+?)\.?\s*$", line)
        if match and len(match.group(1).strip()) > 3:
            subtasks.append(
                {
                    "name": match.group(1).strip(),
                    "description": "",
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


def _parse_diagnose_output(
    model_output: str,
    known_subtasks: list[str],
) -> list[dict[str, str]]:
    """Parse diagnose phase output into [{name, diagnosis}, ...].

    Matches diagnosed subtask names against known subtask names using
    substring matching (the model may abbreviate or rephrase names).
    """
    raw = _parse_subtask_list(model_output)
    known_lower = {k.lower().strip(): k for k in known_subtasks}

    matched: list[dict[str, str]] = []
    for item in raw:
        name = item["name"].lower().strip().rstrip(":")
        diagnosis = item["description"] or item["name"]

        # Exact match
        if name in known_lower:
            matched.append({"name": known_lower[name], "diagnosis": diagnosis})
            continue

        # Substring match: find the known subtask that best matches
        for known_key, known_name in known_lower.items():
            if name in known_key or known_key in name:
                matched.append({"name": known_name, "diagnosis": diagnosis})
                break

    return matched


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
    prompt_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one iteration of the agent loop.

    Args:
        graph: Compiled single-iteration LangGraph.
        project_prompt: Raw task description (not pre-rendered).
        adapter_id: Current adapter to load, or None for base model.
        session_id: Session ID for trajectory tracking.
        iteration: Current iteration number.
        phase: Pipeline phase for template-driven prompts.
        prompt_context: Extra template vars for prompt rendering (retry context).

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
        "test_count": 0,
        "tests_ran": False,
        "trajectory": [],
        "phase": phase,
        "prompt_context": prompt_context,
        "finish_reason": None,
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
        Path to saved adapter directory, or None if no checkpoint provided.

    Raises:
        RuntimeError: If a checkpoint is provided but adapter generation fails.
    """
    if not checkpoint_path or not Path(checkpoint_path).exists():
        logger.warning(
            "No hypernetwork checkpoint at %s — skipping adapter generation",
            checkpoint_path,
        )
        return None

    # Free fragmented GPU memory before loading hypernetwork
    import torch  # noqa: PLC0415

    if device != "cpu" and torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()

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


def _register_adapter(
    registry: Any,
    adapter_id: str,
    task_type: str,
    base_model_id: str,
    file_path: str,
    session_id: str,
    pass_rate: float = 0.0,
    generation: int = 0,
) -> float:
    """Register an adapter in the evolution registry and compute fitness.

    Returns:
        Computed fitness score.
    """
    from datetime import datetime, timezone

    from adapter_registry.models import AdapterRecord
    from evaluation.metrics import evaluate_fitness

    fitness = evaluate_fitness(adapter_id, pass_rate, diversity_score=0.5)
    now = datetime.now(timezone.utc).isoformat()

    record = AdapterRecord(
        id=adapter_id,
        version=1,
        task_type=task_type,
        base_model_id=base_model_id,
        rank=8,
        created_at=now,
        file_path=file_path,
        file_hash="",
        file_size_bytes=0,
        pass_rate=pass_rate,
        fitness_score=fitness,
        source="phased-pipeline",
        session_id=session_id,
        generation=generation,
    )
    try:
        registry.store(record)
    except Exception:
        # Duplicate — update fitness instead
        registry.update_fitness(adapter_id, pass_rate=pass_rate, fitness_score=fitness)
    return fitness


async def run_phased_pipeline(
    project_prompt: str,
    max_iterations: int = 10,
    checkpoint_path: str | None = None,
    base_model_id: str = DEFAULT_BASE_MODEL,
    device: str = "cpu",
    population_size: int = 2,
    max_phase_iterations: int | None = None,
) -> dict[str, Any]:
    """Run the 4-phase pipeline with template-driven adapters and evolution.

    Each phase runs up to its configured max iterations, generating a new
    adapter via H() each iteration. Early stops when the phase succeeds.
    Between iterations, an evolution sweep merges top adapters and prunes
    low-fitness ones.

    Iteration counts are resolved per-phase via env vars:
      - ``RUNE_MAX_ITERATIONS_DECOMPOSE``, ``..._PLAN``, ``..._CODE``, ``..._INTEGRATE``
      - ``RUNE_MAX_PHASE_ITERATIONS`` (global default)
      - ``max_phase_iterations`` kwarg (overrides global env var)
      - Hardcoded fallback: 5

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
        max_phase_iterations: Default max iterations per phase (env vars override).

    Returns:
        Summary dict with phase results, adapters, evolution stats, and final code.
    """
    from adapter_registry.registry import AdapterRegistry
    from rune_agent.graph import create_single_iteration_graph
    from sqlalchemy import create_engine

    graph = create_single_iteration_graph()
    session_id = f"rune-{uuid.uuid4().hex[:8]}"
    adapter_dir = ADAPTER_BASE_DIR / session_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Evolution registry — per-session SQLite DB
    registry_engine = create_engine(f"sqlite:///{adapter_dir}/evolution.db")
    registry = AdapterRegistry(engine=registry_engine)

    # Resolve per-phase iteration limits
    iters_decompose = _get_phase_iterations("decompose", max_phase_iterations)
    iters_plan = _get_phase_iterations("plan", max_phase_iterations)
    iters_code = _get_phase_iterations("code", max_phase_iterations)
    iters_integrate = _get_phase_iterations("integrate", max_phase_iterations)
    iters_repair = _get_phase_iterations("repair", max_phase_iterations)
    logger.info(
        "Phase iteration limits: decompose=%d, plan=%d, code=%d, integrate=%d, repair=%d",
        iters_decompose,
        iters_plan,
        iters_code,
        iters_integrate,
        iters_repair,
    )

    phase_results: dict[str, Any] = {}
    registered_adapters: list[dict[str, str]] = []
    iteration_counter = 0
    evolution_stats: dict[str, Any] = {
        "phase_iterations": {},
        "sweeps": {},
        "best_adapters": {},
    }

    # Import evolution sweep
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from swarm_evolution import evolution_sweep

    # Extract first sentence as project label for prompts.
    # The full project spec flows through adapter weights; prompts only
    # need enough to orient the model on which project it's working on.
    _dot = project_prompt.find(".")
    project_label = project_prompt[: _dot + 1] if _dot > 0 else project_prompt.split("\n")[0]

    # ---------------------------------------------------------------
    # Phase 1: DECOMPOSE (single agent, with evolution)
    # ---------------------------------------------------------------
    logger.info("=== Phase 1: DECOMPOSE ===")
    best_decompose_state: dict[str, Any] | None = None
    best_decompose_adapter_id: str | None = None
    best_decompose_score: float = -1.0
    loaded_adapter_id: str | None = None  # tracks what's currently loaded
    phase1_task_type = "phase1-decompose"

    for evo_iter in range(iters_decompose):
        logger.info("  Decompose iteration %d/%d", evo_iter + 1, iters_decompose)

        # Build trajectory — include prior output on retries
        prior_output = ""
        if best_decompose_state is not None:
            prior_output = best_decompose_state.get("generated_code", "")
        decompose_trajectory = render_trajectory("decompose", project=project_prompt)
        if prior_output and evo_iter > 0:
            decompose_trajectory += (
                f"\n\n[Prior decomposition attempt — improve on this]\n"
                f"{prior_output[:500]}"
            )

        iter_adapter_dir = str(adapter_dir / f"phase1_decompose_v{evo_iter}")
        adapter_path = run_hypernetwork(
            trajectory_text=decompose_trajectory,
            output_dir=iter_adapter_dir,
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        adapter_id: str | None = None
        if adapter_path:
            adapter_id = f"phase1-decompose-v{evo_iter}"
            await _load_adapter(adapter_id, adapter_path, loaded_adapter_id)
            loaded_adapter_id = adapter_id
            _register_adapter(
                registry,
                adapter_id=adapter_id,
                task_type=phase1_task_type,
                base_model_id=base_model_id,
                file_path=adapter_path,
                session_id=session_id,
                pass_rate=0.0,
                generation=evo_iter,
            )
            registered_adapters.append(
                {
                    "adapter_id": adapter_id,
                    "task_type": phase1_task_type,
                    "phase": "decompose",
                }
            )

        iteration_counter += 1
        decompose_phase = "decompose_concise" if adapter_id else "decompose"
        state = await run_iteration(
            graph=graph,
            project_prompt=project_prompt,
            adapter_id=adapter_id,
            session_id=session_id,
            iteration=iteration_counter,
            phase=decompose_phase,
        )

        # Score: reward 2-6 unique subtasks, penalize duplicates
        iter_subtasks = _parse_subtask_list(state.get("generated_code", ""))
        unique_names = {s["name"].lower().strip() for s in iter_subtasks}
        n_unique = len(unique_names)
        is_fallback = n_unique == 1 and iter_subtasks[0]["name"] == "implementation"
        if is_fallback:
            score = 0.05  # parser fallback — model didn't produce a list
        elif n_unique >= 2 and n_unique <= 8:
            score = min(1.0, n_unique / 3.0)
        elif n_unique == 1 and state.get("tests_passed"):
            score = 0.5
        elif n_unique > 8:
            score = 0.2  # too many subtasks indicates repetition
        else:
            score = 0.3
        # Penalize heavy duplication
        if len(iter_subtasks) > n_unique * 2:
            score *= 0.3

        # Update fitness in registry
        if adapter_id:
            registry.update_fitness(adapter_id, pass_rate=score, fitness_score=score)

        if score > best_decompose_score:
            best_decompose_score = score
            best_decompose_state = state
            best_decompose_adapter_id = adapter_id

        logger.info(
            "  Decompose v%d: %d subtasks (%d unique), score=%.2f%s",
            evo_iter,
            len(iter_subtasks),
            n_unique,
            score,
            " (best)" if adapter_id == best_decompose_adapter_id else "",
        )

        # Early stop: got multiple unique subtasks
        if n_unique >= 2 and score >= 0.6:
            logger.info("  Decompose converged at iteration %d", evo_iter + 1)
            break

        # Evolution sweep between iterations
        if evo_iter > 0:
            sweep = evolution_sweep(registry)
            evolution_stats["sweeps"][f"decompose-iter{evo_iter}"] = sweep

    evolution_stats["phase_iterations"]["decompose"] = evo_iter + 1

    # Use best decompose result
    assert best_decompose_state is not None
    raw_subtasks = _parse_subtask_list(best_decompose_state.get("generated_code", ""))
    # Deduplicate subtasks by name
    seen: set[str] = set()
    subtasks: list[dict[str, str]] = []
    for st in raw_subtasks:
        key = st["name"].lower().strip()
        if key not in seen:
            seen.add(key)
            subtasks.append(st)
    evolution_stats["best_adapters"]["decompose"] = best_decompose_adapter_id

    logger.info(
        "Decomposed into %d subtasks: %s",
        len(subtasks),
        [s["name"] for s in subtasks],
    )
    phase_results["decompose"] = {
        "subtasks": subtasks,
        "adapter_id": best_decompose_adapter_id,
        "iterations": evo_iter + 1,
        "best_score": best_decompose_score,
    }

    # ---------------------------------------------------------------
    # Phase 2: PLAN (parallel swarm, with evolution)
    # ---------------------------------------------------------------
    logger.info("=== Phase 2: PLAN ===")
    best_plans: dict[str, str] = {}
    best_plan_score: float = -1.0
    best_plan_adapter_ids: list[str] = []
    phase2_task_type = "phase2-plan"

    for evo_iter in range(iters_plan):
        logger.info("  Plan iteration %d/%d", evo_iter + 1, iters_plan)
        iter_plans: dict[str, str] = {}

        async def _plan_subtask(
            idx: int, subtask: dict[str, str], evo: int
        ) -> tuple[str, str, str | None]:
            """Plan a single subtask — runs as a parallel coroutine."""
            traj = render_trajectory(
                "plan",
                subtask=subtask,
                subtask_index=idx + 1,
                total_subtasks=len(subtasks),
                project=project_prompt,
            )
            # Include prior plan on retries
            if evo > 0 and subtask["name"] in best_plans:
                traj += (
                    f"\n\n[Prior plan — improve on this]\n"
                    f"{best_plans[subtask['name']][:300]}"
                )

            plan_ad = str(adapter_dir / f"phase2_plan_{subtask['name']}_v{evo}")
            plan_path = run_hypernetwork(
                trajectory_text=traj,
                output_dir=plan_ad,
                base_model_id=base_model_id,
                checkpoint_path=checkpoint_path,
                device=device,
            )
            plan_aid: str | None = None
            if plan_path:
                plan_aid = f"phase2-plan-{subtask['name']}-v{evo}"
                await _load_adapter(plan_aid, plan_path, None)
                _register_adapter(
                    registry,
                    adapter_id=plan_aid,
                    task_type=phase2_task_type,
                    base_model_id=base_model_id,
                    file_path=plan_path,
                    session_id=session_id,
                    generation=evo,
                )
                registered_adapters.append(
                    {
                        "adapter_id": plan_aid,
                        "task_type": phase2_task_type,
                        "phase": "plan",
                    }
                )

            plan_state = await run_iteration(
                graph=graph,
                project_prompt=project_prompt,
                adapter_id=plan_aid,
                session_id=session_id,
                iteration=iteration_counter + idx + 1,
                phase="plan",
                prompt_context={"subtask_name": subtask["name"],
                                "project_label": project_label},
            )
            return (
                subtask["name"],
                _parse_plan(plan_state.get("generated_code", "")),
                plan_aid,
            )

        plan_tasks = [_plan_subtask(i, st, evo_iter) for i, st in enumerate(subtasks)]
        plan_results = await asyncio.gather(*plan_tasks)

        iter_adapter_ids: list[str] = []
        for name, plan_text, plan_aid in plan_results:
            iter_plans[name] = plan_text
            if plan_aid:
                iter_adapter_ids.append(plan_aid)

        # Score: fraction of plans with substance
        good_plans = sum(1 for p in iter_plans.values() if len(p) > 50)
        score = good_plans / max(len(iter_plans), 1)

        # Update fitness for all plan adapters
        for aid in iter_adapter_ids:
            registry.update_fitness(aid, pass_rate=score, fitness_score=score)

        if score > best_plan_score:
            best_plan_score = score
            best_plans = dict(iter_plans)
            best_plan_adapter_ids = iter_adapter_ids

        logger.info(
            "  Plan v%d: %d/%d good plans, score=%.2f",
            evo_iter,
            good_plans,
            len(iter_plans),
            score,
        )

        # Early stop: all plans have substance
        if score >= 1.0:
            logger.info("  Plan converged at iteration %d", evo_iter + 1)
            break

        # Evolution sweep
        if evo_iter > 0:
            sweep = evolution_sweep(registry)
            evolution_stats["sweeps"][f"plan-iter{evo_iter}"] = sweep

    iteration_counter += len(subtasks) * (evo_iter + 1)
    evolution_stats["phase_iterations"]["plan"] = evo_iter + 1
    evolution_stats["best_adapters"]["plan"] = best_plan_adapter_ids

    plans = best_plans
    logger.info("Plans generated for %d subtasks", len(plans))
    phase_results["plan"] = {
        "plans": {k: v[:200] for k, v in plans.items()},
        "plan_lengths": {k: len(v) for k, v in plans.items()},
        "iterations": evo_iter + 1,
        "best_score": best_plan_score,
    }

    # ---------------------------------------------------------------
    # Phase 3: CODE (parallel swarm, with evolutionary retries via H())
    # Each subtask gets up to iters_code attempts. Each attempt
    # generates a fresh adapter, and evolution sweeps run between attempts.
    # ---------------------------------------------------------------
    logger.info("=== Phase 3: CODE ===")
    code_outputs: dict[str, str] = {}
    code_subtask_results: dict[str, dict[str, Any]] = {}

    async def _code_subtask(
        idx: int, subtask: dict[str, str]
    ) -> tuple[str, str, dict[str, Any]]:
        """Code a single subtask with evolutionary retry loop."""
        plan = plans.get(subtask["name"], "")
        existing_code = ""
        last_state: dict[str, Any] | None = None
        loaded_code_adapter: str | None = None

        for attempt in range(iters_code):
            if attempt == 0:
                traj = render_trajectory(
                    "code",
                    subtask=subtask,
                    subtask_index=idx + 1,
                    total_subtasks=len(subtasks),
                    plan=plan,
                    existing_code=existing_code,
                    project=project_prompt,
                )
            else:
                assert last_state is not None  # guaranteed by attempt > 0
                is_truncated = last_state.get("finish_reason") == "length"

                if is_truncated:
                    # Continuation: prior output was cut off at max_tokens.
                    # Concatenate prior + new output after generation.
                    traj = render_trajectory(
                        "code_continue",
                        subtask=subtask,
                        attempt=attempt + 1,
                        max_retries=iters_code,
                        plan=plan,
                        existing_code=existing_code,
                        project=project_prompt,
                    )
                else:
                    # Retry: code was complete but tests failed or code errored.
                    passed, total = _count_test_results(
                        last_state.get("stdout", ""),
                        last_state.get("stderr", ""),
                    )
                    error_summary = _extract_error_summary(
                        last_state.get("stderr", "")
                    )
                    failed_tests = _extract_failed_tests(
                        last_state.get("stderr", "")
                    )
                    tests_passed = last_state.get("tests_passed", False)

                    if total == 0 and last_state.get("exit_code", 1) == 0:
                        fix_guidance = (
                            "NO tests detected — include unittest.TestCase tests "
                            "and end with: if __name__ == '__main__': unittest.main()"
                        )
                    elif total == 0:
                        fix_guidance = (
                            "Code failed to execute. "
                            "Check syntax, imports, indentation."
                        )
                    elif passed == 0:
                        fix_guidance = (
                            "No tests pass. Focus on basic structure first."
                        )
                    else:
                        fix_guidance = (
                            f"{total - passed} test(s) failing. "
                            "Fix without breaking passing tests."
                        )

                    traj = render_trajectory(
                        "code_retry",
                        subtask=subtask,
                        attempt=attempt + 1,
                        max_retries=iters_code,
                        plan=plan,
                        existing_code=existing_code,
                        passed=passed,
                        total=total,
                        tests_passed=tests_passed,
                        error_summary=error_summary,
                        failed_tests=failed_tests,
                        fix_guidance=fix_guidance,
                        history=None,
                        project=project_prompt,
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
                await _load_adapter(
                    code_adapter_id, code_adapter_path, loaded_code_adapter
                )
                loaded_code_adapter = code_adapter_id
                _register_adapter(
                    registry,
                    adapter_id=code_adapter_id,
                    task_type=f"phase3-code-{subtask['name']}",
                    base_model_id=base_model_id,
                    file_path=code_adapter_path,
                    session_id=session_id,
                    generation=attempt,
                )
                registered_adapters.append(
                    {
                        "adapter_id": code_adapter_id,
                        "task_type": f"phase3-code-{subtask['name']}",
                        "phase": "code",
                    }
                )

            if attempt == 0:
                code_phase = "code"
                code_ctx: dict[str, Any] | None = {
                    "subtask_name": subtask["name"],
                    "project_label": project_label,
                }
            elif is_truncated:
                code_phase = "code_continue"
                code_ctx = {
                    "subtask_name": subtask["name"],
                    "project_label": project_label,
                }
            else:
                assert last_state is not None
                passed, total = _count_test_results(
                    last_state.get("stdout", ""),
                    last_state.get("stderr", ""),
                )
                error_summary = _extract_error_summary(last_state.get("stderr", ""))

                if total == 0 and last_state.get("exit_code", 1) == 0:
                    fix_guidance_prompt = (
                        "NO tests detected — include unittest.TestCase"
                    )
                elif total == 0:
                    fix_guidance_prompt = "Code failed to execute. Check syntax."
                elif passed == 0:
                    fix_guidance_prompt = "No tests pass. Fix basic structure."
                else:
                    fix_guidance_prompt = f"{total - passed} test(s) failing. Fix them."

                code_phase = "code_retry"
                code_ctx = {
                    "subtask_name": subtask["name"],
                    "project_label": project_label,
                    "passed": passed,
                    "total": total,
                    "error_summary": error_summary,
                    "fix_guidance": fix_guidance_prompt,
                }
            code_state = await run_iteration(
                graph=graph,
                project_prompt=project_prompt,
                adapter_id=code_adapter_id,
                session_id=session_id,
                iteration=iteration_counter + idx * iters_code + attempt + 1,
                phase=code_phase,
                prompt_context=code_ctx,
            )

            last_state = code_state
            new_code = code_state.get("generated_code", "")

            # For continuation: concatenate prior code + new output
            if attempt > 0 and is_truncated:
                existing_code = existing_code + "\n" + new_code
                logger.info(
                    "  Subtask '%s' continuation: appended %d chars (total %d)",
                    subtask["name"],
                    len(new_code),
                    len(existing_code),
                )
            else:
                existing_code = new_code

            # Update fitness with partial credit from test results
            code_passed = code_state.get("tests_passed", False)
            code_p, code_t = _count_test_results(
                code_state.get("stdout", ""),
                code_state.get("stderr", ""),
            )
            if code_passed:
                code_fitness = 1.0
            elif code_t > 0:
                code_fitness = max(0.1, code_p / code_t * 0.8)
            else:
                code_fitness = 0.1
            if code_adapter_id:
                registry.update_fitness(
                    code_adapter_id,
                    pass_rate=code_p / code_t if code_t > 0 else 0.0,
                    fitness_score=code_fitness,
                )

            if code_passed:
                logger.info(
                    "  Subtask '%s' PASSED on attempt %d",
                    subtask["name"],
                    attempt + 1,
                )
                return (
                    subtask["name"],
                    existing_code,
                    {"passed": True, "attempts": attempt + 1},
                )

            logger.info(
                "  Subtask '%s' FAILED attempt %d/%d — retrying via H()",
                subtask["name"],
                attempt + 1,
                iters_code,
            )

            # Evolution sweep between attempts
            if attempt > 0:
                evolution_sweep(registry)

        # Exhausted retries — return last attempt's code
        logger.warning(
            "Subtask '%s' failed after %d attempts",
            subtask["name"],
            iters_code,
        )
        return (
            subtask["name"],
            existing_code,
            {"passed": False, "attempts": iters_code},
        )

    # Run coding in parallel
    code_tasks = [_code_subtask(i, st) for i, st in enumerate(subtasks)]
    code_results = await asyncio.gather(*code_tasks)
    for name, code_text, subtask_result in code_results:
        code_outputs[name] = code_text
        code_subtask_results[name] = subtask_result
    iteration_counter += len(subtasks) * iters_code

    # Post-code evolution sweep
    code_sweep = evolution_sweep(registry)
    evolution_stats["sweeps"]["post-code"] = code_sweep

    # Track Phase 3 in evolution stats
    max_code_attempts = max(
        (r["attempts"] for r in code_subtask_results.values()), default=1
    )
    code_passed_count = sum(1 for r in code_subtask_results.values() if r["passed"])
    evolution_stats["phase_iterations"]["code"] = max_code_attempts
    evolution_stats["best_adapters"]["code"] = {
        name: f"phase3-code-{name}-v{r['attempts'] - 1}"
        for name, r in code_subtask_results.items()
        if r["passed"]
    }

    logger.info(
        "Code generated for %d subtasks (%d/%d passed)",
        len(code_outputs),
        code_passed_count,
        len(code_outputs),
    )
    phase_results["code"] = {
        "outputs": code_outputs,
        "subtask_results": code_subtask_results,
        "iterations": max_code_attempts,
        "passed": code_passed_count,
        "total": len(code_outputs),
    }

    # ---------------------------------------------------------------
    # Phase 4: INTEGRATE (single agent, with evolution)
    # ---------------------------------------------------------------
    logger.info("=== Phase 4: INTEGRATE ===")
    skeletons = {
        name: _extract_code_skeleton(code) for name, code in code_outputs.items()
    }

    # Build integration doc: structured summary of subtasks, plans, and code
    # This flows through the hypernetwork so the integrate adapter "knows"
    # what was built and how pieces connect.
    integration_doc_parts: list[str] = []
    for st in subtasks:
        name = st["name"]
        plan_summary = plans.get(name, "")[:200]
        code_summary = _extract_code_skeleton(code_outputs.get(name, ""))[:300]
        passed = code_subtask_results.get(name, {}).get("passed", False)
        integration_doc_parts.append(
            f"- {name}: {'PASSED' if passed else 'FAILED'}\n"
            f"  Plan: {plan_summary}\n"
            f"  Code: {code_summary}"
        )
    integration_doc = "\n".join(integration_doc_parts)

    best_integrate_state: dict[str, Any] | None = None
    best_integrate_adapter_id: str | None = None
    best_integrate_score: float = -1.0
    loaded_integrate_id: str | None = None  # tracks currently loaded adapter
    phase4_task_type = "phase4-integrate"

    for evo_iter in range(iters_integrate):
        logger.info("  Integrate iteration %d/%d", evo_iter + 1, iters_integrate)

        integrate_trajectory = render_trajectory(
            "integrate",
            project=project_prompt,
            subtask_count=len(subtasks),
            skeletons=skeletons,
            integration_doc=integration_doc,
        )
        # Include prior error context on retries
        if evo_iter > 0 and best_integrate_state is not None:
            error_ctx = _extract_error_summary(best_integrate_state.get("stderr", ""))
            if error_ctx:
                integrate_trajectory += (
                    f"\n\n[Prior integration errors — fix these]\n{error_ctx}"
                )

        iter_adapter_dir = str(adapter_dir / f"phase4_integrate_v{evo_iter}")
        adapter_path = run_hypernetwork(
            trajectory_text=integrate_trajectory,
            output_dir=iter_adapter_dir,
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        integrate_aid: str | None = None
        if adapter_path:
            integrate_aid = f"phase4-integrate-v{evo_iter}"
            await _load_adapter(integrate_aid, adapter_path, loaded_integrate_id)
            loaded_integrate_id = integrate_aid
            _register_adapter(
                registry,
                adapter_id=integrate_aid,
                task_type=phase4_task_type,
                base_model_id=base_model_id,
                file_path=adapter_path,
                session_id=session_id,
                generation=evo_iter,
            )
            registered_adapters.append(
                {
                    "adapter_id": integrate_aid,
                    "task_type": phase4_task_type,
                    "phase": "integrate",
                }
            )

        if evo_iter == 0 or best_integrate_state is None:
            integrate_phase = "integrate"
            integrate_ctx: dict[str, Any] | None = {
                "subtask_count": len(subtasks),
                "project_label": project_label,
            }
        else:
            int_passed, int_total = _count_test_results(
                best_integrate_state.get("stdout", ""),
                best_integrate_state.get("stderr", ""),
            )
            int_error = _extract_error_summary(best_integrate_state.get("stderr", ""))

            if int_total == 0 and best_integrate_state.get("exit_code", 1) == 0:
                int_fix = "NO tests detected — include unittest.TestCase"
            elif int_total == 0:
                int_fix = "Code failed to execute. Check syntax."
            elif int_passed == 0:
                int_fix = "No tests pass. Fix basic structure."
            else:
                int_fix = f"{int_total - int_passed} test(s) failing. Fix them."

            integrate_phase = "integrate_retry"
            integrate_ctx = {
                "subtask_count": len(subtasks),
                "project_label": project_label,
                "passed": int_passed,
                "total": int_total,
                "error_summary": int_error,
                "fix_guidance": int_fix,
            }
        iteration_counter += 1
        integrate_state = await run_iteration(
            graph=graph,
            project_prompt=project_prompt,
            adapter_id=integrate_aid,
            session_id=session_id,
            iteration=iteration_counter,
            phase=integrate_phase,
            prompt_context=integrate_ctx,
        )

        # Score from test results
        passed_count, total_count = _count_test_results(
            integrate_state.get("stdout", ""),
            integrate_state.get("stderr", ""),
        )
        tests_passed = integrate_state.get("tests_passed", False)
        if total_count > 0:
            score = passed_count / total_count
        else:
            # No tests ran — never treat as full success
            score = 0.1 if integrate_state.get("exit_code", 1) == 0 else 0.0

        # Update fitness
        if integrate_aid:
            registry.update_fitness(integrate_aid, pass_rate=score, fitness_score=score)

        if score > best_integrate_score:
            best_integrate_score = score
            best_integrate_state = integrate_state
            best_integrate_adapter_id = integrate_aid

        logger.info(
            "  Integrate v%d: tests_passed=%s, score=%.2f%s",
            evo_iter,
            tests_passed,
            score,
            " (best)" if integrate_aid == best_integrate_adapter_id else "",
        )

        # Early stop on success
        if tests_passed:
            logger.info("  Integrate converged at iteration %d", evo_iter + 1)
            break

        # Evolution sweep between iterations
        if evo_iter > 0:
            sweep = evolution_sweep(registry)
            evolution_stats["sweeps"][f"integrate-iter{evo_iter}"] = sweep

    evolution_stats["phase_iterations"]["integrate"] = evo_iter + 1
    evolution_stats["best_adapters"]["integrate"] = best_integrate_adapter_id

    assert best_integrate_state is not None
    final_code = best_integrate_state.get("generated_code", "")
    final_passed = best_integrate_state.get("tests_passed", False)

    phase_results["integrate"] = {
        "adapter_id": best_integrate_adapter_id,
        "tests_passed": final_passed,
        "iterations": evo_iter + 1,
        "best_score": best_integrate_score,
    }

    # ---------------------------------------------------------------
    # Phase 5: DIAGNOSE → REPAIR → RE-INTEGRATE loop
    # When integration fails, ask the model which subtasks need repair,
    # fix them with targeted adapters, and re-integrate.
    # ---------------------------------------------------------------
    repair_history: list[dict[str, Any]] = []

    for repair_iter in range(iters_repair):
        if final_passed:
            break

        logger.info(
            "=== Phase 5: REPAIR iteration %d/%d ===",
            repair_iter + 1,
            iters_repair,
        )

        # --- DIAGNOSE: ask the model which subtasks need fixing ---
        integration_error = _extract_error_summary(
            best_integrate_state.get("stderr", "")
        )
        if not integration_error:
            integration_error = best_integrate_state.get("stderr", "")[:500]

        diagnose_traj = render_trajectory(
            "diagnose",
            project=project_prompt,
            code_outputs=code_outputs,
            integration_error=integration_error,
            repair_history=repair_history,
        )
        diagnose_adapter_dir = str(
            adapter_dir / f"phase5_diagnose_v{repair_iter}"
        )
        diagnose_adapter_path = run_hypernetwork(
            trajectory_text=diagnose_traj,
            output_dir=diagnose_adapter_dir,
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        diagnose_aid: str | None = None
        if diagnose_adapter_path:
            diagnose_aid = f"phase5-diagnose-v{repair_iter}"
            await _load_adapter(diagnose_aid, diagnose_adapter_path, None)

        iteration_counter += 1
        diagnose_state = await run_iteration(
            graph=graph,
            project_prompt=project_prompt,
            adapter_id=diagnose_aid,
            session_id=session_id,
            iteration=iteration_counter,
            phase="diagnose",
            prompt_context={"project_label": project_label},
        )

        diagnosed = _parse_diagnose_output(
            diagnose_state.get("generated_code", ""),
            list(code_outputs.keys()),
        )

        if not diagnosed:
            logger.warning("  Diagnose produced no actionable items, stopping repair")
            break

        logger.info(
            "  Diagnosed %d subtask(s): %s",
            len(diagnosed),
            [(d["name"], d["diagnosis"][:60]) for d in diagnosed],
        )

        # --- REPAIR: fix each diagnosed subtask ---
        for diag in diagnosed:
            repair_name = diag["name"]
            diagnosis = diag["diagnosis"]
            subtask = next(
                (s for s in subtasks if s["name"] == repair_name), None
            )
            if subtask is None:
                logger.warning("  Diagnosed subtask '%s' not found, skipping", repair_name)
                continue

            sibling_skeletons = {
                n: _extract_code_skeleton(c)
                for n, c in code_outputs.items()
                if n != repair_name
            }

            # Per-subtask repair history
            subtask_history = [
                h.get("diagnosis", "")
                for h in repair_history
                for d in h.get("diagnosed", [])
                if d.get("name") == repair_name
            ]

            repair_traj = render_trajectory(
                "code_repair",
                subtask=subtask,
                existing_code=code_outputs[repair_name],
                diagnosis=diagnosis,
                sibling_skeletons=sibling_skeletons,
                repair_history=subtask_history,
                project=project_prompt,
            )
            repair_adapter_dir = str(
                adapter_dir / f"phase5_repair_{repair_name}_v{repair_iter}"
            )
            repair_adapter_path = run_hypernetwork(
                trajectory_text=repair_traj,
                output_dir=repair_adapter_dir,
                base_model_id=base_model_id,
                checkpoint_path=checkpoint_path,
                device=device,
            )

            repair_aid: str | None = None
            if repair_adapter_path:
                repair_aid = f"phase5-repair-{repair_name}-v{repair_iter}"
                await _load_adapter(repair_aid, repair_adapter_path, None)

            iteration_counter += 1
            repair_state = await run_iteration(
                graph=graph,
                project_prompt=project_prompt,
                adapter_id=repair_aid,
                session_id=session_id,
                iteration=iteration_counter,
                phase="code_repair",
                prompt_context={
                    "subtask_name": repair_name,
                    "project_label": project_label,
                    "diagnosis": diagnosis,
                },
            )

            repaired_code = repair_state.get("generated_code", "")
            if repaired_code.strip():
                code_outputs[repair_name] = repaired_code
                logger.info(
                    "  Repaired '%s' (%d lines)",
                    repair_name,
                    len(repaired_code.splitlines()),
                )

        # --- RE-INTEGRATE with patched code ---
        skeletons = {
            name: _extract_code_skeleton(code)
            for name, code in code_outputs.items()
        }
        integration_doc_parts = []
        for st in subtasks:
            name = st["name"]
            plan_summary = plans.get(name, "")[:200]
            code_summary = _extract_code_skeleton(
                code_outputs.get(name, "")
            )[:300]
            integration_doc_parts.append(
                f"- {name}: REPAIRED\n"
                f"  Plan: {plan_summary}\n"
                f"  Code: {code_summary}"
            )
        integration_doc = "\n".join(integration_doc_parts)

        reintegrate_traj = render_trajectory(
            "integrate",
            project=project_prompt,
            subtask_count=len(subtasks),
            skeletons=skeletons,
            integration_doc=integration_doc,
        )
        reintegrate_adapter_dir = str(
            adapter_dir / f"phase5_reintegrate_v{repair_iter}"
        )
        reintegrate_adapter_path = run_hypernetwork(
            trajectory_text=reintegrate_traj,
            output_dir=reintegrate_adapter_dir,
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        reintegrate_aid: str | None = None
        if reintegrate_adapter_path:
            reintegrate_aid = f"phase5-reintegrate-v{repair_iter}"
            await _load_adapter(reintegrate_aid, reintegrate_adapter_path, None)

        iteration_counter += 1
        reintegrate_state = await run_iteration(
            graph=graph,
            project_prompt=project_prompt,
            adapter_id=reintegrate_aid,
            session_id=session_id,
            iteration=iteration_counter,
            phase="integrate",
            prompt_context={
                "subtask_count": len(subtasks),
                "project_label": project_label,
            },
        )

        ri_passed, ri_total = _count_test_results(
            reintegrate_state.get("stdout", ""),
            reintegrate_state.get("stderr", ""),
        )
        ri_score = ri_passed / ri_total if ri_total > 0 else 0.0
        final_passed = reintegrate_state.get("tests_passed", False)

        logger.info(
            "  Re-integrate: %d/%d tests, score=%.2f, passed=%s",
            ri_passed,
            ri_total,
            ri_score,
            final_passed,
        )

        # Record this repair iteration
        repair_history.append({
            "iteration": repair_iter,
            "diagnosed": diagnosed,
            "integration_error": integration_error,
            "result": ri_score,
        })

        if ri_score > best_integrate_score:
            best_integrate_score = ri_score
            best_integrate_state = reintegrate_state
            best_integrate_adapter_id = reintegrate_aid

        if final_passed:
            logger.info("  Repair loop converged at iteration %d", repair_iter + 1)
            break

    evolution_stats["phase_iterations"]["repair"] = len(repair_history)

    # Update final results after repair loop
    final_code = best_integrate_state.get("generated_code", "")
    final_passed = best_integrate_state.get("tests_passed", False)

    if repair_history:
        phase_results["repair"] = {
            "iterations": len(repair_history),
            "best_score": best_integrate_score,
            "diagnosed_total": sum(
                len(h["diagnosed"]) for h in repair_history
            ),
        }

    # Final evolution sweep
    final_sweep = evolution_sweep(registry)
    evolution_stats["sweeps"]["final"] = final_sweep

    # -------------------------------------------------------------------
    # Save generated code artifacts to session directory
    # -------------------------------------------------------------------
    import json as _json

    output_dir = adapter_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save final integrated code as runnable .py
    final_code_path = output_dir / "final.py"
    final_code_path.write_text(final_code)

    # Save per-subtask code
    for name, code_text in code_outputs.items():
        safe_name = re.sub(r"[^\w\-]", "_", name)
        (output_dir / f"subtask_{safe_name}.py").write_text(code_text)

    # Save run summary with full code and execution results
    run_summary = {
        "session_id": session_id,
        "project_prompt": project_prompt,
        "final_tests_passed": final_passed,
        "subtasks": [s["name"] for s in subtasks],
        "plans": dict(plans),
        "code_outputs": dict(code_outputs),
        "code_subtask_results": code_subtask_results,
        "integration_score": best_integrate_score,
        "phase_iterations": evolution_stats.get("phase_iterations", {}),
    }
    (output_dir / "run_summary.json").write_text(
        _json.dumps(run_summary, indent=2, default=str)
    )

    logger.info("Code artifacts saved to %s", output_dir)
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
        "evolution": evolution_stats,
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
    max_phase_iterations: int | None = None,
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
        max_phase_iterations: Default max iterations per phase (env vars override).

    Returns:
        Summary dict with phase results.
    """
    return await run_phased_pipeline(
        project_prompt=project_prompt,
        max_iterations=max_iterations,
        checkpoint_path=checkpoint_path,
        base_model_id=base_model_id,
        device=device,
        max_phase_iterations=max_phase_iterations,
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
    parser.add_argument(
        "--max-phase-iterations",
        type=int,
        default=5,
        help="Max evolutionary iterations per phase (default: 5, early stops on success)",
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
            max_phase_iterations=args.max_phase_iterations,
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
