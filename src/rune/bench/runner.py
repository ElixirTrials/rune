"""Benchmark runner: task loading, execution, and pass@1 scoring."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rune.bench.lcb import extract_entry_function
from rune.engine.continuation import strip_self_tests
from rune.engine.oracle import merge_public_checks
from rune.engine.state import advisory_kinds_from_state, make_initial_state
from rune.engine.validity import validate_solution
from rune.mining.session_log import write_session
from rune.sandbox.executor import run_in_sandbox

logger = logging.getLogger(__name__)


def _seed_rng(seed: int) -> None:
    """Seed the global torch RNG so in-engine generation is reproducible.

    torch's RNG is process-global, so seeding here propagates to every
    model.generate() call the engine makes for the task.
    """
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class BenchTask:
    """A single benchmark problem.

    Attributes:
        task_id: Unique problem identifier.
        description: Natural-language problem statement.
        test_code: Python test code executed against the solution.
        entry_point: Expected function name in the generated solution.
    """

    task_id: str
    description: str
    test_code: str
    entry_point: str = "solution"
    signature: str = ""  # real def line (reference_* adapter anchor); optional
    public_checks: str = ""  # in-loop oracle only; empty => doctest fallback (MBPP)


@dataclass(frozen=True)
class TaskResult:
    """Result for one benchmark task.

    Attributes:
        task_id: Identifier matching the source BenchTask.
        passed: True if the solution passed all tests.
        code: Generated code that was evaluated.
        stderr: Stderr from the test run.
    """

    task_id: str
    passed: bool
    code: str
    stderr: str


@dataclass
class BenchResult:
    """Aggregate benchmark results.

    Attributes:
        pass_at_1: Fraction of tasks passed on first attempt.
        total_tasks: Total tasks evaluated.
        passed_tasks: Number of tasks that passed.
        per_task: Individual TaskResult for each task.
    """

    pass_at_1: float
    total_tasks: int
    passed_tasks: int
    per_task: list[TaskResult] = field(default_factory=list)


def load_tasks(path: Path) -> list[BenchTask]:
    """Load benchmark tasks from a JSON file.

    Args:
        path: Path to a JSON file containing a list of task dicts.

    Returns:
        List of BenchTask instances.
    """
    data = json.loads(path.read_text())
    return [BenchTask(**t) for t in data]


def dump_tasks(tasks: list[BenchTask], path: Path) -> Path:
    """Write benchmark tasks to a JSON file readable by :func:`load_tasks`.

    Args:
        tasks: Tasks to serialise.
        path: Destination JSON path; parent directories are created.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(t) for t in tasks], indent=2))
    return path


def _default_advisory_kinds() -> frozenset[str]:
    from rune.config import PipelineConfig  # noqa: PLC0415

    return frozenset(PipelineConfig().advisory_requirement_kinds)


def _passes_hard_requirements(
    task: BenchTask,
    code: str,
    spec: str,
    *,
    skip_kinds: frozenset[str] | None = None,
) -> bool:
    """Structural requirements only; advisory kinds are ignored at ship."""
    if skip_kinds is None:
        skip_kinds = _default_advisory_kinds()
    if not task.public_checks.strip():
        return True
    return validate_solution(
        code,
        entry_point=task.entry_point,
        signature=task.signature,
        spec=spec,
        public_checks=task.public_checks,
        skip_kinds=skip_kinds,
    ).ok


def _passes_public_checks(task: BenchTask, code: str) -> bool:
    """True only if *code* actually passes the public checks (quality 3).

    Structural validity is necessary but not sufficient: the engine otherwise
    shipped runnable-but-wrong near-misses (``best_quality == 2``) that score
    pass@1=0 (issue #52). Normalize to the gradeable entry form first so the
    canonical ``class Solution`` shape is graded the same way the LCB grader
    grades it.
    """
    from rune.engine.oracle import build_subtask_probe  # noqa: PLC0415

    normalized = (
        extract_entry_function(code, task.entry_point) if task.entry_point else code
    )
    if not normalized.strip():
        return False
    probe, fired = build_subtask_probe(normalized, task.public_checks)
    if not fired:
        return False
    return run_in_sandbox(probe, timeout=5).exit_code == 0


def _benchmark_shippable(
    task: BenchTask,
    code: str,
    spec: str,
    *,
    skip_kinds: frozenset[str] | None = None,
) -> bool:
    if not task.public_checks.strip():
        return True
    if not _passes_hard_requirements(task, code, spec, skip_kinds=skip_kinds):
        return False
    return _passes_public_checks(task, code)


def _runnable_ship_fallback(
    task: BenchTask,
    code: str,
    spec: str,
    *,
    skip_kinds: frozenset[str] | None = None,
) -> bool:
    """True when *code* is structurally shippable even if public checks fail."""
    if not code.strip():
        return False
    if not task.public_checks.strip():
        return True
    return _passes_hard_requirements(task, code, spec, skip_kinds=skip_kinds)


def _write_task_session(
    sessions_dir: Path | None,
    final_state: dict[str, Any],
    task: BenchTask,
    config: dict[str, Any],
    *,
    pass_at_1: bool,
) -> None:
    """Persist engine trajectory for mining/debug even when nothing ships."""
    if sessions_dir is None:
        return
    write_session(
        final_state,
        {
            "benchmark": config.get("benchmark", "unknown"),
            "problem_id": task.task_id,
            "pass_at_1": pass_at_1,
        },
        sessions_dir / task.task_id,
    )


def resolve_shipped_code(
    final_state: dict[str, Any], task: BenchTask, *, spec: str = ""
) -> str:
    """Pick the code to score: integrated, best per entry, or AST-extracted entry."""
    from rune.config import PipelineConfig  # noqa: PLC0415

    defaults = PipelineConfig()
    entry = task.entry_point
    best = final_state.get("best_code") or {}
    best_entry = str(best.get(entry, "")) if entry else ""
    integrated = final_state.get("integrated_code") or ""
    task_spec = spec or task.description
    skip_kinds = advisory_kinds_from_state(final_state)
    ship_best_on_exhaustion = bool(
        final_state.get("ship_best_on_exhaustion", defaults.ship_best_on_exhaustion)
    )
    ship_best_min_quality = int(
        final_state.get("ship_best_min_quality", defaults.ship_best_min_quality)
    )
    if integrated.strip() and entry and best_entry.strip():
        from rune.engine.oracle import defines_entry_point  # noqa: PLC0415

        bq = int(final_state.get("best_quality", {}).get(entry, -1))
        best_valid = _benchmark_shippable(
            task, best_entry, task_spec, skip_kinds=skip_kinds
        )
        int_valid = _benchmark_shippable(
            task, integrated, task_spec, skip_kinds=skip_kinds
        )
        if (
            task.public_checks
            and bq >= 2
            and defines_entry_point(best_entry, entry)
            and best_valid
            and (not defines_entry_point(integrated, entry) or not int_valid or bq >= 3)
        ):
            return best_entry

    if integrated.strip() and _benchmark_shippable(
        task, integrated, task_spec, skip_kinds=skip_kinds
    ):
        return integrated

    shipped = best or final_state.get("code_results", {})
    if entry and entry in shipped:
        candidate = str(shipped[entry])
        if _benchmark_shippable(task, candidate, task_spec, skip_kinds=skip_kinds):
            return candidate

    blob = "\n\n".join(str(v) for v in shipped.values() if v)
    if entry and blob.strip():
        from rune.engine.oracle import defines_entry_point  # noqa: PLC0415

        extracted = extract_entry_function(blob, entry)
        if (
            extracted.strip()
            and defines_entry_point(extracted, entry)
            and _benchmark_shippable(task, extracted, task_spec, skip_kinds=skip_kinds)
        ):
            return extracted

    if not ship_best_on_exhaustion:
        return ""

    # Budget exhausted with no public-passing answer — ship the best attempt we
    # retained rather than submitting blank (issue #52).
    if entry and best_entry.strip():
        from rune.engine.oracle import defines_entry_point  # noqa: PLC0415

        normalized = (
            extract_entry_function(best_entry, entry)
            if task.entry_point
            else best_entry
        )
        bq = int(final_state.get("best_quality", {}).get(entry, -1))
        if (
            bq >= ship_best_min_quality
            and normalized.strip()
            and defines_entry_point(normalized, entry)
            and _runnable_ship_fallback(
                task, normalized, task_spec, skip_kinds=skip_kinds
            )
        ):
            return normalized

    # Last resort: rather than shipping blank, scan every retained candidate
    # (best_code, code_results, and raw trajectory outputs) for any extractable
    # function that defines the entry point and is structurally runnable. The
    # salvage-capable extractor recovers functions buried under trailing garbage
    # (issue: "ships blank despite valid code"). Prefer the most-recent
    # trajectory output (repair wins).
    if entry:
        from rune.engine.oracle import defines_entry_point  # noqa: PLC0415

        candidates: list[str] = []
        for value in best.values():
            if value:
                candidates.append(str(value))
        for value in (final_state.get("code_results") or {}).values():
            if value:
                candidates.append(str(value))
        for record in final_state.get("trajectory", []):
            gen = getattr(record, "generated_code", None)
            if gen:
                candidates.append(str(gen))
            out = getattr(record, "output_text", "")
            if out:
                candidates.append(str(out))
        # Most-recent / repair-stage candidates first.
        for raw in reversed(candidates):
            if not raw.strip():
                continue
            extracted = extract_entry_function(raw, entry)
            if (
                extracted.strip()
                and defines_entry_point(extracted, entry)
                and _runnable_ship_fallback(
                    task, extracted, task_spec, skip_kinds=skip_kinds
                )
            ):
                return extracted

    return ""


async def run_benchmark(
    tasks: list[BenchTask],
    engine: Any,
    config: dict[str, Any],
    sessions_dir: Path | None = None,
) -> BenchResult:
    """Run the full benchmark suite and return aggregate results.

    Args:
        tasks: Tasks to evaluate.
        engine: Compiled LangGraph engine (CompiledStateGraph).
        config: Configurable dict passed to engine.ainvoke (contains
            ``model`` and ``run_config`` keys).
        sessions_dir: If set, write one session dir per task here (corpus producer).

    Returns:
        BenchResult with pass@1 and per-task details.
    """
    if not tasks:
        return BenchResult(pass_at_1=0.0, total_tasks=0, passed_tasks=0, per_task=[])

    budget = config["run_config"]["max_phase_iterations"]
    results: list[TaskResult] = []

    model = config.get("model")
    seed = config["run_config"].get("seed")
    for i, task in enumerate(tasks):
        if seed is not None:
            _seed_rng(seed + i)
        rc = config.get("run_config", {})
        public_checks = task.public_checks
        from rune.config import PipelineConfig  # noqa: PLC0415

        merge_spec = bool(
            rc.get(
                "merge_spec_public_checks",
                PipelineConfig().merge_spec_public_checks,
            )
        )
        if merge_spec and public_checks.strip() and task.entry_point:
            public_checks = merge_public_checks(
                task.description, public_checks, task.entry_point
            )
        initial_state = make_initial_state(
            task.description,
            budget,
            task.entry_point,
            task.signature,
            public_checks,
            run_config=rc,
        )

        try:
            final_state: dict[str, Any] = await engine.ainvoke(
                initial_state, config={"configurable": config}
            )
        except Exception:
            logger.exception("Engine failed for task %s", task.task_id)
            results.append(
                TaskResult(
                    task_id=task.task_id, passed=False, code="", stderr="engine error"
                )
            )
            if model is not None and hasattr(model, "reset_adapter"):
                model.reset_adapter()
            continue

        generated_code = resolve_shipped_code(final_state, task)

        # Strip the model's own self-tests (incl. __main__ asserts) before
        # appending the held-out tests: otherwise a wrong self-test fails a
        # correct implementation. The recorded `code` below stays full-length.
        if not generated_code.strip():
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    passed=False,
                    code="",
                    stderr="entry_point not found in shipped code",
                )
            )
            _write_task_session(
                sessions_dir, final_state, task, config, pass_at_1=False
            )
            if model is not None and hasattr(model, "reset_adapter"):
                model.reset_adapter()
            continue

        full_code = strip_self_tests(generated_code) + "\n\n" + task.test_code

        try:
            sandbox_result = await asyncio.to_thread(run_in_sandbox, full_code)
        except Exception:
            logger.exception("Sandbox failed for task %s", task.task_id)
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    passed=False,
                    code=generated_code,
                    stderr="sandbox error",
                )
            )
            _write_task_session(
                sessions_dir, final_state, task, config, pass_at_1=False
            )
            if model is not None and hasattr(model, "reset_adapter"):
                model.reset_adapter()
            continue

        passed = sandbox_result.exit_code == 0
        results.append(
            TaskResult(
                task_id=task.task_id,
                passed=passed,
                code=generated_code,
                stderr=sandbox_result.stderr,
            )
        )
        _write_task_session(sessions_dir, final_state, task, config, pass_at_1=passed)

        if model is not None and hasattr(model, "reset_adapter"):
            model.reset_adapter()

    n_passed = sum(1 for r in results if r.passed)
    return BenchResult(
        pass_at_1=n_passed / len(results) if results else 0.0,
        total_tasks=len(results),
        passed_tasks=n_passed,
        per_task=results,
    )
