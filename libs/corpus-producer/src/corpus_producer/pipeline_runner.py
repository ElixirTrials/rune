"""Subprocess wrapper around scripts/rune_runner.py.

Runs one (benchmark, problem) pair through the full 5-phase Rune pipeline
and parses per-phase artifacts from the JSON output file written by the
subprocess.

Design rationale: subprocess mode (vs in-process import) gives clean GPU
state, prevents adapter registry cross-contamination between runs, and
enables future process-level parallelism.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from corpus_producer.models import PhaseArtifact

logger = logging.getLogger(__name__)

# Path to the rune_runner.py entrypoint, resolved relative to this file.
# Walks up: corpus_producer/pipeline_runner.py -> corpus_producer/ -> src/
#   -> corpus-producer/ -> libs/ -> rune/ -> scripts/
_RUNE_ROOT = Path(__file__).resolve().parents[5]
_RUNE_RUNNER = _RUNE_ROOT / "scripts" / "rune_runner.py"


@dataclass
class PipelineRunResult:
    """Output of a single pipeline run for one (benchmark, problem) pair.

    Attributes:
        run_id: UUID of this pipeline run.
        benchmark: Benchmark identifier.
        problem_id: Problem identifier within the benchmark.
        artifacts: Per-phase PhaseArtifacts, in pipeline order.
        final_code: Integrated final code (Phase 4 output).
        success: True if the subprocess exited 0 and produced artifacts.
        error: Subprocess stderr, populated on failure.
    """

    run_id: str
    benchmark: str
    problem_id: str
    artifacts: list[PhaseArtifact] = field(default_factory=list)
    final_code: str = ""
    success: bool = False
    error: str = ""


@runtime_checkable
class PipelineRunnerProtocol(Protocol):
    """Protocol for injectable pipeline runner (enables test doubles)."""

    def __call__(
        self,
        benchmark: str,
        problem_id: str,
        problem_prompt: str,
        *,
        timeout: int = 300,
        base_model_id: str = "Qwen/Qwen3.5-9B",
    ) -> PipelineRunResult:
        """Run the pipeline for one (benchmark, problem) pair."""
        ...


def _parse_artifacts_from_output(
    run_id: str,
    benchmark: str,
    problem_id: str,
    output_data: dict[str, object],
) -> list[PhaseArtifact]:
    """Extract per-phase PhaseArtifacts from rune_runner JSON output.

    rune_runner returns a dict with keys:
      phase_results.decompose.{output}
      phase_results.plan.{outputs: {subtask: text}}
      phase_results.code.{outputs: {subtask: text}}
      phase_results.integrate.{generated_code}
      phase_results.repair (optional, if diagnose fired)

    We flatten to one artifact per phase. For code/plan we concatenate
    subtask outputs into a single text with subtask headers.

    Args:
        run_id: Pipeline run UUID.
        benchmark: Benchmark identifier.
        problem_id: Problem identifier.
        output_data: Parsed JSON dict from rune_runner.

    Returns:
        List of PhaseArtifact, one per phase that produced output.
    """
    artifacts: list[PhaseArtifact] = []
    raw_phase_results = output_data.get("phase_results") or {}
    phase_results: dict[str, object] = (
        raw_phase_results if isinstance(raw_phase_results, dict) else {}
    )

    def _art(
        phase: str,
        inp: str,
        out: str,
        meta: dict[str, str] | None = None,
    ) -> PhaseArtifact:
        return PhaseArtifact(
            phase=phase,
            benchmark=benchmark,
            problem_id=problem_id,
            pipeline_run_id=run_id,
            input_text=inp,
            output_text=out,
            metadata=meta or {},
        )

    # DECOMPOSE
    raw_decompose = phase_results.get("decompose") or {}
    decompose: dict[str, object] = (
        raw_decompose if isinstance(raw_decompose, dict) else {}
    )
    if decompose.get("output"):
        project_prompt = str(output_data.get("project_prompt", ""))
        decompose_out = str(decompose["output"])
        artifacts.append(_art("decompose", inp=project_prompt, out=decompose_out))

    # PLAN (concatenate subtask plans)
    raw_plan = phase_results.get("plan") or {}
    plan: dict[str, object] = raw_plan if isinstance(raw_plan, dict) else {}
    if plan.get("plans"):
        plans_dict: dict[str, str] = plan["plans"]  # type: ignore[assignment]
        out_parts = [f"### {k}\n{v}" for k, v in plans_dict.items()]
        plan_inp = str(decompose.get("output", ""))
        artifacts.append(_art("plan", inp=plan_inp, out="\n\n".join(out_parts)))

    # CODE (concatenate subtask code)
    raw_code = phase_results.get("code") or {}
    code: dict[str, object] = raw_code if isinstance(raw_code, dict) else {}
    if code.get("outputs"):
        code_dict: dict[str, str] = code["outputs"]  # type: ignore[assignment]
        out_parts = [f"### {k}\n{v}" for k, v in code_dict.items()]
        code_inp = str(plan.get("plans", ""))
        artifacts.append(_art("code", inp=code_inp, out="\n\n".join(out_parts)))

    # INTEGRATE
    raw_integrate = phase_results.get("integrate") or {}
    integrate: dict[str, object] = (
        raw_integrate if isinstance(raw_integrate, dict) else {}
    )
    if True:  # always attempt integrate artifact if key present in phase_results
        final = str(output_data.get("final_code", ""))
        integrate_inp = str(code.get("outputs", ""))
        if "integrate" in phase_results:
            artifacts.append(_art("integrate", inp=integrate_inp, out=final))

    # DIAGNOSE (optional — only if repair loop fired)
    raw_repair = phase_results.get("repair") or {}
    repair: dict[str, object] = raw_repair if isinstance(raw_repair, dict) else {}
    if repair.get("diagnosis"):
        diagnose_inp = str(integrate.get("stderr", ""))
        diagnose_out = str(repair["diagnosis"])
        artifacts.append(_art("diagnose", inp=diagnose_inp, out=diagnose_out))

    return artifacts


def run_pipeline_for_problem(
    benchmark: str,
    problem_id: str,
    problem_prompt: str,
    *,
    timeout: int = 300,
    base_model_id: str = "Qwen/Qwen3.5-9B",
) -> PipelineRunResult:
    """Run the full 5-phase Rune pipeline for one problem via subprocess.

    Writes a temporary JSON output file that rune_runner populates, then
    reads and parses artifacts from it.

    Args:
        benchmark: Benchmark identifier (used for artifact tagging only).
        problem_id: Problem identifier (used for artifact tagging).
        problem_prompt: Full problem text passed to rune_runner --project.
        timeout: Process timeout in seconds (default 300).
        base_model_id: Base model HF repo id.

    Returns:
        PipelineRunResult with artifacts and success flag.
    """
    run_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory(prefix="rune_run_") as tmpdir:
        output_json = Path(tmpdir) / "pipeline_output.json"

        cmd = [
            "uv",
            "run",
            str(_RUNE_RUNNER),
            "--project",
            problem_prompt,
            "--output-json",
            str(output_json),
            "--base-model",
            base_model_id,
            "--max-phase-iterations",
            "1",  # one iteration per phase for corpus production
        ]

        logger.info(
            "Running pipeline for %s/%s (run_id=%s)",
            benchmark,
            problem_id,
            run_id,
        )
        try:
            proc = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
                cwd=str(_RUNE_ROOT),
            )
        except subprocess.TimeoutExpired:
            logger.warning("Pipeline timed out for %s/%s", benchmark, problem_id)
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error=f"Timed out after {timeout}s",
            )
        except Exception as exc:
            logger.exception("Subprocess error for %s/%s", benchmark, problem_id)
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error=str(exc),
            )

        if proc.returncode != 0:
            logger.warning(
                "Pipeline failed for %s/%s (exit %d): %s",
                benchmark,
                problem_id,
                proc.returncode,
                proc.stderr[-500:],
            )
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error=proc.stderr[-500:],
            )

        if not output_json.exists():
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error="rune_runner did not write output JSON",
            )

        try:
            output_data = json.loads(output_json.read_text())
        except json.JSONDecodeError as exc:
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error=f"JSON parse error: {exc}",
            )

        artifacts = _parse_artifacts_from_output(
            run_id, benchmark, problem_id, output_data
        )
        final_code = str(output_data.get("final_code", ""))

        return PipelineRunResult(
            run_id=run_id,
            benchmark=benchmark,
            problem_id=problem_id,
            artifacts=artifacts,
            final_code=final_code,
            success=True,
        )
