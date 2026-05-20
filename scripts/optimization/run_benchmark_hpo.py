"""Benchmark HPO: Optuna over Rune pipeline parameters on failed MBPP problems.

Tunes ``scaling_factor``, ``temperature``, ``repetition_penalty``,
``max_tokens``, and ``max_phase_iterations`` by running the real
``run_phased_pipeline()`` from ``rune_runner.py`` per problem and measuring
MBPP pass rate. Best params are validated once on a held-out set and saved
to the ``PipelineConfig``.

Usage:
    uv run python scripts/optimization/run_benchmark_hpo.py \\
        --hypernet-checkpoint s3://.../checkpoint.pt \\
        --failed-ids evaluation_results/paper/mbpp_failed_ids.json \\
        --n-trials 30 --problems-per-trial 8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

from evaluation.benchmarks.protocol import Problem  # noqa: E402

logger = logging.getLogger("benchmark_hpo")

EXPERIMENT_NAME = "benchmark-hpo-mbpp"
DEFAULT_SEED = 42
TUNING_FRACTION = 0.70
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"


def split_problems(
    problems: list[Problem],
    seed: int = DEFAULT_SEED,
    tuning_fraction: float = TUNING_FRACTION,
) -> tuple[list[Problem], list[Problem]]:
    """Split problems into (tuning, validation) sets, seed-deterministic.

    Problems are sorted by ``problem_id`` for a stable starting order, then
    shuffled with a seeded RNG. The first ``tuning_fraction`` become the
    tuning set; the remainder are held out for validation.

    Args:
        problems: Problems to split.
        seed: RNG seed controlling the shuffle.
        tuning_fraction: Fraction assigned to the tuning set.

    Returns:
        A ``(tuning, validation)`` tuple of disjoint problem lists.
    """
    ordered = sorted(problems, key=lambda p: p.problem_id)
    random.Random(seed).shuffle(ordered)
    n_tuning = round(len(ordered) * tuning_fraction)
    return ordered[:n_tuning], ordered[n_tuning:]


def load_failed_ids(path: Path) -> set[str]:
    """Load the set of failed MBPP problem IDs from a JSON file.

    The file must be a JSON list of ``problem_id`` strings (e.g.
    ``"mbpp/123"``). Recovering this list from a prior run is the caller's
    responsibility — it is not computed here.

    Args:
        path: Path to the JSON list of failed problem IDs.

    Returns:
        The failed problem IDs as a set.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Failed-ID file not found: {path}. Supply --failed-ids pointing "
            f"at a JSON list of failed MBPP problem_ids (e.g. ['mbpp/12', ...])."
        )
    return {str(i) for i in json.loads(path.read_text())}


def subsample_problems(
    problems: list[Problem],
    n: int,
    seed: int = DEFAULT_SEED,
) -> list[Problem]:
    """Return a seed-deterministic random subset of ``n`` problems.

    If ``n`` is at least the pool size, all problems are returned. Problems
    are sorted by ``problem_id`` before sampling so the result depends only
    on the pool contents and seed, not on input order.

    Args:
        problems: The pool of problems to sample from.
        n: Number of problems to return.
        seed: RNG seed controlling the sample.

    Returns:
        A list of ``min(n, len(problems))`` problems.
    """
    ordered = sorted(problems, key=lambda p: p.problem_id)
    if n >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, n)


@dataclass
class ProblemVerdict:
    """Outcome of running the pipeline on a single problem.

    Attributes:
        problem_id: The MBPP problem_id evaluated.
        passed: True if the generated code passed all MBPP test asserts.
        code_attempts: Code-phase iterations used (``phases.code.iterations``).
        diagnose_fired: True if the diagnose->repair phase ran.
        n_subtasks: Number of subtasks the decompose phase produced.
        wall_time_s: Wall-clock duration of the pipeline run.
        accumulated_code_len: Length of the final accumulated code string.
        error: Full failure message, or "" if passed. Truncated only at the
            point it is logged as an MLflow param (kept whole in JSONL).
        generation: The final accumulated code the pipeline produced. Kept so
            the per-problem JSONL artifact can show exactly what was generated.
        phase_metrics: Flat phase/evolution metrics from extract_phase_metrics.
    """

    problem_id: str
    passed: bool
    code_attempts: int
    diagnose_fired: bool
    n_subtasks: int
    wall_time_s: float
    accumulated_code_len: int
    error: str
    generation: str = ""
    phase_metrics: dict[str, float] = field(default_factory=dict)


def extract_phase_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Flatten a ``run_phased_pipeline()`` result into MLflow metric values.

    Missing phases default to 0.0 (the phase did not run or did not succeed).

    Args:
        result: The dict returned by ``run_phased_pipeline()``.

    Returns:
        A dict of float metric values keyed by metric name.
    """
    phases = result.get("phases", {})
    decompose = phases.get("decompose", {})
    plan = phases.get("plan", {})
    code = phases.get("code", {})
    integrate = phases.get("integrate", {})
    evolution = result.get("evolution", {})
    return {
        "phase_decompose_ok": float(bool(decompose.get("subtasks"))),
        "phase_plan_ok": float(bool(plan.get("plans"))),
        "phase_code_attempts": float(code.get("iterations", 0)),
        "phase_integrate_ok": float(bool(integrate.get("tests_passed"))),
        "evolution_sweeps": float(len(evolution.get("sweeps", {}))),
        "adapters_generated": float(len(result.get("adapters", []))),
    }


def write_trial_pipeline_config(
    scaling_factor: float, dest_dir: Path, *, max_tokens: int | None = None
) -> Path:
    """Write a temp ``PipelineConfig`` JSON carrying the trial's adapter scaling.

    ``run_phased_pipeline()`` reads ``adapter.scaling`` from the PipelineConfig
    (via ``load_config()``, which honors ``RUNE_PIPELINE_CONFIG``). This temp
    file is the only way to vary scaling per trial without modifying
    ``rune_runner.py``.

    Args:
        scaling_factor: Adapter scaling value for this trial.
        dest_dir: Directory the config file is written into (created if absent).
        max_tokens: Generation token cap for this trial. Written into the
            PipelineConfig so ``rune_runner.py`` picks it up via
            ``os.environ.setdefault("RUNE_MAX_TOKENS", ...)``.

    Returns:
        Path to the written ``pipeline_config.json``.
    """
    from shared.pipeline_config import PipelineConfig

    dest_dir.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, object] = {"adapter.scaling": scaling_factor}
    if max_tokens is not None:
        overrides["generation.max_tokens"] = max_tokens
    cfg = PipelineConfig().override(**overrides)
    return cfg.save(dest_dir / "pipeline_config.json")


def apply_trial_env(
    scaling_factor: float,
    temperature: float,
    repetition_penalty: float,
    max_phase_iterations: int,
    config_dir: Path,
    *,
    max_tokens: int | None = None,
    max_tokens_plan: int | None = None,
    max_tokens_code: int | None = None,
    max_tokens_integrate: int | None = None,
    thinking_budget: int | None = None,
) -> None:
    """Set env vars + temp PipelineConfig so ``run_phased_pipeline()`` sees the trial params.

    - ``scaling_factor`` -> temp PipelineConfig + ``RUNE_PIPELINE_CONFIG``
    - ``temperature`` -> ``RUNE_TEMPERATURE``
    - ``repetition_penalty`` -> ``RUNE_REPETITION_PENALTY``
    - ``max_phase_iterations`` -> ``RUNE_MAX_PHASE_ITERATIONS``
    - ``max_tokens`` -> ``RUNE_MAX_TOKENS`` + PipelineConfig
    - ``max_tokens_plan`` -> ``RUNE_MAX_TOKENS_PLAN``
    - ``max_tokens_code`` -> ``RUNE_MAX_TOKENS_CODE``
    - ``max_tokens_integrate`` -> ``RUNE_MAX_TOKENS_INTEGRATE``
    - ``thinking_budget`` -> ``RUNE_THINKING_BUDGET``

    Env vars are set unconditionally (overwriting any prior trial's values);
    ``run_phased_pipeline()`` uses ``os.environ.setdefault`` so values set here
    take precedence.

    Args:
        scaling_factor: Adapter scaling for this trial.
        temperature: Generation temperature.
        repetition_penalty: Generation repetition penalty.
        max_phase_iterations: Per-phase iteration cap.
        config_dir: Directory for the trial's temp PipelineConfig.
        max_tokens: Generation token cap. Also written to
            ``RUNE_MAX_TOKENS`` so ``rune-agent`` nodes pick it up.
        max_tokens_plan: Token cap for the plan phase.
        max_tokens_code: Token cap for the code phase.
        max_tokens_integrate: Token cap for the integrate phase.
        thinking_budget: Chain-of-thought token budget.
    """
    cfg_path = write_trial_pipeline_config(
        scaling_factor, config_dir, max_tokens=max_tokens
    )
    os.environ["RUNE_PIPELINE_CONFIG"] = str(cfg_path)
    os.environ["RUNE_TEMPERATURE"] = str(temperature)
    os.environ["RUNE_REPETITION_PENALTY"] = str(repetition_penalty)
    os.environ["RUNE_MAX_PHASE_ITERATIONS"] = str(max_phase_iterations)
    if max_tokens is not None:
        os.environ["RUNE_MAX_TOKENS"] = str(max_tokens)
    if max_tokens_plan is not None:
        os.environ["RUNE_MAX_TOKENS_PLAN"] = str(max_tokens_plan)
    if max_tokens_code is not None:
        os.environ["RUNE_MAX_TOKENS_CODE"] = str(max_tokens_code)
    if max_tokens_integrate is not None:
        os.environ["RUNE_MAX_TOKENS_INTEGRATE"] = str(max_tokens_integrate)
    if thinking_budget is not None:
        os.environ["RUNE_THINKING_BUDGET"] = str(thinking_budget)


def score_pipeline_result(
    problem: Problem,
    result: dict[str, Any],
    wall_time_s: float,
) -> ProblemVerdict:
    """Score a ``run_phased_pipeline()`` result against the problem's MBPP tests.

    Args:
        problem: The MBPP problem that was run.
        result: The dict returned by ``run_phased_pipeline()``.
        wall_time_s: Wall-clock duration of the pipeline run.

    Returns:
        A ProblemVerdict capturing pass/fail and per-problem diagnostics.
    """
    from evaluation.benchmarks.mbpp import MBPPAdapter

    code = result.get("accumulated_code", "")
    verdict = MBPPAdapter().score(problem, code)
    phase_metrics = extract_phase_metrics(result)
    return ProblemVerdict(
        problem_id=problem.problem_id,
        passed=verdict.passed,
        code_attempts=int(phase_metrics["phase_code_attempts"]),
        diagnose_fired="repair" in result.get("phases", {}),
        n_subtasks=len(result.get("subtasks", [])),
        wall_time_s=wall_time_s,
        accumulated_code_len=len(code),
        error="" if verdict.passed else (verdict.error or ""),
        generation=code,
        phase_metrics=phase_metrics,
    )


def run_pipeline_on_problem(
    problem: Problem,
    hypernet_checkpoint: str,
    base_model: str,
    device: str,
    pool: Any = None,
) -> ProblemVerdict:
    """Run the full 5-phase pipeline on one problem and return its verdict.

    Each problem run is wrapped in an ``mlflow.start_span`` so the full
    pipeline execution (LangGraph nodes, adapter generation, scoring) appears
    as a trace in the MLflow Traces tab with the problem ID, pass/fail
    verdict, and timing.

    Any pipeline exception is caught and converted into a failed verdict so a
    single bad problem cannot abort an Optuna trial. The per-session adapter
    directory is removed afterwards to bound disk usage.

    Args:
        problem: The MBPP problem to solve.
        hypernet_checkpoint: Path (local or ``s3://``) to the hypernetwork.
        base_model: Base model HuggingFace ID.
        device: Device for pipeline computation (e.g. ``"cuda"``).
        pool: Shared ModelPool to reuse across problems. Avoids
            reloading the base model on every call.

    Returns:
        A ProblemVerdict for the run.
    """
    from rune_runner import run_phased_pipeline  # type: ignore[import-not-found]

    mlflow = _mlflow()
    start = time.time()
    adapter_dir: str | None = None
    with mlflow.start_span(name=f"problem/{problem.problem_id}") as span:
        span.set_inputs(
            {
                "problem_id": problem.problem_id,
                "prompt": problem.prompt[:300],
            }
        )
        verdict = None
        try:
            result = asyncio.run(
                run_phased_pipeline(
                    project_prompt=problem.prompt,
                    checkpoint_path=hypernet_checkpoint,
                    base_model_id=base_model,
                    device=device,
                    pool=pool,
                )
            )
            adapter_dir = result.get("adapter_dir")
            verdict = score_pipeline_result(problem, result, time.time() - start)
        except Exception:
            logger.exception(
                "Problem %s crashed — scoring as failed", problem.problem_id
            )
            verdict = ProblemVerdict(
                problem_id=problem.problem_id,
                passed=False,
                code_attempts=0,
                diagnose_fired=False,
                n_subtasks=0,
                wall_time_s=time.time() - start,
                accumulated_code_len=0,
                error="pipeline crash (see logs)",
                generation="",
                phase_metrics={},
            )
        finally:
            if adapter_dir:
                shutil.rmtree(adapter_dir, ignore_errors=True)
            _flush_gpu()
        if verdict is not None:
            span.set_outputs(
                {
                    "passed": verdict.passed,
                    "code_attempts": verdict.code_attempts,
                    "n_subtasks": verdict.n_subtasks,
                    "wall_time_s": round(verdict.wall_time_s, 1),
                    "error": verdict.error[:200] if verdict.error else "",
                }
            )
    # Flush traces eagerly so each problem appears in the UI immediately
    mlflow.flush_trace_async_logging()
    return verdict


def evaluate_problem_set(
    problems: list[Problem],
    hypernet_checkpoint: str,
    base_model: str,
    device: str,
    pool: Any = None,
) -> list[ProblemVerdict]:
    """Run the pipeline on every problem, returning verdicts in input order.

    Args:
        problems: Problems to evaluate.
        hypernet_checkpoint: Path to the hypernetwork checkpoint.
        base_model: Base model HuggingFace ID.
        device: Device for pipeline computation.
        pool: Shared ModelPool to reuse across problems.

    Returns:
        A list of ProblemVerdict, one per input problem.
    """
    return [
        run_pipeline_on_problem(p, hypernet_checkpoint, base_model, device, pool=pool)
        for p in problems
    ]


def _flush_gpu() -> None:
    """Free cached CUDA memory between pipeline runs."""
    import gc  # noqa: PLC0415

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        torch.cuda.empty_cache()
    except ImportError:
        pass  # torch not installed (CPU-only env)


def _mlflow() -> Any:
    """Return the imported ``mlflow`` module (indirection for test monkeypatching)."""
    import mlflow

    return mlflow


def _mlflow_run(**kwargs: Any) -> Any:
    """Open an MLflow run context (indirection for test monkeypatching)."""
    return _mlflow().start_run(**kwargs)


def write_verdicts_jsonl(verdicts: list[ProblemVerdict], path: Path) -> None:
    """Write one JSON record per verdict to a JSONL file.

    Each line is a full per-problem record — generated code and untruncated
    error included — so a failed run can be investigated after the fact.

    Args:
        verdicts: Verdicts to serialize.
        path: Destination ``.jsonl`` file.
    """
    with path.open("w") as f:
        for v in verdicts:
            f.write(json.dumps(asdict(v)) + "\n")


def log_verdicts_artifact(verdicts: list[ProblemVerdict], label: str) -> None:
    """Write verdicts to a temp JSONL, log it to MLflow, then delete the temp file.

    The JSONL lands in the active MLflow run's artifact store (S3-backed via
    the tracking server) under ``verdicts/``. Nothing per-problem is left on
    the local filesystem — this is what keeps a 30-trial run from filling the
    local disk.

    Args:
        verdicts: Per-problem verdicts to persist for investigation.
        label: Filename stem for the artifact (e.g. ``"trial-007"``).
    """
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="hpo_verdicts_"))
    try:
        jsonl_path = tmp_dir / f"{label}.jsonl"
        write_verdicts_jsonl(verdicts, jsonl_path)
        _mlflow().log_artifact(str(jsonl_path), artifact_path="verdicts")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def log_trial_metrics(verdicts: list[ProblemVerdict], wall_time_s: float) -> None:
    """Log aggregate and per-problem metrics for one trial to the active MLflow run.

    Args:
        verdicts: Per-problem verdicts from this trial.
        wall_time_s: Total wall-clock duration of the trial.
    """
    mlflow = _mlflow()
    n = len(verdicts)
    n_passed = sum(v.passed for v in verdicts)
    mlflow.log_metrics(
        {
            "pass_rate": n_passed / n if n else 0.0,
            "n_passed": n_passed,
            "n_problems": n,
            "wall_time_s": wall_time_s,
            "mean_attempts_used": (
                sum(v.code_attempts for v in verdicts) / n if n else 0.0
            ),
            "diagnose_fire_rate": (
                sum(v.diagnose_fired for v in verdicts) / n if n else 0.0
            ),
        }
    )
    for idx, v in enumerate(verdicts):
        pid = v.problem_id
        per_problem = {
            f"problem/{pid}/passed": float(v.passed),
            f"problem/{pid}/code_attempts": v.code_attempts,
            f"problem/{pid}/diagnose_fired": float(v.diagnose_fired),
            f"problem/{pid}/n_subtasks": v.n_subtasks,
            f"problem/{pid}/wall_time_s": v.wall_time_s,
            f"problem/{pid}/accumulated_code_len": v.accumulated_code_len,
        }
        for key, val in v.phase_metrics.items():
            per_problem[f"problem/{pid}/{key}"] = val
        mlflow.log_metrics(per_problem, step=idx)
        if v.error:
            mlflow.log_param(f"problem/{pid}/error", v.error[:500])


def make_objective(
    tuning_problems: list[Problem],
    *,
    hypernet_checkpoint: str,
    base_model: str,
    device: str,
    problems_per_trial: int,
    seed: int,
    work_dir: Path,
    pool: Any = None,
) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective: sample params, run the pipeline, log, score.

    Args:
        tuning_problems: Problems trials sample from.
        hypernet_checkpoint: Path to the hypernetwork checkpoint.
        base_model: Base model HuggingFace ID.
        device: Device for pipeline computation.
        problems_per_trial: Problems sampled (without replacement) per trial.
        seed: Base seed; trial ``k`` samples with ``Random(seed + k)``.
        work_dir: Directory for per-trial temp PipelineConfig files.
        pool: Shared ModelPool kept resident across trials. Created by
            ``main()`` so the base model is loaded once per study.

    Returns:
        An objective callable returning the trial's MBPP pass rate.
    """

    def objective(trial: optuna.Trial) -> float:
        scaling_factor = trial.suggest_float("scaling_factor", 0.02, 0.50, log=True)
        temperature = trial.suggest_float("temperature", 0.1, 0.4)
        repetition_penalty = trial.suggest_float("repetition_penalty", 1.0, 1.3)
        max_tokens = trial.suggest_categorical("max_tokens", [1024, 2048])
        max_phase_iterations = trial.suggest_int("max_phase_iterations", 1, 3)

        # Per-phase token budgets
        max_tokens_plan = trial.suggest_categorical(
            "max_tokens_plan", [512, 1024]
        )
        max_tokens_code = trial.suggest_categorical(
            "max_tokens_code", [1024, 2048]
        )
        max_tokens_integrate = trial.suggest_categorical(
            "max_tokens_integrate", [1024, 2048]
        )
        thinking_budget = trial.suggest_categorical(
            "thinking_budget", [256, 512]
        )

        n = min(problems_per_trial, len(tuning_problems))
        trial_problems = random.Random(seed + trial.number + 1).sample(
            tuning_problems, n
        )
        apply_trial_env(
            scaling_factor=scaling_factor,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_phase_iterations=max_phase_iterations,
            config_dir=work_dir / f"trial_{trial.number}",
            max_tokens=max_tokens,
            max_tokens_plan=max_tokens_plan,
            max_tokens_code=max_tokens_code,
            max_tokens_integrate=max_tokens_integrate,
            thinking_budget=thinking_budget,
        )

        start = time.time()
        with _mlflow_run(run_name=f"trial-{trial.number:03d}", nested=True):
            _mlflow().log_params(
                {
                    "trial_number": trial.number,
                    "scaling_factor": scaling_factor,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    "max_tokens": max_tokens,
                    "max_phase_iterations": max_phase_iterations,
                    "max_tokens_plan": max_tokens_plan,
                    "max_tokens_code": max_tokens_code,
                    "max_tokens_integrate": max_tokens_integrate,
                    "thinking_budget": thinking_budget,
                }
            )
            verdicts = evaluate_problem_set(
                trial_problems,
                hypernet_checkpoint,
                base_model,
                device,
                pool=pool,
            )
            log_trial_metrics(verdicts, time.time() - start)
            log_verdicts_artifact(verdicts, f"trial-{trial.number:03d}")

        n_passed = sum(v.passed for v in verdicts)
        pass_rate = n_passed / len(verdicts) if verdicts else 0.0
        logger.info(
            "Trial %d: pass_rate=%.3f (%d/%d)",
            trial.number,
            pass_rate,
            n_passed,
            len(verdicts),
        )
        return pass_rate

    return objective


def save_best_params(
    study: optuna.Study,
    out_dir: Path,
    config_path: Path | None = None,
) -> Path:
    """Write ``best_params.json`` and update the PipelineConfig with best params.

    ``max_phase_iterations`` is recorded in ``best_params.json`` only — the
    PipelineConfig has no field for it.

    Args:
        study: The completed Optuna study.
        out_dir: Directory for ``best_params.json`` (created if absent).
        config_path: PipelineConfig destination; ``None`` writes the default
            ``~/.rune/pipeline_config.json``.

    Returns:
        Path to the written ``best_params.json``.
    """
    from shared.pipeline_config import PipelineConfig

    best = study.best_params
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "best_params.json"
    params_path.write_text(json.dumps(best, indent=2))

    overrides: dict[str, object] = {
        "adapter.scaling": best["scaling_factor"],
        "generation.temperature": best["temperature"],
        "generation.repetition_penalty": best["repetition_penalty"],
        "generation.max_tokens": best["max_tokens"],
    }
    # Per-phase token budgets (may not be present in older studies)
    for key in (
        "max_tokens_plan",
        "max_tokens_code",
        "max_tokens_integrate",
        "thinking_budget",
    ):
        if key in best:
            overrides[f"phase_tokens.{key}"] = best[key]

    config = PipelineConfig().override(**overrides)
    config.save(config_path)
    return params_path


def write_validation_results(verdicts: list[ProblemVerdict], path: Path) -> None:
    """Write per-problem validation pass/fail results as JSON.

    Args:
        verdicts: Verdicts from the validation-set evaluation.
        path: Destination JSON file.
    """
    n = len(verdicts)
    n_passed = sum(v.passed for v in verdicts)
    data = {
        "pass_rate": n_passed / n if n else 0.0,
        "n_passed": n_passed,
        "n_problems": n,
        "problems": {
            v.problem_id: {
                "passed": v.passed,
                "code_attempts": v.code_attempts,
                "diagnose_fired": v.diagnose_fired,
                "error": v.error,
            }
            for v in verdicts
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_trial_summary(study: optuna.Study, path: Path) -> None:
    """Write a CSV of every trial: number, state, pass rate, and params.

    Args:
        study: The completed Optuna study.
        path: Destination CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial_number",
                "state",
                "pass_rate",
                "scaling_factor",
                "temperature",
                "repetition_penalty",
                "max_tokens",
                "max_phase_iterations",
            ]
        )
        for t in study.trials:
            writer.writerow(
                [
                    t.number,
                    t.state.name,
                    t.value if t.value is not None else "",
                    t.params.get("scaling_factor", ""),
                    t.params.get("temperature", ""),
                    t.params.get("repetition_penalty", ""),
                    t.params.get("max_tokens", ""),
                    t.params.get("max_phase_iterations", ""),
                ]
            )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the benchmark HPO runner."""
    from datetime import date

    parser = argparse.ArgumentParser(
        description="Optuna HPO over Rune pipeline params on failed MBPP problems"
    )
    parser.add_argument(
        "--hypernet-checkpoint",
        required=True,
        help="Path (local or s3://) to the trained hypernetwork checkpoint",
    )
    parser.add_argument(
        "--failed-ids",
        type=Path,
        default=Path("evaluation_results/paper/mbpp_failed_ids.json"),
        help="JSON list of MBPP problem_ids to use as the working set. "
        "Defaults to the recovered failed-problem list. If the file is "
        "absent, a random subset of --n-problems is used instead.",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=127,
        help="Size of the random MBPP subset used when --failed-ids is absent",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--problems-per-trial", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--study-name",
        default=f"mbpp-hpo-{date.today():%Y%m%d}",
        help="Optuna study name (also the MLflow parent run name)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results/benchmark_hpo"),
        help="Directory for best_params.json, study.db, and result artifacts",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Optuna storage URI (default: sqlite:///<output-dir>/study.db)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: shrink to 1 trial x 2 problems for a fast "
        "end-to-end check of the pipeline + MLflow wiring",
    )
    return parser


def main() -> None:
    """Entry point: load problems, run the Optuna study, validate, save artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    if args.smoke:
        args.n_trials = 1
        args.problems_per_trial = 2
        logger.info("Smoke mode: n_trials=1, problems_per_trial=2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_uri = args.db or f"sqlite:///{args.output_dir / 'study.db'}"
    work_dir = args.output_dir / "trials"

    from evaluation.benchmarks.mbpp import MBPPAdapter

    all_problems = MBPPAdapter().load_problems()
    if args.failed_ids.exists():
        failed = load_failed_ids(args.failed_ids)
        problems = [p for p in all_problems if p.problem_id in failed]
        if not problems:
            raise SystemExit(
                f"No MBPP problems matched the {len(failed)} failed IDs in "
                f"{args.failed_ids}. Check the file's problem_id format."
            )
        logger.info("Using %d problems from %s", len(problems), args.failed_ids)
    else:
        problems = subsample_problems(all_problems, args.n_problems, args.seed)
        logger.info(
            "%s not found - using a random subset of %d MBPP problems",
            args.failed_ids,
            len(problems),
        )
    tuning, validation = split_problems(problems, seed=args.seed)
    logger.info(
        "Loaded %d failed problems: tuning=%d validation=%d",
        len(problems),
        len(tuning),
        len(validation),
    )

    from model_training.training_common import setup_mlflow

    mlflow = _mlflow()
    # setup_mlflow resolves MLFLOW_TRACKING_URI (default http://localhost:5000,
    # the in-pod server), sets the experiment, and returns False if the server
    # is unreachable. This study logs every trial/metric/JSONL artifact to
    # MLflow, so a missing server is fatal rather than a silent local fallback.
    if not setup_mlflow(EXPERIMENT_NAME, tracking_uri=None):
        raise SystemExit(
            "MLflow is unavailable (tracking server unreachable or "
            "RUNE_DISABLE_MLFLOW=1). Start the stack first: "
            "docker compose -f infra/docker-compose.yml up -d mlflow"
        )
    mlflow.langchain.autolog(run_tracer_inline=True)
    # The in-pod server runs with --serve-artifacts, so get_artifact_uri()
    # returns a proxied mlflow-artifacts:/ URI (never s3://) and the server
    # streams artifacts to its S3 --artifacts-destination. The only way
    # artifacts land on local disk instead is a non-http tracking URI.
    tracking_uri = mlflow.get_tracking_uri()
    if not tracking_uri.startswith(("http://", "https://")):
        logger.warning(
            "MLflow tracking URI is %s - not an http(s) server. Artifacts "
            "(per-problem JSONL, study.db) will land on local disk, not S3. "
            "Set MLFLOW_TRACKING_URI=http://localhost:5000.",
            tracking_uri,
        )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=db_uri,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    from model_training.model_pool import ModelPool  # noqa: PLC0415

    # Force NF4 quantization — HPO needs stability over bf16 speed, and
    # the perceiver forward pass peak pushes L4 VRAM past the 15% headroom.
    os.environ["RUNE_POOL_QUANTIZE"] = "1"

    pool = ModelPool.create(
        model_name=args.base_model,
        device=args.device,
        hypernet_checkpoint_path=args.hypernet_checkpoint,
    )

    objective = make_objective(
        tuning,
        hypernet_checkpoint=args.hypernet_checkpoint,
        base_model=args.base_model,
        device=args.device,
        problems_per_trial=args.problems_per_trial,
        seed=args.seed,
        work_dir=work_dir,
        pool=pool,
    )

    t0 = time.time()
    with _mlflow_run(run_name=args.study_name):
        mlflow.log_params(
            {
                "study_name": args.study_name,
                "sampler": "TPE",
                "n_trials": args.n_trials,
                "problems_per_trial": args.problems_per_trial,
                "tuning_set_size": len(tuning),
                "validation_set_size": len(validation),
                "seed": args.seed,
            }
        )
        # catch=() — all exceptions are handled inside
        # run_pipeline_on_problem (scored as failed verdicts).
        study.optimize(objective, n_trials=args.n_trials, catch=())

        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not completed:
            raise SystemExit(
                "Benchmark HPO produced no successful trials. "
                "Refusing to save a best config."
            )

        best = study.best_params
        logger.info("Best trial #%d: %s", study.best_trial.number, best)

        # Validation pass: best params, evaluated once on the held-out set.
        apply_trial_env(
            scaling_factor=best["scaling_factor"],
            temperature=best["temperature"],
            repetition_penalty=best["repetition_penalty"],
            max_phase_iterations=best["max_phase_iterations"],
            config_dir=work_dir / "validation",
            max_tokens=best["max_tokens"],
        )
        val_verdicts = evaluate_problem_set(
            validation,
            args.hypernet_checkpoint,
            args.base_model,
            args.device,
            pool=pool,
        )
        val_pass_rate = (
            sum(v.passed for v in val_verdicts) / len(val_verdicts)
            if val_verdicts
            else 0.0
        )
        log_verdicts_artifact(val_verdicts, "validation")

        params_path = save_best_params(study, args.output_dir)
        validation_path = args.output_dir / "validation_results.json"
        summary_path = args.output_dir / "trial_summary.csv"
        write_validation_results(val_verdicts, validation_path)
        write_trial_summary(study, summary_path)

        mlflow.log_metrics(
            {
                "best_trial_number": study.best_trial.number,
                "best_pass_rate": study.best_value,
                "validation_pass_rate": val_pass_rate,
                "tuning_vs_validation_gap": study.best_value - val_pass_rate,
                "total_wall_time_s": time.time() - t0,
            }
        )
        for key, val in best.items():
            mlflow.log_param(f"best/{key}", val)
        for artifact in (params_path, validation_path, summary_path):
            mlflow.log_artifact(str(artifact))
        if db_uri.startswith("sqlite:///"):
            db_file = db_uri.replace("sqlite:///", "")
            if Path(db_file).exists():
                mlflow.log_artifact(db_file)

    pool.release()
    _flush_gpu()

    logger.info(
        "HPO complete: best_pass_rate=%.3f validation_pass_rate=%.3f",
        study.best_value,
        val_pass_rate,
    )
    print(f"Best params: {best}")
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
