"""Optuna HPO for engine params on the benchmark suite."""

from __future__ import annotations

import asyncio
from typing import Any


async def run_hpo(
    tasks: list[Any],
    engine: Any,
    base_config: Any,
    model: Any,
    n_trials: int = 50,
) -> dict[str, Any]:
    """Run Optuna HPO study tuning engine params to maximise pass@1.

    Args:
        tasks: BenchTask list to evaluate each trial against.
        engine: Compiled LangGraph engine.
        base_config: PipelineConfig used as the baseline; fields are overridden
            per trial.
        model: ModelWrapper passed through to the benchmark runner.
        n_trials: Number of Optuna trials.

    Returns:
        Dict with ``best_params`` and ``best_value`` (best pass@1).
    """
    import optuna  # noqa: PLC0415
    from optuna_integration import MLflowCallback  # noqa: PLC0415

    from rune.bench.runner import run_benchmark  # noqa: PLC0415

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        adapter_scaling = trial.suggest_float("adapter_scaling", 0.01, 0.2)
        temperature = trial.suggest_float("temperature", 0.1, 1.0)
        max_tokens = trial.suggest_int("max_tokens", 512, 4096, step=256)
        max_phase_iterations = trial.suggest_int("max_phase_iterations", 3, 10)

        cfg = base_config.override(
            adapter_scaling=adapter_scaling,
            temperature=temperature,
            max_tokens=max_tokens,
            max_phase_iterations=max_phase_iterations,
        )

        bench_config: dict[str, Any] = {
            "model": model,
            "run_config": cfg.to_dict(),
        }

        result = asyncio.run(run_benchmark(tasks, engine, bench_config))
        return result.pass_at_1

    mlflow_callback = MLflowCallback(mlflow_kwargs={"nested": True})
    study = optuna.create_study(direction="maximize", study_name="rune-bench-hpo")

    # Run optimize in a thread so the outer async event loop stays alive and
    # the sync objective can call asyncio.run() without nesting conflicts.
    await asyncio.to_thread(
        study.optimize,
        objective,
        n_trials,
        callbacks=[mlflow_callback],
    )

    return {"best_params": study.best_params, "best_value": study.best_value}
