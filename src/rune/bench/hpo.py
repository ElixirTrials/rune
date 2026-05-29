"""Optuna HPO for engine params on the benchmark suite."""

from __future__ import annotations

import asyncio
from typing import Any


async def run_hpo(
    tasks: list[Any],
    engine: Any,
    base_config: Any,
    model: Any,
    n_trials: int,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    """Run Optuna HPO study tuning engine params to maximise pass@1.

    All search ranges come from ``base_config.hpo``.
    """
    import optuna  # noqa: PLC0415
    from optuna_integration import MLflowCallback  # noqa: PLC0415

    from rune.bench.runner import run_benchmark  # noqa: PLC0415

    hpo = base_config.hpo
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _suggest(
        trial: optuna.Trial,
        name: str,
        int_param: bool = False,
    ) -> float | int:
        spec = hpo[name]
        if int_param:
            return trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
            )
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            log=spec.get("log", False),
        )

    outer_loop = asyncio.get_running_loop()

    def objective(trial: optuna.Trial) -> float:
        adapter_scaling = _suggest(trial, "adapter_scaling")
        temperature = _suggest(trial, "temperature")
        max_tokens = _suggest(trial, "max_tokens", int_param=True)
        max_phase_iterations = _suggest(
            trial,
            "max_phase_iterations",
            int_param=True,
        )
        cont_multiplier = _suggest(trial, "cont_multiplier")

        cfg = base_config.override(
            adapter_scaling=adapter_scaling,
            temperature=temperature,
            max_tokens=max_tokens,
            max_phase_iterations=max_phase_iterations,
            cont_multiplier=cont_multiplier,
        )

        bench_config: dict[str, Any] = {
            "model": model,
            "run_config": cfg.to_dict(),
        }

        future = asyncio.run_coroutine_threadsafe(
            run_benchmark(tasks, engine, bench_config), outer_loop
        )
        return future.result().pass_at_1

    mlflow_kwargs: dict[str, Any] = {"nested": True}
    if parent_run_id:
        mlflow_kwargs["tags"] = {
            "mlflow.parentRunId": parent_run_id,
        }
    mlflow_callback = MLflowCallback(mlflow_kwargs=mlflow_kwargs)
    study = optuna.create_study(
        direction="maximize",
        study_name="rune-bench-hpo",
    )

    await asyncio.to_thread(
        study.optimize,
        objective,
        n_trials,
        callbacks=[mlflow_callback],
    )

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
    }
