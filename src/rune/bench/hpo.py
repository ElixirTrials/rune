"""Optuna HPO for engine params on the benchmark suite, with a held-out set."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42
_DEFAULT_TUNING_FRACTION = 0.70


def split_tasks(
    tasks: list[Any],
    *,
    seed: int = _DEFAULT_SEED,
    tuning_fraction: float = _DEFAULT_TUNING_FRACTION,
) -> tuple[list[Any], list[Any]]:
    """Split tasks into ``(tuning, validation)``, seed-deterministic.

    Tasks are sorted by ``task_id`` for a stable order, shuffled with a seeded
    RNG, then the first ``tuning_fraction`` become the tuning set and the rest
    are held out for validation. The split is independent of input order.

    Args:
        tasks: Tasks to split.
        seed: RNG seed controlling the shuffle.
        tuning_fraction: Fraction assigned to the tuning set.

    Returns:
        A disjoint ``(tuning, validation)`` tuple.
    """
    ordered = sorted(tasks, key=lambda t: t.task_id)
    random.Random(seed).shuffle(ordered)
    n_tuning = round(len(ordered) * tuning_fraction)
    return ordered[:n_tuning], ordered[n_tuning:]


async def run_hpo(
    tasks: list[Any],
    engine: Any,
    base_config: Any,
    model: Any,
    n_trials: int,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    """Tune engine params to maximise tuning-set pass@1, then score the best
    params once on a held-out validation set.

    Optimising and reporting on the same problems overfits the hyperparameters,
    so the task pool is split (``hpo.seed`` / ``hpo.tuning_fraction``, defaults
    42 / 0.70): trials only ever see the tuning set; ``validation_pass_at_1`` is
    the trustworthy, held-out number. Search ranges come from ``base_config.hpo``.
    """
    import optuna  # noqa: PLC0415
    from optuna_integration import MLflowCallback  # noqa: PLC0415

    from rune.bench.runner import run_benchmark  # noqa: PLC0415

    hpo = base_config.hpo
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tuning, validation = split_tasks(
        tasks,
        seed=hpo.get("seed", _DEFAULT_SEED),
        tuning_fraction=hpo.get("tuning_fraction", _DEFAULT_TUNING_FRACTION),
    )
    logger.info(
        "HPO split: %d tuning / %d held-out validation", len(tuning), len(validation)
    )

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
        cfg = base_config.override(
            adapter_scaling=_suggest(trial, "adapter_scaling"),
            temperature=_suggest(trial, "temperature"),
            max_tokens=_suggest(trial, "max_tokens", int_param=True),
            max_phase_iterations=_suggest(
                trial, "max_phase_iterations", int_param=True
            ),
            cont_multiplier=_suggest(trial, "cont_multiplier"),
        )
        bench_config: dict[str, Any] = {
            "model": model,
            "run_config": cfg.to_dict(),
        }
        # Trials only ever see the tuning set.
        future = asyncio.run_coroutine_threadsafe(
            run_benchmark(tuning, engine, bench_config), outer_loop
        )
        return future.result().pass_at_1

    mlflow_kwargs: dict[str, Any] = {"nested": True}
    if parent_run_id:
        mlflow_kwargs["tags"] = {"mlflow.parentRunId": parent_run_id}
    mlflow_callback = MLflowCallback(mlflow_kwargs=mlflow_kwargs)
    study = optuna.create_study(direction="maximize", study_name="rune-bench-hpo")

    await asyncio.to_thread(
        study.optimize,
        objective,
        n_trials,
        callbacks=[mlflow_callback],
    )

    result: dict[str, Any] = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_tuning": len(tuning),
        "n_validation": len(validation),
        "validation_pass_at_1": None,
    }

    # Held-out: score the best params exactly once on the untouched set.
    if validation:
        best_cfg = base_config.override(**study.best_params)
        val = await run_benchmark(
            validation,
            engine,
            {"model": model, "run_config": best_cfg.to_dict()},
        )
        result["validation_pass_at_1"] = val.pass_at_1
        logger.info("HPO held-out validation pass@1: %.3f", val.pass_at_1)
    else:
        logger.warning("No held-out validation tasks; skipping validation pass")

    return result
