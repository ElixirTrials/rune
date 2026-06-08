"""Optuna HPO for engine params on the benchmark suite, with a held-out set."""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42
_DEFAULT_TUNING_FRACTION = 0.70
_BENCH_HPO_STUDY = "rune-bench-hpo"
_BENCH_HPO_DB = Path("optuna_bench_hpo.db")


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
    *,
    fresh: bool = False,
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
        overrides: dict[str, Any] = {
            "adapter_scaling": _suggest(trial, "adapter_scaling"),
            "temperature": _suggest(trial, "temperature"),
            "presence_penalty": _suggest(trial, "presence_penalty"),
            "max_phase_iterations": _suggest(
                trial, "max_phase_iterations", int_param=True
            ),
            "cont_multiplier": _suggest(trial, "cont_multiplier"),
        }
        # Optional categorical: the prompt/adapter conditioning surface (#52).
        if "prompt_mode" in hpo:
            overrides["prompt_mode"] = trial.suggest_categorical(
                "prompt_mode", hpo["prompt_mode"]["choices"]
            )
        cfg = base_config.override(**overrides)
        bench_config: dict[str, Any] = {
            "model": model,
            "run_config": cfg.to_dict(),
        }
        # Trials only ever see the tuning set.
        future = asyncio.run_coroutine_threadsafe(
            run_benchmark(tuning, engine, bench_config), outer_loop
        )
        return future.result().pass_at_1

    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    mlflow_kwargs: dict[str, Any] = {"nested": True}
    if parent_run_id:
        mlflow_kwargs["tags"] = {"mlflow.parentRunId": parent_run_id}
    # Name the per-trial objective metric clearly (was the generic "value") so
    # each nested trial run is traceable by flavor + scale + tuning_pass_at_1.
    mlflow_callback = MLflowCallback(
        mlflow_kwargs=mlflow_kwargs, metric_name="tuning_pass_at_1"
    )

    # Progress curve: log the running best + completed-trial count to the PARENT
    # run as a metric series, so HPO progress is visible without opening every
    # nested trial. Direct client calls are thread-safe (the callback runs inside
    # study.optimize's worker thread).
    _client = MlflowClient()

    def _log_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if not parent_run_id:
            return
        step = trial.number
        # study.best_value raises ValueError until a trial has COMPLETED; the
        # callback also fires after failed/pruned trials, so guard it (else a
        # failed first trial crashes the whole HPO run).
        has_completed = any(
            t.state == optuna.trial.TrialState.COMPLETE
            for t in study.get_trials(deepcopy=False)
        )
        if has_completed:
            _client.log_metric(
                parent_run_id, "best_tuning_pass_at_1", study.best_value, step=step
            )
        _client.log_metric(parent_run_id, "trials_completed", step + 1, step=step)
        if trial.value is not None:
            _client.log_metric(parent_run_id, "trial_pass_at_1", trial.value, step=step)

    if fresh and _BENCH_HPO_DB.exists():
        _BENCH_HPO_DB.unlink()
        logger.info("Deleted existing Optuna DB: %s", _BENCH_HPO_DB)
    study = optuna.create_study(
        direction="maximize",
        study_name=_BENCH_HPO_STUDY,
        storage=f"sqlite:///{_BENCH_HPO_DB}",
        load_if_exists=not fresh,
    )

    await asyncio.to_thread(
        study.optimize,
        objective,
        n_trials,
        callbacks=[mlflow_callback, _log_progress],
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
