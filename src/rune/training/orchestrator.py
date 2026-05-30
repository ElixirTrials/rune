"""Training pipeline: D2L context distillation → success gate.

All GPU-heavy imports (torch, transformers, peft, trl, mlflow) are deferred
inside function bodies so this module stays importable in CPU-only CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rune.training.d2l_train import D2LTrainConfig

logger = logging.getLogger(__name__)


def _run_hypernetwork_distillation(config: Any) -> None:
    """Stage 2: D2L hypernetwork context distillation.

    Args:
        config: Training configuration.
    """
    from rune.training.hypernet_distill import (  # noqa: PLC0415
        run_hypernet_distillation,
    )

    logger.info("Stage 2: hypernetwork distillation")
    run_hypernet_distillation(config)


def _run_success_gate(
    baseline_scores: dict[str, float],
    new_scores: dict[str, float],
) -> int:
    """Stage 3: evaluate success gate.

    Args:
        baseline_scores: Benchmark scores before training.
        new_scores: Benchmark scores after training.

    Returns:
        0 if gate passes, 1 if gate fails.
    """
    from rune.training.gate import evaluate_gate  # noqa: PLC0415

    result = evaluate_gate(baseline_scores, new_scores)
    if result.passed:
        logger.info(
            "Gate PASSED: %d/%d benchmarks improved",
            result.passing_benchmarks,
            result.total_benchmarks,
        )
        return 0
    logger.warning(
        "Gate FAILED: %d/%d benchmarks improved, regressions=%s",
        result.passing_benchmarks,
        result.total_benchmarks,
        result.regressions,
    )
    return 1


async def run_training_pipeline(
    config: D2LTrainConfig,
    corpus_dir: Path,
    *,
    hpo: bool = False,
    n_trials: int = 50,
) -> int:
    """Run the full three-stage training pipeline.

    Stage 2: D2L hypernetwork context distillation.

    The success gate is evaluated from the bench path (issue #49 Task 13),
    not inline here, so a successful pipeline run returns 0.

    When ``hpo=True``, uses Optuna to tune ``learning_rate``,
    ``warmup_ratio``, ``lora_rank``, and ``neftune_alpha`` over ``n_trials``
    trials, with MLflow logging via MLflowCallback.  All GPU and HPO imports
    are deferred inside this function body.

    Args:
        config: Base training configuration.
        corpus_dir: Directory containing per-bin JSONL corpora.
        hpo: Whether to run Optuna HPO instead of a single training run.
        n_trials: Number of Optuna trials (only used when ``hpo=True``).

    Returns:
        Exit code: 0 on gate pass, 1 on gate failure.
    """
    corpus_dir = Path(corpus_dir)

    if hpo:
        import mlflow  # noqa: PLC0415

        active = mlflow.active_run()
        parent_id = active.info.run_id if active else None
        return await _run_hpo(
            config,
            corpus_dir,
            n_trials=n_trials,
            parent_run_id=parent_id,
        )

    _run_hypernetwork_distillation(config)
    return 0


async def _run_hpo(
    config: D2LTrainConfig,
    corpus_dir: Path,
    *,
    n_trials: int,
    parent_run_id: str | None = None,
) -> int:
    """Run Optuna HPO study over key training hyperparameters.

    Args:
        config: Base configuration whose fields are overridden per trial.
        corpus_dir: Corpus directory forwarded to each trial's distillation stage.
        n_trials: Number of Optuna trials to run.

    Returns:
        Exit code from the best-trial pipeline run (0 pass / 1 fail).
    """
    import optuna  # noqa: PLC0415
    from optuna_integration import MLflowCallback  # noqa: PLC0415

    mlflow_kwargs: dict[str, Any] = {"nested": True}
    if parent_run_id:
        mlflow_kwargs["tags"] = {"mlflow.parentRunId": parent_run_id}
    mlflow_cb = MLflowCallback(
        tracking_uri=None,
        metric_name="gate_exit_code",
        mlflow_kwargs=mlflow_kwargs,
    )

    def objective(trial: Any) -> float:
        trial_config = config.model_copy(
            update={
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-6, 1e-4, log=True
                ),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
                "lora_rank": trial.suggest_categorical("lora_rank", [4, 8, 16, 32]),
                "neftune_alpha": trial.suggest_float("neftune_alpha", 0.0, 10.0),
            }
        )
        _run_hypernetwork_distillation(trial_config)
        # The success gate is evaluated from the bench path (issue #49 Task 13),
        # not inline here. Until that lands every trial returns 0.
        return 0.0

    import asyncio  # noqa: PLC0415

    study = optuna.create_study(direction="minimize")
    await asyncio.to_thread(
        study.optimize, objective, n_trials=n_trials, callbacks=[mlflow_cb]
    )

    logger.info(
        "HPO complete: best_value=%s best_params=%s",
        study.best_value,
        study.best_params,
    )
    return int(study.best_value)
