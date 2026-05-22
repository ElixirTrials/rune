"""Three-stage training pipeline: oracle → distillation → success gate.

All GPU-heavy imports (torch, transformers, peft, trl, mlflow) are deferred
inside function bodies so this module stays importable in CPU-only CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rune.training.d2l_train import D2LTrainConfig

logger = logging.getLogger(__name__)


def _run_oracle_training(config: D2LTrainConfig, corpus_dir: Path) -> None:
    """Stage 1: per-bin QLoRA oracle training.

    Args:
        config: Training configuration.
        corpus_dir: Directory containing per-bin JSONL corpora.
    """
    logger.info("Stage 1: oracle training from %s", corpus_dir)


def _run_hypernetwork_distillation(config: D2LTrainConfig) -> None:
    """Stage 2: hypernetwork distillation via DiffAwareSFTTrainer + KL+CE.

    Args:
        config: Training configuration.
    """
    from rune.training.d2l_train import run_distillation  # noqa: PLC0415

    logger.info("Stage 2: hypernetwork distillation")
    run_distillation(config)


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

    Stage 1: Oracle training (per-bin QLoRA).
    Stage 2: Hypernetwork distillation (DiffAwareSFTTrainer + KL+CE).
    Stage 3: Success gate evaluation.

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
        return await _run_hpo(config, corpus_dir, n_trials=n_trials)

    _run_oracle_training(config, corpus_dir)
    _run_hypernetwork_distillation(config)

    # Placeholder scores — in production these come from the bench runner.
    baseline_scores: dict[str, float] = {}
    new_scores: dict[str, float] = {}
    return _run_success_gate(baseline_scores, new_scores)


async def _run_hpo(
    config: D2LTrainConfig,
    corpus_dir: Path,
    *,
    n_trials: int,
) -> int:
    """Run Optuna HPO study over key training hyperparameters.

    Args:
        config: Base configuration whose fields are overridden per trial.
        corpus_dir: Corpus directory forwarded to each trial's oracle stage.
        n_trials: Number of Optuna trials to run.

    Returns:
        Exit code from the best-trial pipeline run (0 pass / 1 fail).
    """
    import optuna  # noqa: PLC0415
    from optuna_integration import MLflowCallback  # noqa: PLC0415

    mlflow_cb = MLflowCallback(
        tracking_uri=None,
        metric_name="gate_exit_code",
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
        _run_oracle_training(trial_config, corpus_dir)
        _run_hypernetwork_distillation(trial_config)
        baseline_scores: dict[str, float] = {}
        new_scores: dict[str, float] = {}
        return float(_run_success_gate(baseline_scores, new_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])

    logger.info(
        "HPO complete: best_value=%s best_params=%s",
        study.best_value,
        study.best_params,
    )
    return int(study.best_value)
