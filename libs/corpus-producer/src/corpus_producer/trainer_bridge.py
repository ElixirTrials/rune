"""Bridge between the corpus producer and the QLoRA trainer.

Calls ``model_training.trainer.train_and_register`` for a single oracle bin
with Report_2-compliant defaults (rank=64 from DeltaCoder, alpha=32,
lr=2e-4, constant LR schedule, diff_aware_loss=True, warm_start=deltacoder).

GPU imports are deferred inside ``invoke_bin_training`` per INFRA-05 so the
module stays importable in CPU-only CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level sentinel so unittest.mock.patch can find the name.
# In GPU environments this resolves to the real function; in CPU-only CI
# (where model_training is unavailable) it remains None and is patched
# by tests via patch("corpus_producer.trainer_bridge.train_and_register").
try:
    from model_training.trainer import train_and_register  # type: ignore[import]
except ImportError:
    train_and_register: Any = None  # type: ignore[no-redef,assignment]

# Report_2 / DeltaCoder warm-start defaults (Section 2.2)
_DEFAULT_RANK = 64
_DEFAULT_ALPHA = 32  # alpha = rank/2 per DeltaCoder convention
_DEFAULT_LR = 2e-4
_DEFAULT_EPOCHS = 3
_DEFAULT_LR_SCHED = "constant"
_DEFAULT_GRAD_ACCUM = 16
_DEFAULT_WARMUP_RATIO = 0.03
_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"
_MODEL_CONFIG = "qwen3.5-9b"


def invoke_bin_training(
    bin_key: str,
    manifest_path: Path | str,
    *,
    dry_run: bool = False,
    database_url: str | None = None,
    mlflow_experiment: str = "rune-qlora",
    diff_aware_loss: bool = True,
    epochs: int | None = None,
    learning_rate: float = _DEFAULT_LR,
) -> str:
    """Train a QLoRA adapter for one oracle bin and register it.

    Calls ``train_and_register`` with DeltaCoder warm-start and Report_2
    hyperparameter defaults. The adapter_id is deterministic:
    ``oracle_<bin_key>``.

    Args:
        bin_key: Oracle bin identifier (e.g. "decompose_humaneval",
            "diagnose_pooled").
        manifest_path: Path to the JSONL manifest for this bin.
        dry_run: If True, log parameters and return without training.
        database_url: SQLAlchemy URL for AdapterRegistry. Defaults to
            env/default path.
        mlflow_experiment: MLflow experiment name.
        diff_aware_loss: Whether to enable diff-aware loss weighting.
            Default True per Report_2 recommendation.
        epochs: Override training epochs. Defaults to ``_DEFAULT_EPOCHS``.
        learning_rate: Override learning rate.

    Returns:
        The adapter_id registered in AdapterRegistry.

    Raises:
        FileNotFoundError: If ``manifest_path`` does not exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    adapter_id = f"oracle_{bin_key}"
    resolved_epochs = epochs if epochs is not None else _DEFAULT_EPOCHS

    logger.info(
        "Training oracle adapter %r from %s (dry_run=%s)",
        adapter_id,
        manifest_path,
        dry_run,
    )

    if dry_run:
        logger.info(
            "DRY RUN — would call train_and_register("
            "adapter_id=%r, dataset_path=%s, warm_start=%r, "
            "rank=%d, alpha=%d, epochs=%d, lr=%g, diff_aware_loss=%s)",
            adapter_id,
            manifest_path,
            _WARM_START,
            _DEFAULT_RANK,
            _DEFAULT_ALPHA,
            resolved_epochs,
            learning_rate,
            diff_aware_loss,
        )
        return adapter_id

    train_and_register(
        session_id=None,
        adapter_id=adapter_id,
        dataset_path=str(manifest_path),
        task_type=bin_key,
        model_config_name=_MODEL_CONFIG,
        warm_start_adapter_id=_WARM_START,
        rank=_DEFAULT_RANK,
        alpha=_DEFAULT_ALPHA,
        epochs=resolved_epochs,
        learning_rate=learning_rate,
        gradient_accumulation_steps=_DEFAULT_GRAD_ACCUM,
        lr_scheduler_type=_DEFAULT_LR_SCHED,
        warmup_ratio=_DEFAULT_WARMUP_RATIO,
        diff_aware_loss=diff_aware_loss,
        database_url=database_url,
        mlflow_experiment=mlflow_experiment,
        encoding_mode="single_turn",
    )

    logger.info("Registered oracle adapter %r", adapter_id)
    return adapter_id
