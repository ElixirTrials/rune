"""D2LTrainConfig base class for hypernetwork distillation training."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from rune.config import DEFAULT_MODEL_ID, _repo_config_path


class D2LTrainConfig(BaseModel):
    """Base configuration for D2L QLoRA/hypernetwork training runs.

    Attributes:
        model_id: HuggingFace model ID.
        checkpoint_path: Path to an existing checkpoint to resume from.
        corpus_path: Path to JSONL training corpus.
        learning_rate: AdamW learning rate.
        warmup_ratio: Fraction of steps used for LR warmup.
        num_epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Steps before an optimizer update.
        lora_rank: LoRA rank r.
        lora_alpha: LoRA scaling alpha.
        neftune_alpha: NEFTune noise alpha; 0.0 disables it.
        max_seq_length: Maximum tokenized sequence length.
        checkpoint_dir: Directory to save model checkpoints.
        experiment_name: MLflow experiment name for this run.
        logging_steps: Trainer logging cadence in steps.
        save_steps: Checkpoint save cadence in steps.
        eval_steps: Evaluation cadence in steps.
        fp16: Whether to train with FP16 mixed precision.
    """

    model_id: str = DEFAULT_MODEL_ID
    checkpoint_path: str = ""
    corpus_path: str = ""
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_rank: int = 8
    lora_alpha: int = 16
    neftune_alpha: float = 0.0
    max_seq_length: int = 2048
    checkpoint_dir: str = "./checkpoints"
    experiment_name: str = "d2l-qwen3-round1"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    fp16: bool = True


def load_train_config(path: Path | None = None) -> D2LTrainConfig:
    """Load training settings from the unified config.yaml `training:` section.

    The model id is single-sourced: it is taken from the file's top-level
    `model_id` (the same value the inference/engine config uses) unless the
    `training:` section overrides it, and `RUNE_BASE_MODEL` overrides everything.

    Args:
        path: Path to a config YAML. Defaults to the repo-root config.yaml
            (or RUNE_CONFIG). A missing file yields dataclass defaults.

    Returns:
        The parsed D2LTrainConfig.
    """
    import yaml  # noqa: PLC0415

    p = Path(path) if path is not None else _repo_config_path()
    d = yaml.safe_load(p.read_text()) if p.exists() else {}
    d = d or {}
    if not isinstance(d, dict):
        raise ValueError(f"{p} must contain a YAML mapping, got {type(d).__name__}")
    training = dict(d.get("training") or {})
    if "model_id" not in training:
        training["model_id"] = d.get("model_id", DEFAULT_MODEL_ID)
    env_model = os.environ.get("RUNE_BASE_MODEL")
    if env_model:
        training["model_id"] = env_model
    return D2LTrainConfig(**training)
