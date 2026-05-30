"""D2LTrainConfig base class for hypernetwork distillation training."""

from __future__ import annotations

from pydantic import BaseModel


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

    model_id: str = "Qwen/Qwen3.5-9B"
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

