"""D2LTrainConfig base class and round-2 distillation entrypoint.

All GPU imports (torch, transformers, peft, trl) are deferred inside function
bodies so this module stays importable in CPU-only CI.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


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


def run_distillation(config: D2LTrainConfig) -> None:
    """Entrypoint for round-2 distillation training.

    Loads corpus from ``config.corpus_path``, constructs a
    :class:`~rune.training.diff_loss.DiffAwareSFTTrainer` via
    :func:`~rune.training.diff_loss.build_diff_aware_sft_trainer`, and
    runs training.  All GPU-heavy imports are deferred here so the module
    remains importable in CPU-only CI.

    Args:
        config: Training configuration.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import datasets as hf_datasets  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import trl  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from rune.training.diff_loss import build_diff_aware_sft_trainer  # noqa: PLC0415

    logger.info("run_distillation: loading corpus from %s", config.corpus_path)
    corpus_path = Path(config.corpus_path)
    records: list[dict[object, object]] = []
    if corpus_path.exists():
        with corpus_path.open() as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if stripped:
                    records.append(json.loads(stripped))
    logger.info("run_distillation: %d training records loaded", len(records))

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
    )

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        task_type="CAUSAL_LM",
    )

    # SFTConfig (not transformers.TrainingArguments) so build_diff_aware_sft_trainer's
    # getattr(args, "max_length") threads the sequence cap into the collator;
    # with TrainingArguments it was always None and every record reached the GPU
    # at full length (OOM on long records).
    training_args = trl.SFTConfig(  # type: ignore[attr-defined, unused-ignore]
        output_dir=config.checkpoint_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        run_name=config.experiment_name,
        report_to=["mlflow"],
        max_length=config.max_seq_length,
    )

    dataset = hf_datasets.Dataset.from_list(records)

    trainer = build_diff_aware_sft_trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        tokenizer=tokenizer,
    )

    if config.checkpoint_path:
        trainer.train(resume_from_checkpoint=config.checkpoint_path)
    else:
        trainer.train()

    trainer.save_model(config.checkpoint_dir)
    logger.info("run_distillation: model saved to %s", config.checkpoint_dir)
