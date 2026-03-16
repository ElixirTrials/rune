"""KL-divergence context distillation training loop for Qwen3-Coder-Next.

Assembles all Phase 25-28 components (config, data pipeline, activation
extraction, weight transfer, functional LoRA injection) into a complete
distillation training script.

Three execution modes:
- dry-run: loads real base model + hypernet, validates tensor shapes, exits
- smoke-test: 5 training steps, verifies finite loss and decreasing trend
- full: trains from JSONL dataset with tiered checkpointing and MLflow tracking

All heavy GPU imports (torch, transformers, peft) are deferred to function
bodies per INFRA-05 project convention.

Usage:
    uv run python -m model_training.d2l_train --dry-run
    uv run python -m model_training.d2l_train --smoke-test
    uv run python -m model_training.d2l_train --dataset path/to/train.jsonl
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

__all__ = ["train_d2l_qwen3", "D2LTrainConfig"]


class D2LTrainConfig(BaseModel):
    """Pydantic model for D2L training hyperparameters.

    Enables validation, JSON serialization (for checkpoint storage), and
    `.model_dump()` for MLflow experiment logging.

    Attributes:
        base_model_name: HuggingFace model name for the student/teacher base.
        sakana_checkpoint_path: Path to the Sakana hypernet checkpoint.
        num_steps: Total training steps.
        lr: Learning rate for AdamW optimizer.
        alpha: Blending weight for KL vs CE loss (1.0 = pure KL, 0.0 = pure CE).
        temperature: Softmax temperature for KL divergence computation.
        checkpoint_every: Steps between lightweight checkpoint saves.
        full_checkpoint_every: Steps between full checkpoint saves (incl. optimizer).
        checkpoint_dir: Directory for checkpoint output.
        experiment_name: MLflow experiment name.
        dry_run: If True, validate tensor shapes then exit.
        smoke_test: If True, run 5 steps and verify loss trend.
        dataset_path: Path to training JSONL file (required for full training).
        grad_clip: Gradient clipping max norm.
        warmup_steps: Number of linear LR warmup steps.
        lora_r: LoRA rank.
        max_length: Maximum tokenizer sequence length.
    """

    base_model_name: str = Field(default="Qwen/Qwen3-Coder-Next")
    sakana_checkpoint_path: str
    num_steps: int = Field(default=100)
    lr: float = Field(default=2e-4)
    alpha: float = Field(default=0.5)
    temperature: float = Field(default=2.0)
    checkpoint_every: int = Field(default=100)
    full_checkpoint_every: int = Field(default=500)
    checkpoint_dir: str = Field(default="./checkpoints")
    experiment_name: str = Field(default="d2l-qwen3")
    dry_run: bool = Field(default=False)
    smoke_test: bool = Field(default=False)
    dataset_path: str | None = Field(default=None)
    grad_clip: float = Field(default=1.0)
    warmup_steps: int = Field(default=10)
    lora_r: int = Field(default=8)
    max_length: int = Field(default=512)

    @field_validator("lr")
    @classmethod
    def _validate_lr(cls, v: float) -> float:
        """Ensure learning rate is strictly positive."""
        if v <= 0:
            raise ValueError(f"lr must be > 0, got {v}")
        return v

    @field_validator("alpha")
    @classmethod
    def _validate_alpha(cls, v: float) -> float:
        """Ensure alpha is in [0, 1]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {v}")
        return v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        """Ensure temperature is strictly positive."""
        if v <= 0:
            raise ValueError(f"temperature must be > 0, got {v}")
        return v

    @field_validator("num_steps")
    @classmethod
    def _validate_num_steps(cls, v: int) -> int:
        """Ensure num_steps is strictly positive."""
        if v <= 0:
            raise ValueError(f"num_steps must be > 0, got {v}")
        return v


def _compute_kl_ce_loss(
    student_logits: Any,
    teacher_logits: Any,
    answer_start: int,
    config: D2LTrainConfig,
) -> tuple[Any, dict[str, float]]:
    """Compute blended KL-divergence and cross-entropy loss over the answer span.

    Slices both logit tensors to ``[:, answer_start:, :]`` so that loss is
    computed only on answer-span tokens (context tokens are masked out).

    Args:
        student_logits: Student model logits of shape (batch, seq_len, vocab).
        teacher_logits: Teacher model logits of shape (batch, seq_len, vocab).
        answer_start: Token index where the answer span begins.  Tokens before
            this index are excluded from loss computation.
        config: Training configuration supplying alpha and temperature.

    Returns:
        A tuple of:
        - total_loss: Blended scalar tensor ``alpha * kl + (1 - alpha) * ce``.
        - metrics: Dict with keys ``kl_loss``, ``ce_loss``, ``total_loss``
          (Python floats, suitable for MLflow logging).
    """
    import torch.nn.functional as functional  # noqa: PLC0415

    alpha = config.alpha
    temp = config.temperature

    # Slice to answer span only
    s = student_logits[:, answer_start:, :]
    t = teacher_logits[:, answer_start:, :]

    vocab = s.shape[-1]

    # KL divergence: temperature-scaled, batchmean reduction
    kl = functional.kl_div(
        functional.log_softmax(s / temp, dim=-1),
        functional.softmax(t / temp, dim=-1),
        reduction="batchmean",
    )

    # CE: student predictions vs teacher hard labels
    ce = functional.cross_entropy(s.reshape(-1, vocab), t.argmax(-1).reshape(-1))

    # Blended loss
    total = alpha * kl + (1 - alpha) * ce

    return total, {
        "kl_loss": kl.item(),
        "ce_loss": ce.item(),
        "total_loss": total.item(),
    }


def train_d2l_qwen3(config: D2LTrainConfig) -> dict[str, Any]:
    """Run KL-divergence context distillation training.

    Args:
        config: Training configuration.

    Returns:
        Dictionary with training results (final loss, steps completed, etc.).

    Raises:
        NotImplementedError: Full implementation in Plan 02.
    """
    raise NotImplementedError("Implemented in Plan 02")


if __name__ == "__main__":
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="D2L distillation training")
    parser.add_argument("--base-model-name", default="Qwen/Qwen3-Coder-Next")
    parser.add_argument("--sakana-checkpoint-path", required=True)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--full-checkpoint-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--experiment-name", default="d2l-qwen3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dataset", dest="dataset_path", default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    cfg = D2LTrainConfig(
        base_model_name=args.base_model_name,
        sakana_checkpoint_path=args.sakana_checkpoint_path,
        num_steps=args.num_steps,
        lr=args.lr,
        alpha=args.alpha,
        temperature=args.temperature,
        checkpoint_every=args.checkpoint_every,
        full_checkpoint_every=args.full_checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        dataset_path=args.dataset_path,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        lora_r=args.lora_r,
        max_length=args.max_length,
    )

    result = train_d2l_qwen3(cfg)
    logger.info("Training complete: %s", result)
