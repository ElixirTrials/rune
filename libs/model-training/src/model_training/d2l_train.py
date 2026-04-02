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
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

__all__ = ["train_d2l_qwen3", "D2LTrainConfig"]


def _require_probe_cache(model_name: str) -> None:
    """Raise RuntimeError if probe cache is absent for *model_name*.

    Called before build_qwen3_hypernet_config so training never proceeds with
    placeholder feature_sizes that produce incorrect LoRA dimensions.

    Defers the import of load_probe_cache so GPU-free test environments are
    not forced to import the full probe module at module load time.

    Args:
        model_name: Canonical model identifier for cache lookup and error message.

    Raises:
        RuntimeError: If no probe cache is found for model_name.
    """
    from model_training.d2l_probe import load_probe_cache  # noqa: PLC0415

    if load_probe_cache(model_name) is not None:
        return
    msg = (
        f"Probe cache not found for '{model_name}' — "
        "run probe_model() and save_probe_cache() before training. "
        "Training with placeholder feature_sizes produces incorrect LoRA "
        "dimensions."
    )
    raise RuntimeError(msg)


class D2LTrainConfig(BaseModel):
    """Pydantic model for D2L training hyperparameters.

    Enables validation, JSON serialization (for checkpoint storage), and
    `.model_dump()` for MLflow experiment logging.

    Attributes:
        model_config_name: Registry lookup key (e.g. "qwen3.5-9b"). Used to
            resolve base_model_name from the model registry when not set.
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

    model_config_name: str = Field(default="qwen3.5-9b")
    base_model_name: str = Field(default="")
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

    def model_post_init(self, __context: Any) -> None:
        """Resolve base_model_name from registry when not explicitly set."""
        if not self.base_model_name:
            from model_training.model_configs import ModelRegistry

            mc = ModelRegistry.default().get(self.model_config_name)
            object.__setattr__(self, "base_model_name", mc.model_id)

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

    Accounts for the causal LM shift: logits at position ``k`` predict token
    ``k+1``, so we slice starting at ``answer_start - 1`` (clamped to 0).
    Returns a zero-loss tensor when the answer span is empty (e.g. truncation
    pushed ``answer_start`` beyond the sequence length).

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
    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    alpha = config.alpha
    temp = config.temperature

    # Guard against empty answer span (truncation pushed answer beyond seq_len).
    # answer_start >= seq_len means no answer tokens exist in the sequence.
    seq_len = student_logits.shape[1]
    if answer_start >= seq_len:
        logger.warning(
            "Answer span empty: answer_start=%d >= seq_len=%d. Returning zero loss.",
            answer_start,
            seq_len,
        )
        zero = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        return zero, {"kl_loss": 0.0, "ce_loss": 0.0, "total_loss": 0.0}

    # Causal LM shift: logits at position k predict token k+1, so the logit
    # that predicts the first answer token is at position answer_start - 1.
    logit_start = max(0, answer_start - 1)

    # Slice to answer span only (shift-aware)
    s = student_logits[:, logit_start:, :]
    t = teacher_logits[:, logit_start:, :]

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


def _training_step(
    record: dict[str, Any],
    base_model: Any,
    tokenizer: Any,
    hypernet: Any,
    hc: Any,
    config: D2LTrainConfig,
) -> tuple[Any, dict[str, float]]:
    """Execute a single training step with two-pass teacher/student separation.

    Pass 1 extracts activations from activation_text only (no answer tokens).
    Pass 2 runs teacher (no_grad) and student (with LoRA) on teacher_text.

    Args:
        record: Data record with 'activation_text' and 'teacher_text' fields.
        base_model: Base LM in eval mode.
        tokenizer: Tokenizer matching base_model.
        hypernet: HyperLoRA in train mode.
        hc: HypernetConfig with layer_indices and lora_config.
        config: Training configuration.

    Returns:
        Tuple of (loss_tensor, metrics_dict).
    """
    import torch  # noqa: PLC0415

    from model_training.d2l_lora import apply_functional_lora  # noqa: PLC0415
    from model_training.d2l_probe import extract_activations_with_model  # noqa: PLC0415

    # Pass 1: extract activations from activation_text (context only, no answer tokens)
    features, attn_mask = extract_activations_with_model(
        text=record["activation_text"],
        model=base_model,
        tokenizer=tokenizer,
        layer_indices=list(hc.layer_indices),
        max_length=config.max_length,
    )

    # Hypernetwork forward — OUTSIDE torch.no_grad to preserve autograd graph
    lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)

    # Compute answer_start: token offset where answer begins in teacher_text
    answer_start = len(
        tokenizer(
            record["activation_text"],
            truncation=True,
            max_length=config.max_length,
        )["input_ids"]
    )

    # Pass 2 teacher: run base model under no_grad to get teacher logits
    teacher_inputs = tokenizer(
        record["teacher_text"],
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
    )
    try:
        device = next(base_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

    with torch.no_grad():
        teacher_out = base_model(**teacher_inputs, output_hidden_states=False)
    teacher_logits = teacher_out.logits

    # Pass 2 student: run base model with functional LoRA patches
    with apply_functional_lora(base_model, lora_dict, hc):
        student_out = base_model(**teacher_inputs, output_hidden_states=False)
    student_logits = student_out.logits

    return _compute_kl_ce_loss(student_logits, teacher_logits, answer_start, config)


def _save_checkpoint(
    step: int,
    hypernet: Any,
    optimizer: Any,
    scheduler: Any,
    config: D2LTrainConfig,
    hc: Any,
    best_loss: float,
    full: bool = False,
) -> Path:
    """Save a tiered checkpoint to disk.

    Lightweight checkpoint: hypernet weights, config, step, layer indices, best_loss.
    Full checkpoint: additionally includes optimizer state, scheduler state, RNG state.

    Args:
        step: Current training step number.
        hypernet: HyperLoRA model.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler.
        config: Training configuration.
        hc: HypernetConfig with attention_layer_indices.
        best_loss: Best loss seen so far.
        full: If True, save full checkpoint with optimizer/scheduler/RNG state.

    Returns:
        Path to the saved checkpoint file.
    """
    import torch  # noqa: PLC0415

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "hypernet_state_dict": hypernet.state_dict(),
        "config_json": config.model_dump(),
        "step": step,
        "attention_layer_indices": list(hc.layer_indices),
        "best_loss": best_loss,
    }

    if full:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict()
        payload["rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state()

    ckpt_path = ckpt_dir / f"ckpt-{step}.pt"
    torch.save(payload, ckpt_path)
    logger.info("Checkpoint saved: %s (full=%s)", ckpt_path, full)
    return ckpt_path


def _setup_mlflow(config: D2LTrainConfig) -> None:
    """Configure MLflow tracking URI and experiment name.

    Uses MLFLOW_TRACKING_URI environment variable if set, otherwise defaults
    to local './mlruns' directory.

    Args:
        config: Training configuration supplying experiment_name.
    """
    import mlflow  # noqa: PLC0415

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    logger.info(
        "MLflow tracking URI: %s, experiment: %s",
        tracking_uri,
        config.experiment_name,
    )


def _dry_run_validate_shapes(config: D2LTrainConfig) -> dict[str, Any]:
    """Validate tensor shapes with a single forward pass, no optimizer step.

    Loads the base model and hypernet, generates one needle record, runs one
    training step without backward/optimizer, asserts all shapes are correct,
    and returns a shape summary.

    Args:
        config: Training configuration.

    Returns:
        Dict with shape validation results: features, lora_dict, student/teacher logits.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from model_training.d2l_config import build_hypernet_config  # noqa: PLC0415
    from model_training.d2l_data import generate_needle_dataset  # noqa: PLC0415
    from model_training.d2l_probe import (  # noqa: PLC0415
        extract_activations_with_model,
    )
    from model_training.sakana_d2l import (  # noqa: PLC0415
        get_aggregator_config,
        transfer_aggregator_weights,
    )

    # Guard: dry-run also requires probe cache to produce correct LoRA dimensions.
    _require_probe_cache(config.model_config_name)

    logger.info("Dry-run: loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        output_hidden_states=True,
    ).eval()

    logger.info("Dry-run: building hypernet config and transferring weights...")
    hc = build_hypernet_config(
        config.model_config_name,
        aggregator_config=get_aggregator_config(config.sakana_checkpoint_path),
        lora_r=config.lora_r,
    )

    from ctx_to_lora.modeling.hypernet import HyperLoRA  # noqa: PLC0415

    hypernet = HyperLoRA(hc).to(torch.float32)
    hypernet = transfer_aggregator_weights(hypernet, config.sakana_checkpoint_path)
    hypernet.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)
    hypernet = hypernet.to(device)

    records = generate_needle_dataset(n=1)
    record = records[0]

    logger.info("Dry-run: running single forward pass for shape validation...")
    features, attn_mask = extract_activations_with_model(
        text=record["activation_text"],
        model=base_model,
        tokenizer=tokenizer,
        layer_indices=list(hc.layer_indices),
        max_length=config.max_length,
    )

    with torch.no_grad():
        lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)

    tokenizer_out = tokenizer(
        record["teacher_text"],
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
    )
    teacher_inputs = {k: v.to(device) for k, v in tokenizer_out.items()}

    with torch.no_grad():
        teacher_out = base_model(**teacher_inputs, output_hidden_states=False)
    teacher_logits = teacher_out.logits

    from model_training.d2l_lora import apply_functional_lora  # noqa: PLC0415

    with apply_functional_lora(base_model, lora_dict, hc):
        with torch.no_grad():
            student_out = base_model(**teacher_inputs, output_hidden_states=False)
    student_logits = student_out.logits

    # Assert shapes
    assert features.ndim == 4, f"features must be 4D, got {features.ndim}D"  # noqa: S101
    assert student_logits.shape == teacher_logits.shape, (  # noqa: S101
        f"student/teacher logit shapes must match: "
        f"{student_logits.shape} vs {teacher_logits.shape}"
    )

    shape_summary = {
        "features_shape": list(features.shape),
        "attn_mask_shape": list(attn_mask.shape),
        "lora_modules": list(lora_dict.keys()),
        "student_logits_shape": list(student_logits.shape),
        "teacher_logits_shape": list(teacher_logits.shape),
        "status": "OK",
    }

    for mod_name, ab in lora_dict.items():
        shape_summary[f"lora_{mod_name}_A_shape"] = list(ab["A"].shape)
        shape_summary[f"lora_{mod_name}_B_shape"] = list(ab["B"].shape)

    logger.info("Dry-run shape validation passed: %s", shape_summary)
    return shape_summary


def train_d2l_qwen3(config: D2LTrainConfig) -> dict[str, Any]:  # noqa: C901
    """Run KL-divergence context distillation training.

    Three execution modes controlled by config flags:
    - dry_run=True: Validate shapes with single forward pass, no optimizer step.
    - smoke_test=True: Run min(num_steps, 5) steps, assert finite decreasing loss.
    - default: Full training from dataset with checkpointing and MLflow tracking.

    Args:
        config: Training configuration.

    Returns:
        Dictionary with training results:
            - final_loss: Loss at the last step.
            - best_loss: Lowest loss seen during training.
            - num_steps_completed: Number of training steps completed.
            - checkpoint_dir: Path to checkpoint directory.
            - shape_summary (dry_run only): Tensor shape validation results.
    """
    import mlflow  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from ctx_to_lora.modeling.hypernet import HyperLoRA  # noqa: PLC0415
    from torch.nn.utils import clip_grad_norm_  # noqa: PLC0415
    from torch.optim import AdamW  # noqa: PLC0415
    from torch.optim.lr_scheduler import (  # noqa: PLC0415
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from model_training.d2l_config import build_hypernet_config  # noqa: PLC0415
    from model_training.d2l_data import (  # noqa: PLC0415
        generate_needle_dataset,
        load_jsonl,
        split_by_task_id,
    )
    from model_training.sakana_d2l import (  # noqa: PLC0415
        get_aggregator_config,
        transfer_aggregator_weights,
    )

    # Mode dispatch: dry_run exits after shape validation
    if config.dry_run:
        shape_summary = _dry_run_validate_shapes(config)
        return {"shape_summary": shape_summary, "status": "dry_run_complete"}

    # Smoke test caps steps at 5
    if config.smoke_test:
        config = config.model_copy(update={"num_steps": min(config.num_steps, 5)})

    # Guard: require probe cache when not in smoke_test mode.
    # smoke_test uses generate_needle_dataset and may not have a real probe cache.
    if not config.smoke_test:
        _require_probe_cache(config.model_config_name)

    num_steps = config.num_steps
    warmup_steps = config.warmup_steps

    # Load tokenizer and base model
    logger.info("Loading tokenizer: %s", config.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    logger.info("Loading base model: %s", config.base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        output_hidden_states=True,
    ).eval()

    # Build hypernet config with aggregator config from checkpoint
    logger.info(
        "Building hypernet config from checkpoint: %s",
        config.sakana_checkpoint_path,
    )
    hc = build_hypernet_config(
        config.model_config_name,
        aggregator_config=get_aggregator_config(config.sakana_checkpoint_path),
        lora_r=config.lora_r,
    )

    # Create hypernet and transfer aggregator weights (freezes aggregator)
    hypernet = HyperLoRA(hc).to(torch.float32)
    hypernet = transfer_aggregator_weights(hypernet, config.sakana_checkpoint_path)
    hypernet.train()

    # Device selection: cuda > mps > cpu
    from shared.hardware import get_best_device  # noqa: PLC0415

    device = torch.device(get_best_device())
    logger.info("Using device: %s", device)
    base_model = base_model.to(device)
    hypernet = hypernet.to(device)

    # Optimizer: only trainable params (head + projections, not frozen aggregator)
    trainable_params = [p for p in hypernet.parameters() if p.requires_grad]
    logger.info("Trainable params: %d", sum(p.numel() for p in trainable_params))
    optimizer = AdamW(trainable_params, lr=config.lr)

    # Scheduler: linear warmup → cosine annealing
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_steps - warmup_steps),
                eta_min=1e-6,
            ),
        ],
        milestones=[warmup_steps],
    )

    # Data loading
    if config.smoke_test or config.dataset_path is None:
        records = generate_needle_dataset(n=20)
        logger.info("Using needle dataset (%d records)", len(records))
    else:
        all_records = load_jsonl(config.dataset_path)
        records, _ = split_by_task_id(all_records)
        logger.info(
            "Loaded %d training records from %s",
            len(records),
            config.dataset_path,
        )

    if not records:
        raise ValueError("No training records loaded; cannot train on empty dataset.")

    # MLflow setup and training loop
    _setup_mlflow(config)

    best_loss = float("inf")
    final_loss = float("inf")
    step_losses: list[float] = []

    with mlflow.start_run(run_name=f"{config.experiment_name}-step{num_steps}"):
        mlflow.log_params(config.model_dump())

        for step in range(1, num_steps + 1):
            record = records[(step - 1) % len(records)]

            loss, metrics = _training_step(
                record=record,
                base_model=base_model,
                tokenizer=tokenizer,
                hypernet=hypernet,
                hc=hc,
                config=config,
            )

            loss.backward()
            clip_grad_norm_(trainable_params, config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step_loss = metrics["total_loss"]
            step_losses.append(step_loss)
            final_loss = step_loss
            if step_loss < best_loss:
                best_loss = step_loss

            mlflow.log_metrics(metrics, step=step)
            logger.info(
                "Step %d/%d — loss=%.4f (kl=%.4f, ce=%.4f)",
                step,
                num_steps,
                metrics["total_loss"],
                metrics["kl_loss"],
                metrics["ce_loss"],
            )

            # Tiered checkpointing
            if step % config.full_checkpoint_every == 0:
                ckpt_path = _save_checkpoint(
                    step=step,
                    hypernet=hypernet,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    hc=hc,
                    best_loss=best_loss,
                    full=True,
                )
                mlflow.log_artifact(str(ckpt_path))
            elif step % config.checkpoint_every == 0:
                ckpt_path = _save_checkpoint(
                    step=step,
                    hypernet=hypernet,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    hc=hc,
                    best_loss=best_loss,
                    full=False,
                )
                mlflow.log_artifact(str(ckpt_path))

    # Smoke test assertions
    if config.smoke_test:
        for i, sl in enumerate(step_losses):
            assert torch.isfinite(torch.tensor(sl)), (  # noqa: S101
                f"Smoke test: loss at step {i + 1} is not finite: {sl}"
            )
        assert step_losses[-1] < step_losses[0], (  # noqa: S101
            f"Smoke test: final loss {step_losses[-1]:.4f} not less than "
            f"initial loss {step_losses[0]:.4f}"
        )
        assert any(p.grad is not None for p in trainable_params), (  # noqa: S101
            "Smoke test: no trainable param has non-None gradient after training"
        )

    return {
        "final_loss": final_loss,
        "best_loss": best_loss,
        "num_steps_completed": num_steps,
        "checkpoint_dir": config.checkpoint_dir,
    }


if __name__ == "__main__":
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="D2L distillation training")
    parser.add_argument("--model-config-name", default="qwen3.5-9b")
    parser.add_argument("--base-model-name", default="")
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
        model_config_name=args.model_config_name,
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
