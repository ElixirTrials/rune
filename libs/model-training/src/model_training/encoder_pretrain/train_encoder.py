"""Encoder pretraining loop: InfoNCE fine-tuning of all-mpnet-base-v2.

Single-GPU only. Multi-GPU support (DistributedDataParallel / Accelerate
multi-process) is intentionally out of scope for this plan; add a
``--multi-gpu`` flag in a follow-on plan.

All GPU-dependent imports (torch, sentence_transformers, transformers,
accelerate) are deferred inside ``run_training`` (INFRA-05). The module
is CPU-safe to import.

MLflow integration follows the pattern in training_common.mlflow_run.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from model_training.training_common import mlflow_run, setup_mlflow

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EncoderTrainingConfig:
    """Resolved training configuration for the encoder pretraining loop.

    Args:
        augmented_dir: Directory of augmented JSONL files (from augment_corpus).
        output_dir: Where to save the final HF-loadable checkpoint.
        base_encoder: HF model id for the starting encoder.
        temperature: InfoNCE temperature tau.
        batch_size: Training batch size (in-batch negatives count = batch_size - 1).
        learning_rate: AdamW learning rate.
        epochs: Number of training epochs.
        max_length: Tokenizer max sequence length (tokens).
        warmup_steps: Linear warmup steps at the start of training.
        test_fraction: Fraction of task_ids reserved for retrieval eval.
        seed: Random seed for train/test split.
        mlflow_experiment: MLflow experiment name.
        mlflow_tracking_uri: Override for MLFLOW_TRACKING_URI env var.
    """

    augmented_dir: Path
    output_dir: Path
    base_encoder: str = "sentence-transformers/all-mpnet-base-v2"
    temperature: float = 0.07
    batch_size: int = 64
    learning_rate: float = 2e-5
    epochs: int = 5
    max_length: int = 512
    warmup_steps: int = 100
    test_fraction: float = 0.2
    seed: int = 42
    mlflow_experiment: str = "rune-encoder-pretrain"
    mlflow_tracking_uri: str | None = None


def _mean_pool(
    token_embeddings: Any,
    attention_mask: Any,
) -> Any:
    """Mean-pool token embeddings weighted by attention mask.

    Args:
        token_embeddings: ``(B, seq_len, hidden_dim)`` from encoder last_hidden_state.
        attention_mask: ``(B, seq_len)`` binary mask; 1 = real token, 0 = padding.

    Returns:
        Mean-pooled embeddings ``(B, hidden_dim)``.
    """
    import torch  # noqa: PLC0415

    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def _encode_batch(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
) -> Any:
    """Run forward pass on a HuggingFace encoder and mean-pool.

    Args:
        model: HuggingFace AutoModel (encoder-only).
        input_ids: ``(B, seq_len)`` token ids.
        attention_mask: ``(B, seq_len)`` attention mask.

    Returns:
        Mean-pooled embeddings ``(B, hidden_dim)``.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _mean_pool(outputs.last_hidden_state, attention_mask)


def _save_sentence_transformer_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    max_length: int,
) -> None:
    """Save model + tokenizer in a SentenceTransformer-loadable directory layout.

    Writes:
    - HuggingFace model weights + config (via model.save_pretrained)
    - tokenizer files (via tokenizer.save_pretrained)
    - sentence_transformers_config.json
    - modules.json (required by SentenceTransformer for directory loading)
    - 1_Pooling/config.json (mean-pooling configuration)

    Args:
        model: Trained HuggingFace AutoModel.
        tokenizer: Matching HuggingFace tokenizer.
        output_dir: Destination directory (created if needed).
        max_length: Max sequence length to embed in config.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    st_config = {
        "max_seq_length": max_length,
        "do_lower_case": False,
    }
    (output_dir / "sentence_transformers_config.json").write_text(
        json.dumps(st_config, indent=2), encoding="utf-8"
    )

    modules = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]
    (output_dir / "modules.json").write_text(
        json.dumps(modules, indent=2), encoding="utf-8"
    )

    pooling_dir = output_dir / "1_Pooling"
    pooling_dir.mkdir(exist_ok=True)
    pooling_config = {
        "word_embedding_dimension": 768,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }
    (pooling_dir / "config.json").write_text(
        json.dumps(pooling_config, indent=2), encoding="utf-8"
    )


def run_training(config: EncoderTrainingConfig) -> Path:
    """Run the InfoNCE encoder pretraining loop.

    Steps:
        1. Load all augmented pairs from config.augmented_dir.
        2. Task-ID-level train/test split (via d2l_data.split_by_task_id).
        3. Build ContrastivePairDataset + DataLoader for the training split.
        4. Load base encoder (AutoModel) + tokenizer.
        5. Train for config.epochs with AdamW + cosine schedule + linear warmup.
        6. Log per-epoch train loss and retrieval eval metrics to MLflow.
        7. Save the final encoder in HF-loadable format to config.output_dir.

    The saved checkpoint is loadable via ``SentenceTransformer(config.output_dir)``.

    Single-GPU only; place model on ``cuda:0`` when available, else ``cpu``.

    Args:
        config: EncoderTrainingConfig with all resolved hyperparameters.

    Returns:
        Path to the saved checkpoint directory (== config.output_dir).
    """
    import torch  # noqa: PLC0415
    from torch.optim import AdamW  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModel,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    from model_training.d2l_data import load_jsonl, split_by_task_id  # noqa: PLC0415
    from model_training.encoder_pretrain.dataset import (  # noqa: PLC0415
        ContrastiveCollator,
        ContrastivePairDataset,
    )
    from model_training.encoder_pretrain.eval_encoder import (  # noqa: PLC0415
        run_retrieval_eval,
    )
    from model_training.encoder_pretrain.loss import infonce_loss  # noqa: PLC0415

    # ---- device ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("run_training: device=%s", device)

    # ---- load all augmented pairs ----
    all_rows: list[dict[str, Any]] = []
    for path in sorted(config.augmented_dir.glob("*.jsonl")):
        all_rows.extend(load_jsonl(path))
    logger.info("Loaded %d augmented rows from %s", len(all_rows), config.augmented_dir)

    if not all_rows:
        raise ValueError(
            f"No augmented pairs found in {config.augmented_dir}. "
            "Run augment_corpus first."
        )

    # ---- task-ID-level train/test split ----
    train_rows, test_rows = split_by_task_id(
        all_rows, test_fraction=config.test_fraction, seed=config.seed
    )
    logger.info(
        "Split: %d train rows, %d test rows (task-ID level)",
        len(train_rows),
        len(test_rows),
    )

    # ---- dataset + dataloader ----
    tokenizer = AutoTokenizer.from_pretrained(config.base_encoder)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=config.max_length)
    train_ds = ContrastivePairDataset(train_rows)
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_ds,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,  # ensure every batch has exactly batch_size pairs
    )

    # ---- model ----
    model = AutoModel.from_pretrained(config.base_encoder)
    model = model.to(device)
    model.train()

    # ---- optimizer + scheduler ----
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    # ---- MLflow setup ----
    mlflow_enabled = setup_mlflow(config.mlflow_experiment, config.mlflow_tracking_uri)
    mlflow_params = {
        "base_encoder": config.base_encoder,
        "temperature": config.temperature,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "max_length": config.max_length,
        "warmup_steps": config.warmup_steps,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
    }

    with mlflow_run(
        enabled=mlflow_enabled,
        run_name="encoder-pretrain",
        params=mlflow_params,
    ):
        # ---- training loop ----
        for epoch in range(1, config.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            model.train()
            for batch in train_loader:
                anchor_ids = batch["anchor_input_ids"].to(device)
                anchor_mask = batch["anchor_attention_mask"].to(device)
                pos_ids = batch["positive_input_ids"].to(device)
                pos_mask = batch["positive_attention_mask"].to(device)

                anchor_emb = _encode_batch(model, anchor_ids, anchor_mask)
                positive_emb = _encode_batch(model, pos_ids, pos_mask)

                loss = infonce_loss(
                    anchor_emb, positive_emb, temperature=config.temperature
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                "Epoch %d/%d: avg_train_loss=%.4f", epoch, config.epochs, avg_loss
            )

            # ---- per-epoch retrieval eval ----
            eval_metrics = run_retrieval_eval(
                model=model,
                tokenizer=tokenizer,
                test_rows=test_rows,
                max_length=config.max_length,
                batch_size=config.batch_size,
                device=str(device),
            )
            logger.info(
                "Epoch %d/%d eval: MRR@10=%.4f Recall@1=%.4f",
                epoch,
                config.epochs,
                eval_metrics["mrr_at_10"],
                eval_metrics["recall_at_1"],
            )

            # Log to MLflow
            if mlflow_enabled:
                try:
                    import mlflow  # noqa: PLC0415

                    mlflow.log_metrics(
                        {
                            "train_loss": avg_loss,
                            "mrr_at_10": eval_metrics["mrr_at_10"],
                            "recall_at_1": eval_metrics["recall_at_1"],
                        },
                        step=epoch,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("mlflow.log_metrics failed", exc_info=True)

        # ---- save checkpoint ----
        _save_sentence_transformer_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=config.output_dir,
            max_length=config.max_length,
        )
        logger.info("Checkpoint saved to %s", config.output_dir)

    return config.output_dir
