r"""Command-line entrypoint for encoder pretraining.

Accepts all flags that map 1:1 to EncoderTrainingConfig fields and exposes
a ``--dry-run`` mode that resolves arguments and prints them as JSON without
loading any GPU libraries, to support CI validation.

All heavy imports (torch, transformers, sentence_transformers) are deferred
to the call site inside ``run_training`` — this CLI itself is CPU-safe.

Usage::

    uv run python -m model_training.encoder_pretrain.cli \
        --augmented-dir data/pairs_augmented \
        --output-dir data/encoder_checkpoint \
        --epochs 5 \
        --batch-size 64 \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for encoder pretraining CLI.

    Returns:
        Configured ArgumentParser with all EncoderTrainingConfig fields.
    """
    parser = argparse.ArgumentParser(
        prog="encoder_pretrain_cli",
        description="InfoNCE encoder pretraining for trajectory-aware embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required paths ---
    parser.add_argument(
        "--augmented-dir",
        dest="augmented_dir",
        required=True,
        metavar="PATH",
        help="Directory of augmented JSONL files (output of augment_corpus).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        metavar="PATH",
        help="Destination directory for the HF-loadable encoder checkpoint.",
    )

    # --- Encoder ---
    parser.add_argument(
        "--base-encoder",
        dest="base_encoder",
        default="sentence-transformers/all-mpnet-base-v2",
        help="HF model id for the starting encoder.",
    )

    # --- Hyperparameters ---
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="InfoNCE temperature tau.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Training batch size (in-batch negatives = batch_size - 1).",
    )
    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max-length",
        dest="max_length",
        type=int,
        default=512,
        help="Tokenizer max sequence length (tokens).",
    )
    parser.add_argument(
        "--warmup-steps",
        dest="warmup_steps",
        type=int,
        default=100,
        help="Linear warmup steps.",
    )
    parser.add_argument(
        "--test-fraction",
        dest="test_fraction",
        type=float,
        default=0.2,
        help="Fraction of task_ids reserved for retrieval eval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )

    # --- MLflow ---
    parser.add_argument(
        "--experiment-name",
        dest="mlflow_experiment",
        default="rune-encoder-pretrain",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-uri",
        dest="mlflow_tracking_uri",
        default=None,
        help="Override MLFLOW_TRACKING_URI; defaults to ./mlruns.",
    )

    # --- Dry run ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and print JSON; do not load models or train.",
    )

    return parser


def _resolve_config(args: argparse.Namespace) -> dict[str, object]:
    """Build the resolved config dict from parsed args.

    Exposed separately so tests can exercise config resolution without
    running the trainer.

    Args:
        args: Parsed argparse.Namespace.

    Returns:
        Dict of config fields (JSON-serializable; paths as strings).
    """
    return {
        "augmented_dir": args.augmented_dir,
        "output_dir": args.output_dir,
        "base_encoder": args.base_encoder,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "warmup_steps": args.warmup_steps,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "mlflow_experiment": args.mlflow_experiment,
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse argv, dispatch to run_training (or dry-run and exit).

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_dict = _resolve_config(args)

    if args.dry_run:
        print(json.dumps(config_dict, indent=2, sort_keys=True))
        return 0

    # Deferred to keep --dry-run CPU-only (INFRA-05).
    from pathlib import Path  # noqa: PLC0415

    from model_training.encoder_pretrain.train_encoder import (  # noqa: PLC0415
        EncoderTrainingConfig,
        run_training,
    )

    config = EncoderTrainingConfig(
        augmented_dir=Path(args.augmented_dir),
        output_dir=Path(args.output_dir),
        base_encoder=args.base_encoder,
        temperature=args.temperature,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        test_fraction=args.test_fraction,
        seed=args.seed,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
    run_training(config)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess
    sys.exit(main())
