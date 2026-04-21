r"""Command-line entrypoint for QLoRA fine-tuning.

Invoked by ``scripts/train.sh`` so the shell wrapper stays thin. Accepts
all flags that map 1:1 to ``train_and_register`` kwargs and exposes a
``--dry-run`` mode that resolves arguments and prints them as JSON
without loading any GPU libraries, to support CI validation.

All heavy imports (torch, transformers, peft, trl) are deferred to the
call site inside ``train_and_register`` — this CLI itself is CPU-safe.

Usage:
    uv run python -m model_training.trainer_cli \
        --dataset data/pairs/repo.jsonl \
        --adapter-id my-adapter \
        --model qwen3.5-9b \
        --warm-start deltacoder \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Warm-start aliases recognized by --warm-start. "deltacoder" resolves at
# runtime to the qwen3.5-9b model-registry entry; "off" disables warm-start.
_WARM_START_ALIASES: dict[str, str | None] = {
    "deltacoder": "danielcherubini/Qwen3.5-DeltaCoder-9B",
    "off": None,
    "none": None,
    "": None,
}


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for ``train.sh``.

    Flags mirror ``train_and_register`` kwargs. ``--dataset`` and
    ``--session-id`` are mutually exclusive and exactly one must be set.
    """
    parser = argparse.ArgumentParser(
        prog="train_cli",
        description="QLoRA fine-tuning entrypoint for Rune (invoked by train.sh).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data source (mutually exclusive, one required) ---
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset",
        metavar="PATH",
        help="JSONL of mined pair records from scripts/mine_github.py --batch.",
    )
    src.add_argument(
        "--session-id",
        dest="session_id",
        metavar="ID",
        help="Self-generated trajectory session ID (RUNE_TRAJECTORY_DIR lookup).",
    )

    # --- Model + warm-start ---
    parser.add_argument(
        "--model",
        dest="model_config_name",
        default="qwen3.5-9b",
        help="Model registry lookup key (qwen3.5-9b, qwen3-coder-next, ...).",
    )
    parser.add_argument(
        "--warm-start",
        dest="warm_start",
        default="deltacoder",
        help=(
            "Warm-start adapter. Aliases: 'deltacoder' (qwen3.5-9b default), "
            "'off'/'none' to skip warm-start, or an explicit HF repo / local "
            "path to an adapter dir."
        ),
    )

    # --- Adapter identity / output ---
    parser.add_argument(
        "--adapter-id",
        dest="adapter_id",
        required=True,
        help="Unique identifier for the resulting adapter.",
    )

    # --- Training hyperparameters ---
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--lr", dest="learning_rate", type=float, default=2e-4
    )
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument(
        "--grad-accum",
        dest="gradient_accumulation_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler_type",
        choices=["constant", "cosine", "linear"],
        default=None,
    )

    # --- Trajectory encoding ---
    parser.add_argument(
        "--encoding-mode",
        dest="encoding_mode",
        choices=["multi_turn", "single_turn"],
        default="multi_turn",
    )
    parser.add_argument(
        "--diff-aware-loss",
        dest="diff_aware_loss",
        action="store_true",
        help="Scale per-token loss by diff-vs-context weighting.",
    )
    parser.add_argument(
        "--diff-changed-weight",
        dest="diff_changed_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--diff-unchanged-weight",
        dest="diff_unchanged_weight",
        type=float,
        default=0.3,
    )

    # --- MLflow ---
    parser.add_argument(
        "--experiment-name",
        dest="mlflow_experiment",
        default="rune-qlora",
    )
    parser.add_argument(
        "--mlflow-uri",
        dest="mlflow_tracking_uri",
        default=None,
        help="Overrides MLFLOW_TRACKING_URI env; defaults to ./mlruns.",
    )

    # --- Task metadata + registry ---
    parser.add_argument("--task-type", dest="task_type", default="code-gen")
    parser.add_argument(
        "--database-url",
        dest="database_url",
        default=None,
        help="SQLAlchemy URL; defaults to RUNE_DATABASE_URL or sqlite:~/.rune/rune.db.",
    )

    # --- Dry run ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve args and print JSON; do not load models or train.",
    )

    return parser


def _resolve_warm_start(raw: str | None) -> str | None:
    """Map --warm-start alias/string to a concrete adapter id/path or None."""
    if raw is None:
        return None
    key = raw.strip().lower()
    if key in _WARM_START_ALIASES:
        return _WARM_START_ALIASES[key]
    return raw  # treat as explicit HF repo id or local path


def _resolve_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build the kwargs dict passed to train_and_register.

    Exposed as a separate function so tests can exercise resolution logic
    without running the trainer itself.
    """
    return {
        "session_id": args.session_id,
        "adapter_id": args.adapter_id,
        "dataset_path": args.dataset,
        "encoding_mode": args.encoding_mode,
        "model_config_name": args.model_config_name,
        "warm_start_adapter_id": _resolve_warm_start(args.warm_start),
        "task_type": args.task_type,
        "rank": args.rank,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "mlflow_experiment": args.mlflow_experiment,
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
        "database_url": args.database_url,
        "diff_aware_loss": args.diff_aware_loss,
        "diff_changed_weight": args.diff_changed_weight,
        "diff_unchanged_weight": args.diff_unchanged_weight,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse argv, dispatch to train_and_register (or dry-run and exit)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    kwargs = _resolve_kwargs(args)

    if args.dry_run:
        print(json.dumps(kwargs, indent=2, sort_keys=True))
        return 0

    # Deferred to keep --dry-run CPU-only.
    from model_training.trainer import train_and_register  # noqa: PLC0415

    train_and_register(**kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess
    sys.exit(main())
