r"""CLI for building a reconstruction dataset.

Usage:
    uv run python -m model_training.reconstruction.cli \
        --database-url sqlite:///$HOME/.rune/rune.db \
        --out-dir data/recon_v1 \
        --warm-start deltacoder \
        --base-model qwen3.5-9b \
        --emb-model sentence-transformers/all-mpnet-base-v2

Mirrors ``trainer_cli.py`` conventions: ``--dry-run`` resolves args to JSON
and exits without importing torch; heavy imports are deferred.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WARM_START_ALIASES: dict[str, str | None] = {
    "deltacoder": "danielcherubini/Qwen3.5-DeltaCoder-9B",
    "off": None,
    "none": None,
    "": None,
}

_BASE_MODEL_ALIASES: dict[str, str] = {
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3-coder-next": "Qwen/Qwen3-Coder-Next",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recon_dataset_cli",
        description="Build a T2L reconstruction dataset from an AdapterRegistry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--warm-start",
        default="off",
        help=(
            "Warm-start adapter: 'deltacoder' alias, 'off'/'none' for none,"
            " or explicit id."
        ),
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model alias or full HF repo id (overrides adapter_config.json).",
    )
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--min-fitness", type=float, default=None)
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source whitelist, e.g. 'distillation,evolution'.",
    )
    parser.add_argument(
        "--emb-model",
        default="none",
        help="HF repo id, 'default' for the built-in default, or 'none' for one-hot.",
    )
    parser.add_argument(
        "--no-zscore",
        dest="compute_zscore",
        action="store_false",
        default=True,
        help="Skip z-score stats computation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_warm_start(raw: str | None) -> str | None:
    """Resolve a warm-start alias or pass through an explicit adapter id.

    Args:
        raw: Raw CLI value, e.g. ``"deltacoder"``, ``"off"``, or an HF repo id.

    Returns:
        Resolved HF repo id, or ``None`` when no warm-start should be used.
    """
    if raw is None:
        return None
    key = raw.strip().lower()
    if key in _WARM_START_ALIASES:
        return _WARM_START_ALIASES[key]
    return raw


def _resolve_base_model(raw: str) -> str:
    """Resolve a base-model alias or pass through a full HF repo id.

    Args:
        raw: Raw CLI value, e.g. ``"qwen3.5-9b"`` or ``"Qwen/Qwen3.5-9B"``.

    Returns:
        Resolved HF repo id.
    """
    key = raw.strip().lower()
    if key in _BASE_MODEL_ALIASES:
        return _BASE_MODEL_ALIASES[key]
    return raw


def _resolve_sources(raw: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated sources string into a tuple.

    Args:
        raw: Comma-separated source names, or ``None``.

    Returns:
        Tuple of source names, or ``None`` when unfiltered.
    """
    if raw is None:
        return None
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _resolve_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Translate parsed CLI args into ``build_reconstruction_dataset`` kwargs.

    All imports here must remain torch-free to keep ``--dry-run`` lightweight.

    Args:
        args: Parsed :class:`argparse.Namespace`.

    Returns:
        Keyword-argument dict suitable for ``build_reconstruction_dataset``.
    """
    emb_choice = (args.emb_model or "none").strip().lower()
    if emb_choice in {"", "none"}:
        emb_model_name: str | None = None
    elif emb_choice == "default":
        from model_training.reconstruction.task_embeddings import (  # noqa: PLC0415
            DEFAULT_EMBEDDING_MODEL,
        )

        emb_model_name = DEFAULT_EMBEDDING_MODEL
    else:
        emb_model_name = args.emb_model
    return {
        "database_url": args.database_url,
        "out_dir": args.out_dir,
        "warm_start_adapter": _resolve_warm_start(args.warm_start),
        "base_model_id_override": _resolve_base_model(args.base_model),
        "task_type": args.task_type,
        "min_fitness": args.min_fitness,
        "sources": _resolve_sources(args.sources),
        "emb_model_name": emb_model_name,
        "compute_zscore": bool(args.compute_zscore),
    }


def _run(kwargs: dict[str, Any]) -> None:
    """Execute the dataset build with deferred heavy imports.

    All torch / sqlalchemy / sentence-transformers imports are confined here
    so that ``--dry-run`` exits before this function is ever called.

    Args:
        kwargs: Resolved kwargs dict from :func:`_resolve_kwargs`.
    """
    from adapter_registry.registry import AdapterRegistry  # noqa: PLC0415
    from sqlalchemy import create_engine  # noqa: PLC0415

    from model_training.reconstruction.builder import (  # noqa: PLC0415
        build_reconstruction_dataset,
    )
    from model_training.reconstruction.task_embeddings import (  # noqa: PLC0415
        DEFAULT_EMBEDDING_DIM,
        load_default_encoder,
    )

    engine = create_engine(kwargs["database_url"])
    registry = AdapterRegistry(engine=engine)

    emb_model_name = kwargs["emb_model_name"]
    emb_model: Any | None = None
    emb_dim = DEFAULT_EMBEDDING_DIM if emb_model_name else None
    if emb_model_name is not None:
        emb_model = load_default_encoder(emb_model_name)

    def _describe(rec: Any) -> str:
        # Default: prepend task_type to adapter id as a sanity-check description.
        # A real deployment should supply a richer callback by wrapping this CLI.
        return f"task_type={rec.task_type}; adapter_id={rec.id}"

    build_reconstruction_dataset(
        registry=registry,
        out_dir=Path(kwargs["out_dir"]),
        task_description_fn=_describe,
        warm_start_adapter=kwargs["warm_start_adapter"],
        base_model_id_override=kwargs["base_model_id_override"],
        emb_model=emb_model,
        compute_zscore=kwargs["compute_zscore"],
        task_type=kwargs["task_type"],
        min_fitness=kwargs["min_fitness"],
        sources=kwargs["sources"],
        emb_model_name=emb_model_name,
        emb_model_dim=emb_dim,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the reconstruction dataset CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    kwargs = _resolve_kwargs(args)
    if args.dry_run:
        print(json.dumps(kwargs, indent=2, sort_keys=True))
        return 0
    _run(kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
