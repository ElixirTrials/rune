"""Checkpoint loading utilities for hypernetwork checkpoints.

Supports both local paths and S3 URIs via fsspec. Used by rune_runner.py
and sakana_d2l.py for checkpoint consumption.

IMPORTANT: All GPU imports (torch) are deferred inside function bodies
per INFRA-05 pattern — this module is importable in CPU-only CI.
"""

from __future__ import annotations

from typing import Any


def _open_checkpoint(path: str) -> Any:
    """Open a checkpoint from a local path or S3 URI.

    Uses fsspec for S3 URIs so checkpoints can be consumed directly
    without a local download step.

    Args:
        path: Local filesystem path or s3:// URI.

    Returns:
        Deserialized checkpoint dict.
    """
    import torch  # noqa: PLC0415

    # weights_only=False: checkpoints contain a HypernetConfig dataclass, not tensors
    if path.startswith("s3://"):
        import fsspec  # type: ignore[import-untyped]  # noqa: PLC0415

        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)
