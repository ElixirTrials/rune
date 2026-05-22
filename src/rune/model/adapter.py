"""LoRA adapter persistence and hot-swap utilities."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AdapterResult:
    """Adapter weights returned by the hypernetwork.

    Attributes:
        adapter_id: Unique identifier for this adapter.
        state_dict: PEFT-compatible state dict of LoRA weights.
    """

    adapter_id: str
    state_dict: dict[str, Any]


async def persist_adapter(
    state_dict: dict[str, Any],
    adapter_id: str,
    output_dir: Path,
) -> Path:
    """Save adapter weights to a safetensors file asynchronously.

    Args:
        state_dict: PEFT state dict to serialise.
        adapter_id: Used as the file stem.
        output_dir: Directory to write the file into.

    Returns:
        Path to the written safetensors file.
    """
    path = output_dir / f"{adapter_id}.safetensors"

    def _write() -> None:
        from safetensors.torch import save_file  # noqa: PLC0415

        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(path))

    await asyncio.to_thread(_write)
    return path


def hotswap_adapter(model: Any, state_dict: dict[str, Any]) -> None:
    """Load new LoRA weights into a PEFT model in-place.

    Args:
        model: A PEFT-wrapped model.
        state_dict: New adapter weights to apply.
    """
    from peft import set_peft_model_state_dict  # noqa: PLC0415

    set_peft_model_state_dict(model, state_dict)
