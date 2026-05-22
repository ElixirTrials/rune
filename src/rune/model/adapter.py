from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AdapterResult:
    adapter_id: str
    state_dict: dict[str, Any]


async def persist_adapter(
    state_dict: dict[str, Any],
    adapter_id: str,
    output_dir: Path,
) -> Path:
    path = output_dir / f"{adapter_id}.safetensors"

    def _write() -> None:
        from safetensors.torch import save_file  # noqa: PLC0415
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(path))

    await asyncio.to_thread(_write)
    return path


def hotswap_adapter(model: Any, state_dict: dict[str, Any]) -> None:
    from peft import set_peft_model_state_dict  # noqa: PLC0415
    set_peft_model_state_dict(model, state_dict)
