"""LoRA adapter utilities: scaling, hot-swap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AdapterResult:
    """Adapter weights returned by the hypernetwork.

    Attributes:
        adapter_id: Unique identifier for this adapter.
        state_dict: PEFT-compatible state dict of LoRA weights.
    """

    adapter_id: str
    state_dict: dict[str, Any]


def scale_lora_b(state_dict: dict[str, Any], factor: float) -> dict[str, Any]:
    """Scale only lora_B parameters in an adapter state dict.

    Returns a new dict; the original is not mutated.
    """
    return {
        k: v * factor if "lora_B" in k else v
        for k, v in state_dict.items()
    }


def hotswap_adapter(model: Any, state_dict: dict[str, Any]) -> None:
    """Load new LoRA weights into a PEFT model in-place.

    Args:
        model: A PEFT-wrapped model.
        state_dict: New adapter weights to apply.
    """
    from peft import set_peft_model_state_dict  # noqa: PLC0415

    set_peft_model_state_dict(model, state_dict)
