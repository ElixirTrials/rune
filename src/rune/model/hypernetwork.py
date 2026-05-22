from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HypernetworkConfig:
    checkpoint_path: str
    model_config_name: str = "qwen3.5-9b"


def load_hypernetwork(config: HypernetworkConfig) -> Any:
    import torch  # noqa: PLC0415

    logger.info("Loading hypernetwork from %s", config.checkpoint_path)
    sd = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)

    from ctx_to_lora.modeling.hypernet import HyperLoRA  # noqa: PLC0415

    hc = sd.get("hypernet_config") or sd.get("config")
    hypernet = HyperLoRA(hc)
    weights = sd.get("hypernet_state_dict") or sd.get("model_state_dict", sd)
    hypernet.load_state_dict(weights, strict=False)
    return hypernet.eval()


def generate_adapter_weights(
    hypernet: Any,
    trajectory_text: str,
    base_model: Any,
    tokenizer: Any,
    layer_indices: list[int],
    max_length: int = 2048,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    from model_training.d2l_activations import (
        extract_activations_with_model,  # noqa: PLC0415
    )

    features, attn_mask = extract_activations_with_model(
        text=trajectory_text,
        model=base_model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
        max_length=max_length,
    )
    with torch.no_grad():
        lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)
    return lora_dict
