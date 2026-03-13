"""Config helpers for Qwen3-Coder-Next hypernetwork training.

Requires transformers>=5.0 for Qwen3NextConfig (hybrid linear/full attention
architecture). transformers 5.3.0 is installed in this project.

All heavy imports (transformers, ctx_to_lora, peft) are deferred to function
bodies per project convention (INFRA-05) to avoid GPU imports at module level.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["get_d2l_qwen3_config", "build_qwen3_hypernet_config"]


def get_d2l_qwen3_config() -> dict[str, Any]:
    """Return Qwen3-Coder-Next architecture dimensions without loading model weights.

    Uses Qwen3NextConfig defaults which exactly match Qwen3-Coder-Next specs:
    - hidden_size: 2048
    - num_hidden_layers: 48 (12 full_attention + 36 linear_attention)
    - num_attention_heads: 16 (Q heads), num_key_value_heads: 2 (GQA KV)
    - head_dim: 256
    - full_attention layer indices: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
    - vocab_size: 151936
    - model_type: "qwen3_next"

    Returns:
        Dict with keys: hidden_size, num_hidden_layers, num_attention_heads,
        num_key_value_heads, head_dim, attention_layer_indices, vocab_size,
        model_type.
    """
    from transformers import Qwen3NextConfig  # noqa: PLC0415

    cfg = Qwen3NextConfig()
    layer_types: list[str] = cfg.layer_types or []
    attention_layer_indices = [
        i for i, t in enumerate(layer_types) if t == "full_attention"
    ]
    return {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "attention_layer_indices": attention_layer_indices,
        "vocab_size": cfg.vocab_size,
        "model_type": cfg.model_type,
    }


def build_qwen3_hypernet_config(
    lora_r: int = 8,
    target_modules: list[str] | None = None,
) -> Any:
    """Construct HypernetConfig targeting Qwen3-Coder-Next attention layers.

    Discovers full_attention layer indices dynamically from Qwen3NextConfig.layer_types.
    Result has exactly 12 layer indices matching the Qwen3-Coder-Next architecture.

    Phase 26 probe cache integration: if a probe cache exists for
    QWEN3_NEXT_CANONICAL_NAME, uses real per-projection in/out dimensions for
    feature_sizes. Falls back to hidden_size placeholder when no cache is found
    (e.g., in CI where the model has not been probed).

    Note: aggregator_config is set to None. Phase 27 (weight transfer) must call
    get_aggregator_config() with a loaded model and populate this field before
    using the config in training.

    Args:
        lora_r: LoRA rank for the adapter. Defaults to 8.
        target_modules: LoRA target module names. Defaults to ["q_proj", "v_proj"].

    Returns:
        HypernetConfig with layer_indices set to the 12 full_attention indices
        and base_hidden_size=2048.
    """
    from ctx_to_lora.modeling.hypernet import HypernetConfig  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415
    from transformers import Qwen3NextConfig  # noqa: PLC0415

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    cfg = Qwen3NextConfig()
    layer_types: list[str] = cfg.layer_types or []
    layer_indices = [
        i for i, t in enumerate(layer_types) if t == "full_attention"
    ]  # Always [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    from model_training.d2l_probe import (  # noqa: PLC0415
        QWEN3_NEXT_CANONICAL_NAME,
        load_probe_cache,
    )

    cache = load_probe_cache(QWEN3_NEXT_CANONICAL_NAME)
    if cache is not None:
        in_sizes = {mod: cache["feature_sizes"][mod]["in"] for mod in target_modules}
        out_sizes = {mod: cache["feature_sizes"][mod]["out"] for mod in target_modules}
        feature_sizes: tuple[dict[str, int], dict[str, int]] = (in_sizes, out_sizes)
        logger.info("Using probe cache feature_sizes for %s", QWEN3_NEXT_CANONICAL_NAME)
    else:
        hidden: int = cfg.hidden_size or 2048
        _placeholder: dict[str, int] = dict.fromkeys(target_modules, hidden)
        feature_sizes = (_placeholder, dict.fromkeys(target_modules, hidden))
        logger.warning(
            "No probe cache for '%s' — using hidden_size=%d as placeholder. "
            "Run probe_model() and save_probe_cache() to set real dimensions.",
            QWEN3_NEXT_CANONICAL_NAME,
            cfg.hidden_size,
        )

    return HypernetConfig(
        latent_size=512,
        use_light_weight_lora=False,
        light_weight_latent_size=128,
        per_rank_gen=False,
        use_per_rank_bias=False,
        use_bias=True,
        per_layer_processing=False,
        use_token_mixing=False,
        num_pre_head_layers=1,
        dropout_rate=0.0,
        lora_config=lora_config,
        extra_modules=None,
        base_hidden_size=cfg.hidden_size,
        layer_indices=layer_indices,
        feature_sizes=feature_sizes,
        # placeholder — Phase 27 sets real value via get_aggregator_config()
        aggregator_config=None,
    )
