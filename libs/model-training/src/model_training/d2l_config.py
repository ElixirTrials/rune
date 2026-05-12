"""Config helpers for hypernetwork training across model architectures.

Provides both model-specific (Qwen3-Coder-Next) and model-agnostic config
builders. The model-agnostic build_hypernet_config() uses the model registry
and probe cache to support any registered model.

All heavy imports (transformers, ctx_to_lora, peft) are deferred to function
bodies per project convention (INFRA-05) to avoid GPU imports at module level.

Default hyperparameters live in ``hypernet_defaults.yaml`` (same directory).
Use :func:`load_hypernet_defaults` to access them.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULTS_PATH = Path(__file__).parent / "hypernet_defaults.yaml"


@functools.lru_cache(maxsize=1)
def load_hypernet_defaults() -> dict[str, Any]:
    """Load default hypernetwork config from ``hypernet_defaults.yaml``."""
    import yaml  # noqa: PLC0415

    with open(_DEFAULTS_PATH) as f:
        return yaml.safe_load(f)


__all__ = [
    "get_d2l_qwen3_config",
    "build_qwen3_hypernet_config",
    "build_hypernet_config",
    "build_from_scratch_hypernet_config",
    "load_hypernet_defaults",
]


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
    aggregator_config: Any = None,
) -> Any:
    """Construct HypernetConfig targeting Qwen3-Coder-Next attention layers.

    Discovers full_attention layer indices dynamically from Qwen3NextConfig.layer_types.
    Result has exactly 12 layer indices matching the Qwen3-Coder-Next architecture.

    Phase 26 probe cache integration: if a probe cache exists for
    QWEN3_NEXT_CANONICAL_NAME, uses real per-projection in/out dimensions for
    feature_sizes. Falls back to hidden_size placeholder when no cache is found
    (e.g., in CI where the model has not been probed).

    Args:
        lora_r: LoRA rank for the adapter. Defaults to 8.
        target_modules: LoRA target module names. Defaults to ["q_proj", "v_proj"].
        aggregator_config: Perceiver aggregator config from a Sakana checkpoint.
            If None (default / Phase 25 CI), HypernetConfig is built with
            aggregator_config=None as placeholder. Phase 29 populates this via
            get_aggregator_config() with a loaded model.

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
        aggregator_config=aggregator_config,
    )


def build_hypernet_config(
    model_name: str,
    lora_r: int | None = None,
    target_modules: list[str] | None = None,
    aggregator_config: Any = None,
) -> Any:
    """Construct HypernetConfig for any registered model.

    Uses the model registry for architecture expectations and the probe cache
    for actual layer indices and feature dimensions. For Qwen3-Coder-Next,
    delegates to the specialized builder. For other models, builds config
    from probe cache data.

    Args:
        model_name: Canonical model name from the registry (e.g. "qwen3.5-9b").
        lora_r: LoRA rank. Defaults to the registry's default_lora_rank.
        target_modules: LoRA target module names. Defaults to probe cache's
            target_modules or ["q_proj", "v_proj"].
        aggregator_config: Perceiver aggregator config from a Sakana checkpoint.

    Returns:
        HypernetConfig configured for the specified model.

    Raises:
        KeyError: If model_name is not in the registry.
        RuntimeError: If no probe cache exists for a non-Qwen3-Coder-Next model.
    """
    from ctx_to_lora.modeling.hypernet import HypernetConfig  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415

    from model_training.d2l_probe import load_probe_cache  # noqa: PLC0415
    from model_training.model_configs import (  # noqa: PLC0415
        ModelRegistry,
        validate_against_probe,
    )

    mc = ModelRegistry.default().get(model_name)

    # Qwen3-Coder-Next has specialized logic for hybrid attention discovery
    if model_name == "qwen3-coder-next":
        rank = lora_r if lora_r is not None else mc.default_lora_rank
        return build_qwen3_hypernet_config(
            lora_r=rank,
            target_modules=target_modules,
            aggregator_config=aggregator_config,
        )

    # Generic path: requires probe cache for layer indices and dimensions
    cache = load_probe_cache(model_name)
    if cache is None:
        msg = (
            f"No probe cache found for {model_name!r}. "
            "Run probe_model() and save_probe_cache() before building "
            "hypernet config for this model."
        )
        raise RuntimeError(msg)

    validate_against_probe(mc, cache)

    rank = lora_r if lora_r is not None else mc.default_lora_rank
    layer_indices = cache["attention_layer_indices"]

    # Resolve target modules from probe cache or fallback
    if target_modules is None:
        target_modules = cache.get("target_modules", ["q_proj", "v_proj"])

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=mc.default_lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Build feature_sizes from probe cache
    feature_sizes_raw = cache.get("feature_sizes", {})
    in_sizes: dict[str, int] = {}
    out_sizes: dict[str, int] = {}
    for mod in target_modules:
        if mod in feature_sizes_raw:
            in_sizes[mod] = feature_sizes_raw[mod]["in"]
            out_sizes[mod] = feature_sizes_raw[mod]["out"]
        else:
            in_sizes[mod] = mc.expected_hidden_size
            out_sizes[mod] = mc.expected_hidden_size

    feature_sizes: tuple[dict[str, int], dict[str, int]] = (
        in_sizes,
        out_sizes,
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
        base_hidden_size=mc.expected_hidden_size,
        layer_indices=layer_indices,
        feature_sizes=feature_sizes,
        aggregator_config=aggregator_config,
    )


def build_from_scratch_hypernet_config(
    model_name: str = "qwen3.5-9b",
    lora_r: int | None = None,
    target_modules: list[str] | None = None,
    n_latent_queries: int | None = None,
) -> Any:
    """Build HypernetConfig + AggregatorConfig from scratch (no Sakana checkpoint).

    Reads architecture dimensions from the HuggingFace model config directly.
    All tuneable defaults (LoRA rank, target modules, perceiver shape, training
    params) are loaded from ``hypernet_defaults.yaml`` and can be overridden
    via function arguments.

    Args:
        model_name: Registry model name.
        lora_r: LoRA rank (default from YAML).
        target_modules: Projection modules to target (default from YAML).
        n_latent_queries: Number of perceiver latent queries (default from YAML).

    Returns:
        HypernetConfig with a fully populated AggregatorConfig.
    """
    from ctx_to_lora.modeling.aggregator import (  # noqa: PLC0415
        AGGREGATOR_TYPE,
        POOL_FN,
        AggregatorConfig,
    )
    from ctx_to_lora.modeling.hypernet import HypernetConfig  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415
    from transformers import AutoConfig  # noqa: PLC0415

    from model_training.model_configs import ModelRegistry  # noqa: PLC0415

    dfl = load_hypernet_defaults()
    lora_dfl = dfl["lora"]
    perc_dfl = dfl["perceiver"]
    head_dfl = dfl["head"]

    if lora_r is None:
        lora_r = lora_dfl["r"]
    if target_modules is None:
        target_modules = list(lora_dfl["target_modules"])
    if n_latent_queries is None:
        n_latent_queries = perc_dfl["n_latent_queries"]

    mc = ModelRegistry.default().get(model_name)
    hf_cfg = AutoConfig.from_pretrained(mc.model_id)

    # Qwen3.5 wraps text config inside a VL config
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg)

    hidden_size: int = text_cfg.hidden_size
    num_hidden_layers: int = text_cfg.num_hidden_layers
    layer_indices = list(range(num_hidden_layers))

    num_heads: int = text_cfg.num_attention_heads
    num_kv_heads: int = text_cfg.num_key_value_heads
    head_dim: int = getattr(text_cfg, "head_dim", hidden_size // num_heads)

    # Linear-attention config (Qwen3.5 hybrid architecture)
    lin_num_k_heads: int = getattr(text_cfg, "linear_num_key_heads", 0)
    lin_k_head_dim: int = getattr(text_cfg, "linear_key_head_dim", 0)
    lin_num_v_heads: int = getattr(text_cfg, "linear_num_value_heads", 0)
    lin_v_head_dim: int = getattr(text_cfg, "linear_value_head_dim", 0)
    key_dim = lin_num_k_heads * lin_k_head_dim
    value_dim = lin_num_v_heads * lin_v_head_dim

    # Per-projection in/out dimensions for both attention types.
    # Qwen3.5 q_proj outputs 2x (gated attention): num_heads * head_dim * 2.
    _proj_dims: dict[str, tuple[int, int]] = {
        # Full-attention modules (layers where layer_type == "full_attention")
        "q_proj": (hidden_size, num_heads * head_dim * 2),
        "k_proj": (hidden_size, num_kv_heads * head_dim),
        "v_proj": (hidden_size, num_kv_heads * head_dim),
        "o_proj": (num_heads * head_dim, hidden_size),
        # Linear-attention modules (layers where layer_type == "linear_attention")
        "in_proj_qkv": (hidden_size, key_dim * 2 + value_dim),
        "in_proj_z": (hidden_size, value_dim),
        "in_proj_a": (hidden_size, lin_num_v_heads),
        "in_proj_b": (hidden_size, lin_num_v_heads),
        "out_proj": (value_dim, hidden_size),
        # MLP modules (all layers)
        "gate_proj": (hidden_size, text_cfg.intermediate_size),
        "up_proj": (hidden_size, text_cfg.intermediate_size),
        "down_proj": (text_cfg.intermediate_size, hidden_size),
    }

    _default = (hidden_size, hidden_size)
    in_sizes = {m: _proj_dims.get(m, _default)[0] for m in target_modules}
    out_sizes = {m: _proj_dims.get(m, _default)[1] for m in target_modules}
    feature_sizes: tuple[dict[str, int], dict[str, int]] = (in_sizes, out_sizes)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * lora_dfl["alpha_multiplier"],
        target_modules=target_modules,
        lora_dropout=lora_dfl["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    latent_size = perc_dfl["latent_size"]
    per_rank_gen: bool = perc_dfl.get("per_rank_gen", True)

    agg_config = AggregatorConfig(
        aggregator_type=AGGREGATOR_TYPE.PERCEIVER,
        num_layers=len(layer_indices),
        num_modules=len(target_modules),
        num_extra_modules=0,
        output_size=perc_dfl["output_size"],
        feature_size=hidden_size,
        pooling_type=POOL_FN.MEAN,
        num_latent_factor=perc_dfl["num_latent_factor"],
        lora_r=lora_r,
        per_rank_gen=per_rank_gen,
        n_latent_queries=n_latent_queries,
        num_blocks=perc_dfl["num_blocks"],
        num_self_attn_per_block=perc_dfl["num_self_attn_per_block"],
        shared_weights=perc_dfl["shared_weights"],
        layer_to_layer_ctx_encoder=perc_dfl["layer_to_layer"],
    )

    return HypernetConfig(
        latent_size=latent_size,
        use_light_weight_lora=False,
        light_weight_latent_size=128,
        per_rank_gen=per_rank_gen,
        use_per_rank_bias=False,
        use_bias=head_dfl["use_bias"],
        per_layer_processing=False,
        use_token_mixing=False,
        num_pre_head_layers=head_dfl["num_pre_head_layers"],
        dropout_rate=head_dfl["dropout_rate"],
        lora_config=lora_config,
        extra_modules=None,
        base_hidden_size=hidden_size,
        layer_indices=layer_indices,
        feature_sizes=feature_sizes,
        aggregator_config=agg_config,
    )
