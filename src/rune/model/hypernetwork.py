"""Hypernetwork loader and adapter-weight generator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_flash_patched = False

_MLP_CHUNK_SIZE = 2048


def _chunk_gated_mlp(orig_fwd: Any, module: Any, x: Any, chunk_size: int) -> Any:
    """Run a gated MLP in chunks over the flattened token axis to cap peak memory.

    The modality_projection MLP receives ``(n_layers, seq_len, hidden)``; its 4x
    gated intermediate is sized by ``n_layers * seq_len``, so chunking only the
    seq dim leaves the ``n_layers`` (=32) multiplier uncapped.  Flattening every
    leading dim into one token axis bounds the intermediate regardless of how the
    tokens are split across batch and sequence.
    """
    import torch  # noqa: PLC0415

    hidden = x.shape[-1]
    flat = x.reshape(-1, hidden)
    total = flat.shape[0]
    if total <= chunk_size:
        return orig_fwd(module, x)
    parts = [
        orig_fwd(module, flat[i : i + chunk_size]) for i in range(0, total, chunk_size)
    ]
    out = torch.cat(parts, dim=0)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def _patch_flash_attention() -> None:
    """Patch ctx_to_lora's idefics2 to use eager attention on this GPU."""
    global _flash_patched  # noqa: PLW0603
    if _flash_patched:
        return
    _flash_patched = True
    import ctx_to_lora.modeling.idefics2 as idefics2_mod  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from ctx_to_lora.modeling.idefics2 import (  # noqa: PLC0415
        Idefics2MLP,
        Idefics2Perceiver,
        Idefics2PerceiverAttention,
        Idefics2PerceiverConfig,
        Idefics2PerceiverLayer,
        Idefics2PerceiverResampler,
        Idefics2RMSNorm,
    )

    # Chunk the gated MLP forward to cap peak memory.  The modality_projection
    # receives (n_layers, seq_len, dim); the 4× expansion intermediate is sized
    # by n_layers*seq_len, so chunking spans the flattened token count (see
    # _chunk_gated_mlp) — chunking the seq dim alone left the ×32 layer
    # multiplier uncapped and OOMed on long trajectories.
    _orig_mlp_fwd = Idefics2MLP.forward

    def _chunked_mlp_forward(self_: Any, x: Any) -> Any:
        return _chunk_gated_mlp(_orig_mlp_fwd, self_, x, _MLP_CHUNK_SIZE)

    Idefics2MLP.forward = _chunked_mlp_forward  # type: ignore[assignment]

    idefics2_mod.IDEFICS2_PERCEIVER_ATTENTION_CLASSES["eager"] = (
        Idefics2PerceiverAttention
    )
    idefics2_mod.IDEFICS2_PERCEIVER_ATTENTION_CLASSES["flash_attention_2"] = (
        Idefics2PerceiverAttention
    )

    _orig_attn_fwd = Idefics2PerceiverAttention.forward

    def _patched_attn_fwd(
        self_: Any, *args: Any, is_cross_attn: Any = None, **kwargs: Any
    ) -> Any:
        for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            kwargs.pop(k, None)
        return _orig_attn_fwd(self_, *args, **kwargs)

    Idefics2PerceiverAttention.forward = _patched_attn_fwd  # type: ignore[assignment]

    def _eager_resampler_forward(
        self_: Any,
        context: Any,
        attention_mask: Any = None,
        position_ids: Any = None,
    ) -> Any:
        bsz = (
            context.shape[0]
            if position_ids is None
            else int(torch.where(position_ids == 0, 1, 0).sum().item())
        )
        latents = self_.latents_q.unsqueeze(0).expand((bsz, *self_.latents_q.size()))
        compressed = latents
        for layer in self_.layers:
            out = layer(
                latents=compressed,
                context=context,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            compressed = out[0]
        return self_.layernorm(compressed)

    Idefics2PerceiverResampler.forward = _eager_resampler_forward  # type: ignore[assignment]

    def _patched_resampler_init(self_: Any, config: Any) -> None:
        import torch.nn as nn  # noqa: PLC0415

        config._attn_implementation = "eager"
        config._attn_implementation_internal = "eager"
        nn.Module.__init__(self_)
        self_.config = config
        self_.num_blocks = config.num_blocks
        self_.num_self_attn_per_block = config.num_self_attn_per_block
        self_.shared_weights = config.shared_weights
        self_.hidden_size = config.hidden_size
        self_.hidden_act = config.hidden_act
        self_.n_latents = config.n_latents
        self_.rms_norm_eps = config.rms_norm_eps
        self_.latents_q = nn.Parameter(torch.randn(self_.n_latents, self_.hidden_size))
        first_x = [Idefics2PerceiverLayer(config, is_cross_attn=True)]
        first_self = [
            Idefics2PerceiverLayer(config, is_cross_attn=False)
            for _ in range(config.num_self_attn_per_block)
        ]
        self_.layers = nn.ModuleList(first_x + first_self)
        for blk in range(1, config.num_blocks):
            if self_.shared_weights:
                x_attn = (
                    Idefics2PerceiverLayer(config, is_cross_attn=True)
                    if blk == 1
                    else x_attn  # type: ignore[possibly-undefined]  # noqa: F821
                )
            else:
                x_attn = Idefics2PerceiverLayer(config, is_cross_attn=True)
            self_.layers.append(x_attn)
            for i in range(config.num_self_attn_per_block):
                sa = (
                    first_self[i]
                    if self_.shared_weights
                    else Idefics2PerceiverLayer(config, is_cross_attn=False)
                )
                self_.layers.append(sa)
        self_.layernorm = Idefics2RMSNorm(self_.hidden_size, eps=self_.rms_norm_eps)
        self_._use_flash_attention_2 = False

    Idefics2PerceiverResampler.__init__ = _patched_resampler_init  # type: ignore[assignment]

    _orig_perceiver_init = Idefics2Perceiver.__init__

    def _patched_perceiver_init(self_: Any, enc: Any, dec: Any) -> None:
        enc._attn_implementation = "eager"
        enc._attn_implementation_internal = "eager"
        dec._attn_implementation = "eager"
        dec._attn_implementation_internal = "eager"
        _orig_perceiver_init(self_, enc, dec)

    Idefics2Perceiver.__init__ = _patched_perceiver_init  # type: ignore[assignment]

    _orig_cfg_init = Idefics2PerceiverConfig.__init__

    def _patched_cfg_init(self_: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["attn_implementation"] = "eager"
        _orig_cfg_init(self_, *args, **kwargs)
        self_._attn_implementation = "eager"
        self_._attn_implementation_internal = "eager"

    Idefics2PerceiverConfig.__init__ = _patched_cfg_init  # type: ignore[assignment]


@dataclass
class HypernetworkConfig:
    """Configuration for loading a HyperLoRA checkpoint.

    Attributes:
        checkpoint_path: Path to the .pt checkpoint file.
        model_config_name: Base model config identifier used during training.
    """

    checkpoint_path: str
    model_config_name: str = "qwen3.5-9b"


def _resolve_checkpoint_path(path: str) -> str:
    """Resolve an S3 URI to a local cached path, or return local path as-is."""
    if not path.startswith("s3://"):
        return path

    import hashlib  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    cache_dir = Path.home() / ".cache" / "rune" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(path.encode()).hexdigest()[:16]
    cached = cache_dir / f"{key}.pt"

    if cached.exists():
        logger.info("Using cached checkpoint: %s", cached)
        return str(cached)

    from urllib.parse import urlparse  # noqa: PLC0415

    import boto3  # noqa: PLC0415

    parsed = urlparse(path)
    bucket, s3_key = parsed.netloc, parsed.path.lstrip("/")
    logger.info("Downloading %s → %s ...", path, cached)
    tmp = cached.with_suffix(".tmp")
    client = boto3.client("s3")
    with open(tmp, "wb") as dst:
        client.download_fileobj(bucket, s3_key, dst)
    tmp.rename(cached)
    return str(cached)


def load_hypernetwork(config: HypernetworkConfig, device: str = "cpu") -> Any:
    """Load a HyperLoRA model from a checkpoint and return it in eval mode.

    Args:
        config: Checkpoint path and model config name. Supports local paths
            and s3:// URIs (downloaded and cached automatically).
        device: Target device (e.g. "cuda", "cpu").

    Returns:
        HyperLoRA model in eval mode on the requested device.
    """
    import torch  # noqa: PLC0415

    _patch_flash_attention()

    local_path = _resolve_checkpoint_path(config.checkpoint_path)
    logger.info("Loading hypernetwork from %s", local_path)
    # mmap=True maps the 2.5GB checkpoint's tensors from disk (reclaimable page
    # cache) instead of anonymous RAM; on this ~15GB-RAM box a plain torch.load
    # (which also pulls in unused optimizer state) spikes past free RAM and the
    # kernel OOM-kills the process while the base model is also loading.
    try:
        sd = torch.load(local_path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        sd = torch.load(local_path, map_location="cpu", weights_only=False)

    from ctx_to_lora.modeling.hypernet import HyperLoRA  # noqa: PLC0415

    hc = sd.get("hypernet_config") or sd.get("config")
    hypernet_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    if hasattr(hc, "aggregator_config"):
        ac = hc.aggregator_config
        if hasattr(ac, "torch_dtype"):
            ac.torch_dtype = hypernet_dtype
        if hasattr(ac, "_attn_implementation"):
            ac._attn_implementation = "eager"
        if hasattr(ac, "_attn_implementation_internal"):
            ac._attn_implementation_internal = "eager"
    if hasattr(hc, "_attn_implementation"):
        hc._attn_implementation = "eager"
    hypernet = HyperLoRA(hc).to(hypernet_dtype)
    weights = sd.get("hypernet_state_dict") or sd.get("model_state_dict", sd)
    hypernet.load_state_dict(weights, strict=False)
    return hypernet.to(device).eval()


def extract_activations_with_model(
    text: str,
    model: Any,
    tokenizer: Any,
    layer_indices: list[int],
    max_length: int = 2048,
) -> tuple[Any, Any]:
    """Extract per-layer hidden state activations from a pre-loaded model.

    Args:
        text: Input text to tokenize and process.
        model: Pre-loaded model in eval mode.
        tokenizer: Paired tokenizer.
        layer_indices: Which hidden state indices to extract.
        max_length: Max token sequence length.

    Returns:
        Tuple of (features, attention_mask).
        features shape: (1, num_layers, seq_len, hidden_dim)
        attention_mask shape: (1, seq_len)
    """
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    selected = torch.stack([hidden_states[i] for i in layer_indices], dim=1)
    attention_mask = inputs["attention_mask"]
    del outputs, hidden_states
    return selected, attention_mask


_ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}


def _to_peft_state_dict(
    lora_dict: dict[str, dict[str, Any]],
    layer_indices: list[int],
    target_modules: list[str],
) -> dict[str, Any]:
    """Convert HyperLoRA nested output to PEFT flat state_dict."""
    state_dict: dict[str, Any] = {}
    for mod_name, weights in lora_dict.items():
        if mod_name not in target_modules:
            continue
        a_weights = weights["A"]
        b_weights = weights["B"]
        prefix = "self_attn" if mod_name in _ATTN_MODULES else "mlp"
        for layer_pos, layer_idx in enumerate(layer_indices):
            key_a = (
                f"base_model.model.model.layers.{layer_idx}"
                f".{prefix}.{mod_name}.lora_A.weight"
            )
            key_b = (
                f"base_model.model.model.layers.{layer_idx}"
                f".{prefix}.{mod_name}.lora_B.weight"
            )
            state_dict[key_a] = a_weights[0, layer_pos].contiguous()
            state_dict[key_b] = b_weights[0, layer_pos].t().contiguous()
    return state_dict


def generate_adapter_weights(
    hypernet: Any,
    trajectory_text: str,
    base_model: Any,
    tokenizer: Any,
    layer_indices: list[int],
    max_length: int = 2048,
    offload_base: bool = False,
) -> dict[str, Any]:
    """Generate LoRA weight dict from a trajectory string via the hypernetwork.

    Args:
        hypernet: Loaded HyperLoRA model.
        trajectory_text: Serialised coding trajectory used as conditioning.
        base_model: Base language model for activation extraction.
        tokenizer: Tokenizer paired with base_model.
        layer_indices: Which transformer layers to extract activations from.
        max_length: Maximum token length for trajectory encoding.
        offload_base: Move base_model to CPU during the hypernetwork forward
            pass to free GPU memory.  Adds transfer latency but prevents OOM
            when both models don't fit simultaneously.

    Returns:
        PEFT-compatible flat state dict for hot-swap.
    """
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    features, attn_mask = extract_activations_with_model(
        text=trajectory_text,
        model=base_model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
        max_length=max_length,
    )

    base_device: torch.device | None = None
    if offload_base:
        base_device = next(base_model.parameters()).device
        base_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    hypernet_device = next(hypernet.parameters()).device
    hypernet_dtype = next(hypernet.parameters()).dtype
    features = features.to(device=hypernet_device, dtype=hypernet_dtype)
    attn_mask = attn_mask.to(device=hypernet_device)
    with torch.no_grad():
        lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)

    if base_device is not None:
        base_model.to(base_device)

    hc = hypernet.config
    target_modules = list(hc.lora_config.target_modules)
    return _to_peft_state_dict(lora_dict, layer_indices, target_modules)
