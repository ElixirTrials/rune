"""Hypernetwork checkpoint loading and management utilities.

Supports both local paths and S3 URIs via fsspec. Handles downloading
pretrained HyperLoRA perceiver checkpoints from HuggingFace and loading
them into memory with flash-attention compatibility patches.

IMPORTANT: All GPU imports (torch) are deferred inside function bodies
per INFRA-05 pattern — this module is importable in CPU-only CI.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_S3_CACHE_DIR = Path.home() / ".cache" / "rune" / "checkpoints"

# HuggingFace repo for pretrained checkpoints
HF_REPO_ID = "SakanaAI/doc-to-lora"
# Available checkpoints: gemma_2b_d2l, gemma_demo, mistral_7b_d2l, qwen_4b_d2l
DEFAULT_VARIANT = "gemma_demo"
DEFAULT_HF_FILENAME = f"{DEFAULT_VARIANT}/checkpoint-80000/pytorch_model.bin"
LOCAL_CACHE_DIR = Path.home() / ".cache" / "rune" / "hypernetwork"

_flash_attention_patched = False


def _open_checkpoint(path: str) -> Any:
    """Open a checkpoint from a local path or S3 URI.

    S3 URIs are downloaded to a local cache on first access and loaded
    from the cache on subsequent calls, avoiding repeated network
    streams (~34s each for a typical hypernetwork checkpoint).

    Args:
        path: Local filesystem path or s3:// URI.

    Returns:
        Deserialized checkpoint dict.
    """
    import torch  # noqa: PLC0415

    if path.startswith("s3://"):
        path = _ensure_local_s3_cache(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _ensure_local_s3_cache(s3_uri: str) -> str:
    """Download an S3 checkpoint to a local cache, returning the local path.

    The cache key is a SHA-256 of the URI. If the cached file already
    exists, returns immediately.
    """
    _S3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(s3_uri.encode()).hexdigest()[:16]
    suffix = Path(s3_uri).suffix or ".pt"
    cached = _S3_CACHE_DIR / f"{key}{suffix}"

    if cached.exists():
        logger.info("Using cached checkpoint: %s", cached)
        return str(cached)

    import fsspec  # type: ignore[import-untyped]  # noqa: PLC0415

    logger.info("Downloading %s → %s ...", s3_uri, cached)
    tmp = cached.with_suffix(".tmp")
    with fsspec.open(s3_uri, "rb") as src, open(tmp, "wb") as dst:
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)
    tmp.rename(cached)
    return str(cached)


def _patch_flash_attention() -> None:  # noqa: C901
    """Patch the idefics2 module to work without flash_attn.

    Replaces flash attention classes and assertions with eager equivalents
    so the perceiver can run on CPU/MPS/CUDA regardless of flash_attn
    availability.
    """
    global _flash_attention_patched  # noqa: PLW0603
    if _flash_attention_patched:
        return
    _flash_attention_patched = True

    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    # Prefer the real flash_attn package when installed (GPU env).
    # Fall back to a stub so the module stays importable in CPU-only CI.
    if "flash_attn" not in sys.modules:
        try:
            import flash_attn  # noqa: PLC0415, F811, F401
        except ImportError:
            import importlib.machinery  # noqa: PLC0415
            import importlib.metadata as _imeta  # noqa: PLC0415

            stub = types.ModuleType("flash_attn")
            stub.__version__ = "2.6.3"  # type: ignore[attr-defined]
            stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
            sys.modules["flash_attn"] = stub

            # Submodule stub for flash_attn.bert_padding (imported by ctx_to_lora)
            bert_stub = types.ModuleType("flash_attn.bert_padding")
            bert_stub.__spec__ = importlib.machinery.ModuleSpec(
                "flash_attn.bert_padding", None
            )
            bert_stub.unpad_input = lambda *a, **kw: None  # type: ignore[attr-defined]  # noqa: ARG005
            stub.bert_padding = bert_stub  # type: ignore[attr-defined]
            sys.modules["flash_attn.bert_padding"] = bert_stub

            # Patch importlib.metadata.version so transformers'
            # `_is_package_available("flash_attn")` succeeds with the stub.
            _orig_meta_version = _imeta.version

            def _patched_meta_version(name: str) -> str:
                if name == "flash_attn":
                    return "2.6.3"
                return _orig_meta_version(name)

            _imeta.version = _patched_meta_version  # type: ignore[assignment]

    import ctx_to_lora.modeling.idefics2 as idefics2_mod  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from ctx_to_lora.modeling.idefics2 import (  # noqa: PLC0415
        Idefics2Perceiver,
        Idefics2PerceiverAttention,
        Idefics2PerceiverConfig,
        Idefics2PerceiverLayer,
        Idefics2PerceiverResampler,
        Idefics2RMSNorm,
    )

    # Map both eager and flash_attention_2 to the eager attention class
    idefics2_mod.IDEFICS2_PERCEIVER_ATTENTION_CLASSES["eager"] = (
        Idefics2PerceiverAttention
    )
    idefics2_mod.IDEFICS2_PERCEIVER_ATTENTION_CLASSES["flash_attention_2"] = (
        Idefics2PerceiverAttention
    )

    # Patch eager attention forward to accept flash-only kwargs
    _orig_attn_fwd = Idefics2PerceiverAttention.forward

    def _patched_attn_fwd(
        self: Any, *args: Any, is_cross_attn: Any = None, **kwargs: Any
    ) -> Any:
        kwargs.pop("cu_seq_lens_q", None)
        kwargs.pop("cu_seq_lens_k", None)
        kwargs.pop("max_length_q", None)
        kwargs.pop("max_length_k", None)
        return _orig_attn_fwd(self, *args, **kwargs)

    Idefics2PerceiverAttention.forward = _patched_attn_fwd

    # Patch resampler forward to use eager path
    def _eager_resampler_forward(
        self: Any,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_ids is None:
            bsz = context.shape[0]
        else:
            bsz = int(torch.where(position_ids == 0, 1, 0).sum().item())
        latents = self.latents_q.unsqueeze(0).expand((bsz, *self.latents_q.size()))
        compressed_context = latents
        for layer in self.layers:
            layer_outputs = layer(
                latents=compressed_context,
                context=context,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            compressed_context = layer_outputs[0]
        return self.layernorm(compressed_context)

    Idefics2PerceiverResampler.forward = _eager_resampler_forward

    # Patch resampler __init__ to use eager attention (avoids triggering
    # transformers' flash_attention_2 validation chain)
    _orig_resampler_init = Idefics2PerceiverResampler.__init__

    def _patched_resampler_init(self: Any, config: Any) -> None:
        # Bypass PreTrainedModel.__init__ entirely — it validates flash
        # attention support which fails on transformers 5.x. Rebuild the
        # resampler layers directly using eager attention.
        import torch.nn as nn  # noqa: PLC0415

        config._attn_implementation = "eager"
        config._attn_implementation_internal = "eager"
        nn.Module.__init__(self)
        self.config = config

        self.num_blocks = config.num_blocks
        self.num_self_attn_per_block = config.num_self_attn_per_block
        self.shared_weights = config.shared_weights
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.n_latents = config.n_latents
        self.rms_norm_eps = config.rms_norm_eps

        self.latents_q = nn.Parameter(torch.randn(self.n_latents, self.hidden_size))

        first_x_attn = [Idefics2PerceiverLayer(config, is_cross_attn=True)]
        first_self_attn_block = [
            Idefics2PerceiverLayer(config, is_cross_attn=False)
            for _ in range(config.num_self_attn_per_block)
        ]
        self.layers = nn.ModuleList(first_x_attn + first_self_attn_block)

        for layer_idx in range(1, config.num_blocks):
            if self.shared_weights:
                if layer_idx == 1:
                    second_x_attn = Idefics2PerceiverLayer(config, is_cross_attn=True)
                x_attn = second_x_attn
            else:
                x_attn = Idefics2PerceiverLayer(config, is_cross_attn=True)
            self.layers.append(x_attn)

            for i in range(config.num_self_attn_per_block):
                if self.shared_weights:
                    self_attn = first_self_attn_block[i]
                else:
                    self_attn = Idefics2PerceiverLayer(config, is_cross_attn=False)
                self.layers.append(self_attn)

        self.layernorm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self._use_flash_attention_2 = False

    Idefics2PerceiverResampler.__init__ = _patched_resampler_init

    # Patch Perceiver init to force eager on sub-configs
    _orig_perceiver_init = Idefics2Perceiver.__init__

    def _patched_perceiver_init(self: Any, enc_cfg: Any, dec_cfg: Any) -> None:
        enc_cfg._attn_implementation = "eager"
        enc_cfg._attn_implementation_internal = "eager"
        dec_cfg._attn_implementation = "eager"
        dec_cfg._attn_implementation_internal = "eager"
        _orig_perceiver_init(self, enc_cfg, dec_cfg)

    Idefics2Perceiver.__init__ = _patched_perceiver_init

    # Patch PerceiverConfig init to default to eager
    _orig_cfg_init = Idefics2PerceiverConfig.__init__

    def _patched_cfg_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["attn_implementation"] = "eager"
        _orig_cfg_init(self, *args, **kwargs)
        self._attn_implementation = "eager"
        self._attn_implementation_internal = "eager"

    Idefics2PerceiverConfig.__init__ = _patched_cfg_init


def download_checkpoint(
    variant: str = DEFAULT_VARIANT,
) -> Path:
    """Download pretrained checkpoint from HuggingFace.

    Args:
        variant: Which checkpoint variant to download.
            Options: 'gemma_demo', 'gemma_2b_d2l', 'mistral_7b_d2l', 'qwen_4b_d2l'.

    Returns:
        Path to the downloaded checkpoint file.
    """
    # Determine filename based on variant
    if variant == "gemma_demo":
        hf_filename = "gemma_demo/checkpoint-80000/pytorch_model.bin"
    elif variant in ("gemma_2b_d2l", "mistral_7b_d2l", "qwen_4b_d2l"):
        hf_filename = f"{variant}/checkpoint-20000/pytorch_model.bin"
    else:
        msg = f"Unknown variant: {variant}"
        raise ValueError(msg)

    cached = LOCAL_CACHE_DIR / variant / "pytorch_model.bin"
    if cached.exists():
        logger.info("Using cached hypernetwork checkpoint: %s", cached)
        return cached

    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    logger.info(
        "Downloading hypernetwork checkpoint %s from %s...", variant, HF_REPO_ID
    )
    downloaded = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=hf_filename))

    cached.parent.mkdir(parents=True, exist_ok=True)
    import shutil  # noqa: PLC0415

    shutil.copy2(downloaded, cached)
    logger.info("Cached to: %s", cached)
    return cached


def load_hypernetwork(
    checkpoint_path: str | Path | None = None,
    variant: str = DEFAULT_VARIANT,
    device: str = "cpu",
) -> tuple[Any, Any]:
    """Load HyperLoRA perceiver from checkpoint.

    Downloads from HuggingFace if no local path is provided.
    Patches flash attention for CPU/MPS compatibility.

    Args:
        checkpoint_path: Path to local checkpoint. If None, downloads from HF.
        variant: HF checkpoint variant (only used if checkpoint_path is None).
        device: Device to load onto.

    Returns:
        Tuple of (hypernet, hypernet_config).
    """
    import torch  # noqa: PLC0415

    _patch_flash_attention()

    # Pre-import flash_attn before torch.load — the unpickler triggers
    # ctx_to_lora module imports in a context that breaks flash_attn
    # resolution if it hasn't been imported yet.
    try:
        import flash_attn.flash_attn_interface  # noqa: F401,PLC0415
    except ImportError:
        pass

    if checkpoint_path is None:
        checkpoint_path = download_checkpoint(variant)

    logger.info("Loading hypernetwork checkpoint: %s", checkpoint_path)
    sd = _open_checkpoint(str(checkpoint_path))

    hc = sd["hypernet_config"]
    logger.info(
        "HypernetConfig: latent_size=%d, lora_r=%d, base_model=%s",
        hc.latent_size,
        hc.lora_config.r,
        sd["base_model_name_or_path"],
    )

    from ctx_to_lora.modeling.hypernet import HyperLoRA  # noqa: PLC0415
    from shared.hardware import resolve_model_dtype  # noqa: PLC0415

    hypernet_param_count = sum(
        v.numel() for v in sd.values() if isinstance(v, torch.Tensor)
    )
    # On GPU, force bf16 — the hypernetwork shares VRAM with the base LLM
    # and fp32 provides no quality benefit for weight generation.
    hypernet_dtype = resolve_model_dtype(
        param_count=hypernet_param_count,
        device=device,
        dtype_override=None if device == "cpu" else "bfloat16",
    )
    logger.info("HyperLoRA dtype resolved to %s", hypernet_dtype)
    # Suppress "Flash Attention 2 without specifying a torch dtype" warning
    # by setting the dtype on the config before HyperLoRA instantiation.
    if hasattr(hc, "aggregator_config"):
        ac = hc.aggregator_config
        if hasattr(ac, "torch_dtype"):
            ac.torch_dtype = hypernet_dtype
    hypernet = HyperLoRA(hc).to(hypernet_dtype)

    # Load hypernet weights. Our from-scratch checkpoints store weights
    # under a "hypernet_state_dict" key; original checkpoints store them
    # as flat top-level tensors.
    model_keys = set(hypernet.state_dict().keys())
    if "hypernet_state_dict" in sd:
        hypernet_sd = sd["hypernet_state_dict"]
    else:
        hypernet_sd = {k: v for k, v in sd.items() if k in model_keys}

    loaded = hypernet.load_state_dict(hypernet_sd, strict=False)
    logger.info(
        "Loaded %d/%d hypernet weight tensors",
        len(hypernet_sd),
        len(model_keys),
    )
    if loaded.missing_keys:
        logger.warning(
            "Missing keys (%d, will use defaults): %s",
            len(loaded.missing_keys),
            loaded.missing_keys,
        )
    if loaded.unexpected_keys:
        logger.info("Unexpected keys: %d", len(loaded.unexpected_keys))

    hypernet = hypernet.to(device)
    hypernet.eval()

    param_count = sum(p.numel() for p in hypernet.parameters())
    logger.info("HyperLoRA params: %d", param_count)

    return hypernet, hc


def _assert_transfer_integrity(hypernet: Any, loaded: Any) -> None:
    """Assert that partial weight transfer completed correctly.

    Validates the result of load_state_dict(strict=False) after loading only
    aggregator.* weights from a checkpoint. Raises AssertionError on any
    sign of a mismatch so failures are caught early rather than silently producing
    a corrupted model.

    Args:
        hypernet: The HyperLoRA model (used to enumerate expected aggregator keys).
        loaded: The _IncompatibleKeys object returned by load_state_dict(strict=False).

    Raises:
        AssertionError: If any aggregator key is missing or any unexpected key is
            present.
    """
    # Check 1: unexpected keys indicate the checkpoint has keys that don't belong
    if loaded.unexpected_keys:
        msg = (
            f"Transfer produced unexpected keys: {loaded.unexpected_keys!r}. "
            "Check that checkpoint prefixes match model parameter names."
        )
        raise AssertionError(msg)

    # Check 2: every missing key must start with "head." (head intentionally not loaded)
    non_head_missing = [k for k in loaded.missing_keys if not k.startswith("head.")]
    if non_head_missing:
        msg = (
            f"Non-head keys were missing after transfer: {non_head_missing!r}. "
            "Run print(checkpoint.keys()) to verify aggregator.* prefixes "
            "exist in the checkpoint."
        )
        raise AssertionError(msg)

    # Check 3: no aggregator keys from the model should be in missing_keys
    aggregator_keys = {k for k in hypernet.state_dict() if k.startswith("aggregator.")}
    aggregator_missing = aggregator_keys & set(loaded.missing_keys)
    if aggregator_missing:
        msg = (
            f"Aggregator keys were not loaded from checkpoint: {aggregator_missing!r}. "
            "The checkpoint may not contain aggregator.* weights."
        )
        raise AssertionError(msg)

    n_aggregator = len(aggregator_keys)
    n_head_reinit = len([k for k in loaded.missing_keys if k.startswith("head.")])
    logger.info(
        "Transfer integrity OK: %d aggregator keys loaded, %d head keys re-initialized",
        n_aggregator,
        n_head_reinit,
    )


def transfer_aggregator_weights(hypernet: Any, checkpoint_path: str | Path) -> Any:
    """Load aggregator weights from a checkpoint into a HyperLoRA instance.

    Loads only aggregator.* weights from the checkpoint (not head.*), freezes all
    aggregator parameters (requires_grad=False), and leaves head.* at PyTorch default
    initialization for training against the new target model.

    This enables reuse of the pretrained Perceiver aggregator across different target
    model architectures. The aggregator maps document embeddings to LoRA weight space
    and is model-agnostic; only the head needs retraining per target model.

    Args:
        hypernet: The HyperLoRA model to load weights into (mutated in-place).
        checkpoint_path: Path to the checkpoint (.bin file).

    Returns:
        The mutated hypernet (returned for chaining convenience).
    """
    import torch  # noqa: PLC0415

    sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Filter to only aggregator.* tensors that exist in the target model
    model_sd = hypernet.state_dict()
    aggregator_sd = {
        k: v
        for k, v in sd.items()
        if k.startswith("aggregator.") and isinstance(v, torch.Tensor) and k in model_sd
    }

    logger.info(
        "Loading %d aggregator weights from checkpoint: %s",
        len(aggregator_sd),
        checkpoint_path,
    )

    loaded = hypernet.load_state_dict(aggregator_sd, strict=False)
    _assert_transfer_integrity(hypernet, loaded)

    # Freeze all aggregator parameters — only head will be trained
    frozen_count = 0
    trainable_count = 0
    for name, param in hypernet.named_parameters():
        if name.startswith("aggregator."):
            param.requires_grad_(False)
            frozen_count += 1
        else:
            trainable_count += 1

    logger.info(
        "Froze %d aggregator params; %d params (head.*) remain trainable",
        frozen_count,
        trainable_count,
    )

    return hypernet


def get_aggregator_config(checkpoint_path: str | Path) -> Any:
    """Extract the Perceiver aggregator structural config from a checkpoint.

    Reads the aggregator_config from the checkpoint's HypernetConfig so that
    d2l_config.py can populate the aggregator_config=None placeholder.

    Args:
        checkpoint_path: Path to the checkpoint (.bin file).

    Returns:
        The aggregator_config object from the checkpoint's HypernetConfig.

    Raises:
        ValueError: If the checkpoint's aggregator_config is None (predates this field).
    """
    import torch  # noqa: PLC0415

    sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    hc = sd["hypernet_config"]

    if hc.aggregator_config is None:
        msg = (
            "aggregator_config is None in checkpoint — "
            "checkpoint may predate this field"
        )
        raise ValueError(msg)

    return hc.aggregator_config
