"""SakanaAI Doc-to-LoRA integration.

Wraps Sakana's pretrained HyperLoRA perceiver so it can be used through
our hypernetwork interface (load_pretrained → generate_adapter).

The Sakana hypernetwork takes per-layer activations from a base model as
input and produces LoRA adapter weights.  This module handles:
  - Downloading the checkpoint from HuggingFace
  - Patching flash-attention assertions for CPU/MPS/non-flash environments
  - Extracting per-layer activations from the base model
  - Saving the generated LoRA weights in PEFT format

GPU imports are deferred inside function bodies per INFRA-05 pattern.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# HuggingFace repo for Sakana's pretrained checkpoints
HF_REPO_ID = "SakanaAI/doc-to-lora"
# Available checkpoints: gemma_2b_d2l, gemma_demo, mistral_7b_d2l, qwen_4b_d2l
DEFAULT_VARIANT = "gemma_demo"
DEFAULT_HF_FILENAME = f"{DEFAULT_VARIANT}/checkpoint-80000/pytorch_model.bin"
LOCAL_CACHE_DIR = Path.home() / ".cache" / "rune" / "sakana_d2l"


def _patch_flash_attention() -> None:
    """Patch Sakana's idefics2 module to work without flash_attn.

    Replaces flash attention classes and assertions with eager equivalents
    so the perceiver can run on CPU/MPS/CUDA without flash_attn installed.
    """
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    # Install a stub flash_attn module so transformers' import check
    # succeeds.  The actual forward pass uses our patched eager path,
    # so the real package is never called.
    if "flash_attn" not in sys.modules:
        import importlib.machinery  # noqa: PLC0415

        stub = types.ModuleType("flash_attn")
        stub.__version__ = "2.6.3"  # type: ignore[attr-defined]  # satisfies version checks
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

    import ctx_to_lora.modeling.idefics2 as idefics2_mod  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from ctx_to_lora.modeling.idefics2 import (  # noqa: PLC0415
        Idefics2Perceiver,
        Idefics2PerceiverAttention,
        Idefics2PerceiverConfig,
        Idefics2PerceiverResampler,
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

    # Patch resampler __init__ to bypass flash_attention_2 assertion
    _orig_resampler_init = Idefics2PerceiverResampler.__init__

    def _patched_resampler_init(self: Any, config: Any) -> None:
        config._attn_implementation = "flash_attention_2"
        _orig_resampler_init(self, config)
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
    """Download Sakana's pretrained checkpoint from HuggingFace.

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
        logger.info("Using cached Sakana checkpoint: %s", cached)
        return cached

    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    logger.info("Downloading Sakana checkpoint %s from %s...", variant, HF_REPO_ID)
    downloaded = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=hf_filename))

    cached.parent.mkdir(parents=True, exist_ok=True)
    import shutil  # noqa: PLC0415

    shutil.copy2(downloaded, cached)
    logger.info("Cached to: %s", cached)
    return cached


def load_sakana_checkpoint(
    checkpoint_path: str | Path | None = None,
    variant: str = DEFAULT_VARIANT,
    device: str = "cpu",
) -> tuple[Any, Any]:
    """Load Sakana's HyperLoRA perceiver from checkpoint.

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

    logger.info("Loading Sakana checkpoint: %s", checkpoint_path)
    sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

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
    hypernet_dtype = resolve_model_dtype(
        param_count=hypernet_param_count,
        device=device,
    )
    logger.info("HyperLoRA dtype resolved to %s", hypernet_dtype)
    hypernet = HyperLoRA(hc).to(hypernet_dtype)

    # Load ALL hypernet weights from checkpoint (not just a prefix subset).
    # The checkpoint contains aggregator.*, head.*, scaler_{A,B}.*,
    # bias_{A,B}.*, and layers.0.* — all are required for correct
    # adapter generation.  scaler_B in particular defaults to zeros,
    # so skipping it zeroes out every lora_B matrix.
    model_keys = set(hypernet.state_dict().keys())
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
    aggregator.* weights from a Sakana checkpoint. Raises AssertionError on any
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
    """Load aggregator weights from a Sakana checkpoint into a HyperLoRA instance.

    Loads only aggregator.* weights from the checkpoint (not head.*), freezes all
    aggregator parameters (requires_grad=False), and leaves head.* at PyTorch default
    initialization for Phase 29 training against the new target model.

    This enables reuse of the pretrained Perceiver aggregator across different target
    model architectures. The aggregator maps document embeddings to LoRA weight space
    and is model-agnostic; only the head needs retraining per target model.

    Args:
        hypernet: The HyperLoRA model to load weights into (mutated in-place).
        checkpoint_path: Path to the Sakana checkpoint (.bin file).

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
    """Extract the Perceiver aggregator structural config from a Sakana checkpoint.

    Reads the aggregator_config from the checkpoint's HypernetConfig so that
    d2l_config.py can populate the aggregator_config=None placeholder set in Phase 25.

    Args:
        checkpoint_path: Path to the Sakana checkpoint (.bin file).

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


def extract_activations(
    text: str,
    base_model_name: str,
    layer_indices: list[int],
    device: str = "cpu",
    max_length: int = 512,
) -> tuple[Any, Any]:
    """Extract per-layer hidden state activations from the base model.

    Backward-compatible wrapper around extract_activations_with_model().
    Loads model and tokenizer, delegates extraction, then cleans up.

    Args:
        text: Input text to process.
        base_model_name: HuggingFace model ID for the base model.
        layer_indices: Which layers to extract activations from.
        device: Device for computation.
        max_length: Max token sequence length.

    Returns:
        Tuple of (features, attention_mask) ready for HyperLoRA.
        features shape: (1, num_layers, seq_len, hidden_dim)
        attention_mask shape: (1, seq_len)
    """
    import torch  # noqa: PLC0415
    from shared.hardware import resolve_model_dtype  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from model_training.d2l_probe import extract_activations_with_model  # noqa: PLC0415

    logger.info("Loading base model %s for activation extraction...", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Estimate param count from model config to resolve dtype
    from transformers import AutoConfig  # noqa: PLC0415

    config = AutoConfig.from_pretrained(base_model_name)
    estimated_params = getattr(config, "num_parameters", None)
    if estimated_params is None:
        # Rough estimate: vocab_size * hidden + num_layers * 4 * hidden^2
        vocab = getattr(config, "vocab_size", 256000)
        hidden = getattr(config, "hidden_size", 2304)
        n_layers = getattr(config, "num_hidden_layers", 26)
        estimated_params = vocab * hidden + n_layers * 4 * hidden * hidden

    # Account for inference model already on GPU as overhead
    overhead = 0
    if device != "cpu" and torch.cuda.is_available():
        overhead = torch.cuda.memory_allocated(0)

    activation_dtype = resolve_model_dtype(
        param_count=estimated_params,
        device=device,
        overhead_bytes=overhead,
    )
    logger.info("Activation extraction dtype resolved to %s", activation_dtype)

    model: Any = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=activation_dtype,
    )
    model = model.to(device)  # type: ignore[assignment]
    model.eval()

    result = extract_activations_with_model(
        text=text,
        model=model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
        max_length=max_length,
    )

    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return result


def generate_adapter_from_sakana(
    text: str,
    output_dir: str,
    checkpoint_path: str | Path | None = None,
    variant: str = DEFAULT_VARIANT,
    base_model_name: str | None = None,
    device: str = "cpu",
    max_length: int = 512,
    scaling_factor: float = 0.16,
) -> str:
    """End-to-end: text → base model activations → HyperLoRA → PEFT adapter.

    This is the main entry point. It:
      1. Loads the Sakana pretrained perceiver (downloading if needed)
      2. Runs text through the base model to get per-layer activations
      3. Feeds activations through the perceiver to generate LoRA weights
      4. Saves weights in PEFT-compatible format

    Args:
        text: Input text (trajectory, document, context) to encode.
        output_dir: Directory to save the PEFT adapter files.
        checkpoint_path: Path to local checkpoint, or None to download.
        variant: HF checkpoint variant if downloading.
        base_model_name: Override base model. If None, uses the one from checkpoint.
        device: Device for computation.
        max_length: Maximum token sequence length for activation extraction.
        scaling_factor: Adapter scaling multiplier (0-1, default from config).

    Returns:
        Path to the saved adapter directory.
    """
    import torch  # noqa: PLC0415

    hypernet, hc = load_sakana_checkpoint(checkpoint_path, variant, device)

    # Determine base model from checkpoint config
    if base_model_name is None:
        # Load checkpoint just to read base_model_name_or_path
        if checkpoint_path is None:
            checkpoint_path = download_checkpoint(variant)
        sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        base_model_name = sd["base_model_name_or_path"]
        del sd

    logger.info("Base model: %s", base_model_name)

    # Extract activations from base model
    layer_indices = list(hc.layer_indices)
    features, attn_mask = extract_activations(
        text=text,
        base_model_name=base_model_name,
        layer_indices=layer_indices,
        device=device,
        max_length=max_length,
    )

    # Generate LoRA weights via perceiver
    logger.info("Generating LoRA weights via HyperLoRA perceiver...")
    with torch.no_grad():
        lora_dict, layernorm_dict = hypernet.generate_weights(features, attn_mask, None)

    # Combine generated weights with bias — Sakana concatenates bias as extra
    # rank dimensions (rank 8 → 16 for single chunk). This is how the
    # checkpoint was trained and evaluated.
    from ctx_to_lora.modeling.lora_merger import combine_lora as _combine_lora

    n_chunks = torch.ones(1, dtype=torch.int32)
    lora_bias = (
        hypernet.get_head_bias() if hypernet.config.use_bias else None
    )
    lora_dict = _combine_lora(lora_dict, n_chunks, lora_bias=lora_bias)

    # Save as PEFT adapter
    _save_sakana_adapter(
        lora_dict=lora_dict,
        output_dir=output_dir,
        base_model_name=base_model_name,
        hc=hc,
        scaling_factor=scaling_factor,
    )

    # Free hypernet VRAM — it's not needed after adapter weights are saved
    del hypernet, lora_dict, layernorm_dict, features, attn_mask
    if device != "cpu":
        torch.cuda.empty_cache()

    return output_dir


def _save_sakana_adapter(
    lora_dict: dict[str, dict[str, Any]],
    output_dir: str,
    base_model_name: str,
    hc: Any,
    scaling_factor: float = 0.16,
) -> None:
    """Save Sakana's HyperLoRA output as a PEFT-compatible adapter.

    Converts from Sakana's format (dict of module → {A, B} with batch/layer dims)
    to PEFT's flat state_dict format.

    Args:
        lora_dict: Output from HyperLoRA.generate_weights().
        output_dir: Directory to write adapter files.
        base_model_name: Base model identifier for adapter config.
        hc: HypernetConfig with lora rank and target modules.
        scaling_factor: Multiplier for adapter influence strength (0-1).
    """
    from safetensors.torch import save_file  # noqa: PLC0415

    output_path = Path(output_dir)
    safetensors_file = output_path / "adapter_model.safetensors"
    if safetensors_file.exists():
        raise FileExistsError(
            f"Adapter already exists at {safetensors_file}; refusing to overwrite"
        )
    output_path.mkdir(parents=True, exist_ok=True)

    layer_indices = list(hc.layer_indices)
    target_modules = list(hc.lora_config.target_modules)

    # Determine actual rank from the combined weights (may be 2*base_rank
    # after combine_lora concatenates bias as extra rank dimensions).
    first_mod = next(iter(lora_dict))
    actual_rank = lora_dict[first_mod]["A"].shape[-2]

    # Module path prefix: attention modules use self_attn, MLP uses mlp
    _attn_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}

    # Convert Sakana format → PEFT flat state_dict
    # Sakana: lora_dict[module_name]["A"] shape (batch, num_layers, rank, in)
    # PEFT: .layers.{i}.{prefix}.{module}.lora_A.weight (rank, in_features)
    state_dict = {}
    for mod_name, weights in lora_dict.items():
        if mod_name not in target_modules:
            continue
        a_weights = weights["A"]  # (batch, num_layers, rank, in_features)
        b_weights = weights["B"]  # (batch, num_layers, rank, out_features)
        prefix = "self_attn" if mod_name in _attn_modules else "mlp"

        for layer_pos, layer_idx in enumerate(layer_indices):
            key_a = (
                f"base_model.model.model.layers.{layer_idx}"
                f".{prefix}.{mod_name}.lora_A.weight"
            )
            key_b = (
                f"base_model.model.model.layers.{layer_idx}"
                f".{prefix}.{mod_name}.lora_B.weight"
            )
            # Take first batch element, transpose B to (out_features, rank)
            state_dict[key_a] = a_weights[0, layer_pos].contiguous()
            state_dict[key_b] = b_weights[0, layer_pos].t().contiguous()

    save_file(state_dict, str(output_path / "adapter_model.safetensors"))

    # Sakana's lora_forward uses scaling = lora_alpha directly (not alpha/r
    # as PEFT does). To compensate, set PEFT lora_alpha = checkpoint_alpha *
    # actual_rank so that PEFT's alpha/r division recovers the original
    # scaling factor.
    checkpoint_alpha = getattr(
        hc.lora_config, "lora_alpha", hc.lora_config.r * 2
    ) if hc is not None else actual_rank * 2
    peft_alpha = checkpoint_alpha * actual_rank * scaling_factor

    adapter_config = {
        "peft_type": "LORA",
        "r": actual_rank,
        "lora_alpha": peft_alpha,
        "target_modules": target_modules,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_name,
        "inference_mode": True,
        "modules_to_save": None,
        "fan_in_fan_out": False,
    }
    config_json = json.dumps(adapter_config, indent=2)
    (output_path / "adapter_config.json").write_text(config_json)

    logger.info(
        "Saved PEFT adapter: %d tensors, %d layers, rank=%d, targets=%s",
        len(state_dict),
        len(layer_indices),
        actual_rank,
        target_modules,
    )
