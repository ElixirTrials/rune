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
            bsz = torch.where(position_ids == 0, 1, 0).sum().item()
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

    hypernet = HyperLoRA(hc).to(torch.float32)

    # Load perceiver/aggregator/head weights from checkpoint
    hypernet_keys = [
        k for k in sd.keys() if k.startswith(("aggregator.", "head.", "extra_"))
    ]
    hypernet_sd = {k: sd[k] for k in hypernet_keys if k in hypernet.state_dict()}

    if hypernet_sd:
        loaded = hypernet.load_state_dict(hypernet_sd, strict=False)
        logger.info("Loaded %d perceiver weight tensors", len(hypernet_sd))
        if loaded.missing_keys:
            logger.info("Missing keys: %d", len(loaded.missing_keys))
    else:
        full_state = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
        loaded = hypernet.load_state_dict(full_state, strict=False)
        logger.info(
            "Loaded full state: missing=%d, unexpected=%d",
            len(loaded.missing_keys),
            len(loaded.unexpected_keys),
        )

    hypernet = hypernet.to(device)
    hypernet.eval()

    param_count = sum(p.numel() for p in hypernet.parameters())
    logger.info("HyperLoRA params: %d", param_count)

    return hypernet, hc


def extract_activations(
    text: str,
    base_model_name: str,
    layer_indices: list[int],
    device: str = "cpu",
    max_length: int = 512,
) -> tuple[Any, Any]:
    """Extract per-layer hidden state activations from the base model.

    Runs the input text through the base model and captures hidden states
    at the specified layer indices. These activations are what Sakana's
    HyperLoRA perceiver expects as input.

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
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    logger.info("Loading base model %s for activation extraction...", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        output_hidden_states=True,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.hidden_states is a tuple of (num_layers+1,) tensors
    # each of shape (batch, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states

    # Stack selected layers: (batch, num_layers, seq_len, hidden_dim)
    selected = torch.stack([hidden_states[i] for i in layer_indices], dim=1)
    attention_mask = inputs["attention_mask"]

    logger.info(
        "Extracted activations: %s from %d layers", selected.shape, len(layer_indices)
    )

    # Free base model memory
    del model
    if device != "cpu":
        torch.cuda.empty_cache() if "cuda" in device else None

    return selected, attention_mask


def generate_adapter_from_sakana(
    text: str,
    output_dir: str,
    checkpoint_path: str | Path | None = None,
    variant: str = DEFAULT_VARIANT,
    base_model_name: str | None = None,
    device: str = "cpu",
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
        max_length=512,
    )

    # Generate LoRA weights via perceiver
    logger.info("Generating LoRA weights via HyperLoRA perceiver...")
    with torch.no_grad():
        lora_dict, layernorm_dict = hypernet.generate_weights(features, attn_mask, None)

    # Save as PEFT adapter
    _save_sakana_adapter(
        lora_dict=lora_dict,
        output_dir=output_dir,
        base_model_name=base_model_name,
        hc=hc,
    )

    return output_dir


def _save_sakana_adapter(
    lora_dict: dict[str, dict[str, Any]],
    output_dir: str,
    base_model_name: str,
    hc: Any,
) -> None:
    """Save Sakana's HyperLoRA output as a PEFT-compatible adapter.

    Converts from Sakana's format (dict of module → {A, B} with batch/layer dims)
    to PEFT's flat state_dict format.

    Args:
        lora_dict: Output from HyperLoRA.generate_weights().
        output_dir: Directory to write adapter files.
        base_model_name: Base model identifier for adapter config.
        hc: HypernetConfig with lora rank and target modules.
    """
    from safetensors.torch import save_file  # noqa: PLC0415

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    layer_indices = list(hc.layer_indices)
    target_modules = list(hc.lora_config.target_modules)
    rank = hc.lora_config.r

    # Convert Sakana format → PEFT flat state_dict
    # Sakana: lora_dict[module_name]["A"] shape (batch, num_layers, rank, in)
    # PEFT: .layers.{i}.mlp.{module}.lora_A.weight (rank, in_features)
    state_dict = {}
    for mod_name, weights in lora_dict.items():
        if mod_name not in target_modules:
            continue
        a_weights = weights["A"]  # (batch, num_layers, rank, in_features)
        b_weights = weights["B"]  # (batch, num_layers, rank, out_features)

        for layer_pos, layer_idx in enumerate(layer_indices):
            key_a = (
                f"base_model.model.model.layers.{layer_idx}"
                f".mlp.{mod_name}.lora_A.weight"
            )
            key_b = (
                f"base_model.model.model.layers.{layer_idx}"
                f".mlp.{mod_name}.lora_B.weight"
            )
            # Take first batch element, transpose B to (out_features, rank)
            state_dict[key_a] = a_weights[0, layer_pos].contiguous()
            state_dict[key_b] = b_weights[0, layer_pos].t().contiguous()

    save_file(state_dict, str(output_path / "adapter_model.safetensors"))

    adapter_config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank * 2,
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
        rank,
        target_modules,
    )
