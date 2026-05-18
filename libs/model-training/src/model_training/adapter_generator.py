"""Adapter generation with optional ModelPool support.

Replaces the public API of sakana_d2l.py for adapter generation.
Supports two modes:

- Pool mode (fast): borrows resident models from a ModelPool — no load/unload
  per call.  Pass ``pool=<ModelPool instance>``.
- Standalone mode (backwards compat): loads/unloads per call, exactly like
  generate_adapter_from_sakana() in sakana_d2l.py.  Omit ``pool``.

GPU imports are deferred inside function bodies per INFRA-05 pattern.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from model_training.model_pool import ModelPool

logger = logging.getLogger(__name__)


def extract_activations(
    text: str,
    base_model_name: str,
    layer_indices: list[int],
    device: str = "cpu",
    max_length: int = 512,
) -> tuple[Any, Any]:
    """Extract per-layer hidden state activations from the base model.

    Backwards-compatible standalone wrapper around
    ``d2l_probe.extract_activations_with_model()``. Loads the model and
    tokenizer, delegates extraction, then cleans up GPU memory.

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

    Raises:
        ValueError: If text is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError(
            "extract_activations called with empty text; adapter would be meaningless."
        )

    import torch  # noqa: PLC0415
    from shared.hardware import resolve_model_dtype  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from model_training.d2l_probe import extract_activations_with_model  # noqa: PLC0415

    logger.info("Loading base model %s for activation extraction...", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    config = AutoConfig.from_pretrained(base_model_name)
    estimated_params = getattr(config, "num_parameters", None)
    if estimated_params is None:
        vocab = getattr(config, "vocab_size", 256000)
        hidden = getattr(config, "hidden_size", 2304)
        n_layers = getattr(config, "num_hidden_layers", 26)
        estimated_params = vocab * hidden + n_layers * 4 * hidden * hidden

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


def _save_adapter(
    lora_dict: dict[str, dict[str, Any]],
    output_dir: str,
    base_model_name: str,
    hc: Any,
    scaling_factor: float = 0.16,
) -> None:
    """Save HyperLoRA output as a PEFT-compatible adapter.

    Converts from the HyperLoRA format (dict of module → {A, B} with
    batch/layer dims) to PEFT's flat state_dict format, then writes
    ``adapter_model.safetensors`` and ``adapter_config.json``.

    Args:
        lora_dict: Output from HyperLoRA.generate_weights() (after combine_lora).
        output_dir: Directory to write adapter files.
        base_model_name: Base model identifier written into adapter_config.json.
        hc: HypernetConfig with lora rank and target modules.
        scaling_factor: Multiplier for adapter influence strength (0–1).
    """
    from safetensors.torch import save_file  # noqa: PLC0415

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    layer_indices = list(hc.layer_indices)
    target_modules = list(hc.lora_config.target_modules)

    first_mod = next(iter(lora_dict))
    actual_rank = lora_dict[first_mod]["A"].shape[-2]

    _attn_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}

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
            state_dict[key_a] = a_weights[0, layer_pos].contiguous()
            state_dict[key_b] = b_weights[0, layer_pos].t().contiguous()

    save_file(state_dict, str(output_path / "adapter_model.safetensors"))

    checkpoint_alpha = (
        getattr(hc.lora_config, "lora_alpha", hc.lora_config.r * 2)
        if hc is not None
        else actual_rank * 2
    )
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


def generate_adapter(
    text: str,
    output_dir: str,
    pool: "ModelPool | None" = None,
    *,
    checkpoint_path: str | Path | None = None,
    base_model_name: str | None = None,
    variant: str = "gemma_demo",
    device: str = "cpu",
    max_length: int = 512,
    scaling_factor: float = 0.16,
) -> str:
    """Generate a PEFT adapter from text using either pool mode or standalone mode.

    Pool mode (``pool`` is not None): borrows resident models from the
    provided ModelPool — no load/unload per call.  Transient tensors
    (lora_dict, features, etc.) are freed after save but the pool-owned
    model and hypernetwork are left resident.

    Standalone mode (``pool`` is None): loads the hypernetwork from
    ``checkpoint_path``, extracts base-model activations, generates adapter
    weights, saves them, then frees all GPU memory — identical behaviour to
    the legacy ``generate_adapter_from_sakana()``.

    Args:
        text: Input text (trajectory, document, context) to encode.
        output_dir: Directory to save the PEFT adapter files.
        pool: Optional ModelPool that owns resident base-model and
            hypernetwork tensors.  When provided, pool mode is used.
        checkpoint_path: Path to local hypernetwork checkpoint, or None to
            download from HuggingFace.  Ignored in pool mode.
        base_model_name: Override base model name.  If None in standalone
            mode, reads ``base_model_name_or_path`` from the checkpoint.
            Ignored in pool mode (pool owns the model).
        variant: HuggingFace checkpoint variant used when downloading.
            Ignored in pool mode.
        device: Device for computation.  Ignored in pool mode (pool device
            takes precedence).
        max_length: Maximum token sequence length for activation extraction.
        scaling_factor: Adapter scaling multiplier (0–1).

    Returns:
        Path to the saved adapter directory (same as ``output_dir``).
    """
    if pool is not None:
        return _generate_adapter_pool(
            text=text,
            output_dir=output_dir,
            pool=pool,
            max_length=max_length,
            scaling_factor=scaling_factor,
        )

    return _generate_adapter_standalone(
        text=text,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        base_model_name=base_model_name,
        variant=variant,
        device=device,
        max_length=max_length,
        scaling_factor=scaling_factor,
    )


def _generate_adapter_pool(
    text: str,
    output_dir: str,
    pool: "ModelPool",
    max_length: int,
    scaling_factor: float,
) -> str:
    """Pool-mode adapter generation — borrows models from pool, no cleanup."""
    import torch  # noqa: PLC0415

    from model_training.d2l_probe import extract_activations_with_model  # noqa: PLC0415

    hypernet, hc = pool.hypernetwork()
    model, tokenizer = pool.base_model()

    features, attn_mask = extract_activations_with_model(
        text=text,
        model=model,
        tokenizer=tokenizer,
        layer_indices=list(hc.layer_indices),
        max_length=max_length,
    )

    logger.info("Generating LoRA weights via HyperLoRA perceiver (pool mode)...")
    with torch.no_grad():
        lora_dict, layernorm_dict = hypernet.generate_weights(features, attn_mask, None)

    from ctx_to_lora.modeling.lora_merger import (
        combine_lora as _combine_lora,  # noqa: PLC0415
    )

    n_chunks = torch.ones(1, dtype=torch.int32)
    lora_bias = hypernet.get_head_bias() if hypernet.config.use_bias else None
    lora_dict = _combine_lora(lora_dict, n_chunks, lora_bias=lora_bias)

    # Determine base_model_name from pool's model_name
    _save_adapter(
        lora_dict=lora_dict,
        output_dir=output_dir,
        base_model_name=pool.model_name,
        hc=hc,
        scaling_factor=scaling_factor,
    )

    del lora_dict, layernorm_dict, features, attn_mask
    return output_dir


def _generate_adapter_standalone(
    text: str,
    output_dir: str,
    checkpoint_path: str | Path | None,
    base_model_name: str | None,
    variant: str,
    device: str,
    max_length: int,
    scaling_factor: float,
) -> str:
    """Standalone-mode adapter generation — loads/unloads per call."""
    import torch  # noqa: PLC0415

    from model_training.hypernetwork import load_hypernetwork  # noqa: PLC0415

    hypernet, hc = load_hypernetwork(checkpoint_path, variant, device)

    if base_model_name is None:
        if checkpoint_path is None:
            from model_training.hypernetwork import download_checkpoint  # noqa: PLC0415

            checkpoint_path = download_checkpoint(variant)
        from model_training.hypernetwork import _open_checkpoint  # noqa: PLC0415

        sd = _open_checkpoint(str(checkpoint_path))
        base_model_name = sd["base_model_name_or_path"]
        del sd

    logger.info("Base model: %s", base_model_name)

    layer_indices = list(hc.layer_indices)
    features, attn_mask = extract_activations(
        text=text,
        base_model_name=base_model_name,
        layer_indices=layer_indices,
        device=device,
        max_length=max_length,
    )

    logger.info("Generating LoRA weights via HyperLoRA perceiver (standalone mode)...")
    with torch.no_grad():
        lora_dict, layernorm_dict = hypernet.generate_weights(features, attn_mask, None)

    from ctx_to_lora.modeling.lora_merger import (
        combine_lora as _combine_lora,  # noqa: PLC0415
    )

    n_chunks = torch.ones(1, dtype=torch.int32)
    lora_bias = hypernet.get_head_bias() if hypernet.config.use_bias else None
    lora_dict = _combine_lora(lora_dict, n_chunks, lora_bias=lora_bias)

    _save_adapter(
        lora_dict=lora_dict,
        output_dir=output_dir,
        base_model_name=base_model_name,
        hc=hc,
        scaling_factor=scaling_factor,
    )

    del hypernet, lora_dict, layernorm_dict, features, attn_mask
    if device != "cpu":
        torch.cuda.empty_cache()

    return output_dir
