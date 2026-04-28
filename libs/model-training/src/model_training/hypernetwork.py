"""DocToLoraHypernetwork: Perceiver-based instant LoRA adapter generation.

Generates rank-8 LoRA adapter weights from token IDs in a single forward pass.
Distinct from the QLoRA gradient-descent path (Phase 21) — this produces adapters
in <1s by cross-attending over token embeddings with learned latents.

IMPORTANT: All GPU imports (torch, safetensors) are deferred inside function/method
bodies per INFRA-05 pattern — this module is importable in CPU-only CI.

Usage:
    from model_training.hypernetwork import (
        DocToLoraHypernetwork,
        save_hypernetwork_adapter,
    )

    model = DocToLoraHypernetwork(input_dim=DEFAULT_VOCAB_SIZE)
    weights = model(token_ids)
    save_hypernetwork_adapter(weights, "/tmp/adapter", "Qwen/Qwen3.5-9B")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

# Default vocabulary size for the hypernetwork input embedding.
# Used when the tokenizer vocabulary size is not explicitly provided.
DEFAULT_VOCAB_SIZE: int = 32000

if TYPE_CHECKING:
    import torch


def _build_hypernetwork_class() -> type:
    """Build and return DocToLoraHypernetwork as a real nn.Module subclass.

    This function is called the first time DocToLoraHypernetwork is instantiated
    so that torch.nn.Module is only imported when needed (INFRA-05 pattern).
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    class _DocToLoraHypernetwork(nn.Module):
        """Perceiver-based hypernetwork that generates LoRA adapter weights.

        Given a sequence of token IDs (from a document/trajectory), produces a
        PEFT-compatible state_dict with lora_A and lora_B matrices for every
        target_module across all transformer layers — in a single forward pass.

        Architecture:
        - Lightweight learned token embedding (not the 7B base model's embedding)
        - Learnable latent array (num_latents x latent_dim)
        - Cross-attention: latents attend over token embeddings
        - Self-attention stack: process latents
        - Linear head: project to all LoRA weight parameters

        Args:
            input_dim: Vocabulary size for the hypernetwork's own token embedding.
            num_latents: Number of learnable latent vectors. Default: 32.
            latent_dim: Dimensionality of each latent vector. Default: 256.
            depth: Number of self-attention layers over latents. Default: 4.
            heads: Number of attention heads. Default: 8.
            rank: LoRA rank for generated adapters. Default: 8.
            target_modules: LoRA target module names. Default: ("q_proj", "v_proj").
            num_layers: Number of transformer layers in target model. Default: 28.
            hidden_dim: Hidden dim of target model. Default: 4096. Used as
                fallback when module_dims is not provided.
            module_dims: Per-module output dimensions for GQA models where
                q_proj and v_proj have different sizes. Maps module name to
                (in_features, out_features). If None, all modules use
                (hidden_dim, hidden_dim).
        """

        def __init__(
            self,
            input_dim: int,
            num_latents: int = 32,
            latent_dim: int = 256,
            depth: int = 4,
            heads: int = 8,
            rank: int = 8,
            target_modules: Sequence[str] = ("q_proj", "v_proj"),
            num_layers: int = 28,
            hidden_dim: int = 4096,
            module_dims: dict[str, tuple[int, int]] | None = None,
        ) -> None:
            super().__init__()

            self.rank = rank
            self.hidden_dim = hidden_dim
            self.target_modules: list[str] = list(target_modules)
            self.num_layers = num_layers
            self.num_latents = num_latents
            self.latent_dim = latent_dim

            # Per-module dimensions: (in_features, out_features)
            # For GQA: q_proj is (hidden, hidden) but v_proj is (hidden, kv_dim)
            if module_dims is not None:
                self.module_dims = module_dims
            else:
                _default_dims = (hidden_dim, hidden_dim)
                self.module_dims = dict.fromkeys(target_modules, _default_dims)

            # Hypernetwork's own lightweight embedding — does NOT load the 7B model
            self.token_embedding = nn.Embedding(input_dim, latent_dim)

            # Learnable latent array: shape (num_latents, latent_dim)
            self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

            # Cross-attention: latents (Q) attend over token embeddings (K, V)
            self.cross_attend = nn.MultiheadAttention(
                latent_dim, heads, batch_first=True
            )

            # Self-attention stack to process latents
            encoder_layer = nn.TransformerEncoderLayer(
                latent_dim, heads, batch_first=True
            )
            self.self_attend = nn.TransformerEncoder(encoder_layer, num_layers=depth)

            # Project flattened latents to all LoRA weight parameters.
            # Each (layer, module) needs lora_A and lora_B with per-module dims.
            total_params = 0
            for m in target_modules:
                in_f, out_f = self.module_dims[m]
                # lora_A: (rank, in_features), lora_B: (out_features, rank)
                total_params += num_layers * (rank * in_f + out_f * rank)
            self.weight_head = nn.Linear(latent_dim * num_latents, total_params)

        def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
            """Generate PEFT-compatible LoRA adapter weights from token IDs.

            Args:
                token_ids: Integer token ID tensor of shape (batch, seq_len).

            Returns:
                Dict mapping PEFT state_dict keys to weight tensors.
                Keys follow:
                  base_model.model.model.layers.{i}.self_attn.{module}.lora_{A|B}.weight
                Returns first batch element (one adapter per trajectory).
            """
            batch_size = token_ids.shape[0]

            # Embed token IDs with hypernetwork's own lightweight embedding
            x = self.token_embedding(token_ids)  # (batch, seq_len, latent_dim)

            # Expand latents for batch dimension
            latents = self.latents.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # (batch, num_latents, latent_dim)

            # Cross-attention: latents attend over token embeddings
            latents, _ = self.cross_attend(
                latents, x, x
            )  # (batch, num_latents, latent_dim)

            # Self-attention stack over latents
            latents = self.self_attend(latents)  # (batch, num_latents, latent_dim)

            # Flatten latents
            latents = latents.reshape(
                batch_size, -1
            )  # (batch, num_latents * latent_dim)

            # Project to all LoRA weight parameters
            weights = self.weight_head(latents)  # (batch, total_weight_params)

            # Reshape into PEFT state_dict for first batch element (index 0)
            state_dict: dict[str, torch.Tensor] = {}
            offset = 0

            for i in range(self.num_layers):
                for module in self.target_modules:
                    in_f, out_f = self.module_dims[module]
                    a_size = self.rank * in_f
                    b_size = out_f * self.rank

                    # lora_A: shape (rank, in_features)
                    key_a = (
                        f"base_model.model.model.layers.{i}"
                        f".self_attn.{module}.lora_A.weight"
                    )
                    state_dict[key_a] = weights[0, offset : offset + a_size].reshape(
                        self.rank, in_f
                    )
                    offset += a_size

                    # lora_B: shape (out_features, rank)
                    key_b = (
                        f"base_model.model.model.layers.{i}"
                        f".self_attn.{module}.lora_B.weight"
                    )
                    state_dict[key_b] = weights[0, offset : offset + b_size].reshape(
                        out_f, self.rank
                    )
                    offset += b_size

            return state_dict

    return _DocToLoraHypernetwork


# ---------------------------------------------------------------------------
# DocToLoraHypernetwork: lazy proxy that builds the real nn.Module class on
# first instantiation. This keeps the module importable without torch.
# ---------------------------------------------------------------------------


class _LazyHypernetworkProxy:
    """Lazy proxy for DocToLoraHypernetwork.

    Acts as a callable that creates a real nn.Module instance on first call.
    The class is rebuilt (and cached) on first instantiation so that torch is
    only imported when actually needed (INFRA-05 pattern).
    """

    _real_class: type | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._real_class is None:
            self.__class__._real_class = _build_hypernetwork_class()
        return self._real_class(*args, **kwargs)  # type: ignore[misc]

    def __instancecheck__(self, instance: object) -> bool:
        if self._real_class is None:
            return False
        return isinstance(instance, self._real_class)


DocToLoraHypernetwork: _LazyHypernetworkProxy = _LazyHypernetworkProxy()


def load_pretrained(
    checkpoint_path: str,
    device: str = "cpu",
    **kwargs: Any,
) -> Any:
    """Load a pretrained DocToLoraHypernetwork from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load onto ('cpu', 'cuda', 'mps'). Default: 'cpu'.
        **kwargs: Override constructor args (input_dim, num_latents, etc.).

    Returns:
        DocToLoraHypernetwork nn.Module loaded with pretrained weights.
    """
    import torch  # noqa: PLC0415

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Extract constructor args from checkpoint or use defaults/overrides
    ctor_args = checkpoint.get("hypernetwork_config", {})
    ctor_args.update(kwargs)
    if "input_dim" not in ctor_args:
        ctor_args["input_dim"] = DEFAULT_VOCAB_SIZE

    model = DocToLoraHypernetwork(**ctor_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def trajectory_to_tokens(
    trajectory_text: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_length: int = 2048,
) -> "torch.Tensor":
    """Encode trajectory text as token IDs for the hypernetwork.

    Uses a simple hash-based tokenization (character trigrams mapped to
    vocab indices). This is intentionally simple — the hypernetwork learns
    its own embedding, so exact tokenization doesn't matter as long as
    it's consistent.

    Args:
        trajectory_text: Text to encode (plan, code diffs, test results, etc.).
        vocab_size: Size of the hypernetwork's embedding vocabulary.
        max_length: Maximum sequence length (truncates or pads).

    Returns:
        Token ID tensor of shape (1, max_length) ready for hypernetwork forward().
    """
    import logging as _logging  # noqa: PLC0415
    import zlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    min_trajectory_chars = 10
    if len(trajectory_text) < min_trajectory_chars:
        _logging.getLogger(__name__).warning(
            "Trajectory text is very short (%d chars < %d minimum). "
            "Generated adapter may be meaningless.",
            len(trajectory_text),
            min_trajectory_chars,
        )

    tokens: list[int] = []
    for i in range(0, len(trajectory_text) - 2):
        trigram = trajectory_text[i : i + 3]
        token_id = zlib.crc32(trigram.encode()) % vocab_size
        tokens.append(token_id)

    # Pad or truncate to max_length
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    else:
        tokens = tokens[:max_length]

    return torch.tensor([tokens], dtype=torch.long)


def generate_adapter(
    hypernetwork: Any,
    trajectory_text: str,
    output_dir: str,
    base_model_id: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_length: int = 2048,
    device: str = "cpu",
) -> str:
    """End-to-end: encode trajectory, run hypernetwork, save adapter.

    Args:
        hypernetwork: A DocToLoraHypernetwork instance.
        trajectory_text: Text to encode into the adapter.
        output_dir: Directory to save the adapter files.
        base_model_id: HuggingFace model ID of the base model.
        vocab_size: Vocabulary size for tokenization.
        max_length: Max token sequence length.
        device: Device for tensor operations.

    Returns:
        Path to the saved adapter directory.
    """
    import torch  # noqa: PLC0415

    # Infer vocab size from hypernetwork's embedding if available
    h_vocab = getattr(
        getattr(hypernetwork, "token_embedding", None), "num_embeddings", None
    )
    effective_vocab = h_vocab if h_vocab is not None else vocab_size
    tokens = trajectory_to_tokens(trajectory_text, effective_vocab, max_length)
    tokens = tokens.to(device)

    with torch.no_grad():
        weights = hypernetwork(tokens)

    rank = getattr(hypernetwork, "rank", 8)
    target_mods = getattr(hypernetwork, "target_modules", ["q_proj", "v_proj"])
    save_hypernetwork_adapter(
        weights,
        output_dir,
        base_model_id,
        rank=rank,
        target_modules=target_mods,
    )
    return output_dir


def save_hypernetwork_adapter(
    weights: dict[str, "torch.Tensor"],
    output_dir: str,
    base_model_id: str,
    rank: int = 8,
    target_modules: list[str] | None = None,
) -> None:
    """Serialize hypernetwork-generated LoRA weights in PEFT adapter format.

    Writes:
    - adapter_model.safetensors: the LoRA weight tensors
    - adapter_config.json: PEFT-compatible configuration

    Args:
        weights: PEFT state_dict from DocToLoraHypernetwork.forward().
        output_dir: Directory to write adapter files to (created if needed).
        base_model_id: HuggingFace model ID of the base model.
        rank: LoRA rank. Default: 8.
        target_modules: List of module names. Default: ["q_proj", "v_proj"].

    Note:
        Does NOT include embed_tokens or lm_head — vLLM rejects these in adapters
        (per Phase 21-01 decision: no modules_to_save in LoraConfig).
    """
    from safetensors.torch import save_file  # noqa: PLC0415

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write adapter weights as safetensors
    safetensors_path = output_path / "adapter_model.safetensors"
    save_file(weights, str(safetensors_path))

    # Write PEFT-compatible adapter_config.json
    adapter_config: dict[str, object] = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank * 2,
        "target_modules": target_modules,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_id,
        "inference_mode": True,
        "modules_to_save": None,
        "fan_in_fan_out": False,
    }

    config_path = output_path / "adapter_config.json"
    config_path.write_text(json.dumps(adapter_config, indent=2))
