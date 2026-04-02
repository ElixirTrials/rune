"""Architecture probe and activation extraction for hypernetwork training.

Discovers standard attention layers (those with q_proj/k_proj/v_proj/o_proj
children) via model.named_modules(), caches results to JSON, and provides
extract_activations_with_model() that accepts a pre-loaded model and tokenizer.

Phase 26 purpose: eliminate hidden_size placeholders and per-call model loading.
The probe becomes the single source of truth for layer indices and projection
dimensions across the v7.0 pipeline.

All heavy GPU imports (torch, transformers) are deferred to function bodies
per INFRA-05 project convention.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ATTN_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj"}
GDN_PROJECTIONS = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
MLP_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}
ALL_KNOWN_PROJECTIONS = ATTN_PROJECTIONS | GDN_PROJECTIONS | MLP_PROJECTIONS
PROBE_CACHE_DIR = Path.home() / ".cache" / "rune" / "probes"
QWEN3_NEXT_CANONICAL_NAME = "qwen3-coder-next"

__all__ = [
    "probe_model",
    "discover_target_modules",
    "extract_activations_with_model",
    "load_probe_cache",
    "save_probe_cache",
    "QWEN3_NEXT_CANONICAL_NAME",
    "ALL_KNOWN_PROJECTIONS",
]


def _model_name_to_cache_path(model_name: str) -> Path:
    """Map model name to its JSON cache file path.

    Uses SHA-256 truncated to 16 hex chars for the filename.

    Args:
        model_name: Canonical model identifier string.

    Returns:
        Path under PROBE_CACHE_DIR.
    """
    h = hashlib.sha256(model_name.encode()).hexdigest()[:16]
    return PROBE_CACHE_DIR / f"{h}.json"


def discover_target_modules(model: Any) -> list[str]:
    """Discover all LoRA-targetable projection modules in a model.

    Iterates model.named_modules() and collects the short names of any
    Linear children whose names match known projection patterns (attention,
    GDN, or MLP). Returns a sorted, deduplicated list.

    Args:
        model: Any nn.Module (typically a transformer model).

    Returns:
        Sorted list of unique projection names found (e.g.
        ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", ...]).
    """
    import torch.nn as nn  # noqa: PLC0415

    found: set[str] = set()
    for _name, module in model.named_modules():
        for child_name, child in module.named_children():
            short = child_name.split(".")[-1]
            if short in ALL_KNOWN_PROJECTIONS and isinstance(child, nn.Linear):
                found.add(short)
    return sorted(found)


def probe_model(model: Any) -> dict[str, Any]:
    """Probe a model's architecture to discover attention layers and modules.

    Iterates model.named_modules() to find layers that have all four attention
    projection children (q_proj, k_proj, v_proj, o_proj). DeltaNet and other
    linear-attention layers that lack these projections are skipped.

    For each discovered attention layer, captures the in/out dimensions of
    q_proj, k_proj, v_proj, and o_proj weights.

    Also discovers all LoRA-targetable projection modules (attention, GDN,
    MLP) via discover_target_modules().

    Args:
        model: Any nn.Module (typically a transformer model).

    Returns:
        Dict with keys:
            - attention_layer_indices: sorted list of int layer indices
            - feature_sizes: dict mapping projection name to {"in": int, "out": int}
            - target_modules: sorted list of discovered projection names
    """
    attention_layer_indices: list[int] = []
    feature_sizes: dict[str, dict[str, int]] = {}

    for name, module in model.named_modules():
        # Get immediate children names
        child_names = {n.split(".")[-1] for n, _ in module.named_children()}
        if not ATTN_PROJECTIONS.issubset(child_names):
            continue

        # Extract layer index from the last numeric segment in dotted name
        parts = name.split(".")
        layer_idx: int | None = None
        for part in reversed(parts):
            if part.isdigit():
                layer_idx = int(part)
                break
        if layer_idx is None:
            continue

        attention_layer_indices.append(layer_idx)

        # Capture projection dimensions (out_f, in_f = weight.shape)
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            proj = getattr(module, proj_name, None)
            if proj is None or not hasattr(proj, "weight"):
                continue
            out_f, in_f = proj.weight.shape
            if proj_name not in feature_sizes:
                feature_sizes[proj_name] = {"in": in_f, "out": out_f}

    return {
        "attention_layer_indices": sorted(attention_layer_indices),
        "feature_sizes": feature_sizes,
        "target_modules": discover_target_modules(model),
    }


def save_probe_cache(model_name: str, probe_result: dict[str, Any]) -> Path:
    """Persist probe results to JSON cache.

    Adds metadata fields (model_name, model_name_hash, probed_at) to a copy
    of probe_result. Creates PROBE_CACHE_DIR if it does not exist.

    Args:
        model_name: Canonical model identifier (used for cache lookup key).
        probe_result: Output from probe_model().

    Returns:
        Path to the written JSON file.
    """
    cache_path = _model_name_to_cache_path(model_name)
    PROBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = dict(probe_result)
    data["model_name"] = model_name
    data["model_name_hash"] = hashlib.sha256(model_name.encode()).hexdigest()[:16]
    data["probed_at"] = datetime.now(timezone.utc).isoformat()

    cache_path.write_text(json.dumps(data, indent=2))
    logger.info("Probe cache saved: %s → %s", model_name, cache_path)
    return cache_path


def load_probe_cache(model_name: str) -> dict[str, Any] | None:
    """Load probe results from JSON cache.

    Args:
        model_name: Canonical model identifier.

    Returns:
        Probe result dict (including metadata fields) if cached, else None.
        Never raises — returns None on any miss.
    """
    cache_path = _model_name_to_cache_path(model_name)
    if not cache_path.exists():
        logger.debug("Probe cache miss for '%s' (path: %s)", model_name, cache_path)
        return None

    data: dict[str, Any] = json.loads(cache_path.read_text())
    logger.debug("Probe cache hit for '%s'", model_name)
    return data


def extract_activations_with_model(
    text: str,
    model: Any,
    tokenizer: Any,
    layer_indices: list[int] | None = None,
    model_name: str | None = None,
    max_length: int = 512,
) -> tuple[Any, Any]:
    """Extract per-layer hidden state activations from a pre-loaded model.

    Runs text through the model with output_hidden_states=True and stacks
    activations from the specified layer indices. Uses hidden_states[i] directly
    (no +1 offset) — consistent with existing sakana_d2l.py convention.

    Args:
        text: Input text to tokenize and process.
        model: Pre-loaded nn.Module in eval mode.
        tokenizer: Pre-loaded tokenizer.
        layer_indices: Which hidden state indices to extract. If None, loads
            from probe cache via model_name.
        model_name: Canonical model name for cache lookup (required when
            layer_indices is None).
        max_length: Max token sequence length.

    Returns:
        Tuple of (features, attention_mask):
            features shape: (1, num_layers, seq_len, hidden_dim)
            attention_mask shape: (1, seq_len)

    Raises:
        RuntimeError: If layer_indices is None and no probe cache exists for model_name.
    """
    import torch  # noqa: PLC0415

    if layer_indices is None:
        if model_name is None:
            msg = (
                "layer_indices is None and model_name is None — "
                "cannot load from probe cache without a model name."
            )
            raise RuntimeError(msg)
        cache = load_probe_cache(model_name)
        if cache is None:
            msg = (
                f"layer_indices is None but no probe cache found for '{model_name}'. "
                "Run probe_model() and save_probe_cache() first."
            )
            raise RuntimeError(msg)
        layer_indices = cache["attention_layer_indices"]

    # Determine device from model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    # Stack selected layers: (batch, num_layers, seq_len, hidden_dim)
    selected = torch.stack([hidden_states[i] for i in layer_indices], dim=1)
    attention_mask = inputs["attention_mask"]

    logger.info(
        "Extracted activations: %s from %d layers", selected.shape, len(layer_indices)
    )
    return selected, attention_mask
