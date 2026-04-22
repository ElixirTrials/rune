"""Extract per-module LoRA A/B matrices from PEFT state_dicts + adapter dirs.

Mirrors the key-parse logic in T2L's
``hyper_llm_modulator/data.py::get_recon_train_data``. PEFT stores A as
``(rank, in_features)`` and B as ``(out_features, rank)`` per layer; we stack
along a new layer axis (dim 0) without transposing.

Torch is imported inside function bodies (INFRA-05).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Example PEFT keys this regex must match:
#   base_model.model.model.layers.12.self_attn.q_proj.lora_A.weight
#   base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight
_KEY_RE = re.compile(
    r"^base_model\.model\.model\.layers\."
    r"(?P<layer>\d+)\."
    r"(?P<prefix>[^.]+)\."
    r"(?P<module>[^.]+)\."
    r"lora_(?P<ab>[AB])\.weight$"
)


def extract_lora_ab_from_state_dict(
    state_dict: dict[str, Any],
    *,
    target_modules: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Return per-module stacked A/B tensors and sorted layer indices.

    Shape: ``{module: {"A": Tensor[L, r, in], "B": Tensor[L, out, r],
    "layer_indices": LongTensor[L]}}``.

    Args:
        state_dict: Dict mapping PEFT key → tensor. Both ``torch.Tensor``
            and safetensors-loaded tensors work.
        target_modules: Module names to extract. Must appear at the module
            slot of the PEFT key (e.g. ``"q_proj"``, ``"gate_proj"``).

    Raises:
        ValueError: If any requested module has zero matching keys, or if
            the A and B layer sets disagree for a module.
    """
    import torch  # noqa: PLC0415

    # Collect A/B per (module, layer) first, then assemble stacks.
    a_by_mod_layer: dict[str, dict[int, torch.Tensor]] = {m: {} for m in target_modules}
    b_by_mod_layer: dict[str, dict[int, torch.Tensor]] = {m: {} for m in target_modules}

    for key, tensor in state_dict.items():
        match = _KEY_RE.match(key)
        if match is None:
            continue
        module = match.group("module")
        if module not in a_by_mod_layer:
            continue
        layer = int(match.group("layer"))
        if match.group("ab") == "A":
            a_by_mod_layer[module][layer] = tensor
        else:
            b_by_mod_layer[module][layer] = tensor

    out: dict[str, dict[str, Any]] = {}
    for module in target_modules:
        a_layers = a_by_mod_layer[module]
        b_layers = b_by_mod_layer[module]
        if not a_layers and not b_layers:
            raise ValueError(
                f"no lora_A/lora_B keys for module '{module}' in state_dict"
            )
        if set(a_layers) != set(b_layers):
            only_a = sorted(set(a_layers) - set(b_layers))
            only_b = sorted(set(b_layers) - set(a_layers))
            raise ValueError(
                f"lora_A and lora_B layer sets mismatch for '{module}': "
                f"only_A={only_a}, only_B={only_b}"
            )
        sorted_layers = sorted(a_layers.keys())
        a_stack = torch.stack([a_layers[i] for i in sorted_layers], dim=0)
        b_stack = torch.stack([b_layers[i] for i in sorted_layers], dim=0)
        out[module] = {
            "A": a_stack,
            "B": b_stack,
            "layer_indices": torch.tensor(sorted_layers, dtype=torch.long),
        }
    return out


def load_adapter_state_dict(adapter_dir: Path) -> dict[str, Any]:
    """Load ``adapter_model.safetensors`` from a PEFT adapter directory."""
    from safetensors.torch import load_file  # noqa: PLC0415

    st_path = adapter_dir / "adapter_model.safetensors"
    if not st_path.is_file():
        raise FileNotFoundError(f"missing adapter_model.safetensors: {st_path}")
    return load_file(str(st_path))


def load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    """Load ``adapter_config.json`` from a PEFT adapter directory."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing adapter_config.json: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def load_adapter_as_record(
    adapter_dir: Path,
    *,
    task_id: str,
    task_description: str,
    warm_start_adapter: str | None,
    base_model_id_override: str | None,
    created_at: str,
    source_task_hash: str | None = None,
    fitness_score: float | None = None,
) -> dict[str, Any]:
    """Read an adapter dir and return kwargs for ``ReconstructionRecord(**kwargs)``.

    The A/B tensors themselves are NOT returned — they live on disk where the
    trainer can stream them. Only shape / identity metadata flows through.

    Args:
        adapter_dir: Directory containing ``adapter_model.safetensors`` and
            ``adapter_config.json``.
        task_id: Manifest-stable id (usually ``AdapterRecord.id``).
        task_description: Text used to compute the task embedding.
        warm_start_adapter: Warm-start adapter path/repo, or None. Stored
            verbatim on the record — downstream consumers must honor this.
        base_model_id_override: If set, overrides the ``base_model_name_or_path``
            from ``adapter_config.json`` (e.g., when the config records the
            warm-start adapter instead of the true base).
        created_at: ISO-8601 UTC timestamp.
        source_task_hash: Optional dedup key.
        fitness_score: Optional evaluation score.

    Returns:
        Dict suitable for ``ReconstructionRecord(**returned_dict)``.
    """
    cfg = load_adapter_config(adapter_dir)
    target_modules: tuple[str, ...] = tuple(cfg["target_modules"])
    rank = int(cfg["r"])

    state_dict = load_adapter_state_dict(adapter_dir)
    per_mod = extract_lora_ab_from_state_dict(state_dict, target_modules=target_modules)

    # All modules share the same layer set by construction (validate_homogeneity).
    first_mod = target_modules[0]
    layer_indices = tuple(int(i) for i in per_mod[first_mod]["layer_indices"].tolist())

    base_from_cfg = str(cfg.get("base_model_name_or_path", ""))
    base_model_id = base_model_id_override or base_from_cfg
    if not base_model_id:
        raise ValueError(
            f"adapter {adapter_dir} has no base_model_name_or_path and no override"
        )

    return {
        "task_id": task_id,
        "adapter_path": str(adapter_dir.resolve()),
        "task_description": task_description,
        "base_model_id": base_model_id,
        "warm_start_adapter": warm_start_adapter,
        "rank": rank,
        "target_modules": target_modules,
        "layer_indices": layer_indices,
        "created_at": created_at,
        "source_task_hash": source_task_hash,
        "fitness_score": fitness_score,
    }


__all__ = [
    "extract_lora_ab_from_state_dict",
    "load_adapter_state_dict",
    "load_adapter_config",
    "load_adapter_as_record",
]
