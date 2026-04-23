"""Oracle adapter cache and lookup helpers for round-2 training.

This module is intentionally CPU-safe: torch and peft are imported inside
function bodies per INFRA-05 so the module stays importable in CPU-only CI.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# {module_name: {"A": Tensor[L,r,in], "B": Tensor[L,r,out]}}
LoraDict = dict[str, dict[str, Any]]

# Matches flat PEFT safetensors keys produced by peft.LoraModel:
#   base_model.model.model.layers.<layer>.self_attn.<module>.lora_<A|B>.weight
# The leading "base_model.model." prefix is PEFT's wrapper layer; the rest
# mirrors the HF Qwen module path.
_PEFT_LORA_KEY_RE = re.compile(
    r"(?:base_model\.model\.)?model\.layers\.(?P<layer>\d+)\."
    r".*?\.(?P<module>[a-z_]+_proj)\.lora_(?P<ab>[AB])\.weight$"
)

ORACLE_ID_PREFIX: str = "oracle_"
DIAGNOSE_BIN_KEY: str = "diagnose_pooled"


def _bin_key_for_record(record: dict[str, Any]) -> str:
    """Derive the oracle bin key for a training manifest record.

    Prefers ``metadata.phase`` + ``metadata.benchmark`` (authoritative, set
    by the corpus producer). Falls back to parsing ``task_id`` of form
    ``"<benchmark>/<problem_id>/<phase>"``. Diagnose pools across benchmarks
    to a single ``"diagnose_pooled"`` bin.

    Args:
        record: One JSONL manifest record from the corpus producer.

    Returns:
        Bin key of form ``"<phase>_<benchmark>"`` or ``"diagnose_pooled"``.

    Raises:
        ValueError: When neither metadata nor task_id provide enough info.
    """
    meta = record.get("metadata") or {}
    phase = meta.get("phase")
    benchmark = meta.get("benchmark")

    if not phase or not benchmark:
        task_id = str(record.get("task_id", ""))
        parts = task_id.split("/")
        if len(parts) >= 3:
            benchmark = benchmark or parts[0]
            phase = phase or parts[-1]

    if not phase or not benchmark:
        raise ValueError(
            f"cannot derive bin_key from record: task_id={record.get('task_id')!r}, "
            f"metadata={meta!r}"
        )

    if phase == "diagnose":
        return DIAGNOSE_BIN_KEY
    return f"{phase}_{benchmark}"


def lookup_oracle_path(bin_key: str, registry: Any) -> str | None:
    """Resolve a bin key to the on-disk path of its registered oracle adapter.

    The adapter_id scheme follows :mod:`corpus_producer.trainer_bridge`:
    ``oracle_<bin_key>``. Returns ``None`` when the adapter is missing or
    archived so callers can decide whether to fall back to the base model
    or skip the record.

    Args:
        bin_key: Oracle bin key (e.g. ``"decompose_humaneval"``,
            ``"diagnose_pooled"``).
        registry: An :class:`adapter_registry.registry.AdapterRegistry`
            instance (or a mock with a compatible ``retrieve_by_id``).

    Returns:
        The adapter's ``file_path`` string, or ``None`` when missing / archived.
    """
    from adapter_registry.exceptions import AdapterNotFoundError  # noqa: PLC0415

    adapter_id = f"{ORACLE_ID_PREFIX}{bin_key}"
    try:
        record = registry.retrieve_by_id(adapter_id)
    except AdapterNotFoundError:
        logger.warning("Oracle adapter %r not found in registry", adapter_id)
        return None
    if record.is_archived:
        logger.warning("Oracle adapter %r is archived; ignoring", adapter_id)
        return None
    return str(record.file_path)


def audit_oracle_coverage(
    records: list[dict[str, Any]],
    registry: Any,
) -> tuple[float, dict[str, int]]:
    """Compute oracle-coverage ratio and per-bin record counts.

    Iterates the records, derives each bin_key, and checks whether a
    registered (non-archived) oracle exists for it. Caches lookup results
    per bin_key to avoid repeated registry queries.

    Args:
        records: List of JSONL manifest records.
        registry: AdapterRegistry instance.

    Returns:
        Tuple ``(coverage_ratio, bin_counts)`` where:
        - ``coverage_ratio`` is ``covered / len(records)`` (using the
          **original** record count as the denominator). Unroutable records
          — those raising ``ValueError`` from :func:`_bin_key_for_record` —
          are logged and skipped, but still count against the denominator.
          Returns ``0.0`` when ``records`` is empty.
        - ``bin_counts`` maps bin_key → record count for *routable* records
          only; unroutable records are excluded, so
          ``sum(bin_counts.values())`` may be less than ``len(records)``.
    """
    if not records:
        return 0.0, {}

    bin_counts: dict[str, int] = {}
    lookup_cache: dict[str, bool] = {}
    covered = 0

    for record in records:
        try:
            bin_key = _bin_key_for_record(record)
        except ValueError as exc:
            logger.warning("Skipping unroutable record: %s", exc)
            continue
        bin_counts[bin_key] = bin_counts.get(bin_key, 0) + 1

        if bin_key not in lookup_cache:
            lookup_cache[bin_key] = lookup_oracle_path(bin_key, registry) is not None
        if lookup_cache[bin_key]:
            covered += 1

    return covered / len(records), bin_counts


def _load_oracle_as_lora_dict(path: str, hc: Any) -> LoraDict:
    """Load a PEFT safetensors adapter and reshape to functional-LoRA format.

    Parses the flat safetensors state_dict and stacks per-layer LoRA A/B
    tensors into the ``{module: {"A": Tensor[L, ...], "B": Tensor[L, ...]}}``
    shape consumed by :func:`model_training.d2l_lora.apply_functional_lora`.
    Layer order matches ``hc.layer_indices`` (sorted ascending).

    Args:
        path: Path to a PEFT adapter directory (containing
            ``adapter_model.safetensors``) or a safetensors file directly.
        hc: HypernetConfig; ``.layer_indices`` selects which base-model
            layers the stacked tensors correspond to.

    Returns:
        Functional-LoRA dict ready for ``apply_functional_lora``.

    Raises:
        ValueError: When the safetensors file is missing expected keys for
            one of ``hc.layer_indices``.
        FileNotFoundError: When ``path`` does not exist.
    """
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    p = Path(path)
    if p.is_dir():
        p = p / "adapter_model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"Oracle adapter not found at {p}")

    state = load_file(str(p))

    # Group by (module, layer) → {A, B}
    by_module: dict[str, dict[int, dict[str, Any]]] = {}
    for key, tensor in state.items():
        m = _PEFT_LORA_KEY_RE.search(key)
        if m is None:
            continue
        module = m.group("module")
        layer_idx = int(m.group("layer"))
        ab = m.group("ab")
        by_module.setdefault(module, {}).setdefault(layer_idx, {})[ab] = tensor

    target_layers: list[int] = sorted(int(i) for i in hc.layer_indices)

    result: LoraDict = {}
    for module, layers in by_module.items():
        a_list: list[Any] = []
        b_list: list[Any] = []
        for idx in target_layers:
            if idx not in layers or "A" not in layers[idx] or "B" not in layers[idx]:
                raise ValueError(
                    f"Oracle {p} missing layer {idx} for module {module}; "
                    f"present layers: {sorted(layers)}"
                )
            a_list.append(layers[idx]["A"])
            b_list.append(layers[idx]["B"])
        result[module] = {
            "A": torch.stack(a_list, dim=0),
            "B": torch.stack(b_list, dim=0),
        }
    if not result:
        raise ValueError(
            f"Oracle {p} produced no LoRA modules after parsing; "
            f"check the PEFT key format"
        )
    return result


class OracleAdapterCache:
    """LRU cache of oracle LoRA dicts keyed by bin_key.

    Entries are lightweight ``LoraDict`` tensors — NOT ``PeftModel`` wrappers.
    The teacher pass applies them to the base model via
    :func:`model_training.d2l_lora.apply_functional_lora`, the same mechanism
    the student pass uses for hypernet output. The base model is therefore
    never structurally mutated (no LoraLayer wrappers), so there is no
    possibility of hook leakage across passes.

    Missing bins (no registered oracle) return ``None`` so callers can
    ``skip`` the record or fall back to the bare base model.

    Not thread-safe; assumes the training loop is single-threaded per GPU.

    Example:
        >>> cache = OracleAdapterCache(registry=reg, hc=hc, max_loaded=4)
        >>> lora = cache.get("decompose_humaneval")
        >>> if lora is not None:
        ...     with apply_functional_lora(base_model, lora, hc):
        ...         with torch.no_grad():
        ...             teacher_logits = base_model(**inputs).logits
    """

    def __init__(self, *, registry: Any, hc: Any, max_loaded: int) -> None:
        """Initialise the cache.

        Args:
            registry: AdapterRegistry instance (or compatible mock).
            hc: HypernetConfig passed through to ``_load_oracle_as_lora_dict``.
            max_loaded: Maximum number of LoRA dicts to hold in memory.
        """
        if max_loaded < 1:
            raise ValueError(f"max_loaded must be >= 1, got {max_loaded}")
        self._registry = registry
        self._hc = hc
        self._max_loaded = max_loaded
        self._cache: OrderedDict[str, LoraDict] = OrderedDict()
        self._missing: set[str] = set()

    def get(self, bin_key: str) -> LoraDict | None:
        """Return the oracle LoRA dict for ``bin_key`` (loading if needed).

        Returns ``None`` when no registered oracle exists. Moves the entry
        to the MRU position on hit.
        """
        if bin_key in self._missing:
            return None
        if bin_key in self._cache:
            self._cache.move_to_end(bin_key)
            return self._cache[bin_key]

        path = lookup_oracle_path(bin_key, self._registry)
        if path is None:
            self._missing.add(bin_key)
            return None

        logger.info("Loading oracle LoRA dict for %r from %s", bin_key, path)
        lora_dict = _load_oracle_as_lora_dict(path, self._hc)
        self._cache[bin_key] = lora_dict
        self._cache.move_to_end(bin_key)
        self._evict_if_full()
        return lora_dict

    def clear(self) -> None:
        """Drop all cached oracles; next ``.get`` will reload from disk."""
        self._cache.clear()
        self._missing.clear()

    def _evict_if_full(self) -> None:
        while len(self._cache) > self._max_loaded:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.info("Evicting oracle %r from cache (LRU)", evicted_key)
