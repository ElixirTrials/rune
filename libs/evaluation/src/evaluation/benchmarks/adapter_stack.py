"""Adapter stack loader for benchmark evaluation.

Resolves a list of adapter_ids from AdapterRegistry into file paths,
and bundles them with an InferenceProvider into an AdapterStack that
the benchmark runner uses to generate completions.

No GPU imports. CPU-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.provider import InferenceProvider


@dataclass
class AdapterStack:
    """Bundle of base model + ordered adapter stack + provider.

    Attributes:
        base_model: HuggingFace model ID or local path for the base model.
        adapter_ids: Ordered list of adapter IDs to apply (first = innermost).
        adapter_paths: Dict mapping adapter_id -> filesystem path.
        provider: InferenceProvider used to generate completions.
    """

    base_model: str
    adapter_ids: list[str]
    adapter_paths: dict[str, str]
    provider: InferenceProvider

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"AdapterStack(base_model={self.base_model!r}, "
            f"adapter_ids={self.adapter_ids!r})"
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of this stack.

        Returns:
            Dict with keys: base_model, adapter_ids, adapter_paths.
        """
        return {
            "base_model": self.base_model,
            "adapter_ids": list(self.adapter_ids),
            "adapter_paths": dict(self.adapter_paths),
        }


def load_adapter_stack(
    base_model: str,
    adapter_ids: list[str],
    provider: InferenceProvider,
    registry: Any,
) -> AdapterStack:
    """Resolve adapter IDs to file paths and construct an AdapterStack.

    Queries the AdapterRegistry for each adapter_id to retrieve its
    file_path. Raises ValueError for any unknown adapter_id.

    Args:
        base_model: HuggingFace model ID or local path for the base model.
        adapter_ids: Ordered list of adapter registry IDs to load.
            Pass an empty list to use the base model only.
        provider: InferenceProvider instance for generating completions.
        registry: AdapterRegistry instance (or compatible duck-type) with
            a retrieve_by_id(adapter_id: str) -> AdapterRecord method.

    Returns:
        AdapterStack with resolved file paths.

    Raises:
        ValueError: If any adapter_id is not found in the registry.

    Example:
        >>> from sqlalchemy import create_engine
        >>> from adapter_registry.registry import AdapterRegistry
        >>> engine = create_engine("sqlite:///adapters.db")
        >>> reg = AdapterRegistry(engine=engine)
        >>> stack = load_adapter_stack("Qwen/Qwen3.5-9B", ["a1"], provider, reg)
    """
    adapter_paths: dict[str, str] = {}
    for aid in adapter_ids:
        try:
            record = registry.retrieve_by_id(aid)
            adapter_paths[aid] = str(record.file_path)
        except Exception as exc:
            raise ValueError(f"Adapter '{aid}' not found in registry: {exc}") from exc

    return AdapterStack(
        base_model=base_model,
        adapter_ids=list(adapter_ids),
        adapter_paths=adapter_paths,
        provider=provider,
    )
