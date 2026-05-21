"""Adapter strategy resolution for the reasoning loop.

Determines whether to use single-pass encoding or chunk composition
based on phase, artifact size, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdapterStrategy:
    """Base strategy for adapter generation."""

    scaling: float


@dataclass
class SinglePass(AdapterStrategy):
    """Single-pass encoding (default)."""

    truncate: bool = False


@dataclass
class ChunkComposition(AdapterStrategy):
    """Semantic chunk composition for long code."""

    merge_method: str = "ties"


_CODE_PHASES = frozenset({"code", "code_repair", "integrate"})


def resolve_adapter_strategy(
    phase: str,
    artifact_tokens: int,
    chunk_threshold: int,
    base_scaling: float,
    enable_chunk_composition: bool = False,
    code_scaling_boost: float = 1.2,
    default_merge_method: str = "ties",
) -> AdapterStrategy:
    """Resolve adapter strategy from phase and artifact size."""
    is_code_phase = phase in _CODE_PHASES
    exceeds_single_pass = artifact_tokens > chunk_threshold

    if is_code_phase and exceeds_single_pass and enable_chunk_composition:
        return ChunkComposition(
            scaling=base_scaling * code_scaling_boost,
            merge_method=default_merge_method,
        )
    elif exceeds_single_pass:
        return SinglePass(scaling=base_scaling, truncate=True)
    else:
        return SinglePass(scaling=base_scaling)
