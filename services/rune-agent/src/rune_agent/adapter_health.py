"""Adapter health monitoring for the reasoning loop.

Detects adapter collapse via cosine similarity, norm ratio, and output
repetition. All computations use plain Python to avoid requiring torch
in CPU-only CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HealthSignals:
    """Health signals computed for a single turn."""

    cosine_similarity: float
    norm_ratio: float
    output_repetition: float
    is_collapsed: bool
    collapse_reason: str | None


def _flatten_weights(weights: dict[str, list[float]]) -> list[float]:
    """Flatten a dict of weight lists into a single vector."""
    flat: list[float] = []
    for key in sorted(weights.keys()):
        flat.extend(weights[key])
    return flat


def compute_cosine_similarity(
    current: dict[str, list[float]],
    previous: dict[str, list[float]] | None,
) -> float:
    """Cosine similarity between flattened adapter weight vectors."""
    if previous is None:
        return 0.0

    a = _flatten_weights(current)
    b = _flatten_weights(previous)

    if len(a) != len(b) or len(a) == 0:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return dot / (norm_a * norm_b)


def compute_norm_ratio(
    current: dict[str, list[float]],
    first_norm: float | None,
) -> float:
    """L2 norm of current adapter / L2 norm of first adapter."""
    flat = _flatten_weights(current)
    current_norm = math.sqrt(sum(x * x for x in flat))

    if first_norm is None or first_norm < 1e-12:
        return 1.0

    return current_norm / first_norm


def compute_adapter_norm(weights: dict[str, list[float]]) -> float:
    """L2 norm of flattened adapter weights."""
    flat = _flatten_weights(weights)
    return math.sqrt(sum(x * x for x in flat))


def _extract_ngrams(text: str, n: int = 4) -> list[tuple[str, ...]]:
    """Extract n-grams from text."""
    words = text.split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def compute_output_repetition(current_output: str, previous_output: str) -> float:
    """Fraction of current 4-grams that also appear in previous output."""
    if not current_output or not previous_output:
        return 0.0

    current_ngrams = _extract_ngrams(current_output)
    if not current_ngrams:
        return 0.0

    previous_set = set(_extract_ngrams(previous_output))
    if not previous_set:
        return 0.0

    overlap = sum(1 for ng in current_ngrams if ng in previous_set)
    return overlap / len(current_ngrams)


def check_health(
    cosine_sim: float,
    norm_ratio: float,
    output_repetition: float,
    consecutive_high_similarity: int,
    cosine_threshold: float = 0.95,
    norm_min: float = 0.1,
    norm_max: float = 10.0,
    repetition_threshold: float = 0.8,
) -> HealthSignals:
    """Check adapter health against thresholds."""
    collapse_reason: str | None = None

    if norm_ratio < norm_min:
        collapse_reason = f"norm_collapse: ratio={norm_ratio:.4f} < {norm_min}"
    elif norm_ratio > norm_max:
        collapse_reason = f"norm_explosion: ratio={norm_ratio:.4f} > {norm_max}"
    elif cosine_sim > cosine_threshold and consecutive_high_similarity >= 1:
        collapse_reason = (
            f"cosine_collapse: sim={cosine_sim:.4f} > {cosine_threshold} "
            f"for {consecutive_high_similarity + 1} consecutive turns"
        )
    elif output_repetition > repetition_threshold:
        collapse_reason = (
            f"output_repetition: {output_repetition:.4f} > {repetition_threshold}"
        )

    return HealthSignals(
        cosine_similarity=cosine_sim,
        norm_ratio=norm_ratio,
        output_repetition=output_repetition,
        is_collapsed=collapse_reason is not None,
        collapse_reason=collapse_reason,
    )
