"""Tests for the controlled-confound evaluation harness."""

from __future__ import annotations

from scripts.paper.controlled_confound import (
    ConfoundCondition,
    build_injected_history,
    build_memory_stripped,
)


def test_injected_history_grows_with_depth() -> None:
    """Injected history context grows with trajectory depth."""
    base_prompt = "def solve(x):"
    trajectory_steps = [f"step {i}" for i in range(10)]

    short = build_injected_history(base_prompt, trajectory_steps[:2])
    long = build_injected_history(base_prompt, trajectory_steps[:8])
    assert len(long) > len(short)


def test_memory_stripped_is_base_only() -> None:
    """Memory-stripped condition uses only the base prompt."""
    base_prompt = "def solve(x):"
    result = build_memory_stripped(base_prompt)
    assert result == base_prompt


def test_conditions_enum() -> None:
    """All three conditions are defined."""
    assert ConfoundCondition.RUNE is not None
    assert ConfoundCondition.INJECTED_HISTORY is not None
    assert ConfoundCondition.MEMORY_STRIPPED is not None
