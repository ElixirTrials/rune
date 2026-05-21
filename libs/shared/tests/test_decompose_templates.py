"""Tests for decompose template chain-of-thought suppression."""

from __future__ import annotations

from pathlib import Path

TEMPLATE_DIR = Path("libs/shared/src/shared/templates")


def test_decompose_trajectory_has_cot_suppression() -> None:
    """decompose.j2 must contain chain-of-thought suppression."""
    content = (TEMPLATE_DIR / "decompose.j2").read_text()
    assert (
        "Do NOT include your chain-of-thought" in content
        or "do NOT include" in content.lower()
    )


def test_decompose_trajectory_has_simple_task_example() -> None:
    """decompose.j2 must have a simple single-function example."""
    content = (TEMPLATE_DIR / "decompose.j2").read_text()
    assert "Write a function" in content or "write a function" in content


def test_decompose_trajectory_has_negative_example() -> None:
    """decompose.j2 must have a BAD/negative example."""
    content = (TEMPLATE_DIR / "decompose.j2").read_text()
    assert "BAD" in content


def test_prompt_decompose_has_cot_suppression() -> None:
    """prompt_decompose.j2 must contain chain-of-thought suppression."""
    content = (TEMPLATE_DIR / "prompt_decompose.j2").read_text()
    assert "No preamble" in content or "no preamble" in content


def test_prompt_decompose_has_negative_example() -> None:
    """prompt_decompose.j2 must have a negative example."""
    content = (TEMPLATE_DIR / "prompt_decompose.j2").read_text()
    assert "BAD" in content


def test_prompt_decompose_concise_has_cot_suppression() -> None:
    """prompt_decompose_concise.j2 must contain chain-of-thought suppression."""
    content = (TEMPLATE_DIR / "prompt_decompose_concise.j2").read_text()
    assert "No preamble" in content or "no preamble" in content or "ONLY" in content
