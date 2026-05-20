"""Tests for decompose template chain-of-thought suppression."""

from __future__ import annotations

from pathlib import Path

TEMPLATE_DIR = Path("libs/shared/src/shared/templates")


def test_decompose_trajectory_has_cot_suppression() -> None:
    """decompose.j2 must contain chain-of-thought suppression."""
    content = (TEMPLATE_DIR / "decompose.j2").read_text()
    assert "do not" in content.lower() or "do NOT" in content


def test_decompose_trajectory_has_example() -> None:
    """decompose.j2 must have at least one example subtask."""
    content = (TEMPLATE_DIR / "decompose.j2").read_text()
    assert "EXAMPLE" in content or "example" in content.lower()


def test_decompose_trajectory_has_dependency_format() -> None:
    """decompose.j2 must show the [depends: ...] format."""
    content = (TEMPLATE_DIR / "decompose.j2").read_text()
    assert "[depends:" in content


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
