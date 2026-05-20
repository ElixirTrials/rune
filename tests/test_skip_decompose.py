"""Tests for task-complexity gating (_should_skip_decompose)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rune_runner import _should_skip_decompose  # type: ignore[import-not-found]


def test_short_function_prompt_skips() -> None:
    """Short 'write a function' prompt should skip decompose."""
    assert (
        _should_skip_decompose("Write a function to check if a number is prime") is True
    )


def test_long_prompt_does_not_skip() -> None:
    """Prompt over threshold words should not skip."""
    long_prompt = "Build a web application that " + " ".join(["word"] * 250)
    assert _should_skip_decompose(long_prompt) is False


def test_short_prompt_without_function_signal_does_not_skip() -> None:
    """Short prompt without function signals should not skip."""
    assert _should_skip_decompose("Build a REST API with three endpoints") is False


def test_implement_signal_skips() -> None:
    """'implement a function' signal should skip for short prompts."""
    assert (
        _should_skip_decompose("Implement a function that returns fibonacci numbers")
        is True
    )


def test_custom_threshold() -> None:
    """Custom threshold should be respected."""
    prompt = "Write a function to sort a list"  # ~7 words
    assert _should_skip_decompose(prompt, threshold=5) is False
    assert _should_skip_decompose(prompt, threshold=50) is True
