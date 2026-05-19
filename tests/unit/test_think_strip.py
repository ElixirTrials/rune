"""Tests for <think>...</think> stripping in TransformersProvider.generate().

Verifies the two-pass regex that strips closed and dangling thinking blocks
emitted by Qwen 3.5's chat template before output is returned to callers.

The logic under test lives in generate() at:
    libs/inference/src/inference/transformers_provider.py

These tests apply the same regexes directly so they remain fast and
deterministic without needing GPU / transformers.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Helpers — mirror the exact regexes in transformers_provider.py
# ---------------------------------------------------------------------------


def _strip_thinking(text: str) -> str:
    """Apply both think-strip passes as done in generate()."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text


# ---------------------------------------------------------------------------
# Closed block tests
# ---------------------------------------------------------------------------


def test_single_closed_block_removed() -> None:
    raw = "<think>Some internal reasoning here.</think>def hello(): pass"
    assert _strip_thinking(raw) == "def hello(): pass"


def test_closed_block_with_trailing_whitespace_removed() -> None:
    raw = "<think>reasoning</think>\n\ndef hello(): pass"
    assert _strip_thinking(raw) == "def hello(): pass"


def test_multiline_closed_block_removed() -> None:
    raw = "<think>\nline1\nline2\nline3\n</think>\nresult = 42"
    assert _strip_thinking(raw) == "result = 42"


def test_multiple_closed_blocks_removed() -> None:
    raw = "<think>first</think>\n<think>second</think>\nactual output"
    assert _strip_thinking(raw) == "actual output"


def test_no_thinking_block_passthrough() -> None:
    code = "def greet(name):\n    return f'Hello, {name}'"
    assert _strip_thinking(code) == code


def test_empty_string_passthrough() -> None:
    assert _strip_thinking("") == ""


# ---------------------------------------------------------------------------
# Dangling (truncated) block tests
# ---------------------------------------------------------------------------


def test_dangling_block_removed() -> None:
    """A <think> with no closing tag should remove everything from it onward."""
    raw = "preamble\n<think>I started thinking but got cut off"
    assert _strip_thinking(raw) == "preamble\n"


def test_dangling_block_at_start_removed() -> None:
    raw = "<think>dangling with no close"
    assert _strip_thinking(raw) == ""


def test_dangling_multiline_removed() -> None:
    raw = "prefix\n<think>\nline1\nline2\nno close tag"
    assert _strip_thinking(raw) == "prefix\n"


# ---------------------------------------------------------------------------
# Edge-case / mixed tests
# ---------------------------------------------------------------------------


def test_closed_then_dangling_both_removed() -> None:
    """Closed block stripped first, then residual dangling block stripped."""
    raw = "<think>closed</think>\nreal output\n<think>dangling"
    assert _strip_thinking(raw) == "real output\n"


def test_content_without_think_tag_preserved() -> None:
    """Strings containing angle brackets but not <think> are untouched."""
    code = "x: list[int] = []\nreturn x"
    assert _strip_thinking(code) == code


def test_think_in_string_literal_also_stripped() -> None:
    """The regex is intentionally greedy — we accept this trade-off."""
    raw = 'msg = "<think>not a real tag</think>"\nreturn msg'
    # First pass strips the closed <think>...</think> inside the string
    result = _strip_thinking(raw)
    assert "<think>" not in result
    assert "</think>" not in result


def test_strip_is_idempotent() -> None:
    """Applying the strip twice is safe."""
    raw = "<think>some thought</think>\noutput"
    once = _strip_thinking(raw)
    twice = _strip_thinking(once)
    assert once == twice
