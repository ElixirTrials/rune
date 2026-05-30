"""Shared code extraction: validate-then-fallback behavior."""

from __future__ import annotations

from rune.engine.parse import CodeResult, extract_code_from_raw


def test_valid_json_returns_code() -> None:
    raw = '{"code": "def f():\\n    return 1"}'
    assert extract_code_from_raw(raw, CodeResult) == "def f():\n    return 1"


def test_invalid_json_falls_back_to_lenient_extraction() -> None:
    # Not valid CodeResult JSON; lenient extractor recovers the code field.
    raw = 'garbage {"code": "x = 1"} trailing'
    assert "x = 1" in extract_code_from_raw(raw, CodeResult)


def test_fallback_to_raw_when_requested_and_nothing_extracted() -> None:
    raw = "def f():\n    return 2"  # plain python, not JSON at all
    assert extract_code_from_raw(raw, CodeResult, fallback_to_raw=True) == raw


def test_no_raw_fallback_returns_empty_when_nothing_extracted() -> None:
    raw = "def f():\n    return 2"
    assert extract_code_from_raw(raw, CodeResult, fallback_to_raw=False) == ""
