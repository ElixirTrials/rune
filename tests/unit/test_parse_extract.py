"""Shared code extraction: validate-then-fallback behavior."""

from __future__ import annotations

import ast
import json

from rune.engine.continuation import strip_self_tests
from rune.engine.parse import CodeResult, _extract_code_block, extract_code_from_raw


def test_valid_json_returns_code() -> None:
    raw = '{"code": "def f():\\n    return 1"}'
    assert extract_code_from_raw(raw, CodeResult) == "def f():\n    return 1"


def test_fenced_code_inside_json_value_is_stripped() -> None:
    # Verified real failure: the grammar-constrained model puts a ```py fence
    # INSIDE the JSON code value, which then crashes the sandbox on line 1.
    body = (
        "def add_lists(lst, tpl):\n"
        "    return tpl + tuple(lst)\n\n"
        "# Test cases\n"
        "assert add_lists([5, 6, 7], (9, 10)) == (9, 10, 5, 6, 7)"
    )
    raw = json.dumps({"code": f"```py\n{body}\n```"})
    extracted = extract_code_from_raw(raw, CodeResult)
    assert "```" not in extracted
    # de-fenced + self-tests stripped => valid, runnable implementation
    runnable = strip_self_tests(extracted)
    ast.parse(runnable)  # would have raised SyntaxError on the ``` line before
    assert "def add_lists" in runnable


def test_extract_code_block_variants() -> None:
    # bare code (no fence) passes through unchanged
    assert _extract_code_block("def f():\n    return 1") == "def f():\n    return 1"
    # fenced block -> inner content
    assert _extract_code_block("```python\nx = 1\n```") == "x = 1"
    # unterminated fence (truncated output) -> content to EOF
    assert _extract_code_block("```py\ndef g(") == "def g("
    # blank-line + indented Python body is NOT mis-extracted as a code block
    assert _extract_code_block("def f():\n\n    return 1") == "def f():\n\n    return 1"
    # inner ``` inside a string literal is preserved (no leading block fence)
    keep = 's = """```"""\nx = 1'
    assert _extract_code_block(keep) == keep


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
