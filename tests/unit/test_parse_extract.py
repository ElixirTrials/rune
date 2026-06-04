"""Freeform code extraction: de-fence model output, no JSON wrapping.

Code actions emit freeform Python (a ```python fence or bare code), never a JSON
``{"code": ...}`` object — that wrapping let the model over-escape newlines
(``\\n`` -> literal backslash-n) and collapse multi-line code to one line, a phantom
SyntaxError. Extraction is now a single CommonMark de-fence (markdown-it).
"""

from __future__ import annotations

import ast

from rune.engine.continuation import strip_self_tests
from rune.engine.parse import extract_code_block


def test_bare_code_passes_through_unchanged() -> None:
    assert extract_code_block("def f():\n    return 1") == "def f():\n    return 1"


def test_fenced_block_returns_inner_content() -> None:
    assert extract_code_block("```python\nx = 1\n```") == "x = 1"
    # language info string is tolerated
    assert extract_code_block("```py\nx = 1\n```") == "x = 1"


def test_unterminated_fence_returns_content_to_eof() -> None:
    # truncated output: an opening fence with no close still yields the body
    assert extract_code_block("```py\ndef g(") == "def g("


def test_multiline_fenced_code_stays_multiline() -> None:
    # Regression for the over-escape bug: real newlines must survive, so the code
    # is multi-line and compiles (the JSON path collapsed this to one line).
    body = (
        "def add_lists(lst, tpl):\n"
        "    return tpl + tuple(lst)\n\n"
        "# Test cases\n"
        "assert add_lists([5, 6, 7], (9, 10)) == (9, 10, 5, 6, 7)"
    )
    extracted = extract_code_block(f"```python\n{body}\n```")
    assert "```" not in extracted
    assert extracted.count("\n") >= 3  # stayed multi-line
    runnable = strip_self_tests(extracted)
    ast.parse(runnable)  # would raise SyntaxError if collapsed to one line
    assert "def add_lists" in runnable


def test_blank_line_indented_body_not_mis_extracted() -> None:
    # an indented Python body after a blank line is NOT a CommonMark code block
    assert extract_code_block("def f():\n\n    return 1") == "def f():\n\n    return 1"


def test_inner_fence_in_string_literal_preserved() -> None:
    keep = 's = """```"""\nx = 1'
    assert extract_code_block(keep) == keep
