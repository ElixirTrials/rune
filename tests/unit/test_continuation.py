from __future__ import annotations

from rune.engine.continuation import (
    CONT_SYSTEM_PROMPT,
    degeneration_score,
    extract_partial_code,
    validate_syntax,
)
from rune.engine.parse import render_template


def test_cont_system_prompt_is_code_only() -> None:
    assert "Output only Python code" in CONT_SYSTEM_PROMPT
    assert "markdown" in CONT_SYSTEM_PROMPT.lower()


def test_prompt_code_continue_renders() -> None:
    text = render_template("prompt_code_continue", task_description="build a parser")
    assert "build a parser" in text


def test_prompt_code_continue_renders_empty_when_absent() -> None:
    assert render_template("prompt_code_continue").strip() == ""


class TestExtractPartialCode:
    """Code output is freeform (a ```python fence or bare code), never JSON."""

    def test_fenced_block_returns_inner_code(self) -> None:
        raw = "```python\ndef foo():\n    return 42\n```"
        assert extract_partial_code(raw) == "def foo():\n    return 42"

    def test_bare_code_passes_through(self) -> None:
        raw = "def foo():\n    return 42"
        assert extract_partial_code(raw) == raw

    def test_multiline_newlines_preserved(self) -> None:
        # Regression: real newlines survive (the JSON path collapsed code to 1 line).
        raw = "```py\nclass Foo:\n    def bar(self):\n        return 1\n```"
        result = extract_partial_code(raw)
        assert result == "class Foo:\n    def bar(self):\n        return 1"

    def test_unterminated_fence_returns_body(self) -> None:
        raw = "```py\nclass Foo:\n    def b"
        result = extract_partial_code(raw)
        assert "class Foo:" in result
        assert "def b" in result


class TestDegenerationScore:
    def test_unique_text(self) -> None:
        text = "the quick brown fox jumps over the lazy dog today"
        score = degeneration_score(text)
        assert score < 0.2

    def test_heavily_repeated(self) -> None:
        text = " ".join(["hello world foo bar"] * 20)
        score = degeneration_score(text)
        assert score > 0.8

    def test_short_text(self) -> None:
        assert degeneration_score("one two three") == 0.0

    def test_empty_string(self) -> None:
        assert degeneration_score("") == 0.0

    def test_exact_n_words(self) -> None:
        assert degeneration_score("a b c d", n=4) == 0.0

    def test_repeated_4gram(self) -> None:
        # "a b c d a b c d" -> ngrams: (a,b,c,d), (b,c,d,a), (c,d,a,b), (d,a,b,c), (a,b,c,d)
        # 5 ngrams, 4 unique -> 1 - 4/5 = 0.2
        score = degeneration_score("a b c d a b c d")
        assert abs(score - 0.2) < 0.01


class TestValidateSyntax:
    def test_valid_python(self) -> None:
        assert validate_syntax("def foo():\n    return 1\n") is True

    def test_syntax_error(self) -> None:
        assert validate_syntax("def foo(\n") is False

    def test_incomplete_class(self) -> None:
        code = "class Foo:\n    def bar(self):\n        x = 1\n        return"
        assert validate_syntax(code) is True

    def test_return_outside_function(self) -> None:
        code = "class Foo:\n    pass\nreturn 1"
        assert validate_syntax(code) is False

    def test_empty_string(self) -> None:
        assert validate_syntax("") is False

    def test_indentation_error(self) -> None:
        code = "def foo():\nreturn 1"
        assert validate_syntax(code) is False

    def test_valid_multiclass(self) -> None:
        code = (
            "class Node:\n"
            "    def __init__(self, data):\n"
            "        self.data = data\n"
            "\n"
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.head = None\n"
        )
        assert validate_syntax(code) is True
