from __future__ import annotations

import json

from rune.engine.continuation import (
    degeneration_score,
    extract_partial_code,
    validate_syntax,
)


class TestExtractPartialCode:
    def test_valid_json(self) -> None:
        raw = json.dumps({"code": "def foo(): pass"})
        assert extract_partial_code(raw) == "def foo(): pass"

    def test_truncated_json(self) -> None:
        raw = '{"code": "class Foo:\\n    def b'
        result = extract_partial_code(raw)
        assert "class Foo:" in result
        assert "def b" in result

    def test_truncated_json_with_escapes(self) -> None:
        raw = '{"code": "line1\\nline2\\tindented'
        result = extract_partial_code(raw)
        assert result == "line1\nline2\tindented"

    def test_non_json_plaintext(self) -> None:
        raw = "def foo():\n    return 42"
        assert extract_partial_code(raw) == raw


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
