from __future__ import annotations

import json

from rune.engine.continuation import (
    dedup_code,
    degeneration_score,
    extract_code,
    extract_partial_code,
    merge_overlap,
)


class TestExtractCode:
    def test_markdown_fence(self) -> None:
        raw = "```python\ndef foo():\n    return 1\n```"
        assert extract_code(raw) == "def foo():\n    return 1"

    def test_truncated_fence(self) -> None:
        raw = "```python\ndef foo():\n    return 1"
        assert "def foo():" in extract_code(raw)
        assert "return 1" in extract_code(raw)

    def test_think_block_complete(self) -> None:
        raw = "<think>planning...</think>\n```python\nx = 1\n```"
        assert extract_code(raw) == "x = 1"

    def test_think_block_truncated(self) -> None:
        raw = "```python\nx = 1\n```\n<think>still thinking"
        result = extract_code(raw)
        assert "x = 1" in result
        assert "<think>" not in result

    def test_assistant_prefix(self) -> None:
        raw = "assistant\n```python\nx = 2\n```"
        assert extract_code(raw) == "x = 2"

    def test_preamble(self) -> None:
        raw = "Here's the code:\ndef bar():\n    pass"
        assert "def bar():" in extract_code(raw)

    def test_heres_is_preamble(self) -> None:
        raw = "Here is the solution:\ndef baz():\n    pass"
        assert "def baz():" in extract_code(raw)

    def test_plain_code(self) -> None:
        raw = "x = 1\ny = 2"
        assert extract_code(raw) == "x = 1\ny = 2"

    def test_empty_string(self) -> None:
        assert extract_code("") == ""

    def test_multiple_fences(self) -> None:
        raw = "```python\ndef a():\n    pass\n```\n\n```python\ndef b():\n    pass\n```"
        result = extract_code(raw)
        assert "def a():" in result
        assert "def b():" in result


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


class TestDedupCode:
    def test_duplicate_class_removed(self) -> None:
        accumulated = "class Foo:\n    def method(self):\n        pass\n"
        new_code = "class Foo:\n    def method(self):\n        pass\n\nclass Bar:\n    pass\n"
        result = dedup_code(new_code, accumulated)
        assert "class Foo" not in result
        assert "class Bar" in result

    def test_duplicate_function_removed(self) -> None:
        accumulated = "def helper():\n    return 1\n"
        new_code = "def helper():\n    return 1\n\ndef new_func():\n    return 2\n"
        result = dedup_code(new_code, accumulated)
        assert "def helper" not in result
        assert "def new_func" in result

    def test_unique_code_preserved(self) -> None:
        accumulated = "def existing():\n    pass\n"
        new_code = "def brand_new():\n    return 42\n"
        result = dedup_code(new_code, accumulated)
        assert "def brand_new" in result

    def test_main_block_stripped(self) -> None:
        new_code = "def foo():\n    pass\n\nif __name__ == '__main__':\n    foo()\n"
        result = dedup_code(new_code, "")
        assert "def foo" in result
        assert "__name__" not in result

    def test_nested_methods_kept_when_parent_unique(self) -> None:
        accumulated = "x = 1\n"
        new_code = (
            "class NewClass:\n"
            "    def method_a(self):\n"
            "        pass\n"
            "    def method_b(self):\n"
            "        pass\n"
        )
        result = dedup_code(new_code, accumulated)
        assert "class NewClass" in result
        assert "method_a" in result
        assert "method_b" in result


class TestMergeOverlap:
    def test_tail_prefix_overlap_removed(self) -> None:
        accumulated = "line1\nline2\nline3\n"
        new_chunk = "line2\nline3\nline4\n"
        result = merge_overlap(accumulated, new_chunk)
        assert result == "line4\n"

    def test_no_overlap_passes_through(self) -> None:
        accumulated = "aaa\nbbb\n"
        new_chunk = "ccc\nddd\n"
        result = merge_overlap(accumulated, new_chunk)
        assert result == new_chunk

    def test_empty_accumulated(self) -> None:
        assert merge_overlap("", "some code\n") == "some code\n"

    def test_empty_new_chunk(self) -> None:
        assert merge_overlap("some code\n", "") == ""

    def test_both_empty(self) -> None:
        assert merge_overlap("", "") == ""

    def test_full_overlap(self) -> None:
        text = "a\nb\n"
        assert merge_overlap(text, text) == ""

    def test_single_line_overlap(self) -> None:
        accumulated = "line1\nline2\n"
        new_chunk = "line2\nline3\n"
        result = merge_overlap(accumulated, new_chunk)
        assert result == "line3\n"


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
