"""Tests for model_training.d2l_diff module."""

from __future__ import annotations


def test_compress_diff_empty_input() -> None:
    """Empty string passes through unchanged."""
    from model_training.d2l_diff import compress_diff

    assert compress_diff("") == ""


def test_compress_diff_no_sections_truncates() -> None:
    """Content without section headers falls through to truncation."""
    from model_training.d2l_diff import compress_diff

    content = "\n".join(f"+line {i}" for i in range(100))
    result = compress_diff(content, max_lines=10)
    lines = result.split("\n")
    assert len(lines) == 11  # 10 kept + truncation message
    assert "truncated" in lines[-1]


def test_compress_diff_keeps_source_files() -> None:
    """Source code files are kept."""
    from model_training.d2l_diff import compress_diff

    content = "--- src/main.py ---\n+def hello(): pass"
    result = compress_diff(content)
    assert "src/main.py" in result
    assert "+def hello(): pass" in result


def test_compress_diff_filters_lockfiles() -> None:
    """Lockfiles are removed."""
    from model_training.d2l_diff import compress_diff

    content = (
        "--- src/main.py ---\n+real code\n"
        "--- package-lock.json ---\n+lock noise"
    )
    result = compress_diff(content)
    assert "src/main.py" in result
    assert "package-lock.json" not in result
    assert "lock noise" not in result


def test_compress_diff_filters_generated_code() -> None:
    """Generated protobuf files are removed."""
    from model_training.d2l_diff import compress_diff

    content = (
        "--- src/handler.go ---\n+real code\n"
        "--- proto/service_pb2.py ---\n+generated"
    )
    result = compress_diff(content)
    assert "handler.go" in result
    assert "service_pb2.py" not in result


def test_compress_diff_filters_binary_extensions() -> None:
    """Binary and media files are removed."""
    from model_training.d2l_diff import compress_diff

    content = (
        "--- src/app.ts ---\n+code\n"
        "--- assets/logo.png ---\n+binary"
    )
    result = compress_diff(content)
    assert "app.ts" in result
    assert "logo.png" not in result


def test_compress_diff_filters_build_paths() -> None:
    """Build artifact paths are removed."""
    from model_training.d2l_diff import compress_diff

    content = (
        "--- src/lib.rs ---\n+code\n"
        "--- dist/bundle.js ---\n+built\n"
        "--- node_modules/foo/index.js ---\n+dep"
    )
    result = compress_diff(content)
    assert "lib.rs" in result
    assert "dist/" not in result
    assert "node_modules/" not in result


def test_compress_diff_truncates_at_max_lines() -> None:
    """Output respects max_lines limit."""
    from model_training.d2l_diff import compress_diff

    lines = [f"+line {i}" for i in range(200)]
    content = "--- big.py ---\n" + "\n".join(lines)
    result = compress_diff(content, max_lines=50)
    result_lines = result.split("\n")
    assert len(result_lines) <= 51  # 50 + truncation message
    assert "truncated" in result_lines[-1]


def test_compress_diff_all_filtered_returns_empty() -> None:
    """When all sections are filtered, return empty string."""
    from model_training.d2l_diff import compress_diff

    content = (
        "--- package-lock.json ---\n+lock\n"
        "--- yarn.lock ---\n+yarn\n"
        "--- dist/out.js ---\n+built"
    )
    result = compress_diff(content)
    assert result == ""
