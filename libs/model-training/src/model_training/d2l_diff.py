"""RTK-style diff compression for training data preparation.

Filters irrelevant files (lockfiles, generated code, binary assets,
build artifacts) and truncates large diffs to minimize token overhead
in hypernetwork training pairs.

Operates on the concatenated diff format produced by mine_pr_diff_chains:
    --- src/main.py ---
    +real code
    --- package-lock.json ---
    +lockfile noise
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

__all__ = ["compress_diff"]

_SKIP_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "go.sum",
        "Gemfile.lock",
        "composer.lock",
        "flake.lock",
        ".DS_Store",
        ".gitattributes",
        ".editorconfig",
    }
)

_SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".lock",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp3",
        ".mp4",
        ".zip",
        ".tar",
        ".gz",
        ".pdf",
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
    }
)

_SKIP_SUFFIXES: tuple[str, ...] = (
    "_pb2.py",
    "_pb2_grpc.py",
    ".pb.go",
    ".pb.h",
    ".pb.cc",
    ".generated.ts",
    ".generated.js",
    ".g.dart",
    ".min.js",
    ".min.css",
    ".map",
)

_SKIP_PATH_SEGMENTS: tuple[str, ...] = (
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "vendor/",
    "dist/",
    "build/",
    ".next/",
    "coverage/",
    ".idea/",
    ".vscode/",
)

_SECTION_RE = re.compile(r"^(--- .+ ---)$", re.MULTILINE)


def _should_skip(filename: str) -> bool:
    """Check if a file should be filtered from training data."""
    path = PurePosixPath(filename)
    if path.name in _SKIP_FILENAMES:
        return True
    if path.suffix in _SKIP_EXTENSIONS:
        return True
    if any(filename.endswith(s) for s in _SKIP_SUFFIXES):
        return True
    return any(seg in filename for seg in _SKIP_PATH_SEGMENTS)


def _truncate(text: str, max_lines: int) -> str:
    """Truncate text to max_lines, appending a marker if truncated."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    kept.append(f"... ({len(lines) - max_lines} lines truncated)")
    return "\n".join(kept)


def compress_diff(content: str, max_lines: int = 500) -> str:
    """Filter irrelevant files and truncate large diffs.

    Parses the ``--- filename ---`` section format produced by
    ``mine_pr_diff_chains`` and removes lockfiles, generated code,
    binary files, and build artifacts.

    Args:
        content: Concatenated diff content with section headers.
        max_lines: Maximum total output lines.

    Returns:
        Filtered and truncated diff string.
    """
    if not content:
        return content

    parts = _SECTION_RE.split(content)

    # No section headers found -- fall through to truncation only
    if len(parts) == 1:
        return _truncate(content, max_lines)

    # parts = [preamble, header1, body1, header2, body2, ...]
    kept: list[str] = []
    idx = 1  # skip preamble (parts[0])
    while idx < len(parts):
        header = parts[idx]
        body = parts[idx + 1] if idx + 1 < len(parts) else ""
        filename = header[4:-4]  # strip "--- " and " ---"
        if not _should_skip(filename):
            kept.append(header + body)
        idx += 2

    result = "".join(kept).strip()

    if not result:
        return ""

    return _truncate(result, max_lines)
