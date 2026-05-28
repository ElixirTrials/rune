"""Shared continuation utilities for code extraction, dedup, and quality."""

from __future__ import annotations

import re

from rune.engine.parse import CodeResult

_CODE_VALUE_RE = re.compile(r'"code"\s*:\s*"', re.DOTALL)

_ESCAPES: dict[str, str] = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}


def extract_code(raw: str) -> str:
    text = re.sub(r"^assistant\s*", "", raw.strip())
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text[: text.index("<think>")]
    text = text.strip()
    text = re.sub(r"^Here(?:'s| is)[^\n]*\n", "", text).strip()
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if not blocks:
        m = re.search(r"```(?:python)?\n(.*)", text, re.DOTALL)
        if m:
            blocks = [m.group(1)]
    if blocks:
        return "\n".join(b.rstrip() for b in blocks)
    if text:
        lines = text.splitlines()
        return "\n".join(line for line in lines if not line.startswith("```")).rstrip()
    return ""


def extract_partial_code(raw: str) -> str:
    """Extract code from a possibly-truncated CodeResult JSON string."""
    try:
        return CodeResult.model_validate_json(raw).code
    except Exception:
        pass
    m = _CODE_VALUE_RE.search(raw)
    if m:
        after = raw[m.end() :]
        chars: list[str] = []
        i = 0
        while i < len(after):
            ch = after[i]
            if ch == '"':
                break
            if ch == "\\" and i + 1 < len(after):
                nxt = after[i + 1]
                chars.append(_ESCAPES.get(nxt, nxt))
                i += 2
            else:
                chars.append(ch)
                i += 1
        return "".join(chars)
    return raw


def dedup_code(new_code: str, accumulated: str) -> str:
    existing_defs: set[str] = set()
    for line in accumulated.splitlines():
        stripped = line.strip()
        if stripped.startswith("class ") or stripped.startswith("def "):
            name = stripped.split("(")[0].split(":")[0]
            name = name.replace("class ", "").replace("def ", "").strip()
            existing_defs.add(name)
    lines = new_code.splitlines(keepends=True)
    result: list[str] = []
    skip_until_dedent = False
    skip_indent: int | None = None
    for line in lines:
        stripped = line.strip()
        if skip_until_dedent:
            indent = len(line) - len(line.lstrip()) if line.strip() else 999
            if indent <= skip_indent and stripped:  # type: ignore[operator]
                skip_until_dedent = False
                skip_indent = None
            else:
                continue
        if stripped.startswith("if __name__"):
            skip_until_dedent = True
            skip_indent = len(line) - len(line.lstrip())
            continue
        if stripped.startswith("class ") or stripped.startswith("def "):
            name = stripped.split("(")[0].split(":")[0]
            name = name.replace("class ", "").replace("def ", "").strip()
            if name in existing_defs:
                skip_until_dedent = True
                skip_indent = len(line) - len(line.lstrip())
                continue
        result.append(line)
    return "".join(result)


def merge_overlap(accumulated: str, new_chunk: str) -> str:
    """Remove overlapping lines where tail of accumulated matches head of new_chunk."""
    if not accumulated or not new_chunk:
        return new_chunk
    acc_lines = accumulated.splitlines(keepends=True)
    new_lines = new_chunk.splitlines(keepends=True)
    max_overlap = min(len(acc_lines), len(new_lines))
    for k in range(max_overlap, 0, -1):
        if acc_lines[-k:] == new_lines[:k]:
            return "".join(new_lines[k:])
    return new_chunk


def degeneration_score(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)
