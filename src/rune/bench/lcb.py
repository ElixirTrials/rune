"""LiveCodeBench submission normalization for official grading."""

from __future__ import annotations

import ast
import json
from typing import Any


def build_public_assert_checks(
    row: dict[str, Any],
    *,
    merge_spec_public_checks: bool = False,
) -> str:
    """Bare ``assert fn(*args) == expected`` lines from LCB public test cases.

    Matches the engine's top-level ``def entry_point`` contract (not ``Solution()``).
    """
    meta = json.loads(row["metadata"]) if row.get("metadata") else {}
    fn = meta.get("func_name")
    if not fn:
        return ""
    lines: list[str] = []
    for t in json.loads(row["public_test_cases"]):
        try:
            args = [ast.literal_eval(a) for a in t["input"].split("\n") if a.strip()]
            out = ast.literal_eval(t["output"])
        except (ValueError, SyntaxError):
            continue
        call = f"{fn}(*{args!r})"
        lines.append(f"assert {call} == {out!r}, {t['input']!r}")
    wired = "\n".join(lines)
    if not merge_spec_public_checks:
        return wired
    from rune.engine.oracle import merge_public_checks  # noqa: PLC0415

    return merge_public_checks(row.get("question_content", ""), wired, fn)


def extract_entry_function(code: str, entry_point: str) -> str:
    """Return the top-level ``entry_point`` function from generated code.

    When the engine over-decomposes, ``best_code`` may be joined into a blob
    containing helper subtasks. LCB grades only the task's ``entry_point``.
    If the name appears multiple times, the last definition wins (repair wins).
    """
    text = code.strip()
    if not text or not entry_point:
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        salvaged = _salvage_entry_function(text, entry_point)
        return salvaged if salvaged else text

    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == entry_point
    ]
    if funcs:
        return ast.unparse(funcs[-1])

    bare = _class_method_to_bare(tree, entry_point)
    if bare is not None:
        return bare
    return text


def _extract_from_tree(tree: ast.Module, entry_point: str) -> str | None:
    """Bare top-level ``entry_point`` from a parsed tree, or None."""
    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == entry_point
    ]
    if funcs:
        return ast.unparse(funcs[-1])
    return _class_method_to_bare(tree, entry_point)


def _salvage_entry_function(text: str, entry_point: str) -> str | None:
    """Recover ``entry_point`` from a blob whose tail is unparseable garbage.

    A code step can emit a valid function then ramble into prose/pseudo-code,
    making ``ast.parse`` of the whole blob raise. We locate where the entry's
    ``def``/``class`` begins and binary-search for the largest prefix (starting
    there) that parses cleanly and still defines ``entry_point``.

    Complexity: O(log L) parse attempts over O(L) lines, each parse O(slice);
    overall O(L log L) — no O(L^2) per-line scan.
    """
    lines = text.split("\n")
    start = _entry_start_line(lines, entry_point)
    if start is None:
        return None

    # Binary-search the largest end boundary (exclusive) in (start, len(lines)]
    # whose slice both parses and defines entry_point. Parsing is monotone only
    # up to the first syntax error, but the leading valid def is contiguous from
    # `start`, so the largest parseable prefix is what we want.
    lo, hi = start + 1, len(lines)
    best: str | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        slice_text = "\n".join(lines[start:mid])
        candidate = _try_extract(slice_text, entry_point)
        if candidate is not None:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _try_extract(slice_text: str, entry_point: str) -> str | None:
    try:
        tree = ast.parse(slice_text)
    except SyntaxError:
        return None
    return _extract_from_tree(tree, entry_point)


def _entry_start_line(lines: list[str], entry_point: str) -> int | None:
    """Index of the line opening ``def entry_point(`` or its ``class`` host."""
    def_marker = f"def {entry_point}("
    class_start: int | None = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(def_marker):
            indent = len(line) - len(stripped)
            # Top-level def: salvage from here. Indented def: it's a method; the
            # enclosing class start (if seen) is the better salvage anchor.
            if indent == 0:
                return i
            if class_start is not None:
                return class_start
            return i
        if stripped.startswith("class "):
            class_start = i
    return None


def normalize_lcb_submission(
    code: str,
    entry_point: str,
    *,
    _starter_code: str = "",
) -> str:
    """Prepare rune engine output for the official LCB call-based grader."""
    text = code.strip()
    if not text:
        return text
    extracted = extract_entry_function(text, entry_point)
    if extracted != text and extracted:
        return extracted
    if "class Solution" in text and entry_point:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text
        bare = _class_method_to_bare(tree, entry_point)
        if bare is not None:
            return bare
    return text


def _class_method_to_bare(tree: ast.Module, entry_point: str) -> str | None:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Solution":
            continue
        methods = [
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name == entry_point
        ]
        if not methods:
            continue
        method = methods[-1]
        args = method.args
        new_args = ast.arguments(
            posonlyargs=list(args.posonlyargs),
            args=[arg for arg in args.args if arg.arg != "self"],
            kwonlyargs=list(args.kwonlyargs),
            kw_defaults=list(args.kw_defaults),
            defaults=list(args.defaults),
        )
        fn = ast.FunctionDef(
            name=entry_point,
            args=new_args,
            body=list(method.body),
            decorator_list=[],
            returns=method.returns,
            type_comment=getattr(method, "type_comment", None),
            type_params=[],
            lineno=method.lineno,
            col_offset=method.col_offset,
        )
        return ast.unparse(fn)
    return None
