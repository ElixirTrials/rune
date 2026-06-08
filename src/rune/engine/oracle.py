"""Public-example correctness oracle (in-loop repair signal).

The engine's in-loop check is ``run_in_sandbox(strip_self_tests(code))`` — a bare
function definition exits 0, so logic errors never produced a failure signal and
``diagnose -> repair`` only ever fired on module-load crashes. This module derives
a trustworthy in-loop oracle from the spec's *public* doctest examples (the same
ones already shown to the model in the prompt — no held-out-test leakage), so a
wrong or crashing implementation fails the sandbox and routes to repair with an
actual-vs-expected message diagnose can act on.

The held-out ``task.test_code`` is never used here; pass@1 still gates on the full
held-out set at scoring, so the oracle cannot inflate it.
"""

from __future__ import annotations

import ast
import codecs
import contextlib
import doctest
import logging
from typing import Any

logger = logging.getLogger(__name__)


def defines_function(code: str, name: str) -> bool:
    """True iff *code* has a top-level function named *name* (AST, not substring)."""
    if not name:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in tree.body
    )


def defines_entry_point(code: str, entry_point: str) -> bool:
    """True if *code* defines *entry_point* as top-level def or Solution method."""
    if not entry_point:
        return False
    if defines_function(code, entry_point):
        return True
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Solution":
            continue
        if any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == entry_point
            for item in node.body
        ):
            return True
    return False


def _call_names(node: ast.AST) -> set[str]:
    return {
        c.func.id
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }


def _augmented_assert(left_src: str, right_src: str, index: int) -> str:
    """An equality assert with an actual-vs-expected message (not a bare assert).

    Bare ``assert f(x) == y`` hands diagnose only "AssertionError @ line"; the
    message carries the call, the actual value, and the expected value so the
    failure conveyed to the adapter is specific.
    """
    g = f"_oracle_got_{index}"
    return (
        f"{g} = {left_src}\n"
        f"assert {g} == {right_src}, "
        f"{left_src!r} + ' -> ' + repr({g}) + ', want ' + repr({right_src})"
    )


def extract_public_checks(spec: str, entry_point: str) -> str:
    """Executable assert lines from *spec*'s doctest examples that call *entry_point*.

    Handles both the assert form (``>>> assert f(x) == y``) and the
    expression+want form (``>>> f(x)`` / ``y``). Examples that do not reference
    *entry_point* are skipped. Returns ``""`` when nothing usable is found.
    """
    if not entry_point:
        return ""
    try:
        examples = doctest.DocTestParser().get_examples(spec)
    except ValueError:
        return ""

    checks: list[str] = []
    for ex in examples:
        source = ex.source.strip()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if not tree.body:
            continue
        node = tree.body[0]

        left_src: str | None = None
        right_src: str | None = None
        if (
            isinstance(node, ast.Assert)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
        ):
            left_src = ast.unparse(node.test.left)
            right_src = ast.unparse(node.test.comparators[0])
        elif isinstance(node, ast.Expr) and ex.want.strip():
            left_src = ast.unparse(node.value)
            right_src = ex.want.strip()

        if left_src is None or right_src is None:
            continue
        if entry_point not in _call_names(ast.parse(left_src, mode="eval").body):
            continue
        checks.append(_augmented_assert(left_src, right_src, len(checks)))

    return "\n".join(checks)


def _call_arg_key(piece: str, entry_point: str) -> tuple[Any, ...] | None:
    """Literal argument tuple for one public check, including augmented oracle form."""
    for args in parse_public_call_arglists(piece, entry_point):
        return tuple(args)
    for line in piece.splitlines():
        stripped = line.strip()
        prefix = "_oracle_got_"
        if not stripped.startswith(prefix) or f" = {entry_point}(" not in stripped:
            continue
        _, rhs = stripped.split(" = ", 1)
        try:
            call = ast.parse(rhs, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(call, ast.Call):
            continue
        call_args = _literal_args_from_call(call)
        if call_args is not None:
            return tuple(call_args)
    return None


def merge_public_checks(spec: str, public_checks: str, entry_point: str) -> str:
    """Union doctest examples from *spec* with wired benchmark *public_checks*.

    LCB ``public_test_cases`` often ship a single case; the problem statement may
    carry additional doctest examples that surface correctness gaps (issue #52).
    """
    wired = (public_checks or "").strip()
    from_spec = extract_public_checks(spec, entry_point).strip()
    if not from_spec:
        return wired
    if not wired:
        return from_spec
    existing: set[tuple[Any, ...]] = set()
    for raw in split_acceptance_checks(wired):
        key = _call_arg_key(raw, entry_point)
        if key is not None:
            existing.add(key)
    extra: list[str] = []
    for raw in split_acceptance_checks(from_spec):
        piece = raw.strip()
        key = _call_arg_key(piece, entry_point)
        if key is None or key in existing:
            continue
        extra.append(piece)
        existing.add(key)
    if not extra:
        return wired
    return wired + "\n" + "\n".join(extra)


_PROBE_IMPORT_PREAMBLE = (
    "from typing import (List, Dict, Set, Tuple, Optional, Union, Any, "
    "Iterable, Sequence, Mapping, Callable, Deque)\n"
    "import collections, math, heapq, bisect, itertools, functools, re\n"
    "from collections import defaultdict, deque, Counter, OrderedDict\n"
)


def with_probe_imports(code: str) -> str:
    """Prepend the typing/stdlib names the official LCB starter code imports.

    The in-loop probe must define the same names the grader provides (e.g.
    ``List``); otherwise an idiomatic ``def f(x: List[int])`` annotation
    NameErrors and a logically-correct solution is rejected without ever being
    graded (issue #52: 3748/3777/3799 burned half their repair budget on
    ``name 'List' is not defined``). Mirrors the grader, not a behaviour change.
    """
    return _PROBE_IMPORT_PREAMBLE + code


def build_probe(code: str, spec: str, entry_point: str) -> tuple[str, bool]:
    """Return ``(probe_code, oracle_fired)`` for sandbox execution.

    Appends the public-example checks only when *code* actually defines
    *entry_point* (so the example is callable); otherwise returns *code* bare —
    today's module-load-only behaviour — and ``oracle_fired=False``. Callers
    should log the fired/fallback ratio: a task with no parseable doctest silently
    reverts to no-signal, and the thesis re-test slice must be the tasks where the
    oracle actually fired and failed on attempt 1.
    """
    checks = extract_public_checks(spec, entry_point)
    if not checks or not defines_entry_point(code, entry_point):
        return code, False
    return with_probe_imports(f"{code}\n\n{checks}"), True


def _parse_checks_module(check: str) -> ast.Module | None:
    """Parse a subtask ``acceptance_check`` (Python asserts) into an AST module.

    The check is Python, so we parse it rather than splitting text — the AST
    handles newlines, semicolons, multi-line statements, and ``;`` or newlines
    *inside string literals* correctly, where textual splitting would not. Models
    sometimes over-escape the check as one JSON string with literal ``\\n``
    between asserts; if the raw text is not valid Python, decode one level of
    string escapes and retry before giving up. Returns ``None`` if nothing parses.
    """
    src = check.strip()
    if not src:
        return None
    candidates = [src]
    with contextlib.suppress(UnicodeDecodeError, ValueError):
        candidates.append(codecs.decode(src, "unicode_escape"))
    for candidate in candidates:
        try:
            return ast.parse(candidate)
        except (SyntaxError, ValueError):
            continue
    return None


def _literal_args_from_call(call: ast.Call) -> list[Any] | None:
    if len(call.args) == 1 and isinstance(call.args[0], ast.Starred):
        try:
            unpacked = ast.literal_eval(call.args[0].value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(unpacked, list):
            return list(unpacked)
        return None
    args: list[Any] = []
    for arg in call.args:
        try:
            args.append(ast.literal_eval(arg))
        except (ValueError, SyntaxError):
            return None
    return args


def parse_public_call_arglists(public_checks: str, entry_point: str) -> list[list[Any]]:
    """Literal argument lists from public ``assert entry_point(...) == ...`` lines."""
    calls: list[list[Any]] = []
    for raw in split_acceptance_checks(public_checks):
        piece = raw.strip()
        if not piece.startswith("assert "):
            continue
        try:
            tree = ast.parse(piece)
        except SyntaxError:
            continue
        if not tree.body or not isinstance(tree.body[0], ast.Assert):
            continue
        test = tree.body[0].test
        if not isinstance(test, ast.Compare) or not isinstance(test.left, ast.Call):
            continue
        call = test.left
        if not isinstance(call.func, ast.Name) or call.func.id != entry_point:
            continue
        args = _literal_args_from_call(call)
        if args is not None:
            calls.append(args)
    return calls


def split_acceptance_checks(check: str) -> list[str]:
    """Individual top-level statements of a subtask ``acceptance_check``.

    One source string per parsed statement (via the AST — not textual splitting),
    so newline-, semicolon-, and JSON-list-derived multi-assert forms all yield
    the same result, and the common over-escaped form is recovered.
    """
    tree = _parse_checks_module(check)
    if tree is None:
        return []
    return [ast.unparse(stmt) for stmt in tree.body]


def _augment_equality_check(check: str, index: int) -> str | None:
    """Rewrite one equality assert with an actual-vs-expected message, or None."""
    try:
        tree = ast.parse(check.strip())
    except (SyntaxError, ValueError):
        return None
    if not (
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.Assert)
        and isinstance(tree.body[0].test, ast.Compare)
        and len(tree.body[0].test.ops) == 1
        and isinstance(tree.body[0].test.ops[0], ast.Eq)
    ):
        return None
    cmp = tree.body[0].test
    return _augmented_assert(
        ast.unparse(cmp.left), ast.unparse(cmp.comparators[0]), index
    )


def build_subtask_probe(code: str, acceptance_check: str) -> tuple[str, bool]:
    """Append a subtask's ``acceptance_check`` asserts to its candidate.

    Each assert is augmented when it is a simple equality so repair gets
    ``f(x) -> <actual>, want <expected>`` instead of a bare AssertionError.
    Multiple asserts run in order; the first failure stops execution and
    surfaces that check's message. Malformed individual checks are skipped.
    """
    if not code.strip() or not acceptance_check.strip():
        return code, False

    augmented: list[str] = []
    for i, raw in enumerate(split_acceptance_checks(acceptance_check)):
        piece = _augment_equality_check(raw, i) or raw.strip()
        try:
            ast.parse(piece)
        except (SyntaxError, ValueError):
            continue
        augmented.append(piece)

    if not augmented:
        return code, False
    return with_probe_imports(f"{code}\n\n" + "\n".join(augmented)), True
