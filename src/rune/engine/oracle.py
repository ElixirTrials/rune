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
import doctest
import logging

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
    if not checks or not defines_function(code, entry_point):
        return code, False
    return f"{code}\n\n{checks}", True


def build_subtask_probe(code: str, acceptance_check: str) -> tuple[str, bool]:
    """Append a subtask's ``acceptance_check`` to its candidate, with an
    actual-vs-expected failure message so repair gets a usable signal.

    The episodic per-subtask check is the in-loop signal. A *bare* ``assert f(x)
    == y`` only yields "AssertionError @ line", which repair cannot act on (e.g.
    a correct algorithm that returns a tuple where a list is expected). So an
    equality check is rewritten to run the call and report ``f(x) -> <actual>,
    want <expected>``. Non-equality / unparseable checks fall through unchanged
    (or skip, so a malformed check never crashes the sandbox into spurious repair).
    """
    if not code.strip() or not acceptance_check.strip():
        return code, False
    try:
        tree = ast.parse(acceptance_check.strip())
    except (SyntaxError, ValueError):
        return code, False
    check = acceptance_check
    if (
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.Assert)
        and isinstance(tree.body[0].test, ast.Compare)
        and len(tree.body[0].test.ops) == 1
        and isinstance(tree.body[0].test.ops[0], ast.Eq)
    ):
        cmp = tree.body[0].test
        check = _augmented_assert(
            ast.unparse(cmp.left), ast.unparse(cmp.comparators[0]), 0
        )
    return f"{code}\n\n{check}", True
