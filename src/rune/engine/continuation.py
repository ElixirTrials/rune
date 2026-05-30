"""Shared continuation utilities for code extraction and quality."""

from __future__ import annotations

import ast
import re

from rune.engine.json_repair import extract_code_value
from rune.engine.parse import CodeResult

CONT_SYSTEM_PROMPT = (
    "Output only Python code. No commentary, no explanations, "
    "no markdown fences. Continue exactly from where the code "
    "left off."
)


def extract_partial_code(raw: str) -> str:
    """Extract code from a possibly-truncated CodeResult JSON string.

    Falls back to *raw* when input isn't JSON at all (e.g. continuation
    rounds that emit plain Python).
    """
    try:
        return CodeResult.model_validate_json(raw).code
    except Exception:
        return extract_code_value(raw) or raw


_NUM_RE = re.compile(r"\b\d+\b")


def degeneration_score(text: str, n: int = 4) -> float:
    normalized = _NUM_RE.sub("<N>", text)
    words = normalized.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def validate_syntax(code: str, *, language: str = "python") -> bool:
    if not code or not code.strip():
        return False
    if language != "python":
        raise NotImplementedError(f"syntax validation not implemented for {language!r}")
    try:
        compile(code, "<check>", "exec")
        return True
    except SyntaxError:
        return False


def _is_test_function(node: ast.stmt) -> bool:
    return isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)
    ) and node.name.startswith("test")


def _is_test_class(node: ast.stmt) -> bool:
    if not isinstance(node, ast.ClassDef):
        return False
    if node.name.startswith("Test"):
        return True
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "TestCase":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "TestCase":
            return True
    return False


def _is_if_name_main(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    # Match both `__name__ == "__main__"` and `"__main__" == __name__`.
    name_left = isinstance(left, ast.Name) and left.id == "__name__"
    main_right = isinstance(right, ast.Constant) and right.value == "__main__"
    name_right = isinstance(right, ast.Name) and right.id == "__name__"
    main_left = isinstance(left, ast.Constant) and left.value == "__main__"
    return (name_left and main_right) or (main_left and name_right)


def _is_runner_main_call(node: ast.stmt) -> bool:
    """Match top-level bare calls: unittest.main(...) or pytest.main(...)."""
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "main"
        and isinstance(func.value, ast.Name)
        and func.value.id in {"unittest", "pytest"}
    )


def _is_bare_test_call(node: ast.stmt) -> bool:
    """Match bare module-level calls whose name starts with 'test', e.g. test_add().

    Conservative: only bare ast.Name calls (not attribute calls like obj.test_x()
    and not assignments like r = test_x()).
    """
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    return isinstance(func, ast.Name) and func.id.startswith("test")


def strip_self_tests(code: str) -> str:
    """Strip model self-authored tests from generated code for sandbox execution.

    Only top-level (module-level) test constructs are removed. Asserts and test
    helpers nested inside function/class bodies are preserved — those are
    implementation logic, not tests.

    Constructs removed:
    - Module-level ``assert`` statements
    - Functions/async functions whose name starts with ``test``
    - Classes that subclass ``TestCase`` (bare or ``unittest.TestCase``) or whose
      name starts with ``Test``
    - ``if __name__ == "__main__":`` blocks (both operand orders)
    - Bare top-level ``unittest.main()`` / ``pytest.main()`` calls
    - Bare top-level ``test*()`` calls (orphan references to stripped test defs)

    On ``SyntaxError`` (or ``ValueError`` from embedded null bytes), the original
    code is returned unchanged so that a syntax-broken implementation still fails
    the sandbox — a genuine repair signal.

    If stripping leaves an empty or whitespace-only module, the original code is
    returned so a tests-only-no-impl blob fails with ``NameError`` rather than
    becoming a vacuous empty-file pass.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code

    kept: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assert):
            continue
        if _is_test_function(node):
            continue
        if _is_test_class(node):
            continue
        if _is_if_name_main(node):
            continue
        if _is_runner_main_call(node):
            continue
        if _is_bare_test_call(node):
            continue
        kept.append(node)

    if not kept:
        return code

    tree.body = kept
    stripped = ast.unparse(tree)
    if not stripped.strip():
        return code
    return stripped
