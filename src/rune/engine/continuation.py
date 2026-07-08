"""Shared continuation utilities for code extraction and quality."""

from __future__ import annotations

import ast
import re
import textwrap

from rune.engine.parse import extract_code_block

CONT_SYSTEM_PROMPT = (
    "Output only Python code. No commentary, no explanations, "
    "no markdown fences. Continue exactly from where the code "
    "left off."
)


def extract_partial_code(raw: str) -> str:
    """De-fence freeform model code output (a ```python fence or bare code).

    Code actions are freeform — never JSON — so extraction is a single CommonMark
    de-fence; bare code (continuation rounds emit plain Python) passes through.
    """
    return extract_code_block(raw)


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


def _chunk_has_code(chunk: str) -> bool:
    """True when a continuation chunk contains at least one code statement.

    A prose ramble (issue #52 q3754: "Given the ambiguity ... I will output: 0")
    parses to nothing but bare string/number literals; a real code tail parses —
    either as-is or after dedenting an indented body fragment — to a statement
    that is not a bare literal. Cheap: at most two ``ast.parse`` attempts.
    """
    stripped = chunk.strip()
    if not stripped:
        return False
    for candidate in (stripped, textwrap.dedent(chunk).strip()):
        try:
            module = ast.parse(candidate)
        except SyntaxError:
            continue
        for node in module.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # bare literal == prose, not code
            return True
    return False


def _salvageable_entry(code: str, entry_point: str) -> str | None:
    """Largest recoverable definition of ``entry_point`` in ``code``, or None.

    Reuses the LCB salvage/extract helpers so the guard's notion of "complete
    entry function" matches what the ship gate would recover. Deferred import
    avoids an engine->bench import cycle.
    """
    from rune.bench.lcb import (  # noqa: PLC0415
        _salvage_entry_function,
        extract_entry_function,
    )

    text = (code or "").strip()
    if not text or not entry_point:
        return None
    try:
        ast.parse(text)
    except SyntaxError:
        salvaged = _salvage_entry_function(text, entry_point)
    else:
        salvaged = extract_entry_function(text, entry_point)
    if not salvaged or not salvaged.strip():
        return None
    try:
        tree = ast.parse(salvaged)
    except SyntaxError:
        return None
    defines = any(
        isinstance(node, ast.FunctionDef) and node.name == entry_point
        for node in ast.walk(tree)
    )
    return salvaged if defines else None


def continuation_should_abort(
    new_chunk: str, accumulated_code: str, entry_point: str
) -> bool:
    """Structural stop for the continuation sub-loop (issue #52 §4 lever 4).

    Returns True when the freshly generated chunk is NOT a plausible code
    continuation (no parseable statement — it is prose) AND the code accumulated
    so far already yields a salvageable definition of ``entry_point``. In that
    state further continuation only appends prose to an already-recoverable
    function, so stop. Independent of the 0.5 degeneration threshold; cheap
    (``ast.parse`` only).
    """
    if not entry_point:
        return False
    if _chunk_has_code(new_chunk):
        return False
    return _salvageable_entry(accumulated_code, entry_point) is not None


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
