"""Reliability tests for MathSandbox.

Covers the failure modes that matter for large-scale LLM training and
inference: long turn sequences, concurrent worker pools, error recovery,
heavy computation, state correctness, and timeout resilience.

Run the full suite:
    uv run pytest tests/test_mathbox.py -v

Skip the slower stress tests:
    uv run pytest tests/test_mathbox.py -v -m "not slow"
"""

from __future__ import annotations

import threading
from typing import Generator

import pytest

from shared.mathbox import MathConfig, MathSandbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def box() -> Generator[MathSandbox, None, None]:
    """Fresh sandbox for each test, closed on teardown."""
    with MathSandbox() as b:
        yield b


# ---------------------------------------------------------------------------
# 1. Basic execution
# ---------------------------------------------------------------------------


def test_basic_print(box: MathSandbox) -> None:
    assert box.execute('print("hello")').strip() == "hello"


def test_no_output_warning(box: MathSandbox) -> None:
    out = box.execute("x = 1")
    assert "[WARN]" in out


def test_expression_result_captured(box: MathSandbox) -> None:
    # execute_result / display_data path (repr, not print)
    out = box.execute("1 + 1")
    assert "2" in out


def test_multiline_block(box: MathSandbox) -> None:
    out = box.execute(
        "total = 0\n"
        "for i in range(10):\n"
        "    total += i\n"
        "print(total)"
    )
    assert out.strip() == "45"


# ---------------------------------------------------------------------------
# 2. State persistence across calls (critical for multi-turn inference)
# ---------------------------------------------------------------------------


def test_variable_persists(box: MathSandbox) -> None:
    box.execute("x = 42")
    assert box.execute("print(x)").strip() == "42"


def test_function_definition_persists(box: MathSandbox) -> None:
    box.execute("def square(n): return n * n")
    assert box.execute("print(square(7))").strip() == "49"


def test_class_definition_persists(box: MathSandbox) -> None:
    box.execute(
        "class Counter:\n"
        "    def __init__(self): self.n = 0\n"
        "    def inc(self): self.n += 1; return self.n"
    )
    box.execute("c = Counter()")
    box.execute("c.inc(); c.inc()")
    assert box.execute("print(c.inc())").strip() == "3"


def test_accumulation_over_many_calls(box: MathSandbox) -> None:
    """Simulates a model that incrementally builds a computation over turns."""
    box.execute("acc = []")
    for i in range(20):
        box.execute(f"acc.append({i})")
    out = box.execute("print(sum(acc))")
    assert out.strip() == str(sum(range(20)))


def test_variable_overwrite(box: MathSandbox) -> None:
    box.execute("val = 'first'")
    box.execute("val = 'second'")
    assert box.execute("print(val)").strip() == "second"


# ---------------------------------------------------------------------------
# 3. Pre-imported math libraries
# ---------------------------------------------------------------------------


def test_sympy_available(box: MathSandbox) -> None:
    out = box.execute("print(sympy.factorint(360))")
    assert "2" in out and "3" in out and "5" in out


def test_numpy_available(box: MathSandbox) -> None:
    out = box.execute("print(numpy.dot([1, 2, 3], [4, 5, 6]))")
    assert out.strip() == "32"


def test_math_available(box: MathSandbox) -> None:
    out = box.execute("print(math.factorial(10))")
    assert out.strip() == "3628800"


def test_mpmath_precision(box: MathSandbox) -> None:
    out = box.execute("print(mpmath.mp.dps)")
    assert out.strip() == "64"


def test_itertools_available(box: MathSandbox) -> None:
    out = box.execute("print(list(itertools.combinations([1,2,3], 2)))")
    assert "(1, 2)" in out


def test_collections_available(box: MathSandbox) -> None:
    out = box.execute("c = collections.Counter('aabbbc'); print(c['b'])")
    assert out.strip() == "3"


# ---------------------------------------------------------------------------
# 4. Additional imports inside the kernel
# ---------------------------------------------------------------------------


def test_in_kernel_import(box: MathSandbox) -> None:
    out = box.execute("import fractions; print(fractions.Fraction(1, 3))")
    assert "1/3" in out


def test_in_kernel_import_persists(box: MathSandbox) -> None:
    box.execute("import fractions")
    out = box.execute("print(fractions.Fraction(2, 6))")
    assert "1/3" in out


# ---------------------------------------------------------------------------
# 5. Error handling — kernel must survive and stay usable
# ---------------------------------------------------------------------------


def test_runtime_error_does_not_kill_kernel(box: MathSandbox) -> None:
    out = box.execute("1 / 0")
    assert "ZeroDivisionError" in out
    # Kernel still alive
    assert box.execute("print('alive')").strip() == "alive"


def test_syntax_error_does_not_kill_kernel(box: MathSandbox) -> None:
    out = box.execute("def f(: pass")
    assert "SyntaxError" in out or "ERROR" in out
    assert box.execute("print('still alive')").strip() == "still alive"


def test_name_error_recovery(box: MathSandbox) -> None:
    out = box.execute("print(undefined_variable)")
    assert "NameError" in out
    box.execute("undefined_variable = 99")
    assert box.execute("print(undefined_variable)").strip() == "99"


def test_import_error_recovery(box: MathSandbox) -> None:
    out = box.execute("import nonexistent_package_xyz")
    assert "ModuleNotFoundError" in out or "ImportError" in out
    assert box.execute("print(1 + 1)").strip() == "2"


def test_stderr_included_in_output(box: MathSandbox) -> None:
    out = box.execute("import sys; sys.stderr.write('err line\\n')")
    assert "err line" in out


# ---------------------------------------------------------------------------
# 6. Reset behaviour
# ---------------------------------------------------------------------------


def test_reset_clears_variables(box: MathSandbox) -> None:
    box.execute("secret = 12345")
    box.reset()
    out = box.execute("print('secret' in dir())")
    assert out.strip() == "False"


def test_reset_restores_prelude_imports(box: MathSandbox) -> None:
    box.reset()
    assert box.execute("print(math.pi > 3)").strip() == "True"
    assert box.execute("print(mpmath.mp.dps)").strip() == "64"


def test_reset_multiple_times(box: MathSandbox) -> None:
    for _ in range(5):
        box.execute("x = 42")
        box.reset()
    out = box.execute("print('x' in dir())")
    assert out.strip() == "False"


# ---------------------------------------------------------------------------
# 7. Timeout — kernel must survive and stay usable after interruption
# ---------------------------------------------------------------------------


def test_timeout_returns_error_message(box: MathSandbox) -> None:
    out = box.execute("import time; time.sleep(60)", timeout=2.0)
    assert "[ERROR]" in out and "timed out" in out


def test_kernel_alive_after_timeout(box: MathSandbox) -> None:
    box.execute("import time; time.sleep(60)", timeout=2.0)
    assert box.execute("print('recovered')").strip() == "recovered"


def test_custom_timeout_via_config() -> None:
    cfg = MathConfig(default_timeout=3.0)
    with MathSandbox(config=cfg) as b:
        out = b.execute("import time; time.sleep(60)")
        assert "[ERROR]" in out and "timed out" in out
        assert b.execute("print('ok')").strip() == "ok"


# ---------------------------------------------------------------------------
# 8. Heavy / realistic math computations
# ---------------------------------------------------------------------------


def test_large_prime_check(box: MathSandbox) -> None:
    out = box.execute("print(sympy.isprime(2**31 - 1))")
    assert out.strip() == "True"


def test_numpy_matrix_multiply(box: MathSandbox) -> None:
    out = box.execute(
        "A = numpy.ones((50, 50))\n"
        "B = numpy.ones((50, 50))\n"
        "print(int((A @ B)[0, 0]))"
    )
    assert out.strip() == "50"


def test_sympy_solve_system(box: MathSandbox) -> None:
    out = box.execute(
        "x, y = sympy.symbols('x y')\n"
        "sol = sympy.solve([x + y - 10, x - y - 2], [x, y])\n"
        "print(sol[x], sol[y])"
    )
    assert "6" in out and "4" in out


def test_mpmath_high_precision_pi(box: MathSandbox) -> None:
    out = box.execute("print(str(mpmath.pi)[:7])")
    assert out.strip() == "3.14159"


def test_combinatorics_count(box: MathSandbox) -> None:
    out = box.execute(
        "print(sum(1 for _ in itertools.combinations(range(20), 3)))"
    )
    assert out.strip() == "1140"


# ---------------------------------------------------------------------------
# 9. Large output (inference can produce verbose traces)
# ---------------------------------------------------------------------------


def test_large_stdout(box: MathSandbox) -> None:
    out = box.execute("for i in range(500): print(i)")
    lines = [l for l in out.strip().splitlines() if l]
    assert len(lines) == 500
    assert lines[-1] == "499"


# ---------------------------------------------------------------------------
# 10. Concurrent sandboxes (parallel inference workers)
# ---------------------------------------------------------------------------


def test_concurrent_sandboxes_isolated() -> None:
    """Two sandboxes run simultaneously; their state must not bleed."""
    results: dict[int, str] = {}
    errors: list[Exception] = []

    def worker(idx: int, value: int) -> None:
        try:
            with MathSandbox() as b:
                b.execute(f"x = {value}")
                b.execute("import time; time.sleep(0.1)")
                results[idx] = b.execute("print(x)").strip()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i, i * 10)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    for i in range(4):
        assert results[i] == str(i * 10), f"worker {i}: got {results[i]}"


# ---------------------------------------------------------------------------
# 11. Multi-turn inference simulation (stress)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_long_inference_session(box: MathSandbox) -> None:
    """Simulates 50 turns of a model that builds on prior context."""
    box.execute("results = []")
    for i in range(50):
        box.execute(f"results.append(sympy.isprime({i}))")
    out = box.execute("print(len(results))")
    assert out.strip() == "50"
    out = box.execute("print(results[2])")  # 2 is prime
    assert out.strip() == "True"
    out = box.execute("print(results[4])")  # 4 is not
    assert out.strip() == "False"


@pytest.mark.slow
def test_repeated_reset_and_execute_cycles(box: MathSandbox) -> None:
    """Simulates a worker that resets between problems (training loop)."""
    for problem_idx in range(20):
        box.reset()
        box.execute(f"n = {problem_idx + 1}")
        out = box.execute("print(n * n)")
        assert out.strip() == str((problem_idx + 1) ** 2)
