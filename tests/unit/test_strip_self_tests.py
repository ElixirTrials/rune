"""Tests for strip_self_tests — TDD for the in-loop self-test stripping feature."""

from __future__ import annotations

import textwrap

from rune.engine.continuation import strip_self_tests
from rune.sandbox.executor import run_in_sandbox

# ---------------------------------------------------------------------------
# Unit tests for strip_self_tests
# ---------------------------------------------------------------------------


class TestStripSelfTests:
    """Each construct that should be stripped or preserved."""

    def _strip(self, src: str) -> str:
        return strip_self_tests(textwrap.dedent(src).strip())

    def test_module_level_assert_removed(self) -> None:
        code = textwrap.dedent("""\
            def add(a, b):
                return a + b
            assert add(1, 2) == 3
        """)
        result = strip_self_tests(code)
        assert "assert" not in result
        assert "def add" in result

    def test_test_function_removed(self) -> None:
        code = textwrap.dedent("""\
            def add(a, b):
                return a + b
            def test_add():
                assert add(1, 2) == 3
        """)
        result = strip_self_tests(code)
        assert "def test_add" not in result
        assert "def add" in result

    def test_async_test_function_removed(self) -> None:
        code = textwrap.dedent("""\
            def add(a, b):
                return a + b
            async def test_add_async():
                assert add(1, 2) == 3
        """)
        result = strip_self_tests(code)
        assert "test_add_async" not in result
        assert "def add" in result

    def test_unittest_testcase_subclass_by_base_removed(self) -> None:
        code = textwrap.dedent("""\
            import unittest
            def add(a, b):
                return a + b
            class TestAdd(unittest.TestCase):
                def test_it(self):
                    self.assertEqual(add(1, 2), 3)
        """)
        result = strip_self_tests(code)
        assert "TestAdd" not in result
        assert "def add" in result

    def test_testcase_subclass_bare_base_removed(self) -> None:
        """class Foo(TestCase): ... should also be removed."""
        code = textwrap.dedent("""\
            def mul(a, b):
                return a * b
            class SomeTests(TestCase):
                def test_mul(self):
                    pass
        """)
        result = strip_self_tests(code)
        assert "SomeTests" not in result
        assert "def mul" in result

    def test_class_name_starting_with_test_removed(self) -> None:
        code = textwrap.dedent("""\
            def sub(a, b):
                return a - b
            class TestSub:
                def run(self):
                    assert sub(5, 3) == 2
        """)
        result = strip_self_tests(code)
        assert "TestSub" not in result
        assert "def sub" in result

    def test_if_name_main_block_removed(self) -> None:
        code = textwrap.dedent("""\
            def greet(name):
                return f"Hello, {name}"
            if __name__ == "__main__":
                print(greet("World"))
        """)
        result = strip_self_tests(code)
        assert '__name__ == "__main__"' not in result
        assert "def greet" in result

    def test_unittest_main_call_removed(self) -> None:
        code = textwrap.dedent("""\
            import unittest
            def add(a, b):
                return a + b
            unittest.main()
        """)
        result = strip_self_tests(code)
        assert "unittest.main()" not in result
        assert "def add" in result

    def test_pytest_main_call_removed(self) -> None:
        code = textwrap.dedent("""\
            import pytest
            def add(a, b):
                return a + b
            pytest.main([__file__])
        """)
        result = strip_self_tests(code)
        assert "pytest.main" not in result
        assert "def add" in result

    def test_nested_assert_inside_impl_function_preserved(self) -> None:
        """Asserts inside function bodies are implementation logic — must not be stripped."""
        code = textwrap.dedent("""\
            def divide(a, b):
                assert b != 0, "division by zero"
                return a / b
        """)
        result = strip_self_tests(code)
        assert "assert b != 0" in result
        assert "def divide" in result

    def test_impl_function_preserved(self) -> None:
        code = textwrap.dedent("""\
            def helper(x):
                return x * 2
            def main_logic(x):
                return helper(x) + 1
        """)
        result = strip_self_tests(code)
        assert "def helper" in result
        assert "def main_logic" in result

    def test_impl_class_preserved(self) -> None:
        code = textwrap.dedent("""\
            class Stack:
                def __init__(self):
                    self._items = []
                def push(self, item):
                    self._items.append(item)
        """)
        result = strip_self_tests(code)
        assert "class Stack" in result

    def test_imports_preserved(self) -> None:
        code = textwrap.dedent("""\
            import os
            from pathlib import Path
            def get_cwd():
                return Path.cwd()
        """)
        result = strip_self_tests(code)
        assert "import os" in result
        assert "from pathlib import Path" in result
        assert "def get_cwd" in result

    def test_syntax_error_returns_original_unchanged(self) -> None:
        bad_code = "def foo(\n    # incomplete"
        result = strip_self_tests(bad_code)
        assert result == bad_code

    def test_empty_after_strip_returns_original(self) -> None:
        """A tests-only blob with no impl should return the original, not empty string."""
        tests_only = textwrap.dedent("""\
            assert 1 == 2
            assert True
        """)
        result = strip_self_tests(tests_only)
        # stripped result would be empty -> must return original (with trailing newline)
        assert result == tests_only

    def test_impl_class_not_starting_with_test_preserved(self) -> None:
        """Classes like 'Tester' or 'testing_helper' should NOT be stripped."""
        code = textwrap.dedent("""\
            class Tester:
                def run(self):
                    pass
        """)
        result = strip_self_tests(code)
        assert "class Tester" in result

    def test_multiple_constructs_stripped_together(self) -> None:
        code = textwrap.dedent("""\
            def add(a, b):
                return a + b
            def test_add():
                assert add(1, 2) == 3
            assert add(0, 0) == 0
            if __name__ == "__main__":
                test_add()
        """)
        result = strip_self_tests(code)
        assert "def add" in result
        assert "def test_add" not in result
        assert "assert add(0, 0)" not in result
        assert "__main__" not in result

    def test_if_name_main_reversed_operand_stripped(self) -> None:
        """if '"__main__" == __name__:' (reversed) must also be stripped."""
        code = textwrap.dedent("""\
            def greet(name):
                return f"Hello, {name}"
            if "__main__" == __name__:
                print(greet("World"))
        """)
        result = strip_self_tests(code)
        assert "__main__" not in result
        assert "def greet" in result

    def test_bare_test_call_stripped_non_test_call_preserved(self) -> None:
        """Bare module-level test_*() calls are stripped; non-test calls are kept."""
        code = textwrap.dedent("""\
            def add(a, b):
                return a + b
            def test_add():
                assert add(1, 2) == 3
            test_add()
            setup()
        """)
        result = strip_self_tests(code)
        assert "test_add()" not in result
        assert "setup()" in result
        assert "def add" in result

    def test_orphan_bare_test_call_sandbox_exit_zero(self) -> None:
        """Impl + wrong def test_add + bare test_add() call: strip removes both,
        sandbox must exit 0 (no NameError, no AssertionError).
        Without fix: unstripped blob exits non-zero (AssertionError: 1+2 != 5).
        """
        blob = textwrap.dedent("""\
            def add(a, b):
                return a + b
            def test_add():
                assert add(1, 2) == 5
            test_add()
        """)
        # Sanity check: unstripped blob fails (AssertionError from wrong test)
        raw_result = run_in_sandbox(blob)
        assert raw_result.exit_code != 0, (
            f"Expected unstripped blob to fail, got exit_code={raw_result.exit_code!r}"
        )
        # After stripping: both the test def and the orphan call are gone → exit 0
        stripped_result = run_in_sandbox(strip_self_tests(blob))
        assert stripped_result.exit_code == 0, (
            f"Expected exit_code 0 after stripping orphan test call, "
            f"got {stripped_result.exit_code!r}. stderr={stripped_result.stderr!r}"
        )

    def test_reversed_guard_orphan_sandbox_exit_zero(self) -> None:
        """Reversed guard + test_x() inside it + correct impl → strip → sandbox exit 0."""
        blob = textwrap.dedent("""\
            def add(a, b):
                return a + b
            def test_x():
                assert False
            if "__main__" == __name__:
                test_x()
        """)
        result = run_in_sandbox(strip_self_tests(blob))
        assert result.exit_code == 0, (
            f"Expected exit_code 0 after stripping reversed guard, "
            f"got {result.exit_code!r}. stderr={result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# Reproduction test: correct impl + wrong self-test → sandbox passes
# ---------------------------------------------------------------------------


def test_repro_correct_impl_with_wrong_self_test_no_longer_fails_sandbox() -> None:
    """Regression: a correct implementation with a wrong module-level self-test
    used to cause spurious sandbox failure → repair loop exhaustion (~8 min on
    mbpp/279). After stripping, the sandbox must exit 0.
    """
    blob = textwrap.dedent("""\
        def add(a, b):
            return a + b
        assert add(1, 2) == 5
    """)
    stripped = strip_self_tests(blob)
    result = run_in_sandbox(stripped)
    assert result.exit_code == 0, (
        f"Expected exit_code 0 after stripping wrong self-test, "
        f"got {result.exit_code!r}. stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Inverse: tests-only blob must NOT become a vacuous pass
# ---------------------------------------------------------------------------


def test_tests_only_blob_is_not_vacuous_pass() -> None:
    """A blob that contains only an assert against an undefined function should:
    1. Return the original code from strip_self_tests (since stripping would
       leave it empty).
    2. Fail the sandbox (NameError → nonzero exit_code).
    """
    tests_only = "assert undefined_fn() == 1\n"
    stripped = strip_self_tests(tests_only)
    # strip must return the original (not empty)
    assert "undefined_fn" in stripped, (
        "strip_self_tests should return original when stripped result is empty"
    )
    result = run_in_sandbox(stripped)
    assert result.exit_code != 0, (
        f"Tests-only blob should fail the sandbox, got exit_code={result.exit_code!r}"
    )
