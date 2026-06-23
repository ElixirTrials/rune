from rune.bench.runner import BenchTask, build_graded_program
from rune.sandbox.executor import run_in_sandbox

_CODE = "def f(x: List[int]) -> int:\n    return x[0]"  # signature needs `List`
_TEST = "assert f([1, 2]) == 1"


def _task(preamble: str = "") -> BenchTask:
    return BenchTask(
        task_id="t",
        description="",
        test_code=_TEST,
        entry_point="f",
        grading_preamble=preamble,
    )


def test_preamble_prepended_before_solution() -> None:
    prog = build_graded_program(_task("from typing import List"), _CODE)
    assert prog.startswith("from typing import List\n")
    assert "def f(x: List[int])" in prog
    assert _TEST in prog


def test_no_preamble_is_noop() -> None:
    prog = build_graded_program(_task(), "def g():\n    return 2")
    assert prog.startswith("def g():")  # nothing prepended


def test_preamble_fixes_nameerror_end_to_end() -> None:
    # The bug: a List-annotated signature NameErrors without the prompt import.
    assert run_in_sandbox(build_graded_program(_task(), _CODE)).exit_code != 0
    # The fix: the preamble supplies the import, the same code passes.
    fixed = build_graded_program(_task("from typing import List"), _CODE)
    assert run_in_sandbox(fixed).exit_code == 0


def test_self_tests_still_stripped() -> None:
    # a wrong model self-test must not fail a correct solution
    code = "def f(x: List[int]) -> int:\n    return x[0]\nassert f([9]) == 999"
    prog = build_graded_program(_task("from typing import List"), code)
    assert run_in_sandbox(prog).exit_code == 0
