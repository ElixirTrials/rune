from rune.bench.runner import BenchTask, build_graded_program
from rune.engine.oracle import _PROBE_IMPORT_PREAMBLE
from rune.sandbox.executor import run_in_sandbox

_CODE = "def f(x: List[int]) -> int:\n    return x[0]"  # signature needs `List`
_TEST = "assert f([1, 2]) == 1"


def _task(preamble: str = "") -> BenchTask:
    return BenchTask(
        task_id="t", description="", test_code=_TEST, entry_point="f",
        grading_preamble=preamble,
    )


def test_preamble_prepended_before_solution() -> None:
    prog = build_graded_program(_task("from typing import List"), _CODE)
    assert "from typing import List\n" in prog
    assert "def f(x: List[int])" in prog
    assert _TEST in prog
    # grading_preamble sits before the solution body (the mirror imports precede
    # both — see test_no_preamble_supplies_mirror_imports).
    assert prog.index("from typing import List") < prog.index("def f(")


def test_no_preamble_supplies_mirror_imports() -> None:
    # With no grading_preamble, the grader-mirror imports are the only prefix
    # (issue #52 §2.A2): the ship gate always mirrors the official star-imports.
    prog = build_graded_program(_task(), "def g():\n    return 2")
    assert prog.startswith(_PROBE_IMPORT_PREAMBLE)
    assert prog[len(_PROBE_IMPORT_PREAMBLE):].startswith("def g():")


def test_mirror_imports_fix_nameerror_end_to_end() -> None:
    # The grader-mirror preamble supplies ``List``, so a List-annotated signature
    # grades cleanly even with no HumanEval grading_preamble (issue #52 §2.A2).
    assert run_in_sandbox(build_graded_program(_task(), _CODE)).exit_code == 0
    # A redundant grading_preamble import is harmless.
    fixed = build_graded_program(_task("from typing import List"), _CODE)
    assert run_in_sandbox(fixed).exit_code == 0


def test_self_tests_still_stripped() -> None:
    # a wrong model self-test must not fail a correct solution
    code = "def f(x: List[int]) -> int:\n    return x[0]\nassert f([9]) == 999"
    prog = build_graded_program(_task("from typing import List"), code)
    assert run_in_sandbox(prog).exit_code == 0
