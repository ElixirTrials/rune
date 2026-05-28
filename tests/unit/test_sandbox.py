from rune.engine.continuation import extract_code
from rune.sandbox.executor import ExecutionResult


class TestExtractCode:
    def test_extract_fenced_python(self) -> None:
        raw = "Here is code:\n```python\nprint('hello')\n```\nDone."
        assert extract_code(raw) == "print('hello')"

    def test_extract_unfenced(self) -> None:
        raw = "print('hello')"
        assert extract_code(raw) == "print('hello')"

    def test_extract_multiple_blocks_takes_longest(self) -> None:
        raw = "```python\nx = 1\n```\n\n```python\nx = 1\ny = 2\nz = 3\n```"
        assert "z = 3" in extract_code(raw)


class TestExecutionResult:
    def test_passed(self) -> None:
        r = ExecutionResult(stdout="ok", stderr="", exit_code=0)
        assert r.exit_code == 0
