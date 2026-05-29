from rune.sandbox.executor import ExecutionResult


class TestExecutionResult:
    def test_passed(self) -> None:
        r = ExecutionResult(stdout="ok", stderr="", exit_code=0)
        assert r.exit_code == 0
