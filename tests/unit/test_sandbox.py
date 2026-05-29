from rune.sandbox.executor import ExecutionResult, run_in_sandbox


class TestExecutionResult:
    def test_passed(self) -> None:
        r = ExecutionResult(stdout="ok", stderr="", exit_code=0)
        assert r.exit_code == 0


class TestRunInSandbox:
    def test_runs_basic_code(self) -> None:
        r = run_in_sandbox("print('hello')")
        assert r.exit_code == 0
        assert "hello" in r.stdout

    def test_non_utf8_output_does_not_raise(self) -> None:
        # Untrusted code writing raw bytes must degrade to an ExecutionResult,
        # not raise UnicodeDecodeError out of run_in_sandbox.
        code = r"import sys; sys.stdout.buffer.write(b'\xff\xfe'); print('done')"
        r = run_in_sandbox(code)
        assert isinstance(r, ExecutionResult)
        assert r.exit_code == 0
