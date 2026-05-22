from rune.sandbox.executor import run_in_sandbox


class TestRunInSandbox:
    def test_passing_code(self) -> None:
        result = run_in_sandbox("print('hello')")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_failing_code(self) -> None:
        result = run_in_sandbox("raise ValueError('boom')")
        assert result.exit_code == 1
        assert "ValueError" in result.stderr

    def test_timeout(self) -> None:
        result = run_in_sandbox("import time; time.sleep(10)", timeout=1)
        assert result.exit_code != 0

    def test_syntax_error(self) -> None:
        result = run_in_sandbox("def (broken")
        assert result.exit_code != 0
