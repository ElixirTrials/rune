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

    def test_future_import_submission_passes(self) -> None:
        # The mem-guard must not displace header-only constructs: a submission
        # starting with `from __future__ import ...` graded as SyntaxError was
        # an incorrect pass@1 failure.
        r = run_in_sandbox("from __future__ import annotations\nprint('ok')")
        assert r.exit_code == 0
        assert "ok" in r.stdout

    def test_module_docstring_preserved(self) -> None:
        code = '"""doc"""\nprint(__doc__)'
        r = run_in_sandbox(code)
        assert r.exit_code == 0
        assert "doc" in r.stdout

    def test_mem_guard_still_binds(self) -> None:
        # A >4GB allocation must raise MemoryError inside the sandbox instead
        # of OOM-killing the host.
        code = "x = bytearray(6 * 1024**3)\nprint('allocated')"
        r = run_in_sandbox(code)
        assert r.exit_code != 0
        assert "MemoryError" in r.stderr

    def test_dunder_main_semantics(self) -> None:
        r = run_in_sandbox("print(__name__)")
        assert r.exit_code == 0
        assert "__main__" in r.stdout
