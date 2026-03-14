"""Tests for shared.sandbox — SubprocessBackend, NsjailBackend, get_sandbox_backend."""

from shared.sandbox import (
    NsjailBackend,
    SubprocessBackend,
    get_sandbox_backend,
)


def test_subprocess_runs_code() -> None:
    backend = SubprocessBackend()
    result = backend.run('print("hello")')
    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0
    assert result.timed_out is False


def test_subprocess_captures_error() -> None:
    backend = SubprocessBackend()
    result = backend.run('raise ValueError("boom")')
    assert result.exit_code != 0
    assert "ValueError" in result.stderr


def test_subprocess_timeout() -> None:
    backend = SubprocessBackend()
    result = backend.run("import time; time.sleep(60)", timeout=2)
    assert result.timed_out is True
    assert result.exit_code == 1


def test_nsjail_falls_back_when_missing(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    backend = NsjailBackend()
    result = backend.run('print("fallback")')
    assert result.stdout.strip() == "fallback"
    assert result.exit_code == 0


def test_get_backend_default() -> None:
    backend = get_sandbox_backend()
    assert isinstance(backend, SubprocessBackend)


def test_get_backend_nsjail_env(monkeypatch) -> None:
    monkeypatch.setenv("RUNE_EXEC_BACKEND", "nsjail")
    backend = get_sandbox_backend()
    assert isinstance(backend, NsjailBackend)
