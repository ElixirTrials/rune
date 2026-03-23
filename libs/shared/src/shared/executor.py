"""Base executor abstraction and subprocess-based implementation.

Defines the ``BaseExecutor`` ABC that all execution backends must implement,
plus ``SubprocessExecutor``, a zero-dependency implementation that runs Python
code in isolated ``tempfile`` directories via ``subprocess``.

Executors support async context managers for clean lifecycle management::

    async with SubprocessExecutor() as ex:
        output = await ex.execute("print(42)")
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class BaseExecutor(ABC):
    """Abstract base class for code execution backends.

    All executor implementations must be usable as async context managers.
    The ``start`` / ``stop`` pair handles resource provisioning and teardown.

    The ``dispatch`` helper provides a synchronous entry-point that is
    compatible with sync agent loops — it delegates to ``_dispatch_async``
    via the running event loop.
    """

    async def __aenter__(self) -> "BaseExecutor":
        """Start the executor and return self."""
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop the executor on context exit."""
        await self.stop()

    @abstractmethod
    async def start(self) -> None:
        """Provision the execution environment.

        Must be called (or entered via async-with) before any other method.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Tear down the execution environment and release resources."""

    @abstractmethod
    async def execute(self, code: str) -> str:
        """Execute Python code and return combined stdout / stderr output.

        Args:
            code: Valid Python source code to execute.

        Returns:
            A string combining stdout, stderr (prefixed with ``[stderr]``),
            and any result value.  Returns ``"(no output)"`` when all streams
            are empty.
        """

    @abstractmethod
    async def write_file(self, path: str, content: str) -> str:
        """Write a file to the execution environment's filesystem.

        Args:
            path: Absolute or session-relative file path.
            content: UTF-8 text content to write.

        Returns:
            Confirmation string, e.g. ``"written: /tmp/data.csv"``.
        """

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read a file from the execution environment's filesystem.

        Args:
            path: Absolute or session-relative file path.

        Returns:
            File contents as a UTF-8 string.
        """

    def dispatch(self, name: str, args: dict) -> str:  # type: ignore[type-arg]
        """Synchronous tool dispatcher for use in non-async agent loops.

        Routes a tool call by name to the corresponding async method and
        blocks until the result is available.  If your agent loop is already
        async, call the async methods directly instead.

        Args:
            name: Tool name — one of ``"execute"``, ``"write_file"``,
                ``"read_file"``.
            args: Parsed tool arguments as a plain dict.

        Returns:
            Tool result as a string, or an error message string.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._dispatch_async(name, args))

    async def _dispatch_async(self, name: str, args: dict) -> str:  # type: ignore[type-arg]
        """Async dispatcher — routes tool name to the appropriate method.

        Args:
            name: Tool name.
            args: Tool arguments.

        Returns:
            Tool result string.
        """
        match name:
            case "execute":
                # Accept common aliases the model may use for the code param
                code = (
                    args.get("code")
                    or args.get("command")
                    or args.get("script")
                    or args.get("bash")
                    or args.get("cmd")
                    or ""
                )
                if not code:
                    return "[error] execute: missing 'code' parameter"
                return await self.execute(code)
            case "write_file":
                path = args.get("path") or args.get("filename") or args.get("file") or ""
                content = args.get("content") or args.get("text") or args.get("data") or ""
                if not path:
                    return "[error] write_file: missing 'path' parameter"
                return await self.write_file(path, content)
            case "read_file":
                path = args.get("path") or args.get("filename") or args.get("file") or ""
                if not path:
                    return "[error] read_file: missing 'path' parameter"
                return await self.read_file(path)
            case _:
                return f"[error] unknown tool: {name}"


class SubprocessExecutor(BaseExecutor):
    """Subprocess-based bash executor using Python's standard library only.

    Runs each ``execute()`` call as a bash script via ``subprocess.run``.
    A persistent ``tempfile.TemporaryDirectory`` acts as the working directory
    for the session lifetime.

    Because each ``execute()`` call spawns a fresh shell process, shell
    variables do NOT persist between calls.  However, the filesystem persists:
    files written in one call (or via ``write_file``) are available to later
    calls.  Packages installed with ``pip install`` also persist within the
    session since they go into the system site-packages.

    This executor has no external dependencies and works in any environment.
    It is the default fallback when OpenSandbox / Docker is unavailable.

    Example::

        async with SubprocessExecutor(timeout=30.0) as ex:
            await ex.execute("pip install pandas -q")
            out = await ex.execute("python3 -c 'import pandas; print(pandas.__version__)'")
            # out == "2.x.x"
    """

    def __init__(self, timeout: float = 30.0) -> None:
        """Initialise a SubprocessExecutor.

        Args:
            timeout: Maximum seconds a single ``execute`` call may run before
                being killed.  Defaults to 30 seconds.
        """
        self._timeout = timeout
        self._tmpdir: tempfile.TemporaryDirectory | None = None  # type: ignore[type-arg]
        self._workdir: Path | None = None

    async def start(self) -> None:
        """Create a temporary working directory for this session."""
        self._tmpdir = tempfile.TemporaryDirectory(prefix="rune_exec_")
        self._workdir = Path(self._tmpdir.name)

    async def stop(self) -> None:
        """Delete the temporary working directory."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._workdir = None

    async def execute(self, code: str) -> str:
        """Execute a bash script in a subprocess within the session workdir.

        The script runs with ``bash``, so any shell command works: ``ls``,
        ``grep``, ``pip install``, ``python3 -c "..."``, ``curl``, etc.
        The working directory persists across calls — files written in one
        call are available to subsequent calls.

        Args:
            code: A bash script (one or more shell commands).

        Returns:
            Combined stdout / stderr output string.

        Raises:
            RuntimeError: If the executor has not been started.
        """
        if self._workdir is None:
            raise RuntimeError(
                "SubprocessExecutor not started. "
                "Call `await executor.start()` or use `async with`."
            )

        script_path = self._workdir / "_exec.sh"
        # Silence any injected shell hooks from the host environment
        wrapped = "unset PROMPT_COMMAND BASH_ENV 2>/dev/null\n" + code
        script_path.write_text(wrapped, encoding="utf-8")

        loop = asyncio.get_event_loop()
        try:
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["bash", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=str(self._workdir),
                ),
            )
        except subprocess.TimeoutExpired:
            return f"[error] execution timed out after {self._timeout:.0f}s"

        parts: list[str] = []
        if proc.stdout.strip():
            parts.append(proc.stdout.rstrip())
        if proc.stderr.strip():
            parts.append("[stderr]\n" + proc.stderr.rstrip())
        if proc.returncode != 0 and not parts:
            parts.append(f"(no output, exit code {proc.returncode})")
        elif proc.returncode != 0:
            parts.append(f"[exit code {proc.returncode}]")
        return "\n".join(parts) if parts else "(no output)"

    async def write_file(self, path: str, content: str) -> str:
        """Write a file inside the session working directory.

        Absolute paths are written as-is; relative paths are resolved against
        the session working directory.

        Args:
            path: File path (absolute or relative).
            content: UTF-8 text content.

        Returns:
            Confirmation message with the resolved path.

        Raises:
            RuntimeError: If the executor has not been started.
        """
        if self._workdir is None:
            raise RuntimeError("SubprocessExecutor not started.")
        target = Path(path) if os.path.isabs(path) else self._workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"written: {target}"

    async def read_file(self, path: str) -> str:
        """Read a file from the session working directory.

        Args:
            path: File path (absolute or relative).

        Returns:
            File contents as a UTF-8 string.

        Raises:
            RuntimeError: If the executor has not been started.
            FileNotFoundError: If the file does not exist.
        """
        if self._workdir is None:
            raise RuntimeError("SubprocessExecutor not started.")
        target = Path(path) if os.path.isabs(path) else self._workdir / path
        return target.read_text(encoding="utf-8")
