"""Stateful Jupyter kernel sandbox for math competition execution.

Provides MathSandbox, a long-lived IPython kernel that preserves state across
multiple execute() calls within a session.  Designed to be usable both from
scripts and from inside a notebook environment (a competition requirement).

Typical usage::

    with MathSandbox() as box:
        box.execute("x = 42")
        result = box.execute("print(x * 2)")   # → "84\\n"

The sandbox pre-imports math, numpy, sympy, itertools, collections, and mpmath
(with 64 decimal-place precision) so competition code can use them immediately.
"""

from __future__ import annotations

import contextlib
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MathConfig:
    """Runtime knobs for MathSandbox.

    Attributes:
        default_timeout: Seconds before a single execute() call is interrupted.
        port_start: First port number in the five-port block assigned to the
            kernel.  Each sandbox claims five consecutive ports.
        mpmath_dps: Decimal places of precision for mpmath.
    """

    default_timeout: float = 30.0
    port_start: int | None = None
    mpmath_dps: int = 64


# ---------------------------------------------------------------------------
# Thread-safe port allocator (class-level, shared across all instances)
# ---------------------------------------------------------------------------


class _PortAllocator:
    _lock = threading.Lock()
    _next = 50000

    @classmethod
    def claim(cls, count: int = 5) -> list[int]:
        with cls._lock:
            ports = list(range(cls._next, cls._next + count))
            cls._next += count
            return ports


# ---------------------------------------------------------------------------
# MathSandbox
# ---------------------------------------------------------------------------

_PRELUDE = """\
import math
import numpy
import sympy
import itertools
import collections
import mpmath
mpmath.mp.dps = {dps}
"""

_RESET_MAGIC = "%reset -f\n" + _PRELUDE


class MathSandbox:
    """Stateful IPython kernel sandbox tuned for math competition problems.

    State (variables, imports, definitions) persists across execute() calls
    within the same session, mirroring an interactive Jupyter notebook.

    Args:
        config: Optional MathConfig instance.  Defaults to MathConfig().

    Example::

        box = MathSandbox()
        box.execute("from sympy import factorint")
        print(box.execute("print(factorint(360))"))
        box.close()

    Can also be used as a context manager::

        with MathSandbox() as box:
            box.execute("n = 100")
            print(box.execute("print(sum(range(n)))"))
    """

    def __init__(self, config: MathConfig | None = None) -> None:
        self._cfg = config or MathConfig()
        self._owns_kernel = False
        self._km = None
        self._client = None

        ports = (
            list(range(self._cfg.port_start, self._cfg.port_start + 5))
            if self._cfg.port_start is not None
            else _PortAllocator.claim(5)
        )

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        env["JUPYTER_PLATFORM_DIRS"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        # Deferred import so the module stays importable without jupyter_client
        # installed (INFRA-05 pattern).
        from jupyter_client import KernelManager  # noqa: PLC0415

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(
            env=env,
            extra_arguments=["--Application.log_level=CRITICAL"],
        )

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._cfg.default_timeout)
        self._owns_kernel = True

        self.execute(_PRELUDE.format(dps=self._cfg.mpmath_dps))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, code: str, timeout: float | None = None) -> str:
        """Run *code* in the kernel and return captured output.

        State is preserved between calls.  Execution is interrupted (not
        killed) when *timeout* expires so the kernel stays alive.

        Args:
            code: Python source to execute.
            timeout: Per-call timeout in seconds.  Falls back to
                ``config.default_timeout`` when omitted.

        Returns:
            Combined stdout + any error text.  Returns a warning string when
            the execution produces no output.
        """
        effective_timeout = timeout if timeout is not None else self._cfg.default_timeout
        client = self._client

        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        start = time.monotonic()

        while True:
            if time.monotonic() - start > effective_timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Execution timed out after {effective_timeout}s"

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type == "error":
                stderr_parts.append(self._format_error(content.get("traceback", [])))

            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr

        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def reset(self) -> None:
        """Clear all kernel state and re-import the standard math libraries."""
        self.execute(_RESET_MAGIC.format(dps=self._cfg.mpmath_dps))

    def close(self) -> None:
        """Shut down the kernel and release all resources."""
        try:
            if self._client:
                self._client.stop_channels()
        except Exception:
            pass

        if self._owns_kernel and self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass
            try:
                self._km.cleanup_resources()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MathSandbox":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _format_error(traceback: list[str]) -> str:
        clean: list[str] = []
        for frame in traceback:
            frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            # Drop host-filesystem frames; keep ipython-input frames.
            if 'File "' in frame and "ipython-input" not in frame:
                continue
            clean.append(frame)
        return "".join(clean)
