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
    # Start well below the Linux default ephemeral range (32768–60999) so
    # kernel ports never collide with OS-assigned client-side TCP ports (e.g.
    # the outgoing connections to a vLLM server).  10100–32767 gives space for
    # (32767 - 10100) / 5 ≈ 4500 concurrent sandboxes before wrapping.
    _next = 10100

    @classmethod
    def claim(cls, count: int = 5) -> list[int]:
        with cls._lock:
            ports = list(range(cls._next, cls._next + count))
            cls._next += count
            return ports


# Semaphore that limits how many IPython kernels may go through their
# startup sequence (start_kernel → wait_for_ready) concurrently.
# Even with unique ports the kernel subprocess races to *bind* before
# start_kernel() returns; serialising to ≤8 at a time eliminates the window.
_KERNEL_STARTUP_SEM = threading.Semaphore(8)


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
import scipy
import networkx
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

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        env["JUPYTER_PLATFORM_DIRS"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"
        # Each sandbox kernel runs one problem at a time; multi-threaded BLAS /
        # OpenMP gives no benefit and balloons thread count when many kernels
        # run concurrently (default 64 OpenBLAS threads × N kernels exhausts
        # the OS thread limit with even modest concurrency).
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        # Deferred import so the module stays importable without jupyter_client
        # installed (INFRA-05 pattern).
        from jupyter_client import KernelManager  # noqa: PLC0415

        self._km = KernelManager()

        if self._cfg.port_start is not None:
            # Explicit fixed port block requested — assign directly.
            ports = list(range(self._cfg.port_start, self._cfg.port_start + 5))
            self._km.shell_port = ports[0]
            self._km.iopub_port = ports[1]
            self._km.stdin_port = ports[2]
            self._km.hb_port = ports[3]
            self._km.control_port = ports[4]
        else:
            # Let KernelManager pick its own open ports via select_random_ports().
            # Pre-assigning from _PortAllocator risks colliding with OS-assigned
            # ephemeral ports (Linux default range 32768–60999), especially when
            # many concurrent HTTP connections are open (e.g. to a vLLM server).
            # The _KERNEL_STARTUP_SEM above serialises startup so KernelManager's
            # own port-selection does not race against sibling kernels.
            pass

        # Serialise kernel startup to avoid ZMQ port-binding races when many
        # sandboxes are created concurrently.  The semaphore is released once
        # wait_for_ready() confirms the kernel has bound its sockets, so
        # running kernels never hold it.
        with _KERNEL_STARTUP_SEM:
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


# ---------------------------------------------------------------------------
# Tool schema + agentic-loop helpers
# ---------------------------------------------------------------------------

# JSON schema forwarded to apply_chat_template so models can call the sandbox.
MATH_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code in a stateful IPython kernel. "
                "State persists across calls. "
                "Pre-imported: math, numpy, sympy, itertools, collections, "
                "mpmath (mp.dps=64). Always use print() to display results."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
]

# ── Tool-call wire formats ────────────────────────────────────────────────────
# Two formats appear depending on how the model is served:
#
#   XML  (in-process vLLM / older Qwen chat templates):
#       <tool_call><function=execute_python>
#         <parameter=code>print(1+1)</parameter>
#       </function></tool_call>
#
#   Hermes JSON  (vLLM server / Qwen3 chat template via OpenAI endpoint):
#       <tool_call>
#       {"name": "execute_python", "arguments": {"code": "print(1+1)"}}
#       </tool_call>
#
# parse_tool_calls() tries XML first; falls back to Hermes JSON so both
# serving modes work without any change to the calling code.

# XML format
_TOOL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)

# Hermes JSON format — capture the JSON object inside the tags
_HERMES_TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_FLOAT_RE = re.compile(r"[-+]?\d+\.\d{4,}")

# Guard: refuse code blocks longer than this before submitting to the sandbox.
MAX_CODE_LINES = 300

# Tolerance for detecting float results that are essentially integers.
_NEAR_INT_TOL = 1e-4


def parse_tool_calls(text: str) -> list[dict]:
    """Parse ``<tool_call>…</tool_call>`` blocks from a model response.

    Handles two wire formats automatically:

    * **XML** — ``<tool_call><function=name><parameter=arg>val</parameter>…``
      emitted by in-process vLLM with the Qwen chat template.
    * **Hermes JSON** — ``<tool_call>{"name": …, "arguments": {…}}</tool_call>``
      emitted by the vLLM OpenAI server (Qwen3 chat template).

    Args:
        text: Raw model output that may contain one or more tool-call blocks.

    Returns:
        List of ``{"name": str, "arguments": dict}`` dicts, one per call.
        Returns an empty list when no recognised tool-call block is found.
    """
    import json as _json

    # ── XML format ────────────────────────────────────────────────────────────
    xml_calls = [
        {
            "name": m.group(1),
            "arguments": {
                p.group(1): p.group(2).strip()
                for p in _PARAM_RE.finditer(m.group(2))
            },
        }
        for m in _TOOL_RE.finditer(text)
    ]
    if xml_calls:
        return xml_calls

    # ── Hermes JSON format ────────────────────────────────────────────────────
    json_calls: list[dict] = []
    for m in _HERMES_TC_RE.finditer(text):
        try:
            payload = _json.loads(m.group(1))
        except _json.JSONDecodeError:
            continue
        name = payload.get("name") or payload.get("function", "")
        args = payload.get("arguments") or payload.get("parameters") or {}
        if name:
            json_calls.append({"name": name, "arguments": args})
    return json_calls


def near_int_hint(result: str) -> str:
    """Return a boxed-integer hint when a sandbox result is very close to an int.

    Appended to tool output so the model knows it can state a clean integer
    answer rather than a rounded float.

    Args:
        result: Captured stdout from a sandbox execution.

    Returns:
        A hint string starting with ``"\\n[HINT] …"`` if a near-integer float
        was detected, otherwise an empty string.
    """
    for m in _FLOAT_RE.finditer(result):
        try:
            val = float(m.group())
        except ValueError:
            continue
        n = round(val)
        if abs(val - n) < _NEAR_INT_TOL and 0 <= n <= 99999:
            return (
                f"\n[HINT] {val:.6f} ≈ {n}. "
                f"If your analysis supports this, use \\boxed{{{n}}}."
            )
    return ""
