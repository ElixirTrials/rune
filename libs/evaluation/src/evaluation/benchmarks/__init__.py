"""Unified benchmark harness for Pass@1 evaluation.

Public API re-exported for convenience:

    from evaluation.benchmarks import (
        Problem,
        PassVerdict,
        BenchmarkAdapter,
        BenchmarkConfig,
        BenchmarkResult,
        run_benchmark,
        load_adapter_stack,
    )
"""

from __future__ import annotations

from evaluation.benchmarks.protocol import (
    BenchmarkAdapter,
    BenchmarkConfig,
    BenchmarkResult,
    PassVerdict,
    Problem,
)

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkConfig",
    "BenchmarkResult",
    "PassVerdict",
    "Problem",
    "run_benchmark",
    "load_adapter_stack",
]


def __getattr__(name: str) -> object:
    """Lazy-load run_benchmark and load_adapter_stack on first access.

    These are deferred so that importing submodules (e.g.
    evaluation.benchmarks.protocol) does not require runner.py and
    adapter_stack.py to exist. Once those modules are implemented, direct
    imports like ``from evaluation.benchmarks import run_benchmark`` work
    exactly as if they were eagerly imported.

    Args:
        name: Attribute name requested.

    Returns:
        The requested object.

    Raises:
        AttributeError: If name is not a known lazy export.
    """
    if name == "run_benchmark":
        from evaluation.benchmarks.runner import run_benchmark  # noqa: PLC0415

        return run_benchmark
    if name == "load_adapter_stack":
        from evaluation.benchmarks.adapter_stack import (
            load_adapter_stack,  # noqa: PLC0415
        )

        return load_adapter_stack
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
