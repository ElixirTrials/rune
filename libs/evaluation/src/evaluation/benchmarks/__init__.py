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

from evaluation.benchmarks.adapter_stack import load_adapter_stack
from evaluation.benchmarks.protocol import (
    BenchmarkAdapter,
    BenchmarkConfig,
    BenchmarkResult,
    PassVerdict,
    Problem,
)
from evaluation.benchmarks.runner import run_benchmark

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkConfig",
    "BenchmarkResult",
    "PassVerdict",
    "Problem",
    "run_benchmark",
    "load_adapter_stack",
]
