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
        load_problems,
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
    "load_problems",
]


def load_problems(
    benchmark_id: str,
    problem_ids: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[Problem]:
    """Load problems for a benchmark via its registered adapter.

    For APPS, passing ``max_samples`` triggers the adapter's stratified-random
    sampler by difficulty (identical to what ``run_benchmark`` would do — the
    caller gets Plan A's APPSAdapter._stratified_sample output, not a separate
    code path). This is the APPS stratification parity point between Plan A
    and Plan C.

    Args:
        benchmark_id: One of the registered benchmark IDs:
            humaneval, mbpp, apps, bigcodebench, ds_1000, livecodebench,
            swe_bench_lite, codecontests.
        problem_ids: Optional list of explicit problem ids to keep. When
            provided, the adapter's own sampling is bypassed and only matching
            ids are returned.
        max_samples: Adapter-level cap. For APPS this drives stratified
            sampling; for other adapters it is a simple head-cap.
        seed: Sampling seed (APPS uses this for stratification).

    Returns:
        List of Problem instances.

    Raises:
        ValueError: If benchmark_id is not in the runner's registry.
    """
    from evaluation.benchmarks.runner import (  # noqa: PLC0415
        _ADAPTER_REGISTRY,
        _import_adapter,
    )

    if benchmark_id not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown benchmark_id {benchmark_id!r}. "
            f"Known benchmarks: {sorted(_ADAPTER_REGISTRY)}"
        )

    adapter = _import_adapter(_ADAPTER_REGISTRY[benchmark_id])
    problems: list[Problem] = adapter.load_problems(
        max_samples=max_samples, seed=seed
    )

    if problem_ids is not None:
        id_set = set(problem_ids)
        problems = [p for p in problems if p.problem_id in id_set]

    return problems


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
