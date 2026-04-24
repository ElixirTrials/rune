"""Benchmark runner — orchestrates sampling + scoring for run_benchmark().

Uses ThreadPoolExecutor for parallel per-problem evaluation. Supports
all eight benchmark adapters via a registry dict keyed by benchmark_id.

No GPU imports. All heavy lifting (model inference, sandbox execution)
happens inside the provider and adapter.score() calls, which are
already CPU-safe at import time.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from evaluation.benchmarks.adapter_stack import AdapterStack
from evaluation.benchmarks.protocol import (
    BenchmarkConfig,
    BenchmarkResult,
    PassVerdict,
    Problem,
)

logger = logging.getLogger(__name__)

# Registry of benchmark_id -> dotted adapter class path (lazy import)
_ADAPTER_REGISTRY: dict[str, str] = {
    "humaneval": "evaluation.benchmarks.humaneval.HumanEvalAdapter",
    "mbpp": "evaluation.benchmarks.mbpp.MBPPAdapter",
    "apps": "evaluation.benchmarks.apps.APPSAdapter",
    "bigcodebench": "evaluation.benchmarks.bigcodebench.BigCodeBenchAdapter",
    "ds_1000": "evaluation.benchmarks.ds1000.DS1000Adapter",
    "livecodebench": "evaluation.benchmarks.livecodebench.LiveCodeBenchAdapter",
    "swe_bench_lite": "evaluation.benchmarks.swe_bench.SWEBenchLiteAdapter",
    "codecontests": "evaluation.benchmarks.codecontests.CodeContestsAdapter",
}


def _import_adapter(dotted_path: str) -> Any:
    """Import and instantiate an adapter class from a dotted module path.

    Args:
        dotted_path: e.g. "evaluation.benchmarks.humaneval.HumanEvalAdapter"

    Returns:
        An instantiated adapter object.
    """
    module_path, cls_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls()


def _generate_completion(
    adapter_stack: AdapterStack,
    problem: Problem,
    max_tokens: int = 512,
) -> str:
    """Synchronously call provider.generate() from a thread.

    Creates a new event loop per thread. ThreadPoolExecutor threads have
    no running loop by default, so we create one per call. This works
    with any InferenceProvider that is not bound to a specific loop at
    construction time (open question #4 from plan).

    Args:
        adapter_stack: AdapterStack with provider and model config.
        problem: Problem whose prompt is sent to the model.
        max_tokens: Generation token cap.

    Returns:
        Generated text string.
    """
    adapter_id = adapter_stack.adapter_ids[0] if adapter_stack.adapter_ids else None
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            adapter_stack.provider.generate(
                prompt=problem.prompt,
                model=adapter_stack.base_model,
                adapter_id=adapter_id,
                max_tokens=max_tokens,
            )
        )
        return str(result.text)
    finally:
        loop.close()


def _evaluate_one(
    adapter: Any,
    adapter_stack: AdapterStack,
    problem: Problem,
    config: BenchmarkConfig,
) -> PassVerdict:
    """Generate a completion and score it for a single problem.

    Args:
        adapter: Benchmark adapter instance (has .score()).
        adapter_stack: AdapterStack for generation.
        problem: Problem to evaluate.
        config: BenchmarkConfig with timeout_s.

    Returns:
        PassVerdict for this problem.
    """
    try:
        generation = _generate_completion(adapter_stack, problem)
        return adapter.score(problem, generation, timeout_s=config.timeout_s)  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning("Error evaluating problem %s: %s", problem.problem_id, exc)
        return PassVerdict(
            problem_id=problem.problem_id,
            passed=False,
            generation="",
            error=str(exc),
            timed_out=False,
        )


def run_benchmark(
    adapter_stack: AdapterStack,
    benchmark_id: str,
    problem_ids: list[str] | None = None,
    max_samples: int | None = None,
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Run a full benchmark evaluation pass and return aggregate Pass@1.

    Orchestrates:
    1. Load problems from the benchmark adapter (with optional ID filter).
    2. Fan out (generate + score) via ThreadPoolExecutor.
    3. Aggregate verdicts into a BenchmarkResult.

    Args:
        adapter_stack: AdapterStack describing base model + adapters + provider.
        benchmark_id: One of the registered benchmark IDs:
            humaneval, mbpp, apps, bigcodebench, ds_1000,
            livecodebench, swe_bench_lite, codecontests.
        problem_ids: Optional list of problem_id strings to restrict
            evaluation to a subset. If None, evaluates all loaded problems.
        max_samples: Cap on total problems evaluated.
        config: BenchmarkConfig overriding defaults (timeout, workers, seed).

    Returns:
        BenchmarkResult with per-problem verdicts and aggregate pass_at_1.

    Raises:
        ValueError: If benchmark_id is not in the known registry.

    Example:
        >>> result = run_benchmark(stack, "humaneval", max_samples=50)
        >>> print(f"Pass@1: {result.pass_at_1:.2%}")
    """
    if benchmark_id not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown benchmark_id {benchmark_id!r}. "
            f"Known benchmarks: {sorted(_ADAPTER_REGISTRY)}"
        )

    # Build effective config: merge caller's max_samples into config
    if config is None:
        cfg = BenchmarkConfig(max_samples=max_samples)
    else:
        cfg = config
    if max_samples is not None:
        cfg = BenchmarkConfig(
            timeout_s=cfg.timeout_s,
            max_workers=cfg.max_workers,
            max_samples=max_samples,
            seed=cfg.seed,
        )

    adapter = _import_adapter(_ADAPTER_REGISTRY[benchmark_id])
    problems: list[Problem] = adapter.load_problems(
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )

    # Apply problem_ids filter
    if problem_ids is not None:
        id_set = set(problem_ids)
        problems = [p for p in problems if p.problem_id in id_set]

    if not problems:
        return BenchmarkResult(benchmark_id=benchmark_id, verdicts=[])

    logger.info(
        "run_benchmark: benchmark=%s n_problems=%d max_workers=%d",
        benchmark_id,
        len(problems),
        cfg.max_workers,
    )

    verdicts: list[PassVerdict] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {
            executor.submit(_evaluate_one, adapter, adapter_stack, p, cfg): p
            for p in problems
        }
        for future in as_completed(futures):
            verdict = future.result()
            verdicts.append(verdict)
            status = "PASS" if verdict.passed else "FAIL"
            logger.debug("  [%s] %s", status, verdict.problem_id)

    # Restore original problem order
    id_order = {p.problem_id: i for i, p in enumerate(problems)}
    verdicts.sort(key=lambda v: id_order.get(v.problem_id, 9999))

    return BenchmarkResult(benchmark_id=benchmark_id, verdicts=verdicts)
