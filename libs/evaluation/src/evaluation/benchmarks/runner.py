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
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evaluation.benchmarks.adapter_stack import AdapterStack
from evaluation.benchmarks.protocol import (
    BenchmarkConfig,
    BenchmarkResult,
    PassVerdict,
    Problem,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 2.0
# Stop sequences per benchmark type (bigcode-evaluation-harness).
# HumanEval: model completes a function body → \ndef stops at next function.
# MBPP: model generates the entire function → \ndef would kill it.
_HUMANEVAL_STOP = [
    "\nclass",
    "\ndef",
    "\n#",
    "\n@",
    "\nprint",
    "\nif",
]
_MBPP_STOP = [
    "\nclass",
    "\nassert",
    '\n"""',
    "\nprint",
    "\nif",
]
_DEFAULT_CODE_STOP = _HUMANEVAL_STOP
_STOP_BY_BENCHMARK: dict[str, list[str]] = {
    "mbpp": _MBPP_STOP,
    "humaneval": _HUMANEVAL_STOP,
}


def _stop_sequences_for(benchmark_id: str) -> list[str]:
    return _STOP_BY_BENCHMARK.get(benchmark_id, _DEFAULT_CODE_STOP)


try:
    from openai import APIConnectionError, APITimeoutError

    _RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
        ConnectionError,
        OSError,
        APIConnectionError,
        APITimeoutError,
    )
except ImportError:
    _RETRYABLE_ERRORS = (ConnectionError, OSError)

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


def _truncate_at_stop_token(text: str, stop_tokens: list[str]) -> str:
    """Post-hoc truncation at the first stop token (bigcode pattern).

    Safety net in case the provider doesn't honor stop sequences.
    """
    min_idx = len(text)
    for token in stop_tokens:
        idx = text.find(token)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx]


def _generate_completion(
    adapter_stack: AdapterStack,
    problem: Problem,
    max_tokens: int = 512,
    stop: list[str] | None = None,
) -> str:
    """Synchronously call provider.generate() from a thread.

    When adapter_generator is set on the stack, generates a per-problem
    adapter via the hypernetwork, loads it into the provider, generates,
    then unloads it. Otherwise uses the first static adapter_id.

    Args:
        adapter_stack: AdapterStack with provider and model config.
        problem: Problem whose prompt is sent to the model.
        max_tokens: Generation token cap.
        stop: Stop sequences for truncation.

    Returns:
        Generated text string.
    """
    effective_stop = stop or _DEFAULT_CODE_STOP

    if adapter_stack.completion_override is not None:
        prompt = problem.prompt
        if adapter_stack.prompt_augmenter is not None:
            prompt = adapter_stack.prompt_augmenter(prompt)
        return adapter_stack.completion_override(prompt, max_tokens)

    loop = asyncio.new_event_loop()
    try:
        if adapter_stack.adapter_generator is not None:
            return _generate_with_hypernet(
                adapter_stack, problem, max_tokens, loop, effective_stop
            )

        effective_prompt = problem.prompt
        if adapter_stack.prompt_augmenter is not None:
            effective_prompt = adapter_stack.prompt_augmenter(effective_prompt)

        adapter_id = None
        for aid in adapter_stack.adapter_ids:
            if aid in adapter_stack.adapter_paths:
                adapter_id = aid
                break

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                result = loop.run_until_complete(
                    adapter_stack.provider.complete_text(
                        prompt=effective_prompt,
                        model=adapter_stack.base_model,
                        adapter_id=adapter_id,
                        max_tokens=max_tokens,
                        stop=effective_stop,
                    )
                )
                return _truncate_at_stop_token(str(result.text), effective_stop)
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF_S * (2**attempt)
                logger.warning(
                    "Connection error on %s (attempt %d/%d), retrying in %.1fs: %s",
                    problem.problem_id,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                    exc,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]
    finally:
        loop.close()


def _generate_with_hypernet(
    adapter_stack: AdapterStack,
    problem: Problem,
    max_tokens: int,
    loop: asyncio.AbstractEventLoop,
    stop: list[str],
) -> str:
    """Generate a completion using a per-problem hypernetwork adapter.

    Args:
        adapter_stack: AdapterStack with adapter_generator set.
        problem: Problem whose prompt drives adapter generation.
        max_tokens: Generation token cap.
        loop: Event loop for async provider calls.
        stop: Stop sequences for truncation.

    Returns:
        Generated text string.
    """
    effective_stop = stop
    assert adapter_stack.adapter_generator is not None

    effective_prompt = problem.prompt
    if adapter_stack.prompt_augmenter is not None:
        effective_prompt = adapter_stack.prompt_augmenter(effective_prompt)

    adapter_path = adapter_stack.adapter_generator(effective_prompt)
    if adapter_path is None:
        logger.warning(
            "adapter_generator returned None for %s, using base",
            problem.problem_id,
        )
        result = loop.run_until_complete(
            adapter_stack.provider.complete_text(
                prompt=effective_prompt,
                model=adapter_stack.base_model,
                max_tokens=max_tokens,
                stop=effective_stop,
            )
        )
        return _truncate_at_stop_token(str(result.text), effective_stop)

    adapter_id = f"hypernet_{problem.problem_id}"
    try:
        loop.run_until_complete(
            adapter_stack.provider.load_adapter(adapter_id, adapter_path)
        )
        result = loop.run_until_complete(
            adapter_stack.provider.complete_text(
                prompt=effective_prompt,
                model=adapter_stack.base_model,
                adapter_id=adapter_id,
                max_tokens=max_tokens,
                stop=effective_stop,
            )
        )
        return _truncate_at_stop_token(str(result.text), effective_stop)
    finally:
        try:
            loop.run_until_complete(adapter_stack.provider.unload_adapter(adapter_id))
        except Exception:
            logger.warning(
                "Failed to unload adapter %s; GPU memory may leak",
                adapter_id,
                exc_info=True,
            )


def _evaluate_one(
    adapter: Any,
    adapter_stack: AdapterStack,
    problem: Problem,
    config: BenchmarkConfig,
    stop: list[str] | None = None,
) -> PassVerdict:
    """Generate a completion and score it for a single problem.

    Args:
        adapter: Benchmark adapter instance (has .score()).
        adapter_stack: AdapterStack for generation.
        problem: Problem to evaluate.
        config: BenchmarkConfig with timeout_s.
        stop: Benchmark-specific stop sequences.

    Returns:
        PassVerdict for this problem.
    """
    generation = _generate_completion(adapter_stack, problem, stop=stop)
    return adapter.score(problem, generation, timeout_s=config.timeout_s)  # type: ignore[no-any-return]


def _load_checkpoint(path: Path) -> list[PassVerdict]:
    """Load previously saved verdicts from a JSONL checkpoint file."""
    if not path.exists():
        return []
    verdicts: list[PassVerdict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        verdicts.append(
            PassVerdict(
                problem_id=d["problem_id"],
                passed=d["passed"],
                generation=d["generation"],
                error=d.get("error"),
                timed_out=d["timed_out"],
            )
        )
    return verdicts


_checkpoint_lock = threading.Lock()


def _append_checkpoint(path: Path, verdict: PassVerdict) -> None:
    """Append a single verdict to the JSONL checkpoint file (thread-safe)."""
    line = json.dumps(
        {
            "problem_id": verdict.problem_id,
            "passed": verdict.passed,
            "generation": verdict.generation,
            "error": verdict.error,
            "timed_out": verdict.timed_out,
        }
    )
    with _checkpoint_lock:
        with path.open("a") as f:
            f.write(line + "\n")


def run_benchmark(  # noqa: C901
    adapter_stack: AdapterStack,
    benchmark_id: str,
    problem_ids: list[str] | None = None,
    max_samples: int | None = None,
    config: BenchmarkConfig | None = None,
    checkpoint_dir: Path | str | None = None,
    on_verdict: Any | None = None,
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
        checkpoint_dir: Directory for per-problem JSONL checkpoints. When set,
            completed verdicts are appended incrementally and subsequent runs
            resume from the checkpoint.
        on_verdict: Optional callback ``(benchmark_id, verdict, running_pass_at_1,
            n_completed, n_total) -> None`` invoked after each problem completes.
            Useful for streaming metrics to MLflow.

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

    # Checkpoint setup: resume from prior run if checkpoint_dir is set
    ckpt_path: Path | None = None
    cached_verdicts: list[PassVerdict] = []
    cached_ids: set[str] = set()
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{benchmark_id}.jsonl"
        cached_verdicts = _load_checkpoint(ckpt_path)
        cached_ids = {v.problem_id for v in cached_verdicts}
        if cached_ids:
            logger.info(
                "Resumed %d cached verdicts for %s",
                len(cached_ids),
                benchmark_id,
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

    remaining = [p for p in problems if p.problem_id not in cached_ids]

    if not problems:
        return BenchmarkResult(benchmark_id=benchmark_id, verdicts=[])

    if not remaining:
        logger.info(
            "All %d problems already cached for %s",
            len(problems),
            benchmark_id,
        )
        id_order = {p.problem_id: i for i, p in enumerate(problems)}
        cached_verdicts.sort(key=lambda v: id_order.get(v.problem_id, 9999))
        return BenchmarkResult(benchmark_id=benchmark_id, verdicts=cached_verdicts)

    logger.info(
        "run_benchmark: benchmark=%s n_problems=%d remaining=%d max_workers=%d",
        benchmark_id,
        len(problems),
        len(remaining),
        cfg.max_workers,
    )

    bench_stop = _stop_sequences_for(benchmark_id)
    n_total = len(problems)
    _pass_count = len([v for v in cached_verdicts if v.passed])
    _completed = len(cached_verdicts)
    _verdict_lock = threading.Lock()

    def _evaluate_and_checkpoint(problem: Problem) -> PassVerdict:
        nonlocal _pass_count, _completed
        verdict = _evaluate_one(adapter, adapter_stack, problem, cfg, stop=bench_stop)
        if ckpt_path is not None:
            _append_checkpoint(ckpt_path, verdict)
        with _verdict_lock:
            _completed += 1
            if verdict.passed:
                _pass_count += 1
            running_p1 = _pass_count / _completed if _completed else 0.0
        if on_verdict is not None:
            on_verdict(benchmark_id, verdict, running_p1, _completed, n_total)
        return verdict

    # Warm up the provider connection with the first problem (single-threaded)
    # before fanning out.
    first_verdict = _evaluate_and_checkpoint(remaining[0])

    verdicts: list[PassVerdict] = list(cached_verdicts) + [first_verdict]
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {
            executor.submit(_evaluate_and_checkpoint, p): p for p in remaining[1:]
        }
        for future in as_completed(futures):
            verdict = future.result()
            verdicts.append(verdict)
            status = "PASS" if verdict.passed else "FAIL"
            logger.debug("  [%s] %s", status, verdict.problem_id)

    # Restore original problem order
    id_order = {p.problem_id: i for i, p in enumerate(problems)}
    verdicts.sort(key=lambda v: id_order.get(v.problem_id, 9999))

    result = BenchmarkResult(benchmark_id=benchmark_id, verdicts=verdicts)

    # Warn only on infrastructure-type failures (connection, timeout, empty gen),
    # not on model-quality failures (assertion, syntax, name errors).
    infra_errors = sum(
        1
        for v in verdicts
        if not v.passed
        and v.error
        and ("Connection" in v.error or "Timeout" in v.error or v.generation == "")
    )
    if infra_errors > len(verdicts) * 0.1:
        logger.warning(
            "HIGH INFRA ERROR RATE: %s has %d/%d infrastructure errors — "
            "check vLLM server health",
            benchmark_id,
            infra_errors,
            result.n_problems,
        )
    return result
