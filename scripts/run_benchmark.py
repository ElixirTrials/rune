r"""CLI entrypoint for benchmark Pass@1 evaluation.

Mirrors trainer_cli.py: all heavy imports (torch, transformers, datasets)
are deferred inside main(). This script is CPU-safe and supports
--dry-run mode for CI validation without loading any models.

Usage:
    uv run python scripts/run_benchmark.py \
        --benchmark humaneval \
        --base-model Qwen/Qwen3.5-9B \
        --max-samples 50 \
        --dry-run

    uv run python scripts/run_benchmark.py \
        --benchmark humaneval \
        --base-model Qwen/Qwen3.5-9B \
        --adapter-ids adapter-001 adapter-002 \
        --max-samples 50 \
        --timeout 30 \
        --workers 4 \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

_KNOWN_BENCHMARKS = [
    "humaneval",
    "mbpp",
    "apps",
    "bigcodebench",
    "ds_1000",
    "livecodebench",
    "swe_bench_lite",
    "codecontests",
]


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for run_benchmark.py."""
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description=(
            "Run a benchmark Pass@1 evaluation on a (base_model, adapter_stack) pair."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=_KNOWN_BENCHMARKS,
        help="Benchmark to evaluate.",
    )
    parser.add_argument(
        "--base-model",
        dest="base_model",
        required=True,
        help="HuggingFace model ID or local path for the base model.",
    )
    parser.add_argument(
        "--adapter-ids",
        dest="adapter_ids",
        nargs="*",
        default=[],
        metavar="ID",
        help="Zero or more adapter registry IDs to load on top of base model.",
    )
    parser.add_argument(
        "--max-samples",
        dest="max_samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap on problems evaluated. None = all available.",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout_s",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Per-problem sandbox timeout in seconds.",
    )
    parser.add_argument(
        "--workers",
        dest="max_workers",
        type=int,
        default=4,
        metavar="N",
        help="ThreadPoolExecutor worker count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for problem subsampling.",
    )
    parser.add_argument(
        "--problem-ids",
        dest="problem_ids",
        nargs="*",
        default=None,
        metavar="ID",
        help="Optional list of specific problem IDs to evaluate.",
    )
    parser.add_argument(
        "--registry-db",
        dest="registry_db",
        default="~/.rune/adapters.db",
        metavar="PATH",
        help="Path to AdapterRegistry SQLite database.",
    )
    parser.add_argument(
        "--provider",
        default="vllm",
        choices=["vllm", "ollama"],
        help="Inference provider backend.",
    )
    parser.add_argument(
        "--provider-url",
        dest="provider_url",
        default="http://localhost:8000",
        metavar="URL",
        help="Base URL for the inference provider.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write JSON results to this file. Defaults to stdout.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "Resolve and print arguments as JSON without loading any models. "
            "CPU-only; safe to run in CI."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def _dry_run_output(args: argparse.Namespace) -> dict[str, Any]:
    """Produce a JSON-serialisable summary of resolved arguments.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Dict suitable for JSON serialisation and stdout printing.
    """
    return {
        "dry_run": True,
        "benchmark": args.benchmark,
        "base_model": args.base_model,
        "adapter_ids": args.adapter_ids,
        "max_samples": args.max_samples,
        "timeout_s": args.timeout_s,
        "max_workers": args.max_workers,
        "seed": args.seed,
        "problem_ids": args.problem_ids,
        "registry_db": args.registry_db,
        "provider": args.provider,
        "provider_url": args.provider_url,
        "output": args.output,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    if args.dry_run:
        resolved = _dry_run_output(args)
        print(json.dumps(resolved, indent=2))
        return 0

    # --- Heavy imports deferred here (INFRA-05 pattern) ---
    try:
        from adapter_registry.registry import AdapterRegistry  # deferred
        from evaluation.benchmarks.adapter_stack import load_adapter_stack  # deferred
        from evaluation.benchmarks.protocol import BenchmarkConfig  # deferred
        from evaluation.benchmarks.runner import run_benchmark  # deferred
        from sqlalchemy import create_engine  # deferred: heavy
    except ImportError as exc:
        logger.error("Missing dependency: %s. Use --dry-run for CPU-only mode.", exc)
        return 1

    # Build inference provider
    from inference.provider import InferenceProvider  # deferred

    provider: InferenceProvider
    try:
        if args.provider == "vllm":
            from inference.vllm_provider import VLLMProvider  # deferred

            provider = VLLMProvider(base_url=args.provider_url)
        else:
            from inference.ollama_provider import OllamaProvider  # deferred

            provider = OllamaProvider(base_url=args.provider_url)
    except ImportError as exc:
        logger.error("Could not import provider %s: %s", args.provider, exc)
        return 1

    # Build registry and adapter stack
    import os

    db_path = os.path.expanduser(args.registry_db)
    engine = create_engine(f"sqlite:///{db_path}")
    registry = AdapterRegistry(engine=engine)

    try:
        stack = load_adapter_stack(
            base_model=args.base_model,
            adapter_ids=args.adapter_ids,
            provider=provider,
            registry=registry,
        )
    except ValueError as exc:
        logger.error("Adapter stack error: %s", exc)
        return 1

    cfg = BenchmarkConfig(
        timeout_s=args.timeout_s,
        max_workers=args.max_workers,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    logger.info(
        "Running benchmark=%s on base_model=%s adapters=%s max_samples=%s",
        args.benchmark,
        args.base_model,
        args.adapter_ids,
        args.max_samples,
    )

    result = run_benchmark(
        adapter_stack=stack,
        benchmark_id=args.benchmark,
        problem_ids=args.problem_ids,
        config=cfg,
    )

    output: dict[str, Any] = {
        "benchmark_id": result.benchmark_id,
        "n_problems": result.n_problems,
        "n_passed": result.n_passed,
        "pass_at_1": result.pass_at_1,
        "verdicts": [
            {
                "problem_id": v.problem_id,
                "passed": v.passed,
                "timed_out": v.timed_out,
                "error": v.error,
            }
            for v in result.verdicts
        ],
    }

    output_str = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        logger.info("Results written to %s", args.output)
    else:
        print(output_str)

    logger.info(
        "Pass@1: %.4f (%d/%d)", result.pass_at_1, result.n_passed, result.n_problems
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
