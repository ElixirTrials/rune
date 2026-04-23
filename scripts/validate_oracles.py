r"""Oracle validation runner — per-oracle "beat base by >=3% absolute" gate.

For each oracle adapter (identified by bin key ``<phase>_<benchmark>`` or
``diagnose_pooled``), this runner:

  1. Evaluates the *base* model on the bin's benchmark via ``run_benchmark``.
  2. Evaluates the *base + oracle adapter* stack on the same benchmark.
  3. Reports Pass@1 delta and marks the oracle as PASS iff
     ``stack - base >= THRESHOLD`` (default 0.03 = 3 absolute points).

The runner is a thin CLI around ``evaluation.benchmarks.run_benchmark``.
It is CPU-safe at import time (heavy imports deferred into ``main``).

Usage:
    uv run python scripts/validate_oracles.py \
        --base-model Qwen/Qwen3.5-9B \
        --oracle decompose humaneval:adapter-id-123 \
        --oracle plan humaneval:adapter-id-456 \
        --max-samples 50 \
        --output oracles.json

Oracle spec syntax: ``<bin_key>:<adapter_id>`` where ``bin_key`` is one of
``<phase>_<benchmark>`` (e.g. ``decompose_humaneval``) or ``diagnose_pooled``.
The benchmark to evaluate against is derived from the bin_key.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.03
DEFAULT_POOLED_BENCHMARK = "humaneval"


def _parse_oracle_spec(spec: str) -> tuple[str, str, str]:
    """Parse ``<bin_key>:<adapter_id>`` into (bin_key, benchmark, adapter_id).

    Args:
        spec: "<bin_key>:<adapter_id>" string.

    Returns:
        Tuple of (bin_key, benchmark_id, adapter_id). For ``diagnose_pooled``
        the benchmark defaults to DEFAULT_POOLED_BENCHMARK.

    Raises:
        ValueError: If the spec is malformed.
    """
    if ":" not in spec:
        raise ValueError(
            f"Oracle spec {spec!r} must be '<bin_key>:<adapter_id>' "
            "e.g. 'decompose_humaneval:adapter-42'"
        )
    bin_key, adapter_id = spec.split(":", 1)
    if bin_key == "diagnose_pooled":
        benchmark = DEFAULT_POOLED_BENCHMARK
    else:
        parts = bin_key.split("_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"bin_key {bin_key!r} must be '<phase>_<benchmark>' "
                "or 'diagnose_pooled'"
            )
        benchmark = parts[1]
    return bin_key, benchmark, adapter_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_oracles",
        description=(
            "Per-oracle 'beat base by >=THRESHOLD' validator over run_benchmark."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base HF model repo id (e.g. Qwen/Qwen3.5-9B).",
    )
    parser.add_argument(
        "--oracle",
        action="append",
        required=True,
        metavar="BIN_KEY:ADAPTER_ID",
        help="Repeatable. Each oracle to validate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Absolute Pass@1 improvement required (default: 0.03 = 3 pts).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap per-benchmark problem count.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-problem timeout (seconds).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="ThreadPoolExecutor worker count.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON results path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model loading; emit a stub result per oracle.",
    )
    return parser


def _dry_run_result(
    bin_key: str, benchmark: str, adapter_id: str, threshold: float
) -> dict[str, Any]:
    """Stub result for --dry-run mode."""
    return {
        "bin_key": bin_key,
        "benchmark": benchmark,
        "adapter_id": adapter_id,
        "base_pass_at_1": 0.0,
        "stack_pass_at_1": 0.0,
        "delta": 0.0,
        "threshold": threshold,
        "passed": False,
        "dry_run": True,
    }


def _evaluate_one(
    base_model: str,
    bin_key: str,
    benchmark: str,
    adapter_id: str,
    threshold: float,
    max_samples: int | None,
    timeout_s: int,
    workers: int,
) -> dict[str, Any]:
    """Evaluate a single oracle against the base. Heavy imports deferred."""
    from evaluation.benchmarks import BenchmarkConfig, run_benchmark
    from evaluation.benchmarks.adapter_stack import load_adapter_stack

    cfg = BenchmarkConfig(
        timeout_s=timeout_s,
        max_workers=workers,
        max_samples=max_samples,
    )

    base_stack = load_adapter_stack(base_model=base_model, adapter_ids=[])
    stack = load_adapter_stack(base_model=base_model, adapter_ids=[adapter_id])

    base_result = run_benchmark(
        adapter_stack=base_stack,
        benchmark_id=benchmark,
        max_samples=max_samples,
        config=cfg,
    )
    stack_result = run_benchmark(
        adapter_stack=stack,
        benchmark_id=benchmark,
        max_samples=max_samples,
        config=cfg,
    )
    delta = stack_result.pass_at_1 - base_result.pass_at_1
    return {
        "bin_key": bin_key,
        "benchmark": benchmark,
        "adapter_id": adapter_id,
        "base_pass_at_1": base_result.pass_at_1,
        "stack_pass_at_1": stack_result.pass_at_1,
        "delta": delta,
        "threshold": threshold,
        "passed": delta >= threshold,
        "dry_run": False,
    }


def validate_oracles(
    base_model: str,
    oracle_specs: list[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_samples: int | None = None,
    timeout_s: int = 30,
    workers: int = 4,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Validate a list of oracle specs and return per-oracle results.

    Args:
        base_model: Base HF repo id.
        oracle_specs: List of "<bin_key>:<adapter_id>" strings.
        threshold: Absolute Pass@1 delta required.
        max_samples: Per-benchmark cap.
        timeout_s: Per-problem timeout.
        workers: ThreadPoolExecutor workers.
        dry_run: If True, emit stub results without loading models.

    Returns:
        List of result dicts, one per oracle.
    """
    results: list[dict[str, Any]] = []
    for spec in oracle_specs:
        bin_key, benchmark, adapter_id = _parse_oracle_spec(spec)
        if dry_run:
            results.append(
                _dry_run_result(bin_key, benchmark, adapter_id, threshold)
            )
            continue
        results.append(
            _evaluate_one(
                base_model=base_model,
                bin_key=bin_key,
                benchmark=benchmark,
                adapter_id=adapter_id,
                threshold=threshold,
                max_samples=max_samples,
                timeout_s=timeout_s,
                workers=workers,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)

    results = validate_oracles(
        base_model=args.base_model,
        oracle_specs=args.oracle,
        threshold=args.threshold,
        max_samples=args.max_samples,
        timeout_s=args.timeout,
        workers=args.workers,
        dry_run=args.dry_run,
    )

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        logger.info("Wrote %d results to %s", len(results), args.output)

    n_passed = sum(1 for r in results if r["passed"])
    print(f"Oracles validated: {n_passed}/{len(results)} PASS (>= {args.threshold})")
    for r in results:
        marker = "PASS" if r["passed"] else "FAIL"
        print(
            f"  [{marker}] {r['bin_key']:<30s} delta={r['delta']:+.3f} "
            f"(base={r['base_pass_at_1']:.3f}, stack={r['stack_pass_at_1']:.3f})"
        )

    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
