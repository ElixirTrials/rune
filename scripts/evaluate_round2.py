"""Run all 6 benchmarks for round-2 adapter vs round-1 baseline, apply gate.

Usage:
    uv run scripts/evaluate_round2.py \\
        --round2-adapter-id round2_<hex8> \\
        --baseline-report path/to/round1_scores.json \\
        --output-report path/to/round2_report.json

The baseline report is expected to have the shape::

    {
        "humaneval": 0.60,
        "mbpp": 0.50,
        "apps": 0.20,
        "bigcodebench": 0.35,
        "ds_1000": 0.40,
        "livecodebench": 0.15
    }

The output report carries the full gate verdict including per-benchmark deltas.
Exits 0 on PASS, 1 on FAIL so CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from model_training.round2_gate import REQUIRED_BENCHMARKS, evaluate_round2_gate


def _run_benchmarks_for_adapter(adapter_id: str) -> dict[str, float]:
    """Run HumanEval, MBPP, APPS, BigCodeBench, DS-1000, LiveCodeBench for an adapter.

    Delegates to ``evaluation.benchmarks.run_benchmark`` (Plan A harness).
    Each call returns ``{"pass_at_1": float, ...}``; we collect the pass_at_1
    into a ``{bench: score}`` map.
    """
    from evaluation.benchmarks import run_benchmark  # noqa: PLC0415

    result: dict[str, float] = {}
    logger = logging.getLogger(__name__)
    for bench in REQUIRED_BENCHMARKS:
        logger.info("Evaluating %s on %s", adapter_id, bench)
        out = run_benchmark(benchmark_id=bench, adapter_id=adapter_id)
        result[bench] = float(out["pass_at_1"])
    return result


def main(argv: Sequence[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round2-adapter-id", required=True)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    ns = parser.parse_args(argv)

    baseline: dict[str, float] = json.loads(Path(ns.baseline_report).read_text())
    round2_scores: dict[str, float] = _run_benchmarks_for_adapter(ns.round2_adapter_id)

    scores: dict[str, dict[str, float]] = {
        b: {"baseline": float(baseline[b]), "round2": float(round2_scores[b])}
        for b in REQUIRED_BENCHMARKS
    }
    report: dict[str, Any] = evaluate_round2_gate(scores)
    report["round2_adapter_id"] = ns.round2_adapter_id
    report["scores"] = scores

    Path(ns.output_report).write_text(json.dumps(report, indent=2, sort_keys=True))
    logging.info(
        "Gate verdict: %s — report at %s",
        "PASS" if report["passed"] else "FAIL",
        ns.output_report,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
