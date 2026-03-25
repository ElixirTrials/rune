#!/usr/bin/env python3
"""CLI entrypoint for the evaluation benchmark runner.

Loads a YAML config from ``libs/evaluation/src/evaluation/configs/`` (or any
path) and runs the full benchmark pipeline.

Usage::

    # Run a YAML config
    uv run python scripts/run_benchmark.py \\
        --config libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml

    # Quick inline run (override config values via flags)
    uv run python scripts/run_benchmark.py \\
        --config libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml \\
        --n-problems 10      \\
        --output-dir /tmp/bench_out

    # Run with two prompts back-to-back (each gets its own run_id)
    uv run python scripts/run_benchmark.py \\
        --config libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml \\
        --prompt-templates math_prompt_v2.j2 math_prompt.j2

    # Enable debug logging
    uv run python scripts/run_benchmark.py \\
        --config libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml \\
        --log-level DEBUG
"""
# ruff: noqa: T201
# mypy: ignore-errors
from __future__ import annotations

import argparse
import copy
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "libs" / "evaluation" / "src"))
sys.path.insert(0, str(REPO_ROOT / "libs" / "shared" / "src"))
sys.path.insert(0, str(REPO_ROOT / "libs" / "inference" / "src"))


def _fresh_run_id(suffix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}_{suffix}"


def _print_summaries(summaries: list[dict], label: str) -> None:
    print(f"\n── Benchmark complete  [{label}] ────────────────────────────")
    for s in summaries:
        print(
            f"  {s['dataset']:<30}  "
            f"{s['correct']:>4}/{s['total']:<4}  "
            f"acc={s['accuracy']:.1%}  "
            f"errors={s['errors']}  "
            f"avg_tok/s={s['avg_tok_per_sec']:.0f}  "
            f"wall={s['wall_time_s']:.1f}s"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a benchmark from a YAML config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML benchmark config file.",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=None,
        metavar="N",
        help="Cap every dataset at N problems (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output_dir from the config.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override the run_id (single-prompt runs only).",
    )
    parser.add_argument(
        "--prompt-templates",
        nargs="+",
        metavar="TEMPLATE",
        default=None,
        help=(
            "One or more Jinja2 prompt templates to run sequentially "
            "(e.g. math_prompt_v2.j2 math_prompt.j2). "
            "Each gets its own auto-generated run_id. "
            "Overrides system_prompt_template from the config."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from evaluation.benchmark_runner import BenchmarkRunner
    from evaluation.config import load_config

    base_cfg = load_config(args.config)

    if args.output_dir is not None:
        base_cfg.output_dir = args.output_dir
    if args.n_problems is not None:
        for ds in base_cfg.datasets:
            ds.n_samples = args.n_problems

    if args.prompt_templates:
        all_summaries: list[tuple[str, list[dict]]] = []
        for template in args.prompt_templates:
            cfg = copy.deepcopy(base_cfg)
            cfg.system_prompt_template = template
            cfg.system_prompt = None  # clear any pre-rendered prompt
            stem = Path(template).stem
            cfg.run_id = _fresh_run_id(stem)
            logging.getLogger(__name__).info(
                "Starting run  template=%s  run_id=%s", template, cfg.run_id
            )
            summaries = BenchmarkRunner(cfg).run()
            all_summaries.append((template, summaries))

        for template, summaries in all_summaries:
            _print_summaries(summaries, label=template)
    else:
        cfg = base_cfg
        if args.run_id is not None:
            cfg.run_id = args.run_id
        summaries = BenchmarkRunner(cfg).run()
        _print_summaries(summaries, label=cfg.system_prompt_template or "default")


if __name__ == "__main__":
    main()
