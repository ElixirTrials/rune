"""Thin GPU entry wrapper: run Rune's pass@1 bench programmatically.

Mirrors ``rune.cli.bench`` (non-HPO path) so that

    tools/run_guarded.sh <log> tools/_bench_entry.py --tasks-file t.json \
        --adapter-scaling 0.0

works (run_guarded takes a script path, not ``python -m``). Prints exactly one
JSON line to stdout: {"pass_at_1": ..., "passed_tasks": ..., "total": ...}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rune.bench.runner import load_tasks, run_benchmark
from rune.config import PipelineConfig, load_config
from rune.engine.graph import create_engine
from rune.model.wrapper import ModelWrapper


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Rune pass@1 bench (single config).")
    p.add_argument("--tasks-file", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--adapter-scaling", type=float, default=None)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--checkpoint-path", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = load_config(args.config) if args.config else PipelineConfig()

    # None-sentinels (not truthiness): --adapter-scaling 0.0 must survive, it is
    # the no-adapter baseline arm. Apply only the args that were actually passed.
    overrides: dict[str, object] = {}
    if args.model_id is not None:
        overrides["model_id"] = args.model_id
    if args.checkpoint_path is not None:
        overrides["checkpoint_path"] = args.checkpoint_path
    if args.adapter_scaling is not None:
        overrides["adapter_scaling"] = args.adapter_scaling
    if overrides:
        cfg = cfg.override(**overrides)

    tasks = load_tasks(args.tasks_file)
    model = ModelWrapper.from_config(cfg)
    engine = create_engine()

    bench_config: dict[str, object] = {
        "model": model,
        "run_config": cfg.to_dict(),
    }

    result = asyncio.run(run_benchmark(tasks, engine, bench_config))

    # Best-effort MLflow logging (experiment "issue52-recipe", one run per gate).
    # An unreachable server degrades to a stderr note; never breaks the run or
    # pollutes the single stdout JSON line.
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("issue52-recipe")
        with mlflow.start_run(run_name="bench"):
            mlflow.log_params(
                {
                    "model_id": cfg.model_id,
                    "checkpoint_path": cfg.checkpoint_path,
                    "adapter_scaling": cfg.adapter_scaling,
                }
            )
            mlflow.log_metric("pass_at_1", result.pass_at_1)
            mlflow.log_metric("passed_tasks", result.passed_tasks)
            mlflow.log_metric("total_tasks", result.total_tasks)
    except Exception as exc:  # noqa: BLE001
        print(f"MLflow logging skipped: {exc}", file=sys.stderr)

    print(
        json.dumps(
            {
                "pass_at_1": result.pass_at_1,
                "passed_tasks": result.passed_tasks,
                "total": result.total_tasks,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
