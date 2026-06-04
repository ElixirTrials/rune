"""Launch the GOAL-3 flavor x adapter_scaling HPO under the guard.

REMOVE-BEFORE-MERGE. Mirrors `rune bench --hpo` but as a script so the RAM/disk
watchdog (run_guarded.sh) can supervise the multi-hour overnight run, and writes
the result to /tmp/goal3/hpo_result.json for recovery if the instance dies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--n-trials", type=int, default=None, dest="n_trials")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    import mlflow  # noqa: PLC0415

    from rune.bench.hpo import run_hpo  # noqa: PLC0415
    from rune.bench.runner import load_tasks  # noqa: PLC0415
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import (  # noqa: PLC0415
        configure_mlflow,
        log_dataset,
        tracked_run,
    )

    cfg = load_rune_config(Path(args.config))
    tasks = load_tasks(Path(args.tasks))
    model = ModelWrapper.from_config(cfg)
    engine = create_engine()
    configure_mlflow("rune-bench")
    trials = args.n_trials or cfg.hpo["n_trials"]
    print(
        f"HPO start: {trials} trials fresh={args.fresh} "
        f"tasks={len(tasks)} model_judge={cfg.model_judge}",
        flush=True,
    )
    with tracked_run("goal3-flavor-hpo", params=cfg.to_dict()) as parent:
        log_dataset(Path(args.tasks), name=Path(args.tasks).name, context="test")
        best = asyncio.run(
            run_hpo(
                tasks,
                engine,
                cfg,
                model,
                trials,
                parent_run_id=parent.info.run_id,
                fresh=args.fresh,
            )
        )
        mlflow.log_metric("tuning_best_pass_at_1", best["best_value"])
        if best["validation_pass_at_1"] is not None:
            mlflow.log_metric("validation_pass_at_1", best["validation_pass_at_1"])

    Path("/tmp/goal3").mkdir(parents=True, exist_ok=True)
    Path("/tmp/goal3/hpo_result.json").write_text(
        json.dumps(best, indent=2, default=str)
    )
    print(
        f"HPO DONE best_tuning_pass@1={best['best_value']:.3f} "
        f"val_pass@1={best['validation_pass_at_1']} params={best['best_params']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
