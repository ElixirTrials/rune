"""Typer CLI entry point: run, train, mine, bench commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)

app = typer.Typer(
    name="rune", help="Local-first coding agent with hypernetwork LoRA adapters"
)


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description"),
    config: Path | None = typer.Option(None, help="Path to config JSON"),
    checkpoint: str | None = typer.Option(None, help="Hypernetwork checkpoint path"),
) -> None:
    """Run a single task through the engine."""
    import asyncio  # noqa: PLC0415

    from rune.config import PipelineConfig, load_config  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415
    from rune.engine.state import make_initial_state  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    cfg = load_config(config) if config else PipelineConfig()
    if checkpoint:
        cfg = cfg.override(checkpoint_path=checkpoint)

    configure_mlflow("rune-run")
    typer.echo(f"Running task: {task}")

    model = ModelWrapper.from_config(cfg)

    initial_state = make_initial_state(task, cfg.max_phase_iterations)

    engine = create_engine()
    with tracked_run("run", params=cfg.to_dict()):
        final_state = asyncio.run(
            engine.ainvoke(
                initial_state,
                config={
                    "configurable": {
                        "model": model,
                        "run_config": cfg.to_dict(),
                    }
                },
            )
        )

    output = final_state.get("integrated_code") or ""
    if output:
        typer.echo(output)
    else:
        typer.echo("Done (no integrated code produced)")


def _load_train_config(path: Path) -> Any:
    """Load D2LTrainConfig from YAML."""
    import yaml  # noqa: PLC0415

    from rune.training.d2l_train import D2LTrainConfig  # noqa: PLC0415

    return D2LTrainConfig(**yaml.safe_load(path.read_text()))


@app.command()
def train(
    corpus_dir: Path | None = typer.Option(None, help="Training corpus directory"),
    config: Path | None = typer.Option(None, help="Config YAML path"),
    hpo: bool = typer.Option(False, help="Run Optuna HPO"),
    n_trials: int = typer.Option(50, help="Number of HPO trials"),
) -> None:
    """Train hypernetwork (oracle → distillation → gate)."""
    import asyncio  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415
    from rune.training.d2l_train import D2LTrainConfig  # noqa: PLC0415
    from rune.training.orchestrator import run_training_pipeline  # noqa: PLC0415

    typer.echo(f"Training {'with HPO' if hpo else 'single run'}")

    train_cfg = _load_train_config(config) if config is not None else D2LTrainConfig()

    if corpus_dir is None:
        corpus_dir = _Path("./corpus")

    configure_mlflow("rune-train")
    with tracked_run("train", params=train_cfg.model_dump()):
        exit_code = asyncio.run(
            run_training_pipeline(
                train_cfg,
                corpus_dir,
                hpo=hpo,
                n_trials=n_trials,
            )
        )
    raise typer.Exit(exit_code)


@app.command()
def mine(
    sessions_dir: Path = typer.Option(..., help="Directory of coding sessions"),
    output_dir: Path = typer.Option(..., help="Output corpus directory"),
) -> None:
    """Mine coding sessions into training corpus."""
    import mlflow as _mlflow  # noqa: PLC0415

    from rune.mining.miner import mine_corpus  # noqa: PLC0415
    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    configure_mlflow("rune-mine")
    typer.echo(f"Mining {sessions_dir} → {output_dir}")
    mine_params = {"sessions_dir": str(sessions_dir), "output_dir": str(output_dir)}
    with tracked_run("mine", params=mine_params):
        counts = mine_corpus(sessions_dir, output_dir)
        _mlflow.log_metrics({f"bin/{k}": v for k, v in counts.items()})
    for bin_key, count in sorted(counts.items()):
        typer.echo(f"  {bin_key}: {count} records")


@app.command()
def bench(
    tasks_file: Path | None = typer.Option(None, help="Benchmark tasks JSON"),
    config: Path | None = typer.Option(None, help="Config YAML path"),
    hpo: bool = typer.Option(False, help="Run Optuna HPO"),
    n_trials: int | None = typer.Option(None, help="Override hpo.n_trials from config"),
) -> None:
    """Run benchmark suite, optionally with HPO."""
    import asyncio  # noqa: PLC0415

    from rune.bench.runner import load_tasks, run_benchmark  # noqa: PLC0415
    from rune.config import PipelineConfig, load_config  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    if tasks_file is None:
        typer.echo("Error: --tasks-file is required", err=True)
        raise typer.Exit(1)

    import mlflow as _mlflow  # noqa: PLC0415

    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    cfg = load_config(config) if config else PipelineConfig()
    tasks = load_tasks(tasks_file)
    model = ModelWrapper.from_config(cfg)
    engine = create_engine()
    configure_mlflow("rune-bench")

    if hpo:
        from rune.bench.hpo import run_hpo  # noqa: PLC0415

        trials = n_trials or cfg.hpo["n_trials"]
        typer.echo(f"Running HPO: {trials} trials")
        with tracked_run("bench-hpo", params=cfg.to_dict()) as parent:
            best = asyncio.run(
                run_hpo(
                    tasks, engine, cfg, model, trials, parent_run_id=parent.info.run_id
                )
            )
            _mlflow.log_metric("tuning_best_pass_at_1", best["best_value"])
            if best["validation_pass_at_1"] is not None:
                _mlflow.log_metric("validation_pass_at_1", best["validation_pass_at_1"])
        typer.echo(
            f"Tuning best pass@1: {best['best_value']:.3f} "
            f"({best['n_tuning']} tuning tasks)"
        )
        val = best["validation_pass_at_1"]
        typer.echo(
            f"Held-out validation pass@1: {val:.3f} ({best['n_validation']} tasks)"
            if val is not None
            else "Held-out validation: skipped (too few tasks)"
        )
        typer.echo(f"Best params: {best['best_params']}")
        return

    typer.echo(f"Benchmarking {len(tasks)} task(s)")

    bench_config: dict[str, Any] = {
        "model": model,
        "run_config": cfg.to_dict(),
    }

    with tracked_run("bench", params=cfg.to_dict()):
        result = asyncio.run(run_benchmark(tasks, engine, bench_config))
        _mlflow.log_metric("pass_at_1", result.pass_at_1)
        _mlflow.log_metric("passed_tasks", result.passed_tasks)
        _mlflow.log_metric("total_tasks", result.total_tasks)

    typer.echo(
        f"pass@1: {result.pass_at_1:.3f}  ({result.passed_tasks}/{result.total_tasks})"
    )
    for tr in result.per_task:
        status = "PASS" if tr.passed else "FAIL"
        typer.echo(f"  [{status}] {tr.task_id}")


@app.command(name="gen-tasks")
def gen_tasks(
    out: Path = typer.Option(..., help="Output benchmark tasks JSON path"),
    ids_file: Path | None = typer.Option(
        None, help='JSON list of task_ids to keep, e.g. ["mbpp/1", "mbpp/2"]'
    ),
    limit: int | None = typer.Option(None, help="Keep at most N tasks"),
) -> None:
    """Generate an MBPP benchmark tasks JSON for `rune bench --tasks-file`."""
    import json  # noqa: PLC0415

    from rune.bench.mbpp import load_mbpp_tasks  # noqa: PLC0415
    from rune.bench.runner import dump_tasks  # noqa: PLC0415

    ids = set(json.loads(ids_file.read_text())) if ids_file is not None else None
    tasks = load_mbpp_tasks(ids=ids, limit=limit)
    dump_tasks(tasks, out)
    typer.echo(f"Wrote {len(tasks)} MBPP task(s) to {out}")


if __name__ == "__main__":
    app()
