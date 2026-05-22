"""Typer CLI entry point: run, train, mine, bench commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

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
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    cfg = load_config(config) if config else PipelineConfig()
    if checkpoint:
        cfg = cfg.override(checkpoint_path=checkpoint)

    typer.echo(f"Running task: {task}")

    model = ModelWrapper.from_config(cfg)

    initial_state: dict[str, Any] = {
        "task": task,
        "subtasks": [],
        "interfaces": {},
        "plans": {},
        "code_results": {},
        "code_passed": {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": None,
        "diagnosis": None,
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": cfg.max_phase_iterations,
    }

    engine = create_engine()
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


@app.command()
def train(
    corpus_dir: Path | None = typer.Option(None, help="Training corpus directory"),
    config: Path | None = typer.Option(None, help="Config JSON path"),
    hpo: bool = typer.Option(False, help="Run Optuna HPO"),
    n_trials: int = typer.Option(50, help="Number of HPO trials"),
) -> None:
    """Train hypernetwork (oracle → distillation → gate)."""
    import asyncio  # noqa: PLC0415
    import json  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    from rune.training.d2l_train import D2LTrainConfig  # noqa: PLC0415
    from rune.training.orchestrator import run_training_pipeline  # noqa: PLC0415

    typer.echo(f"Training {'with HPO' if hpo else 'single run'}")

    if config is not None:
        raw = json.loads(_Path(config).read_text())
        train_cfg = D2LTrainConfig(**raw)
    else:
        train_cfg = D2LTrainConfig()

    if corpus_dir is None:
        corpus_dir = _Path("./corpus")

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
    from rune.mining.miner import mine_corpus  # noqa: PLC0415

    typer.echo(f"Mining {sessions_dir} → {output_dir}")
    counts = mine_corpus(sessions_dir, output_dir)
    for bin_key, count in sorted(counts.items()):
        typer.echo(f"  {bin_key}: {count} records")


@app.command()
def bench(
    tasks_file: Path | None = typer.Option(None, help="Benchmark tasks JSON"),
    config: Path | None = typer.Option(None, help="Config JSON path"),
    hpo: bool = typer.Option(False, help="Run Optuna HPO"),
    n_trials: int = typer.Option(50, help="Number of HPO trials"),
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

    cfg = load_config(config) if config else PipelineConfig()
    tasks = load_tasks(tasks_file)
    model = ModelWrapper.from_config(cfg)
    engine = create_engine()

    if hpo:
        from rune.bench.hpo import run_hpo  # noqa: PLC0415

        typer.echo(f"Running HPO: {n_trials} trials")
        best = asyncio.run(run_hpo(tasks, engine, cfg, model, n_trials))
        typer.echo(f"Best pass@1: {best['best_value']:.3f}")
        typer.echo(f"Best params: {best['best_params']}")
        return

    typer.echo(f"Benchmarking {len(tasks)} task(s)")

    bench_config: dict[str, Any] = {
        "model": model,
        "run_config": cfg.to_dict(),
    }

    result = asyncio.run(run_benchmark(tasks, engine, bench_config))

    typer.echo(
        f"pass@1: {result.pass_at_1:.3f}  ({result.passed_tasks}/{result.total_tasks})"
    )
    for tr in result.per_task:
        status = "PASS" if tr.passed else "FAIL"
        typer.echo(f"  [{status}] {tr.task_id}")


if __name__ == "__main__":
    app()
