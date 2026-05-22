"""Typer CLI entry point: run, train, mine, bench commands."""

from __future__ import annotations

from pathlib import Path

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

    from typing import Any  # noqa: PLC0415

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
    typer.echo(f"Training {'with HPO' if hpo else 'single run'}")
    raise NotImplementedError("Training not yet implemented")


@app.command()
def mine(
    sessions_dir: Path = typer.Option(..., help="Directory of coding sessions"),
    output_dir: Path = typer.Option(..., help="Output corpus directory"),
) -> None:
    """Mine coding sessions into training corpus."""
    typer.echo(f"Mining {sessions_dir} → {output_dir}")
    raise NotImplementedError("Mining not yet implemented")


@app.command()
def bench(
    tasks_file: Path | None = typer.Option(None, help="Benchmark tasks JSON"),
    config: Path | None = typer.Option(None, help="Config JSON path"),
    hpo: bool = typer.Option(False, help="Run Optuna HPO"),
    n_trials: int = typer.Option(50, help="Number of HPO trials"),
) -> None:
    """Run benchmark suite, optionally with HPO."""
    typer.echo(f"Benchmarking {'with HPO' if hpo else 'single pass'}")
    raise NotImplementedError("Benchmarking not yet implemented")


if __name__ == "__main__":
    app()
