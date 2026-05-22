from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="rune", help="Local-first coding agent with hypernetwork LoRA adapters")


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description"),
    config: Optional[Path] = typer.Option(None, help="Path to config JSON"),
    checkpoint: Optional[str] = typer.Option(None, help="Hypernetwork checkpoint path"),
) -> None:
    """Run a single task through the engine."""
    from rune.config import PipelineConfig, load_config  # noqa: PLC0415

    cfg = load_config(config) if config else PipelineConfig()
    if checkpoint:
        cfg = cfg.override(checkpoint_path=checkpoint)

    typer.echo(f"Running task: {task}")
    raise NotImplementedError("Engine invocation not yet implemented")


@app.command()
def train(
    corpus_dir: Optional[Path] = typer.Option(None, help="Training corpus directory"),
    config: Optional[Path] = typer.Option(None, help="Config JSON path"),
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
    tasks_file: Optional[Path] = typer.Option(None, help="Benchmark tasks JSON"),
    config: Optional[Path] = typer.Option(None, help="Config JSON path"),
    hpo: bool = typer.Option(False, help="Run Optuna HPO"),
    n_trials: int = typer.Option(50, help="Number of HPO trials"),
) -> None:
    """Run benchmark suite, optionally with HPO."""
    typer.echo(f"Benchmarking {'with HPO' if hpo else 'single pass'}")
    raise NotImplementedError("Benchmarking not yet implemented")


if __name__ == "__main__":
    app()
