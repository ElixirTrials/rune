"""Overnight optimization runner: find optimal adapter-prompt parameters.

Uses Optuna TPE sampler to search across scaling, prompt style, trajectory
style, temperature, max_tokens, repetition_penalty, and trajectory length.

Evaluates each trial across a random subset of diverse tasks to prevent
overfitting. Results are persisted to SQLite for crash recovery.

Usage:
    uv run python scripts/optimization/run_optimization.py
    uv run python scripts/optimization/run_optimization.py --n-trials 50
    uv run python scripts/optimization/run_optimization.py --tasks event_store,regex_nfa
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("INFERENCE_PROVIDER", "transformers")
os.environ.setdefault("TRANSFORMERS_MODEL_NAME", "google/gemma-2-2b-it")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path

setup_path()

import shutil

import optuna
import torch
from model_training.sakana_d2l import (
    download_checkpoint,
    generate_adapter_from_sakana,
)
from peft import PeftModel
from shared.hardware import get_best_device
from shared.pipeline_config import PipelineConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import score_code
from task_pool import EvalTask, get_tasks
from template_library import get_prompt, get_trajectory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("optimization")

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = get_best_device()
CODE_SYSTEM = "You are a Python code generator. Output only code, no explanation."

# Prompt and trajectory style choices
PROMPT_STYLES = ["minimal", "must_include", "skeleton", "hybrid", "open"]
TRAJECTORY_STYLES = ["prose", "exemplar", "signatures", "full_context", "minimal"]

# ---------------------------------------------------------------------------
# Model + checkpoint management (load once, reuse)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_checkpoint_path = None


def _setup() -> None:
    global _model, _tokenizer, _checkpoint_path
    if _model is not None:
        return

    from shared.hardware import resolve_model_dtype

    logger.info("Loading model %s on %s", MODEL_NAME, DEVICE)
    _checkpoint_path = str(download_checkpoint(variant="gemma_demo"))

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    config = AutoModelForCausalLM.from_pretrained(MODEL_NAME).config
    h = getattr(config, "hidden_size", 2048)
    v = getattr(config, "vocab_size", 32000)
    n = getattr(config, "num_hidden_layers", 24)
    param_count = v * h + n * 12 * h * h
    dtype = resolve_model_dtype(param_count=param_count, device=DEVICE)

    _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    _model.to(DEVICE)  # type: ignore[arg-type]
    _model.eval()
    logger.info("Model loaded (dtype=%s)", dtype)


# ---------------------------------------------------------------------------
# Single-trial evaluation
# ---------------------------------------------------------------------------


def _generate_with_adapter(
    trajectory: str,
    prompt: str,
    scaling: float,
    max_length: int,
    temperature: float,
    max_tokens: int,
    repetition_penalty: float,
    use_bias: bool,
) -> str:
    """Generate code using an adapter built from the trajectory."""
    assert _model is not None, "call _setup() first"
    assert _tokenizer is not None, "call _setup() first"
    tmpdir = tempfile.mkdtemp()
    try:
        adapter_dir = str(Path(tmpdir) / "adapter")

        # Generate adapter
        gc.collect()
        torch.cuda.empty_cache()
        adapter_path = generate_adapter_from_sakana(
            text=trajectory,
            output_dir=adapter_dir,
            checkpoint_path=_checkpoint_path,
            base_model_name=MODEL_NAME,
            device=DEVICE,
            max_length=max_length,
        )

        # Apply scaling override
        config_path = Path(adapter_path) / "adapter_config.json"
        cfg = json.loads(config_path.read_text())
        cfg["lora_alpha"] = cfg["lora_alpha"] * scaling
        config_path.write_text(json.dumps(cfg, indent=2))

        # Load adapter
        safe_name = f"opt_{id(trajectory) % 10000}"
        model = PeftModel.from_pretrained(_model, adapter_path, adapter_name=safe_name)  # type: ignore[arg-type]
        model.to(DEVICE)  # type: ignore[arg-type]
        model.eval()

        # Generate
        content = f"{CODE_SYSTEM}\n\n{prompt}"
        messages = [{"role": "user", "content": content}]
        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=8192
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                top_p=0.9,
                repetition_penalty=repetition_penalty,
                pad_token_id=_tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][input_len:]
        # decode() returns str | list[str]; 1D tensor → always str.
        text = cast(str, _tokenizer.decode(new_tokens, skip_special_tokens=True))

        del model
        gc.collect()
        torch.cuda.empty_cache()

        return text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def evaluate_trial(
    tasks: list[EvalTask],
    scaling: float,
    prompt_style: str,
    trajectory_style: str,
    max_length: int,
    temperature: float,
    max_tokens: int,
    repetition_penalty: float,
    use_bias: bool,
) -> float:
    """Evaluate a parameter set across multiple tasks. Returns average score."""
    scores: list[float] = []

    for task in tasks:
        prompt = get_prompt(prompt_style, task)
        trajectory = get_trajectory(trajectory_style, task)

        code = _generate_with_adapter(
            trajectory=trajectory,
            prompt=prompt,
            scaling=scaling,
            max_length=max_length,
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            use_bias=use_bias,
        )

        result = score_code(code, task)
        scores.append(result.total)

        logger.info(
            "  %s: %.1f (pass=%s, domain=%d, struct=%d, degen=%s) %s",
            task.name,
            result.total,
            result.tests_pass,
            result.domain_hits,
            result.structure_hits,
            not result.not_degenerate,
            result.error[:60] if result.error else "",
        )

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def make_objective(
    all_tasks: list[EvalTask],
    tasks_per_trial: int = 3,
) -> Any:
    """Create an Optuna objective that samples tasks per trial."""

    import mlflow  # noqa: PLC0415

    def objective(trial: optuna.Trial) -> float:
        scaling = trial.suggest_float("scaling", 0.02, 0.20)
        prompt_style = trial.suggest_categorical("prompt_style", PROMPT_STYLES)
        trajectory_style = trial.suggest_categorical(
            "trajectory_style", TRAJECTORY_STYLES
        )
        max_length = trial.suggest_categorical("max_length", [512, 1024, 2048])
        temperature = trial.suggest_float("temperature", 0.1, 0.7)
        max_tokens = trial.suggest_categorical("max_tokens", [512, 1024, 2048])
        repetition_penalty = trial.suggest_float("repetition_penalty", 1.0, 1.3)
        use_bias = trial.suggest_categorical("use_bias", [True, False])

        # Sample tasks for this trial
        n = min(tasks_per_trial, len(all_tasks))
        trial_tasks = random.sample(all_tasks, n)

        logger.info(
            "Trial %d: scaling=%.3f prompt=%s traj=%s len=%d temp=%.2f "
            "max_tok=%d rep=%.2f bias=%s tasks=%s",
            trial.number,
            scaling,
            prompt_style,
            trajectory_style,
            max_length,
            temperature,
            max_tokens,
            repetition_penalty,
            use_bias,
            [t.name for t in trial_tasks],
        )

        # One MLflow run per trial; Optuna's catch=(Exception,) at the
        # study level marks any raised exception as a FAILED trial so the
        # study-level best-trial selection excludes failures — no silent
        # zero-score fallbacks.
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}"):
            mlflow.set_tags(
                {
                    "hpo.kind": "pipeline-hpo-trial",
                    "hpo.trial_number": str(trial.number),
                    "hpo.tasks_evaluated": ",".join(t.name for t in trial_tasks),
                }
            )
            mlflow.log_params(
                {
                    "scaling": scaling,
                    "prompt_style": prompt_style,
                    "trajectory_style": trajectory_style,
                    "max_length": max_length,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "repetition_penalty": repetition_penalty,
                    "use_bias": use_bias,
                    "tasks_per_trial": n,
                }
            )
            score = evaluate_trial(
                tasks=trial_tasks,
                scaling=scaling,
                prompt_style=prompt_style,
                trajectory_style=trajectory_style,
                max_length=max_length,
                temperature=temperature,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                use_bias=use_bias,
            )
            mlflow.log_metric("fitness", score)

        logger.info("Trial %d: avg_score=%.2f", trial.number, score)
        return score

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize pipeline parameters with Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=200, help="Number of optimization trials"
    )
    parser.add_argument(
        "--tasks-per-trial",
        type=int,
        default=3,
        help="Tasks to evaluate per trial",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names to include (default: all)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="sqlite:///optuna_pipeline.db",
        help="Optuna storage URI",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="rune-pipeline-v1",
        help="Optuna study name",
    )
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=15,
        help="Random trials before TPE kicks in",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="rune-pipeline-hpo",
        help="MLflow experiment name for per-trial runs",
    )
    args = parser.parse_args()

    # Load model
    _setup()

    # Task pool
    task_names = args.tasks.split(",") if args.tasks else None
    all_tasks = get_tasks(task_names)
    logger.info("Task pool: %s", [t.name for t in all_tasks])

    # Point MLflow at the requested experiment; per-trial runs are opened
    # inside the objective function.
    import mlflow  # noqa: PLC0415

    mlflow.set_experiment(args.experiment_name)

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.db,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=args.startup_trials,
            seed=42,
        ),
    )

    objective = make_objective(all_tasks, args.tasks_per_trial)

    logger.info(
        "Starting optimization: %d trials, %d tasks/trial, %d startup",
        args.n_trials,
        args.tasks_per_trial,
        args.startup_trials,
    )
    t0 = time.time()

    # catch=(Exception,) so per-trial failures are marked FAILED in Optuna
    # (excluded from best_trial) instead of being silently scored 0.0.
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
        catch=(Exception,),
    )

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    failed = [t for t in study.trials if t.state.name == "FAIL"]
    if not completed:
        msg = (
            f"Pipeline HPO study '{study.study_name}' produced no successful "
            f"trials ({len(failed)} failed, {len(study.trials)} total). "
            "Refusing to save a 'best config'."
        )
        logger.error(msg)
        raise SystemExit(msg)

    elapsed = time.time() - t0
    logger.info("Optimization complete in %.0fs", elapsed)
    logger.info("Best score: %.2f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Save optimal config
    best = study.best_params
    config = PipelineConfig().override(
        **{
            "adapter.scaling": best["scaling"],
            "adapter.use_bias": best["use_bias"],
            "adapter.max_length": best["max_length"],
            "generation.temperature": best["temperature"],
            "generation.max_tokens": best["max_tokens"],
            "generation.repetition_penalty": best["repetition_penalty"],
            "prompt.style": best["prompt_style"],
            "trajectory.style": best["trajectory_style"],
        }
    )
    config_path = config.save()
    logger.info("Optimal config saved to %s", config_path)

    # Study-level summary run in its own experiment so it lives alongside
    # (but doesn't nest under) the per-trial runs.
    mlflow.set_experiment(f"{args.experiment_name}-studies")
    with mlflow.start_run(run_name=f"study-{args.study_name}"):
        mlflow.set_tags(
            {
                "hpo.kind": "pipeline-hpo-study-summary",
                "hpo.study_name": args.study_name,
                "hpo.db": args.db,
                "hpo.config_path": str(config_path),
            }
        )
        mlflow.log_params(
            {
                "n_trials_requested": args.n_trials,
                "tasks_per_trial": args.tasks_per_trial,
                "startup_trials": args.startup_trials,
                "task_pool": ",".join(t.name for t in all_tasks),
                **{f"best.{k}": v for k, v in best.items()},
            }
        )
        mlflow.log_metrics(
            {
                "best_fitness": study.best_value,
                "best_trial_number": study.best_trial.number,
                "n_trials_completed": len(completed),
                "n_trials_failed": len(failed),
                "n_trials_total": len(study.trials),
                "elapsed_seconds": elapsed,
            }
        )

    # Print top 10 trials
    print("\n=== TOP 10 TRIALS ===")
    top_trials = sorted(completed, key=lambda t: t.value or 0, reverse=True)[:10]
    for t in top_trials:
        print(
            f"  #{t.number:3d} score={t.value:.2f} "
            f"scaling={t.params.get('scaling', 0):.3f} "
            f"prompt={t.params.get('prompt_style', '')} "
            f"traj={t.params.get('trajectory_style', '')} "
            f"temp={t.params.get('temperature', 0):.2f} "
            f"rep={t.params.get('repetition_penalty', 0):.2f}"
        )

    print(f"\nBest config saved to: {config_path}")


if __name__ == "__main__":
    main()
