# tools/adapter_scaling_hpo.py
"""Optuna HPO for adapter_scaling (B-only) to find the sweet spot.

Run:
  uv run python tools/adapter_scaling_hpo.py \
      [--config benchmarks/bench.yaml] [--n-trials 30]

All generation params and search ranges come from bench.yaml.
Results logged to MLflow experiment 'adapter-scaling-hpo'.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from rune.model.adapter import scale_lora_b

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("benchmarks/bench.yaml")


@dataclass(frozen=True)
class Prompt:
    name: str
    task: str
    fn_name: str
    system_prompt: str = "You are a code generator."


PROMPTS = [
    Prompt("add", "Write a Python function add(a, b) that returns a + b.", "add"),
    Prompt(
        "sort",
        "Write a Python function sort_list(items) that sorts a list of integers.",
        "sort_list",
    ),
    Prompt(
        "reverse",
        "Write a Python function reverse_str(s) that reverses a string.",
        "reverse_str",
    ),
]


def _trajectory(role: str, task: str, plan: str) -> str:
    return f"ROLE: {role}\nTASK: {task}\nPLAN: {plan}"


def _enriched_trajectory(task: str, fn_name: str) -> str:
    return (
        f"ROLE: coder\nTASK: {task}\n"
        f"PLAN: Implement {fn_name} with type hints.\n"
        f"INTERFACE: def {fn_name}(...) -> ...\n"
        "ERROR: NameError on first attempt.\n"
        "FIX: Corrected function name typo."
    )


def _contradictory_trajectory() -> str:
    return (
        "ROLE: coder\nTASK: Sort a list of integers in ascending order.\n"
        "PLAN: Implement bubble sort with early termination.\n"
        "INTERFACE: def sort_list(items: list[int]) -> list[int]"
    )


def _edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    return 1.0 - ratio


def _coherence(output: str, fn_name: str) -> float:
    if len(output) < 10:
        return 0.0
    printable = sum(1 for c in output if c.isprintable() or c in "\n\t")
    ascii_ratio = printable / len(output)
    has_fn = 1.0 if f"def {fn_name}" in output else 0.0
    return has_fn * 0.6 + ascii_ratio * 0.4




def _run_trial(
    model: Any,
    scaling: float,
    gen_kwargs: dict[str, Any],
) -> dict[str, float]:
    import torch  # noqa: PLC0415

    metrics: dict[str, float] = {}
    diff_scores: list[float] = []
    sensitivity_scores: list[float] = []
    contradiction_scores: list[float] = []
    coherence_scores: list[float] = []

    for p in PROMPTS:
        traj = _trajectory(
            "coder",
            p.task,
            f"Implement {p.fn_name}.",
        )
        adapter = model.generate_adapter(traj)
        raw_sd = adapter.state_dict

        zero_sd = {k: torch.zeros_like(v) for k, v in raw_sd.items()}
        model.hotswap_adapter(zero_sd)
        baseline = asyncio.run(
            model.generate(
                p.task,
                system_prompt=p.system_prompt,
                **gen_kwargs,
            )
        )

        model.hotswap_adapter(scale_lora_b(raw_sd, scaling))
        adapted = asyncio.run(
            model.generate(
                p.task,
                system_prompt=p.system_prompt,
                **gen_kwargs,
            )
        )

        e_traj = _enriched_trajectory(p.task, p.fn_name)
        enriched_adapter = model.generate_adapter(e_traj)
        model.hotswap_adapter(
            scale_lora_b(enriched_adapter.state_dict, scaling),
        )
        enriched = asyncio.run(
            model.generate(
                p.task,
                system_prompt=p.system_prompt,
                **gen_kwargs,
            )
        )

        contra_adapter = model.generate_adapter(
            _contradictory_trajectory(),
        )
        model.hotswap_adapter(
            scale_lora_b(contra_adapter.state_dict, scaling),
        )
        contradictory = asyncio.run(
            model.generate(
                p.task,
                system_prompt=p.system_prompt,
                **gen_kwargs,
            )
        )

        diff = _edit_distance(baseline.text, adapted.text)
        sens = _edit_distance(adapted.text, enriched.text)
        contra = _edit_distance(adapted.text, contradictory.text)
        coh = _coherence(adapted.text, p.fn_name)

        diff_scores.append(diff)
        sensitivity_scores.append(sens)
        contradiction_scores.append(contra)
        coherence_scores.append(coh)

        metrics[f"diff/{p.name}"] = round(diff, 4)
        metrics[f"sensitivity/{p.name}"] = round(sens, 4)
        metrics[f"contradiction/{p.name}"] = round(contra, 4)
        metrics[f"coherence/{p.name}"] = round(coh, 4)

    avg_diff = sum(diff_scores) / len(diff_scores)
    avg_sens = sum(sensitivity_scores) / len(sensitivity_scores)
    avg_contra = sum(contradiction_scores) / len(contradiction_scores)
    avg_coh = sum(coherence_scores) / len(coherence_scores)

    influence = (avg_diff + avg_sens + avg_contra) / 3.0
    objective = avg_coh * influence

    metrics["avg/differentiation"] = round(avg_diff, 4)
    metrics["avg/sensitivity"] = round(avg_sens, 4)
    metrics["avg/contradiction"] = round(avg_contra, 4)
    metrics["avg/coherence"] = round(avg_coh, 4)
    metrics["avg/influence"] = round(influence, 4)
    metrics["objective"] = round(objective, 4)
    return metrics


def main() -> None:
    from rune.config import load_config  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Adapter scaling HPO")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Config YAML path (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.checkpoint_path:
        parser.error("Config must set checkpoint_path")

    bench = cfg.bench
    hpo = cfg.hpo
    hpo_as = hpo["adapter_scaling"]
    gen_kwargs = {
        "max_tokens": bench["gen_max_tokens"],
        "temperature": bench["gen_temperature"],
        "repetition_penalty": cfg.repetition_penalty,
        "top_p": cfg.top_p,
    }
    n_trials = args.n_trials or hpo["n_trials"]

    import mlflow  # noqa: PLC0415
    import optuna  # noqa: PLC0415

    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import configure_mlflow  # noqa: PLC0415

    configure_mlflow("adapter-scaling-hpo")
    model = ModelWrapper.from_config(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        scaling = trial.suggest_float(
            "adapter_scaling",
            hpo_as["low"],
            hpo_as["high"],
            log=hpo_as.get("log", False),
        )
        logger.info(
            "Trial %d: adapter_scaling=%.3f",
            trial.number,
            scaling,
        )

        metrics = _run_trial(model, scaling, gen_kwargs)

        with mlflow.start_run(
            run_name=f"trial-{trial.number}",
            nested=True,
        ):
            mlflow.log_params(
                {"adapter_scaling": scaling, "trial": trial.number},
            )
            mlflow.log_metrics(metrics)

        logger.info(
            "Trial %d: obj=%.4f (coh=%.2f, diff=%.2f, sens=%.2f, contra=%.2f)",
            trial.number,
            metrics["objective"],
            metrics["avg/coherence"],
            metrics["avg/differentiation"],
            metrics["avg/sensitivity"],
            metrics["avg/contradiction"],
        )
        return metrics["objective"]

    study = optuna.create_study(
        direction="maximize",
        study_name="adapter-scaling-hpo",
    )

    with mlflow.start_run(run_name="adapter-scaling-hpo"):
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "search_low": hpo_as["low"],
                "search_high": hpo_as["high"],
                "n_prompts": len(PROMPTS),
                "max_tokens": gen_kwargs["max_tokens"],
                "temperature": gen_kwargs["temperature"],
            }
        )
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params({f"best/{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best/objective", study.best_value)

    print("\n=== BEST ===")
    print(f"  adapter_scaling: {study.best_params['adapter_scaling']:.4f}")
    print(f"  objective: {study.best_value:.4f}")
    print("\n=== ALL TRIALS ===")
    for t in sorted(
        study.trials,
        key=lambda t: t.value or 0,
        reverse=True,
    ):
        print(
            f"  #{t.number}:"
            f" scaling={t.params['adapter_scaling']:.3f}"
            f" objective={t.value:.4f}"
        )


if __name__ == "__main__":
    main()
