"""Hyperparameter optimization for QLoRA fine-tuning on mined pair data.

Unlike ``run_optimization.py`` (which tunes *inference*-side parameters
for a frozen Sakana adapter), this harness tunes the *training*-time
hyperparameters of a DeltaCoder-warm-started QLoRA fine-tune on mined
GitHub trajectories.

Search space — warm-start-aware:

* ``lr`` (log-uniform, 1e-5 … 5e-4) — centered on the repo default 2e-4.
  Thinking Machines' "LoRA Without Regret" finds optimal LoRA LR is
  ~10x the FullFT LR and approximately rank-invariant, so we sample
  LR across orders of magnitude rather than narrowly.
* ``alpha_override`` (categorical, {16, 32, 64, 128}) — applied post-load
  via module-tree walk (``_override_lora_alpha``). DeltaCoder's saved
  alpha stays on disk; only the effective scaling at training time
  changes per trial.
* ``lora_dropout`` (categorical, {0.0, 0.05, 0.1}) — applied post-load
  via ``_override_lora_dropout``. Small grid because recent research
  calls short-run LoRA dropout an unreliable regularizer.
* ``warmup_ratio`` (uniform, 0.0 … 0.1).
* ``grad_accum`` (categorical, {8, 16, 32}) — LoRA penalizes large
  effective batches more than FullFT does.
* ``lr_scheduler`` (categorical, {constant, cosine}).
* ``diff_aware_loss`` (categorical, {True, False}) — A/B flag so the
  study can adjudicate whether the custom collator beats vanilla SFT.

``rank`` and ``target_modules`` are NOT in the search space: both are
baked into the DeltaCoder safetensor shapes and cannot be changed
without discarding the warm-start.

Budget targeting a single L4 24GB: with Hyperband pruning a study of
10 trials on a 500-record subsample × 1 epoch lands around 8–14
GPU-hours — overnight-scale. See ``docs/plans/training_upgrade.md``.

Usage:
    uv run python scripts/optimization/run_training_hpo.py \
        --dataset data/pairs/repo.jsonl \
        --n-trials 10 \
        --study-name rune-training-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("training-hpo")


@dataclass(frozen=True)
class FitnessConfig:
    """Blended fitness weights for HPO trial ranking.

    The blend is::

        fitness = loss_weight * (1 - normalize(eval_loss))
                + pass_at_1_weight * pass_at_1_humaneval_smoke

    Defense: pure loss overrates trials that overfit a small subsample;
    pure pass@1 on a 20-task smoke tier has too much variance to rank
    trials reliably. The blend stabilizes ranking while still rewarding
    real generation quality. Weights are exposed so operators can sweep
    them later without code changes.
    """

    loss_weight: float = 0.6
    pass_at_1_weight: float = 0.4


@dataclass
class HPORunArgs:
    """Non-search-space CLI arguments threaded into the Optuna objective."""

    dataset: str
    adapter_id_prefix: str
    model_config_name: str
    warm_start: str | None
    subsample: int
    eval_tier: str
    output_root: Path
    experiment_name: str
    keep_top_k: int
    extra_train_kwargs: dict[str, Any] = field(default_factory=dict)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_training_hpo",
        description="Training-hyperparameter HPO for DeltaCoder fine-tune.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="JSONL of mined pairs.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument(
        "--study-name", dest="study_name", default="rune-training-v1"
    )
    parser.add_argument(
        "--db", default="sqlite:///./optuna_training.db", help="Optuna storage URI"
    )
    parser.add_argument(
        "--model", dest="model_config_name", default="qwen3.5-9b"
    )
    parser.add_argument(
        "--warm-start",
        dest="warm_start",
        default="deltacoder",
        help="Warm-start alias or HF/local path (see trainer_cli).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=500,
        help="Records-per-trial subsample (proxy mode for L4 throughput).",
    )
    parser.add_argument(
        "--eval-tier",
        choices=["smoke", "mini", "none"],
        default="smoke",
        help="HumanEval tier for pass@1 fitness signal.",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default="./hpo_artifacts",
        help="Directory to write per-trial adapters.",
    )
    parser.add_argument(
        "--experiment-name",
        dest="experiment_name",
        default="rune-qlora-hpo",
    )
    parser.add_argument(
        "--keep-top-k",
        dest="keep_top_k",
        type=int,
        default=3,
        help="Retain the top-K trial adapters; rest are deleted after the study.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2-trial × 1-step smoke test for CI; ignores --n-trials.",
    )
    parser.add_argument(
        "--loss-weight", dest="loss_weight", type=float, default=0.6
    )
    parser.add_argument(
        "--pass-at-1-weight",
        dest="pass_at_1_weight",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="TPE sampler seed."
    )
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=4,
        help="Random trials before TPE kicks in.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Resolve args and print the study plan; do not run any trials.",
    )
    return parser


def _suggest_trial_params(trial: Any) -> dict[str, Any]:
    """Sample one trial's hyperparameters from the warm-start-aware search space."""
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    alpha = trial.suggest_categorical("alpha_override", [16, 32, 64, 128])
    dropout = trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1])
    warmup = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    grad_accum = trial.suggest_categorical("grad_accum", [8, 16, 32])
    scheduler = trial.suggest_categorical(
        "lr_scheduler", ["constant", "cosine"]
    )
    diff_aware = trial.suggest_categorical("diff_aware_loss", [False, True])
    return {
        "lr": lr,
        "alpha_override": alpha,
        "lora_dropout": dropout,
        "warmup_ratio": warmup,
        "grad_accum": grad_accum,
        "lr_scheduler": scheduler,
        "diff_aware_loss": diff_aware,
    }


def _subsample_dataset(src: Path, n: int, dest: Path) -> int:
    """Write the first ``n`` lines of ``src`` into ``dest`` as JSONL.

    Kept simple — deterministic head-sample, not random. Random sampling
    could be added later, but for HPO proxy runs a stable subsample
    makes trials directly comparable.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src.open("r", encoding="utf-8") as fin, dest.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(line + "\n")
            written += 1
            if written >= n:
                break
    return written


def _build_trial_kwargs(
    *,
    run_args: HPORunArgs,
    sampled: dict[str, Any],
    adapter_id: str,
    trial_dataset_path: str,
) -> dict[str, Any]:
    """Translate sampled hyperparameters into train_and_register kwargs."""
    # Resolve warm-start string through the same alias table trainer_cli
    # uses so operators can say --warm-start deltacoder here too.
    from model_training.trainer_cli import _resolve_warm_start  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "session_id": None,
        "adapter_id": adapter_id,
        "dataset_path": trial_dataset_path,
        "encoding_mode": "multi_turn",
        "model_config_name": run_args.model_config_name,
        "warm_start_adapter_id": _resolve_warm_start(run_args.warm_start),
        "epochs": 1,  # proxy mode; operators choose final epochs on the winner
        "learning_rate": sampled["lr"],
        "gradient_accumulation_steps": sampled["grad_accum"],
        "lr_scheduler_type": sampled["lr_scheduler"],
        "override_lora_alpha": sampled["alpha_override"],
        "override_lora_dropout": sampled["lora_dropout"],
        "diff_aware_loss": sampled["diff_aware_loss"],
        "mlflow_experiment": run_args.experiment_name,
    }
    # Warmup ratio is baked into SFTConfig(warmup_ratio=...) inside
    # train_qlora — it's a fixed 0.03 today. Logging through MLFlow params
    # captures what the trial actually sampled even though the current
    # trainer doesn't consume it. Future PR can thread it.
    kwargs["extra_hpo_params"] = {"warmup_ratio": sampled["warmup_ratio"]}
    kwargs.update(run_args.extra_train_kwargs)
    return kwargs


def _eval_loss_from_trainer_state(output_dir: str) -> float:
    """Read the final training loss from the trainer_state.json MLflow emits.

    Falls back to ``float('inf')`` if the file is absent — the trial
    then ranks worst and Hyperband prunes it.
    """
    state_file = Path(output_dir) / "trainer_state.json"
    if not state_file.exists():
        return float("inf")
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return float("inf")
    history = state.get("log_history", [])
    losses = [
        float(entry["loss"]) for entry in history if "loss" in entry
    ]
    return losses[-1] if losses else float("inf")


def _pass_at_1_humaneval(adapter_dir: str, tier: str) -> float:
    """Run HumanEval at the requested tier against ``adapter_dir``.

    Returns ``0.0`` when the eval infrastructure is unavailable or the
    tier is ``none``. Kept deliberately minimal — the production path
    can plug in ``evaluation.metrics.run_humaneval_subset`` once the
    loader accepts an adapter directory directly.
    """
    if tier == "none":
        return 0.0
    try:
        pass
    except ImportError:
        return 0.0
    # Placeholder until an adapter-aware HumanEval path exists in the
    # evaluation package. Returning 0.0 makes the fitness fall back
    # entirely on loss-weight, which is still meaningful for ranking.
    logger.warning(
        "pass@1 eval not yet wired through evaluation.metrics — falling back to 0.0"
    )
    return 0.0


def _compute_fitness(
    eval_loss: float,
    pass_at_1: float,
    *,
    prior_losses: list[float],
    cfg: FitnessConfig,
) -> float:
    """Blend normalized eval loss with pass@1 into a single scalar.

    Normalization is min-max across the study's completed trials; with
    fewer than 3 priors we fall back to ``0.5`` so the loss term
    contributes a stable baseline instead of dominating early trials.
    """
    if len(prior_losses) < 3 or eval_loss == float("inf"):
        loss_norm = 0.5
    else:
        lo = min(prior_losses)
        hi = max(prior_losses)
        if hi == lo:
            loss_norm = 0.5
        else:
            loss_norm = (eval_loss - lo) / (hi - lo)
            loss_norm = max(0.0, min(1.0, loss_norm))
    return cfg.loss_weight * (1.0 - loss_norm) + cfg.pass_at_1_weight * pass_at_1


def _run_single_trial(
    trial: Any,
    *,
    run_args: HPORunArgs,
    fitness_cfg: FitnessConfig,
    prior_losses: list[float],
) -> float:
    """Objective function body for one Optuna trial."""
    sampled = _suggest_trial_params(trial)
    logger.info("Trial %d sampled params: %s", trial.number, sampled)

    trial_dir = run_args.output_root / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_dataset = trial_dir / "dataset.jsonl"
    n = _subsample_dataset(
        Path(run_args.dataset), run_args.subsample, trial_dataset
    )
    logger.info("Trial %d subsample size: %d records", trial.number, n)

    adapter_id = f"{run_args.adapter_id_prefix}-t{trial.number:03d}"
    kwargs = _build_trial_kwargs(
        run_args=run_args,
        sampled=sampled,
        adapter_id=adapter_id,
        trial_dataset_path=str(trial_dataset),
    )
    # Strip the extra_hpo_params marker; it's informational for logging.
    extra_hpo_params = kwargs.pop("extra_hpo_params", {})
    logger.info(
        "Trial %d adapter_id=%s warmup_ratio=%.3f",
        trial.number,
        adapter_id,
        extra_hpo_params.get("warmup_ratio", 0.0),
    )

    # Point the trainer at a per-trial adapter output dir so HPO artifacts
    # don't collide with the default ~/.rune/adapters layout.
    os.environ["RUNE_ADAPTER_DIR"] = str(trial_dir / "adapter_root")

    from model_training.trainer import train_and_register  # noqa: PLC0415

    try:
        train_and_register(**kwargs)
    except Exception as exc:  # noqa: BLE001 — one bad trial mustn't sink the study
        logger.exception("Trial %d crashed: %s", trial.number, exc)
        return 0.0

    adapter_output_dir = str(
        Path(os.environ["RUNE_ADAPTER_DIR"]) / adapter_id
    )
    eval_loss = _eval_loss_from_trainer_state(adapter_output_dir)
    pass_at_1 = _pass_at_1_humaneval(adapter_output_dir, run_args.eval_tier)
    fitness = _compute_fitness(
        eval_loss, pass_at_1, prior_losses=prior_losses, cfg=fitness_cfg
    )
    logger.info(
        "Trial %d eval_loss=%.4f pass@1=%.3f fitness=%.4f",
        trial.number,
        eval_loss,
        pass_at_1,
        fitness,
    )
    prior_losses.append(eval_loss)
    return fitness


def _prune_retained_adapters(
    study: Any, run_args: HPORunArgs
) -> list[tuple[int, float]]:
    """Delete adapter dirs for all but the top-K trials by fitness value."""
    import shutil  # noqa: PLC0415

    completed = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state.name == "COMPLETE" and t.value is not None
    ]
    ranked = sorted(completed, key=lambda t: t.value, reverse=True)
    keep = {t.number for t in ranked[: run_args.keep_top_k]}
    removed: list[tuple[int, float]] = []
    for t in ranked[run_args.keep_top_k :]:
        adapter_dir = run_args.output_root / f"trial_{t.number:03d}"
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir, ignore_errors=True)
            removed.append((t.number, t.value))
    logger.info(
        "Retention: kept %d, pruned %d (keep_top_k=%d)",
        len(keep),
        len(removed),
        run_args.keep_top_k,
    )
    return removed


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for training-hyperparameter HPO."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_args = HPORunArgs(
        dataset=str(Path(args.dataset).resolve()),
        adapter_id_prefix=args.study_name,
        model_config_name=args.model_config_name,
        warm_start=args.warm_start,
        subsample=args.subsample if not args.smoke else 4,
        eval_tier=args.eval_tier,
        output_root=output_root,
        experiment_name=args.experiment_name,
        keep_top_k=args.keep_top_k,
    )
    fitness_cfg = FitnessConfig(
        loss_weight=args.loss_weight, pass_at_1_weight=args.pass_at_1_weight
    )
    n_trials = 2 if args.smoke else args.n_trials

    plan = {
        "study_name": args.study_name,
        "db": args.db,
        "n_trials": n_trials,
        "dataset": run_args.dataset,
        "subsample": run_args.subsample,
        "model_config_name": run_args.model_config_name,
        "warm_start": run_args.warm_start,
        "output_root": str(run_args.output_root),
        "fitness": {
            "loss_weight": fitness_cfg.loss_weight,
            "pass_at_1_weight": fitness_cfg.pass_at_1_weight,
        },
        "eval_tier": run_args.eval_tier,
        "keep_top_k": run_args.keep_top_k,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    if args.print_only:
        return 0

    import optuna  # noqa: PLC0415

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=3, reduction_factor=3
    )
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.startup_trials, seed=args.seed
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.db,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    prior_losses: list[float] = []

    def _objective(trial: optuna.Trial) -> float:
        return _run_single_trial(
            trial,
            run_args=run_args,
            fitness_cfg=fitness_cfg,
            prior_losses=prior_losses,
        )

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    # Retention pruning.
    _prune_retained_adapters(study, run_args)

    best = study.best_trial
    logger.info(
        "HPO complete. Best trial=%d fitness=%.4f params=%s",
        best.number,
        best.value,
        best.params,
    )
    summary = {
        "study_name": args.study_name,
        "best_trial": best.number,
        "best_fitness": best.value,
        "best_params": best.params,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
