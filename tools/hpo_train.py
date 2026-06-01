"""Optuna HPO over D2L training hyperparameters (issue #49).

Objective = held-out val_diff_agreement on the near-dup-filtered CLEAN val split
(the honest generalization signal), penalized if val_preservation collapses below
0.7 (a broad-perturbation adapter that wrecks the agreement region is not a win).
Each trial runs a SHORT training (few-step) at sampled hyperparams, then reads the
last val metrics that run_hypernet_distillation logged to MLflow.

Resumable: Optuna study persisted to a sqlite RDB, so an overnight run survives
restarts and `--report` prints the best config. Run under tools/run_guarded.sh.

NOT a final-quality run — it ranks configs cheaply; extend the best trial's config
to a full multi-epoch final train afterwards.
"""

from __future__ import annotations

import argparse
import gc
import sys

S3_CKPT = (
    "s3://elixirtrials-949678234935-eu-west-2-artifacts/"
    "checkpoints/hypernet_hpo/checkpoint.pt"
)


def _last_val(experiment_name: str) -> tuple[float, float]:
    """Read last (val_diff_agreement, val_preservation) for a trial's MLflow run."""
    import mlflow  # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    mlflow.set_tracking_uri("http://localhost:5000")
    c = MlflowClient()
    e = c.get_experiment_by_name(experiment_name)
    if e is None:
        return 0.0, 1.0
    runs = c.search_runs(
        [e.experiment_id], order_by=["attribute.start_time DESC"], max_results=1
    )
    if not runs:
        return 0.0, 1.0

    def last(metric: str, default: float) -> float:
        h = c.get_metric_history(runs[0].info.run_id, metric)
        return h[-1].value if h else default

    return last("val_diff_agreement", 0.0), last("val_preservation", 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train", default="/tmp/rune-corpus/external_codereview.train.jsonl"
    )
    ap.add_argument(
        "--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl"
    )
    ap.add_argument("--storage", default="sqlite:////tmp/rune-hpo.db")
    ap.add_argument("--study", default="issue49-d2l-hpo")
    ap.add_argument("--n-trials", type=int, default=24)
    ap.add_argument("--trial-steps", type=int, default=80)
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--report", action="store_true", help="print best config and exit")
    a = ap.parse_args()

    import optuna  # noqa: PLC0415

    if a.report:
        study = optuna.load_study(study_name=a.study, storage=a.storage)
        print(f"trials={len(study.trials)} best_value={study.best_value:.4f}")
        print(f"best_params={study.best_params}")
        print(
            f"best_val_preservation={study.best_trial.user_attrs.get('val_preservation')}"
        )
        return 0

    from rune.training.hypernet_distill import (  # noqa: PLC0415
        DistillConfig,
        run_hypernet_distillation,
    )

    def objective(trial: object) -> float:
        hp = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
            "scaler_b_init": trial.suggest_float("scaler_b_init", 0.05, 0.3),
            "train_scaling": trial.suggest_float("train_scaling", 0.25, 1.5),
            "topk": trial.suggest_int("topk", 25, 100),
            "grad_accum_steps": trial.suggest_categorical(
                "grad_accum_steps", [4, 8, 16]
            ),
        }
        exp = f"{a.study}-t{trial.number}"
        cfg = DistillConfig(
            corpus_path=a.train,
            val_corpus_path=a.val,
            checkpoint_path=S3_CKPT,
            checkpoint_dir=f"/tmp/rune-hpo/t{trial.number}",
            max_steps=a.trial_steps,
            val_steps=max(a.trial_steps // 2, 20),
            max_seq_length=a.max_seq_length,
            num_epochs=1,
            experiment_name=exp,
            save_steps=0,
            log_steps=10,
            **hp,
        )
        run_hypernet_distillation(cfg)
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:
            pass
        val_da, val_pres = _last_val(exp)
        trial.set_user_attr("val_preservation", val_pres)
        # maximize held-out diff_agreement; penalize collapsed preservation
        return val_da if val_pres >= 0.7 else val_da - (0.7 - val_pres)

    study = optuna.create_study(
        study_name=a.study,
        storage=a.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(objective, n_trials=a.n_trials)
    print(f"BEST value={study.best_value:.4f} params={study.best_params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
