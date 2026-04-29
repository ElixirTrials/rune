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

        fitness = hunk_loss_weight          * (1 - normalize(hunk_loss))
                + hunk_accuracy_weight      * hunk_accuracy
                + adapter_improvement_weight * max(0, adapter_improvement)

    ``hunk_loss`` and ``hunk_accuracy`` are diff-restricted metrics: NLL and
    top-1 accuracy computed only on assistant tokens that fall inside a
    ``+`` / replace hunk (per :func:`model_training.diff_loss._compute_hunk_ranges`).
    This directly rewards trials whose adapters encode the revision delta —
    aligned with the episodic-memory thesis — instead of overrating trials
    that minimize total loss but do not internalize the edit.

    ``adapter_improvement`` is the relative reduction in hunk loss from the
    adapter relative to the frozen base model.  When the CLI flag
    ``--no-adapter-improvement-eval`` disables that second forward pass, the
    weight collapses to ``0.0`` and the remaining two weights rebalance to
    ``(0.6, 0.4)``.
    """

    hunk_loss_weight: float = 0.5
    hunk_accuracy_weight: float = 0.3
    adapter_improvement_weight: float = 0.2


@dataclass(frozen=True)
class ScreeningFitnessConfig:
    """Stage-1 screening-fitness weights and pruning thresholds.

    See ``instructions/hpo_improvements.md`` §1–2.5 for derivation.

    The screening blend is::

        screening_fitness = loss_weight * (1 - loss_norm)
                          + accuracy_weight * accuracy_score

    where both inputs are EMA-smoothed across ``smoothing_window`` evals.
    Entropy is a hard floor (prune below ``entropy_floor`` nats), not a
    weighted term. ``min_steps_before_pruning`` defers all pruning calls
    until enough steps have accumulated for the EMA to stabilise.
    """

    loss_weight: float = 0.6
    accuracy_weight: float = 0.4
    entropy_floor: float = 0.3  # nats; calibrated per-run when possible
    minimum_screening_fitness: float = 0.3
    smoothing_window: int = 25
    min_steps_before_pruning: int = 150
    delta_normalize_accuracy: bool = True


def _rebalanced_fitness_config(cfg: FitnessConfig) -> FitnessConfig:
    """Rebalance the weights when the adapter-improvement eval is disabled.

    The hunk_loss and hunk_accuracy weights are renormalized so they sum to
    ``1.0`` (defaulting to ``(0.6, 0.4)`` when the input still uses the
    class defaults), and ``adapter_improvement_weight`` is forced to ``0.0``.
    """
    total = cfg.hunk_loss_weight + cfg.hunk_accuracy_weight
    if total <= 0.0:
        return FitnessConfig(
            hunk_loss_weight=0.6,
            hunk_accuracy_weight=0.4,
            adapter_improvement_weight=0.0,
        )
    return FitnessConfig(
        hunk_loss_weight=cfg.hunk_loss_weight / total,
        hunk_accuracy_weight=cfg.hunk_accuracy_weight / total,
        adapter_improvement_weight=0.0,
    )


@dataclass
class HPORunArgs:
    """Non-search-space CLI arguments threaded into the Optuna objective."""

    dataset: str
    adapter_id_prefix: str
    model_config_name: str
    warm_start: str | None
    subsample: int
    output_root: Path
    experiment_name: str
    keep_top_k: int
    heldout_fraction: float = 0.1
    heldout_strategy: str = "step_index"
    compute_adapter_delta: bool = True
    seed: int = 42
    extra_train_kwargs: dict[str, Any] = field(default_factory=dict)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_training_hpo",
        description="Training-hyperparameter HPO for DeltaCoder fine-tune.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="JSONL of mined pairs.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--study-name", dest="study_name", default="rune-training-v1")
    parser.add_argument(
        "--db", default="sqlite:///./optuna_training.db", help="Optuna storage URI"
    )
    parser.add_argument("--model", dest="model_config_name", default="qwen3.5-9b")
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
        "--hunk-loss-weight",
        dest="hunk_loss_weight",
        type=float,
        default=0.5,
        help="Fitness weight for (1 - normalized hunk_loss).",
    )
    parser.add_argument(
        "--hunk-accuracy-weight",
        dest="hunk_accuracy_weight",
        type=float,
        default=0.3,
        help="Fitness weight for hunk-restricted top-1 accuracy.",
    )
    parser.add_argument(
        "--adapter-improvement-weight",
        dest="adapter_improvement_weight",
        type=float,
        default=0.2,
        help="Fitness weight for adapter-vs-base hunk-loss delta.",
    )
    parser.add_argument(
        "--adapter-improvement-eval",
        dest="adapter_improvement_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the second forward pass with the adapter disabled to "
            "compute the adapter-vs-base hunk-loss delta. When off, weights "
            "rebalance to (0.6, 0.4, 0.0)."
        ),
    )
    parser.add_argument(
        "--heldout-fraction",
        dest="heldout_fraction",
        type=float,
        default=0.1,
        help="Fraction of the trial subsample reserved for held-out eval.",
    )
    parser.add_argument(
        "--heldout-strategy",
        dest="heldout_strategy",
        choices=["step_index", "random"],
        default="step_index",
        help=(
            "step_index: hold out the largest-step_index pair per sampled task. "
            "random: hold out all pairs from a random sample of tasks."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="TPE sampler seed.")
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
    parser.add_argument(
        "--stage",
        choices=["screen", "refine", "auto", "single"],
        default="single",
        help=(
            "HPO stage. 'single' (default) runs the legacy single-stage flow "
            "with zero behaviour change. 'screen' runs cheap training-time "
            "screening. 'refine' loads a Stage-1 study and re-evaluates the "
            "top-K under the hunk-restricted fitness. 'auto' chains both."
        ),
    )
    parser.add_argument(
        "--screen-loss-weight", dest="screen_loss_weight",
        type=float, default=0.6,
        help="Stage-1 weight on (1 - loss_norm).",
    )
    parser.add_argument(
        "--screen-accuracy-weight", dest="screen_accuracy_weight",
        type=float, default=0.4,
        help="Stage-1 weight on accuracy_score.",
    )
    parser.add_argument(
        "--entropy-floor", dest="entropy_floor",
        type=float, default=None,
        help="Override the calibrated entropy floor (nats). None = use config default or calibrator.",
    )
    parser.add_argument(
        "--screen-top-k", dest="screen_top_k",
        type=int, default=5,
        help="Stage-1 survivors passed to Stage-2.",
    )
    parser.add_argument(
        "--stage1-study-name", dest="stage1_study_name",
        type=str, default=None,
        help="Stage-2 / 'refine' only: study-name to seed Stage-2 from.",
    )
    parser.add_argument(
        "--screen-subsample", dest="screen_subsample",
        type=int, default=500,
        help="Per-trial subsample size for Stage-1 screening.",
    )
    parser.add_argument(
        "--screen-epochs", dest="screen_epochs",
        type=int, default=2,
        help="Stage-1 epochs per trial. Raised from the implicit 1 to avoid step-starvation.",
    )
    parser.add_argument(
        "--screen-smoothing-window", dest="screen_smoothing_window",
        type=int, default=25,
        help="EMA window (in eval-step units) for screening metrics.",
    )
    parser.add_argument(
        "--screen-min-steps", dest="screen_min_steps",
        type=int, default=None,
        help="Override the calibrated min-steps-before-pruning. None = use config default or calibrator.",
    )
    parser.add_argument(
        "--min-screening-fitness", dest="min_screening_fitness",
        type=float, default=0.3,
        help="Kill-switch: refuse to seed Stage-2 if best Stage-1 fitness is below this.",
    )
    parser.add_argument(
        "--calibrate-from-mlflow", dest="calibrate_from_mlflow",
        action="store_true",
        help="Before Stage-1, derive entropy_floor and min_steps from recent MLflow runs.",
    )
    parser.add_argument(
        "--force-uncalibrated", dest="force_uncalibrated",
        action="store_true",
        help="Allow Stage-1 to launch even when calibrator validation reports rho<0.6.",
    )
    return parser


def _suggest_trial_params(trial: Any) -> dict[str, Any]:
    """Sample one trial's hyperparameters from the warm-start-aware search space."""
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    alpha = trial.suggest_categorical("alpha_override", [16, 32, 64, 128])
    dropout = trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1])
    warmup = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    grad_accum = trial.suggest_categorical("grad_accum", [8, 16, 32])
    scheduler = trial.suggest_categorical("lr_scheduler", ["constant", "cosine"])
    diff_aware = trial.suggest_categorical("diff_aware_loss", [False, True])
    neftune = trial.suggest_categorical("neftune_noise_alpha", [None, 5.0, 10.0])
    return {
        "lr": lr,
        "alpha_override": alpha,
        "lora_dropout": dropout,
        "warmup_ratio": warmup,
        "grad_accum": grad_accum,
        "lr_scheduler": scheduler,
        "diff_aware_loss": diff_aware,
        "neftune_noise_alpha": neftune,
    }


def _subsample_dataset(src: Path, n: int, dest: Path) -> int:
    """Write up to ``n`` records from ``src`` into ``dest`` as JSONL.

    Deterministic and task-aware: does a first-pass round-robin over
    ``task_id`` (preferring ``metadata.source_task_id``) so that small
    subsamples span multiple tasks — required by the no-leakage held-out
    split in ``_stratify_heldout_split``. Record order within each task
    is preserved. Input order determines task visitation order, so the
    sample is still stable across trials.
    """
    import json as _json  # noqa: PLC0415
    from collections import OrderedDict  # noqa: PLC0415

    dest.parent.mkdir(parents=True, exist_ok=True)
    buckets: OrderedDict[str, list[str]] = OrderedDict()
    with src.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            meta = rec.get("metadata") or {}
            tid = str(meta.get("source_task_id") or rec.get("task_id") or "")
            buckets.setdefault(tid, []).append(line)

    written = 0
    with dest.open("w", encoding="utf-8") as fout:
        while written < n and any(buckets.values()):
            for tid in list(buckets.keys()):
                if not buckets[tid]:
                    continue
                fout.write(buckets[tid].pop(0) + "\n")
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
        "warmup_ratio": sampled["warmup_ratio"],
        "neftune_noise_alpha": sampled["neftune_noise_alpha"],
        "mlflow_experiment": run_args.experiment_name,
    }
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
    losses = [float(entry["loss"]) for entry in history if "loss" in entry]
    return losses[-1] if losses else float("inf")


def _load_pairs_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a pairs JSONL into a list of dicts (stdlib only, CPU-safe)."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _stratify_heldout_split(
    pairs: list[dict[str, Any]],
    *,
    fraction: float,
    strategy: str,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split pairs into (train, heldout) with no task_id leakage.

    ``step_index``: pick ``max(1, ⌈fraction * N_tasks⌉)`` tasks; for each,
    move the pair with the largest ``metadata.step_index`` into heldout and
    leave the earlier steps in train.  Training never sees a held-out task's
    terminal revision, so there is no pair-level leakage.

    ``random``: pick ``max(1, ⌈fraction * N_tasks⌉)`` tasks and move *all*
    of their pairs into heldout.  Task-level partitioning; the train and
    eval splits have disjoint ``source_task_id`` sets.
    """
    import math  # noqa: PLC0415
    import random as _rand  # noqa: PLC0415
    from collections import defaultdict  # noqa: PLC0415

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        meta = p.get("metadata") or {}
        tid = meta.get("source_task_id") or p.get("task_id", "")
        groups[str(tid)].append(p)

    task_ids = sorted(groups.keys())
    if fraction <= 0:
        return list(pairs), []

    rng = _rand.Random(seed)
    rng.shuffle(task_ids)
    n_tasks = len(task_ids)
    if n_tasks <= 1:
        raise ValueError(
            f"Heldout split would leave train set empty: got N_tasks={n_tasks}; "
            "at least 2 tasks are required when fraction > 0."
        )
    raw_heldout = math.ceil(n_tasks * fraction)
    if raw_heldout >= n_tasks:
        raise ValueError(
            f"Heldout split would leave train set empty: fraction={fraction} with "
            f"N_tasks={n_tasks} would hold out all tasks. Reduce fraction so at "
            "least one task remains for training."
        )
    n_heldout = min(
        max(1, raw_heldout),
        n_tasks - 1,
    )
    heldout_task_ids = set(task_ids[:n_heldout])

    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []

    for tid, group in groups.items():
        if tid not in heldout_task_ids:
            train.extend(group)
            continue
        if strategy == "step_index":
            ordered = sorted(
                group,
                key=lambda p: (p.get("metadata") or {}).get("step_index", 0),
            )
            # Terminal revision → heldout; earlier steps stay in train.
            heldout.append(ordered[-1])
            train.extend(ordered[:-1])
        else:  # random
            heldout.extend(group)
    return train, heldout


def _tokenize_for_eval(tokenizer: Any, text: str) -> dict[str, Any]:
    """Tokenize a single eval pair with truncation.

    The eval forward pass uses eager attention (no flash-attn for Qwen3.5),
    whose softmax matrix is O(L²) at fp32 — at L=4096 a single sample
    consumes ~64 MB of attention plus activations. Mined GitHub pairs can
    easily exceed 8 k tokens; uncapped tokenization is the proximate kill
    shot in RCA-2 Cause 2 (848 MiB allocation observed at OOM with 545 MiB
    free). Override via ``RUNE_EVAL_MAX_LENGTH`` env var if your training
    distribution skews longer and you have VRAM headroom — the cap is
    policy, not arithmetic.
    """
    max_length = int(os.environ.get("RUNE_EVAL_MAX_LENGTH", "2048"))
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )


def _flush_gpu_between_phases() -> None:
    """Force a deterministic GPU flush between training and eval.

    SFTTrainer holds cyclic refs to its model and optimizer; del-then-GC is
    not synchronous. The ``paged_adamw_8bit`` optimizer keeps small
    CUDA-resident bookkeeping tensors alive until the trainer object is
    finalised. Without this explicit flush the cached base re-enters
    PeftModel.from_pretrained on top of training residuals (RCA-2 Cause 3).
    """
    import gc  # noqa: PLC0415

    gc.collect()
    gc.collect()  # second pass clears generations promoted by the first
    try:
        import torch  # noqa: PLC0415

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 — torch may be missing on CPU CI
        pass


class _EMA:
    """Exponential moving average that ignores non-finite updates.

    The smoothing factor is ``alpha = 2 / (window + 1)`` (the standard
    pandas / financial EMA), so ``window`` is the half-life-ish span
    rather than a literal sample count. Used by
    :class:`OptunaScreeningCallback` to smooth per-eval metrics before
    Hyperband pruning decisions — the in-flight 3-epoch run showed
    per-step values are too noisy for direct pruning.
    """

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError(f"_EMA window must be positive, got {window}")
        self._alpha = 2.0 / (window + 1.0)
        self._value: float | None = None

    def update(self, x: float) -> None:
        """Append one sample. Non-finite inputs are silently ignored."""
        import math  # noqa: PLC0415
        if not math.isfinite(x):
            return
        if self._value is None:
            self._value = x
        else:
            self._value = self._alpha * x + (1.0 - self._alpha) * self._value

    @property
    def value(self) -> float | None:
        """Current smoothed value, or ``None`` before the first update."""
        return self._value


class OptunaScreeningCallback:
    """Stage-1 ``TrainerCallback`` driving Hyperband and the entropy guard.

    Hooks ``on_evaluate`` (HuggingFace Trainer evaluation event). At each
    eval step:

    1. Push ``eval_loss``, ``eval/token_accuracy``, ``eval/entropy`` into
       per-trial EMA buffers (``smoothing_window`` from cfg).
    2. If ``state.global_step < cfg.min_steps_before_pruning``, return
       silently — the buffers are still warming up and noisy decisions
       made now would falsely kill good trials.
    3. If smoothed entropy is below ``cfg.entropy_floor``, raise
       ``optuna.TrialPruned`` with a diagnostic message (entropy
       collapse: GEM, CurioSFT).
    4. Otherwise call ``trial.report(smoothed_accuracy, step)`` and
       ``trial.should_prune()`` to drive Hyperband. Loss is reported
       post-trial after global min-max normalisation across completed
       trials, not here — Hyperband only needs *a* monotone signal,
       and accuracy is the bounded-range one.

    NOTE: We don't subclass ``transformers.TrainerCallback`` directly to
    keep this module CPU-importable in CI (INFRA-05). The Trainer will
    duck-type our class — ``on_evaluate`` is the only hook it calls.
    """

    def __init__(self, trial: Any, cfg: ScreeningFitnessConfig) -> None:
        self.trial = trial
        self.cfg = cfg
        self._loss_ema = _EMA(cfg.smoothing_window)
        self._acc_ema = _EMA(cfg.smoothing_window)
        self._ent_ema = _EMA(cfg.smoothing_window)

    @property
    def smoothed_loss(self) -> float | None:
        return self._loss_ema.value

    @property
    def smoothed_accuracy(self) -> float | None:
        return self._acc_ema.value

    @property
    def smoothed_entropy(self) -> float | None:
        return self._ent_ema.value

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        import optuna  # noqa: PLC0415

        if not metrics:
            return

        self._loss_ema.update(float(metrics.get("eval_loss", float("inf"))))
        self._acc_ema.update(float(metrics.get("eval/token_accuracy", 0.0)))
        if "eval/entropy" in metrics:
            self._ent_ema.update(float(metrics["eval/entropy"]))

        # Step floor: don't act until buffers have warmed up.
        if state.global_step < self.cfg.min_steps_before_pruning:
            return

        ent = self._ent_ema.value
        if ent is not None and ent < self.cfg.entropy_floor:
            raise optuna.TrialPruned(
                f"Entropy {ent:.3f} < floor {self.cfg.entropy_floor:.3f} "
                f"(step={state.global_step})"
            )

        acc = self._acc_ema.value
        if acc is None:
            return
        self.trial.report(acc, step=state.global_step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Hyperband prune at step={state.global_step} acc_ema={acc:.3f}"
            )


def _evaluate_adapter_on_heldout(
    adapter_path: str,
    pairs: list[dict[str, Any]],
    *,
    base_model_id: str,
    compute_adapter_delta: bool = True,
) -> dict[str, float]:
    """Teacher-forced hunk-restricted LM eval on a heldout pair split.

    For each pair, we compute + / replace hunk character ranges via
    :func:`model_training.diff_loss._compute_hunk_ranges`, tokenize the
    full ``teacher_text`` with ``return_offsets_mapping=True``, and shift
    the hunk ranges into the teacher-text coordinate system so the
    character→token mapping stays correct.  Metrics aggregate only over
    assistant tokens whose offset intersects a hunk range:

    - ``hunk_loss``: mean NLL over hunk tokens (lower is better).
    - ``hunk_accuracy``: mean top-1 accuracy over hunk tokens.
    - ``hunk_entropy``: mean predictive entropy over hunk tokens
      (diagnostic; not in the fitness blend).
    - ``adapter_improvement``: ``1 - (adapter_hunk_loss / base_hunk_loss)``
      when ``compute_adapter_delta=True``, else ``0.0``.  Positive means
      the adapter reduces hunk loss relative to the frozen base model.

    Returns all zeros when ``pairs`` is empty — lets upstream code call us
    blindly without guarding.
    """
    if not pairs:
        return {
            "hunk_loss": 0.0,
            "hunk_accuracy": 0.0,
            "adapter_improvement": 0.0,
            "hunk_entropy": 0.0,
        }

    # Deferred GPU imports keep the module CPU-importable (INFRA-05).
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from model_training.d2l_data import (  # noqa: PLC0415
        _extract_post_revision,
        _extract_pre_revision,
    )
    from model_training.diff_loss import _compute_hunk_ranges  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # Reuse the trainer's cached NF4 base when RUNE_PERSIST_BASE_MODEL=1 is
    # set by the HPO runner — avoids a second from_pretrained that doubles
    # VRAM during eval. Use the trainer's _get_or_load_base so the cache is
    # shared across train+eval within a study.
    from model_training.trainer import _get_or_load_base  # noqa: PLC0415

    base_model, tokenizer = _get_or_load_base(
        base_model_id,
        bnb_config=bnb_config,
        attn_impl=None,
        auto_model_cls=AutoModelForCausalLM,
        auto_tokenizer_cls=AutoTokenizer,
    )
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
    adapter_model.eval()

    def _forward_hunk_metrics(
        model: Any, disable: bool
    ) -> tuple[float, float, float, int]:
        total_loss = 0.0
        total_acc = 0.0
        total_ent = 0.0
        total_tok = 0

        cm = model.disable_adapter() if disable else _NullContext()
        with cm, torch.no_grad():
            for pair in pairs:
                act = pair.get("activation_text", "")
                teach = pair.get("teacher_text", "")
                pre = _extract_pre_revision(act)
                post = _extract_post_revision(act, teach)
                if not post:
                    continue
                hunks = _compute_hunk_ranges(pre, post)
                if not hunks:
                    continue
                post_start = teach.rfind(post)
                if post_start == -1:
                    continue
                shifted = [(s + post_start, e + post_start) for s, e in hunks]

                enc = _tokenize_for_eval(tokenizer, teach)
                # When tokenization truncated, the post-string offset
                # boundary may end mid-hunk. Scan offsets backwards to find
                # the last real (non-zero) offset — fast tokenizers append
                # special tokens (EOS / pad) with offset (0, 0), so taking
                # the literal last offset would silently drop near-end
                # hunks on non-truncated sequences.
                offsets_list = enc["offset_mapping"][0].tolist()
                byte_cap = len(teach)
                for off in reversed(offsets_list):
                    if int(off[1]) > 0:
                        byte_cap = int(off[1])
                        break
                shifted = [(s, min(e, byte_cap)) for s, e in shifted if s < byte_cap]
                if not shifted:
                    continue
                input_ids = enc["input_ids"].to(model.device)
                attention_mask = enc["attention_mask"].to(model.device)
                offsets = offsets_list

                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits[0]
                shift_logits = logits[:-1]
                shift_ids = input_ids[0][1:]
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                probs = log_probs.exp()

                for i in range(shift_logits.size(0)):
                    tok_offset = offsets[i + 1]
                    ts, te = tok_offset
                    if ts == 0 and te == 0:
                        continue
                    in_hunk = any(ts < he and te > hs for hs, he in shifted)
                    if not in_hunk:
                        continue
                    tgt = int(shift_ids[i].item())
                    nll = -float(log_probs[i, tgt].item())
                    pred = int(shift_logits[i].argmax().item())
                    ent = float(-(probs[i] * log_probs[i]).sum().item())
                    total_loss += nll
                    total_acc += 1.0 if pred == tgt else 0.0
                    total_ent += ent
                    total_tok += 1
        if total_tok == 0:
            return 0.0, 0.0, 0.0, 0
        return (
            total_loss / total_tok,
            total_acc / total_tok,
            total_ent / total_tok,
            total_tok,
        )

    import gc as _gc  # noqa: PLC0415

    try:
        # Adapter-active pass.
        hunk_loss, hunk_acc, hunk_ent, n_tok = _forward_hunk_metrics(
            adapter_model, disable=False
        )

        adapter_improvement = 0.0
        if compute_adapter_delta and n_tok > 0:
            base_loss, _, _, _ = _forward_hunk_metrics(adapter_model, disable=True)
            if base_loss > 0.0 and math.isfinite(base_loss):
                adapter_improvement = 1.0 - (hunk_loss / base_loss)
    finally:
        # Detach the trial's adapter from the (possibly cached) base BEFORE
        # propagating any forward-pass exception so the next trial sees a
        # clean cached base. unload() returns the restored base — capture it
        # and strip lingering peft_config (RCA-3, RCA-5 H1).
        try:
            restored = adapter_model.unload()
            inner = (
                restored
                if restored is not None
                else getattr(adapter_model, "model", None)
            )
            if inner is not None and hasattr(inner, "peft_config"):
                try:
                    delattr(inner, "peft_config")
                except AttributeError:
                    pass
        except Exception:  # noqa: BLE001 — never break HPO on cleanup
            logger.exception("Heldout eval: PeftModel.unload() failed")
        del adapter_model
        _gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — torch may be unavailable on CPU CI
            pass

    return {
        "hunk_loss": float(hunk_loss),
        "hunk_accuracy": float(hunk_acc),
        "adapter_improvement": float(adapter_improvement),
        "hunk_entropy": float(hunk_ent),
    }


class _NullContext:  # pragma: no cover - trivial
    """Minimal stdlib-free ``contextlib.nullcontext`` clone for the forward pass."""

    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _compute_fitness(
    hunk_loss: float,
    hunk_accuracy: float,
    adapter_improvement: float,
    *,
    prior_losses: list[float],
    cfg: FitnessConfig,
) -> float:
    """Blend hunk_loss / hunk_accuracy / adapter_improvement into one scalar.

    Normalization is min-max across the study's completed trials' hunk
    losses; with fewer than 3 priors we fall back to ``0.5`` so the loss
    term contributes a stable baseline instead of dominating early trials.
    ``adapter_improvement`` is floored at ``0.0`` — a regressing adapter
    earns zero credit, not a negative penalty, so retained-top-K ranking
    stays sane when the base model happens to win on a single pair.
    """
    if len(prior_losses) < 3 or hunk_loss == float("inf"):
        loss_norm = 0.5
    else:
        lo = min(prior_losses)
        hi = max(prior_losses)
        if hi == lo:
            loss_norm = 0.5
        else:
            loss_norm = (hunk_loss - lo) / (hi - lo)
            loss_norm = max(0.0, min(1.0, loss_norm))
    delta = max(0.0, adapter_improvement)
    return (
        cfg.hunk_loss_weight * (1.0 - loss_norm)
        + cfg.hunk_accuracy_weight * hunk_accuracy
        + cfg.adapter_improvement_weight * delta
    )


def _compute_screening_fitness(
    eval_loss: float,
    accuracy_score: float,
    *,
    prior_losses: list[float],
    cfg: ScreeningFitnessConfig,
) -> float:
    """Stage-1 fitness scalar from smoothed eval loss and accuracy.

    Mirrors ``_compute_fitness``'s min-max normalisation pattern but with
    only two terms (loss + accuracy). With fewer than 3 prior losses, or
    when the current loss is non-finite, falls back to ``loss_norm = 0.5``
    so early trials get a stable midpoint instead of dominating the blend.

    ``accuracy_score`` is expected pre-clamped to ``[0, 1]`` by the caller
    (raw or delta-normalised); we do not re-clamp here.
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
    return cfg.loss_weight * (1.0 - loss_norm) + cfg.accuracy_weight * accuracy_score


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
    n = _subsample_dataset(Path(run_args.dataset), run_args.subsample, trial_dataset)
    logger.info("Trial %d subsample size: %d records", trial.number, n)

    # Split the trial subsample into train / heldout with no task leakage.
    full_pairs = _load_pairs_jsonl(str(trial_dataset))
    train_pairs, heldout_pairs = _stratify_heldout_split(
        full_pairs,
        fraction=run_args.heldout_fraction,
        strategy=run_args.heldout_strategy,
        seed=run_args.seed + trial.number,
    )
    # Overwrite the trial dataset with the train split so the trainer
    # never sees the heldout pairs.
    with trial_dataset.open("w", encoding="utf-8") as fh:
        for rec in train_pairs:
            fh.write(json.dumps(rec) + "\n")
    logger.info(
        "Trial %d heldout split: train=%d heldout=%d strategy=%s",
        trial.number,
        len(train_pairs),
        len(heldout_pairs),
        run_args.heldout_strategy,
    )

    adapter_id = f"{run_args.adapter_id_prefix}-t{trial.number:03d}"
    kwargs = _build_trial_kwargs(
        run_args=run_args,
        sampled=sampled,
        adapter_id=adapter_id,
        trial_dataset_path=str(trial_dataset),
    )
    logger.info(
        "Trial %d adapter_id=%s warmup_ratio=%.3f",
        trial.number,
        adapter_id,
        sampled["warmup_ratio"],
    )

    # Point the trainer at a per-trial adapter output dir so HPO artifacts
    # don't collide with the default ~/.rune/adapters layout.
    os.environ["RUNE_ADAPTER_DIR"] = str(trial_dir / "adapter_root")

    import mlflow  # noqa: PLC0415
    from model_training.trainer import train_and_register  # noqa: PLC0415

    # Open an HPO-owned MLflow run BEFORE training so the hpo.* tags are
    # attached even if training crashes (e.g. CUDA OOM). The trainer's
    # TRL MLflowCallback attaches to the active run, so its params and
    # training metrics land inside this same run. Using the context
    # manager ensures the run is terminated on exception.
    #
    # Pin the tracking URI BEFORE set_experiment so MLflow doesn't fall
    # back to its default filesystem backend (which emits a v2.x deprecation
    # warning and would orphan our run when the trainer later resets the URI
    # to sqlite). Precedence mirrors training_common.setup_mlflow.
    mlflow.set_tracking_uri(
        kwargs.get("mlflow_tracking_uri")
        or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment(kwargs.get("mlflow_experiment") or run_args.experiment_name)
    mlflow.start_run(
        run_name=f"{run_args.adapter_id_prefix}-t{trial.number:03d}",
    )
    try:
        mlflow.set_tags(
            {
                "hpo.study_name": run_args.adapter_id_prefix,
                "hpo.trial_number": str(trial.number),
                "hpo.dataset": run_args.dataset,
                "hpo.warm_start": run_args.warm_start,
                "hpo.diff_aware_loss": str(sampled["diff_aware_loss"]),
                "hpo.heldout_strategy": run_args.heldout_strategy,
                "hpo.heldout_fraction": str(run_args.heldout_fraction),
                "hpo.subsample_size": str(n),
                "hpo.train_pairs": str(len(train_pairs)),
                "hpo.heldout_pairs": str(len(heldout_pairs)),
                "hpo.adapter_id": adapter_id,
            }
        )

        train_and_register(**kwargs)

        # Force a deterministic GPU flush before eval — the trainer's
        # paged_adamw_8bit + cyclic SFTTrainer↔model refs are not freed
        # synchronously by del+GC, so eval can re-wrap the cached base on
        # top of training residuals (RCA-2 Cause 3).
        _flush_gpu_between_phases()

        adapter_output_dir = str(Path(os.environ["RUNE_ADAPTER_DIR"]) / adapter_id)
        # Resolve base model ID the same way train_and_register does.
        base_model_id = kwargs.get("base_model_id") or os.environ.get(
            "RUNE_BASE_MODEL", "Qwen/Qwen3.5-9B"
        )
        eval_metrics = _evaluate_adapter_on_heldout(
            adapter_output_dir,
            heldout_pairs,
            base_model_id=base_model_id,
            compute_adapter_delta=run_args.compute_adapter_delta,
        )
        mlflow.log_metrics(
            {f"eval/{k}": v for k, v in eval_metrics.items()},
            step=trial.number,
        )
    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")

    fitness = _compute_fitness(
        eval_metrics["hunk_loss"],
        eval_metrics["hunk_accuracy"],
        eval_metrics["adapter_improvement"],
        prior_losses=prior_losses,
        cfg=fitness_cfg,
    )
    logger.info(
        "Trial %d hunk_loss=%.4f hunk_acc=%.3f"
        " adapter_imp=%.3f entropy=%.3f fitness=%.4f",
        trial.number,
        eval_metrics["hunk_loss"],
        eval_metrics["hunk_accuracy"],
        eval_metrics["adapter_improvement"],
        eval_metrics["hunk_entropy"],
        fitness,
    )
    prior_losses.append(eval_metrics["hunk_loss"])
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
    """CLI entrypoint for training-hyperparameter HPO.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when
            ``None``.

    Returns:
        Exit code: 0 on success.

    Raises:
        SystemExit: Raised by argparse on invalid arguments or ``--help``.
    """
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
        output_root=output_root,
        experiment_name=args.experiment_name,
        keep_top_k=args.keep_top_k,
        heldout_fraction=args.heldout_fraction,
        heldout_strategy=args.heldout_strategy,
        compute_adapter_delta=args.adapter_improvement_eval,
        seed=args.seed,
    )
    fitness_cfg = FitnessConfig(
        hunk_loss_weight=args.hunk_loss_weight,
        hunk_accuracy_weight=args.hunk_accuracy_weight,
        adapter_improvement_weight=args.adapter_improvement_weight,
    )
    if not args.adapter_improvement_eval:
        fitness_cfg = _rebalanced_fitness_config(fitness_cfg)
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
        "fitness_formula": (
            "w_L * (1 - norm(hunk_loss)) + w_A * hunk_accuracy "
            "+ w_D * max(0, adapter_improvement)"
        ),
        "fitness": {
            "hunk_loss_weight": fitness_cfg.hunk_loss_weight,
            "hunk_accuracy_weight": fitness_cfg.hunk_accuracy_weight,
            "adapter_improvement_weight": fitness_cfg.adapter_improvement_weight,
        },
        "heldout": {
            "fraction": run_args.heldout_fraction,
            "strategy": run_args.heldout_strategy,
            "adapter_improvement_eval": run_args.compute_adapter_delta,
        },
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

    # Tell Optuna that a per-trial exception is a *failed trial*, not a
    # study-halting bug and not a zero-fitness completion. Failed trials
    # are excluded from ``study.best_trial`` — preventing a crashed run
    # from being reported as "best".
    study.optimize(
        _objective,
        n_trials=n_trials,
        show_progress_bar=False,
        catch=(Exception,),
    )

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    failed = [t for t in study.trials if t.state.name == "FAIL"]
    if not completed:
        msg = (
            f"HPO study '{study.study_name}' produced no successful trials "
            f"({len(failed)} failed, {len(study.trials)} total). Refusing to "
            "emit a 'best trial' summary."
        )
        logger.error(msg)
        raise SystemExit(msg)

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
        "n_trials_completed": len(completed),
        "n_trials_failed": len(failed),
        "n_trials_total": len(study.trials),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    # Emit a study-level MLflow summary run so the UI shows one canonical
    # artifact per study alongside the per-trial runs. Placed after the
    # trials finish so it doesn't collide with any trial's active run.
    _log_study_summary_to_mlflow(
        experiment_name=f"{args.experiment_name}-studies",
        summary=summary,
        args=args,
        run_args=run_args,
        fitness_cfg=fitness_cfg,
    )
    return 0


def _log_study_summary_to_mlflow(
    *,
    experiment_name: str,
    summary: dict[str, Any],
    args: argparse.Namespace,
    run_args: HPORunArgs,
    fitness_cfg: FitnessConfig,
) -> None:
    """Log a study-level parent run aggregating best-trial stats."""
    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"study-{args.study_name}"):
        mlflow.set_tags(
            {
                "hpo.study_name": args.study_name,
                "hpo.dataset": run_args.dataset,
                "hpo.warm_start": run_args.warm_start,
                "hpo.model_config_name": run_args.model_config_name,
                "hpo.db": args.db,
                "hpo.output_root": str(run_args.output_root),
                "hpo.kind": "training-hpo-study-summary",
            }
        )
        mlflow.log_params(
            {
                "n_trials_requested": args.n_trials if not args.smoke else 2,
                "subsample": run_args.subsample,
                "heldout_fraction": run_args.heldout_fraction,
                "heldout_strategy": run_args.heldout_strategy,
                "keep_top_k": run_args.keep_top_k,
                "startup_trials": args.startup_trials,
                "hunk_loss_weight": fitness_cfg.hunk_loss_weight,
                "hunk_accuracy_weight": fitness_cfg.hunk_accuracy_weight,
                "adapter_improvement_weight": fitness_cfg.adapter_improvement_weight,
                **{f"best.{k}": v for k, v in summary["best_params"].items()},
            }
        )
        mlflow.log_metrics(
            {
                "best_fitness": summary["best_fitness"],
                "n_trials_completed": summary["n_trials_completed"],
                "n_trials_failed": summary["n_trials_failed"],
                "n_trials_total": summary["n_trials_total"],
                "best_trial_number": summary["best_trial"],
            }
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
