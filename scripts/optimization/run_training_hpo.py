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
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Local fallback for adapters when the MLflow/S3 upload fails. Picked under
# ~/.rune/ so it shares the volume already mounted into the dev container and
# isn't on the trial's ephemeral output dir.
_LOCAL_FALLBACK_ROOT = Path.home() / ".rune" / "hpo-adapter-fallback"

# Free-space safety margin we refuse to consume when copying an adapter to the
# local fallback. ~1 GiB leaves headroom for OS / log writes / next trial's
# trainer_state.json so a fallback save can't be the thing that fills the disk.
_FALLBACK_FREE_SPACE_MARGIN_BYTES = 1 * 1024 * 1024 * 1024

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
    upload_adapters_to_mlflow: bool = True
    cleanup_local_adapters: bool = True
    encoding_mode: str = "multi_turn"
    extra_train_kwargs: dict[str, Any] = field(default_factory=dict)
    proxy_epochs: int = 3
    force_diff_aware_loss: bool | None = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_training_hpo",
        description="Training-hyperparameter HPO for DeltaCoder fine-tune.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="JSONL of mined pairs.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument(
        "--proxy-epochs",
        dest="proxy_epochs",
        type=int,
        default=3,
        help=(
            "Epochs per HPO trial. Was hardcoded to 1 historically; "
            "the 2026-05-03 deep-dive established 1 epoch is below the "
            "visibility threshold for code-edit fine-tunes against a "
            "warm-start prior, so the historical HPO winners were tuned "
            "under near-zero signal. Default 3 matches the canonical "
            "Magicoder-style recipe."
        ),
    )
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
        "--encoding-mode",
        dest="encoding_mode",
        choices=["multi_turn", "single_turn"],
        default="multi_turn",
        help="Chat encoding mode passed to train_and_register.",
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
        "--upload-adapters-to-mlflow",
        dest="upload_adapters_to_mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After eval, upload each trial's adapter as an MLflow artifact. "
            "When the MLflow server is configured with --serve-artifacts and "
            "--default-artifact-root s3://… (the in-pod default), the upload "
            "lands in S3 transparently — no boto3 calls from the client. "
            "Disable to keep adapters local-only."
        ),
    )
    parser.add_argument(
        "--cleanup-local-adapters",
        dest="cleanup_local_adapters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Delete each trial's local adapter directory immediately after "
            "the MLflow upload succeeds. Avoids letting hundreds of trial "
            "adapters fill the host disk during long studies. Has no effect "
            "when --no-upload-adapters-to-mlflow is set, since dropping the "
            "only copy would lose the adapter entirely."
        ),
    )
    parser.add_argument(
        "--force-diff-aware-loss",
        dest="force_diff_aware_loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force diff_aware_loss to True (or False with --no-). "
            "Removes it from the search space so every trial uses the "
            "fixed value. Useful for A/B comparison across studies."
        ),
    )
    return parser


def _suggest_trial_params(
    trial: Any,
    *,
    force_diff_aware_loss: bool | None = None,
) -> dict[str, Any]:
    """Sample one trial's hyperparameters from the warm-start-aware search space."""
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    alpha = trial.suggest_categorical("alpha_override", [16, 32, 64, 128])
    dropout = trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1])
    warmup = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    grad_accum = trial.suggest_categorical("grad_accum", [8, 16, 32])
    scheduler = trial.suggest_categorical("lr_scheduler", ["constant", "cosine"])
    if force_diff_aware_loss is not None:
        diff_aware = trial.suggest_categorical(
            "diff_aware_loss",
            [force_diff_aware_loss],
        )
    else:
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
        "encoding_mode": run_args.encoding_mode,
        "model_config_name": run_args.model_config_name,
        "warm_start_adapter_id": _resolve_warm_start(run_args.warm_start),
        "epochs": run_args.proxy_epochs,
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


class _RunningTopK:
    """Track the running top-K trials by fitness for selective S3 uploads.

    The HPO study can run 30+ trials. Uploading every adapter to S3 would
    persist 30× as many checkpoints as the operator actually wants
    ("save a few" in the original ask), and waste S3 lifecycle later.
    Instead we mirror the post-study ``--keep-top-k`` semantics inline:
    only adapters that currently rank in the top-K get uploaded; when a
    new trial displaces an older top-K member, the displaced run's
    artifacts are dropped from MLflow so the bucket converges to ≤ K
    adapters per study.

    The set is small (K ≤ 10 in practice), so a sorted list with a linear
    scan is faster and clearer than a heap.
    """

    def __init__(self, k: int) -> None:
        if k < 0:
            raise ValueError(f"_RunningTopK: k must be >= 0, got {k}")
        self.k = k
        # Entries sorted by fitness ascending — entries[0] is the worst-of-top-K.
        self._entries: list[tuple[float, int, str]] = []

    def offer(
        self, fitness: float, trial_number: int, run_id: str
    ) -> tuple[bool, str | None]:
        """Decide whether ``trial_number`` enters the top-K.

        Returns:
            ``(should_upload, displaced_run_id)``:
              * ``should_upload`` is ``True`` when this trial should be
                pushed to S3.
              * ``displaced_run_id`` is the MLflow run id of the trial
                that just got knocked out of the top-K (caller cleans
                up its artifacts), or ``None`` when nothing was
                displaced.
        """
        if self.k == 0:
            return False, None
        if len(self._entries) < self.k:
            self._entries.append((fitness, trial_number, run_id))
            self._entries.sort(key=lambda e: e[0])
            return True, None
        worst_fitness, _, worst_run_id = self._entries[0]
        if fitness <= worst_fitness:
            return False, None
        self._entries[0] = (fitness, trial_number, run_id)
        self._entries.sort(key=lambda e: e[0])
        return True, worst_run_id


def _delete_mlflow_run(run_id: str) -> bool:
    """Best-effort delete of a displaced MLflow run.

    Marks the run as ``deleted`` in the tracking store and instructs the
    server to delete its artifacts; on the in-pod stack the artifact
    store is S3, so this clears the S3 prefix for the run. Returns
    ``False`` (and logs) on any failure rather than aborting the trial —
    a leftover orphaned run is annoying but recoverable, while a thrown
    exception here would mask the actual training-side outcome of the
    new trial.

    Note: stale orphans (e.g. from earlier studies) are not addressed
    here — operators should periodically run ``mlflow gc --tracking-uri
    …`` to physically purge artifacts of runs marked deleted long ago.
    """
    try:
        from mlflow.exceptions import MlflowException  # noqa: PLC0415
        from mlflow.tracking import MlflowClient  # noqa: PLC0415

        MlflowClient().delete_run(run_id)
    except (ImportError, OSError, RuntimeError, MlflowException):
        logger.exception(
            "Failed to delete displaced MLflow run %s; "
            "artifact will linger until `mlflow gc` runs.",
            run_id,
        )
        return False
    return True


def _dir_size_bytes(path: Path) -> int:
    """Sum of regular-file sizes under *path*. Unreadable entries are skipped."""
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _save_to_local_fallback(
    adapter_path: Path,
    *,
    run_id: str,
    artifact_path: str,
    fallback_root: Path = _LOCAL_FALLBACK_ROOT,
    margin_bytes: int = _FALLBACK_FREE_SPACE_MARGIN_BYTES,
) -> Path | None:
    """Copy an adapter to a stable local location, gated on free disk.

    Used when ``mlflow.log_artifacts`` fails so the adapter survives even if
    the trial's output dir is later cleaned up. Returns the destination path
    on success, ``None`` if the disk is too full to absorb the copy without
    breaching ``margin_bytes`` of headroom (or if the copy itself errored).
    The caller is responsible for tagging the MLflow run with the outcome.
    """
    adapter_size = _dir_size_bytes(adapter_path)
    # Probe disk usage on whichever ancestor of fallback_root currently exists,
    # so the check works on a fresh install where the root has never been made.
    probe = fallback_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        logger.exception(
            "Could not stat disk usage at %s — skipping local fallback.", probe
        )
        return None
    required = adapter_size + margin_bytes
    if usage.free < required:
        logger.error(
            "Local fallback skipped: free=%d B at %s, need %d B "
            "(adapter=%d B + %d B safety margin). Adapter remains at %s.",
            usage.free,
            probe,
            required,
            adapter_size,
            margin_bytes,
            adapter_path,
        )
        return None
    dest = fallback_root / run_id / artifact_path
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(adapter_path, dest)
    except OSError:
        logger.exception("Local fallback copy %s -> %s failed.", adapter_path, dest)
        return None
    logger.warning(
        "MLflow upload failed; adapter copied to local fallback %s "
        "(size=%d B, free_after=%d B).",
        dest,
        adapter_size,
        usage.free - adapter_size,
    )
    return dest


def _upload_adapter_and_cleanup(
    adapter_dir: str,
    *,
    upload: bool,
    cleanup: bool,
    artifact_path: str = "adapter",
) -> bool:
    """Log ``adapter_dir`` to the active MLflow run, then optionally delete it.

    Why MLflow over a direct ``boto3`` upload: the in-pod MLflow server is
    started with ``--serve-artifacts`` and ``--default-artifact-root
    s3://…/mlflow/artifacts/`` (see ``infra/docker-compose.yml``), so any
    ``mlflow.log_artifacts`` call from the HPO host streams through the
    server and lands in the team S3 bucket *exactly once*, scoped to the
    current run id. The client never holds AWS credentials and the run's
    ``artifact_uri`` is the canonical pointer — no second copy elsewhere.

    Returns ``True`` when the upload (and optional cleanup) succeeded so
    the caller can record an MLflow tag for observability. Returns
    ``False`` on any failure path; the local copy is preserved in that
    case so the trial is not lost.

    The ``cleanup=True`` path is gated on ``upload=True``: we never delete
    the only existing copy.
    """
    if not upload:
        return False

    import mlflow  # noqa: PLC0415

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        logger.warning("Adapter dir %s missing — skipping MLflow upload.", adapter_dir)
        return False

    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else "<no-active-run>"
    artifact_uri = active_run.info.artifact_uri if active_run else "<unknown>"
    target_uri = f"{artifact_uri.rstrip('/')}/{artifact_path}"

    try:
        file_count = sum(1 for _ in adapter_path.rglob("*") if _.is_file())
    except OSError:
        file_count = -1

    try:
        mlflow.log_artifacts(str(adapter_path), artifact_path=artifact_path)
    except Exception as exc:
        logger.exception(
            "MLflow log_artifacts failed: src=%s (%d files) -> dst=%s "
            "(run_id=%s, error=%s: %s); keeping local copy at %s. "
            "If the tracking server is configured with --serve-artifacts, "
            "verify its IAM role has s3:PutObject AND s3:ListBucket on the "
            "artifact bucket prefix.",
            adapter_dir,
            file_count,
            target_uri,
            run_id,
            type(exc).__name__,
            exc,
            adapter_dir,
        )
        _save_to_local_fallback(
            adapter_path, run_id=run_id, artifact_path=artifact_path
        )
        raise

    if cleanup:
        try:
            shutil.rmtree(adapter_path, ignore_errors=False)
        except OSError:
            logger.exception(
                "Failed to remove local adapter dir %s after MLflow upload "
                "(adapter is safely persisted in S3 via MLflow).",
                adapter_dir,
            )
    return True


def _all_masked_batch_frac_from_trainer_state(output_dir: str) -> float:
    """Mean ``train/all_masked_batch_frac`` across this trial's log_history.

    Surfaces the RCA-5 H2 zero-gradient-batch frequency to the HPO scoreboard.
    A trial whose adapter looks "good" on hunk metrics but trained against
    largely empty gradients should be discounted; this number is what tells
    the operator whether to trust the result. ``0.0`` is healthy; ``>0.05``
    means at least 5 % of micro-batches contributed nothing to the loss
    (almost always a dataset/truncation issue, not a hyperparameter issue).

    Returns ``0.0`` when the metric was never logged (non-diff-aware trials,
    older trainer versions, or a missing trainer_state.json) — those trials
    should not be ranked-against the diff-aware ones on this dimension.
    """
    state_file = Path(output_dir) / "trainer_state.json"
    if not state_file.exists():
        return 0.0
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0
    history = state.get("log_history", [])
    fracs = [
        float(entry["train/all_masked_batch_frac"])
        for entry in history
        if "train/all_masked_batch_frac" in entry
    ]
    if not fracs:
        return 0.0
    return sum(fracs) / len(fracs)


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
    except ImportError:
        pass


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

                # Gather hunk-token positions first, then run log_softmax
                # only on those rows. Materialising the full ``[L-1, V]``
                # log_probs + probs would peak at ~2.5 GB at L=2048,
                # V=151k — tight on the L4 with the adapter+base resident
                # (mirrors the training-side fix in diff_loss.py).
                hunk_idx: list[int] = []
                for i in range(shift_logits.size(0)):
                    ts, te = offsets[i + 1]
                    if ts == 0 and te == 0:
                        continue
                    if any(ts < he and te > hs for hs, he in shifted):
                        hunk_idx.append(i)
                if not hunk_idx:
                    continue
                idx_tensor = torch.tensor(
                    hunk_idx, device=shift_logits.device, dtype=torch.long
                )
                hunk_logits = shift_logits.index_select(0, idx_tensor)
                hunk_targets = shift_ids.index_select(0, idx_tensor)
                hunk_log_probs = torch.log_softmax(hunk_logits.float(), dim=-1)
                nll_per_token = -hunk_log_probs.gather(
                    1, hunk_targets.unsqueeze(1)
                ).squeeze(1)
                hunk_preds = hunk_logits.argmax(dim=-1)
                entropy_per_token = -(hunk_log_probs.exp() * hunk_log_probs).sum(dim=-1)

                total_loss += float(nll_per_token.sum().item())
                total_acc += float((hunk_preds == hunk_targets).sum().item())
                total_ent += float(entropy_per_token.sum().item())
                total_tok += len(hunk_idx)
                del hunk_logits, hunk_log_probs, nll_per_token, entropy_per_token
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
        except (RuntimeError, AttributeError, OSError):
            logger.exception("Heldout eval: PeftModel.unload() failed")
        del adapter_model
        _gc.collect()
        try:
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
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


def _gate_and_upload_adapter(
    adapter_output_dir: str,
    *,
    run_id: str,
    trial_number: int,
    fitness: float,
    run_args: HPORunArgs,
    top_k: _RunningTopK,
) -> str:
    """Decide whether this trial's adapter goes to S3, and act on it.

    Three exit states, returned as the string the caller writes to the
    ``hpo.adapter_uploaded_to_mlflow`` MLflow tag:

    - ``"disabled"``: ``--no-upload-adapters-to-mlflow`` was passed; the
      adapter stays local untouched.
    - ``"true"``: trial entered the running top-K and the adapter was
      uploaded successfully (local copy removed when
      ``--cleanup-local-adapters`` is on).
    - ``"skipped_not_top_k"``: trial did not earn a slot; the adapter is
      cleaned up locally (when cleanup is on) without ever being
      uploaded — this is what bounds total S3 checkpoints to ≤ K per
      study.
    - ``"upload_failed"``: top-K slot was earned but the upload itself
      failed; the local copy is preserved so the operator can recover.

    Side effects: when the top-K offer displaces an older trial, that
    trial's MLflow run is deleted via :func:`_delete_mlflow_run` so the
    bucket converges to ≤ K live runs per study.
    """
    import shutil  # noqa: PLC0415

    if not run_args.upload_adapters_to_mlflow:
        return "disabled"

    should_upload, displaced_run_id = top_k.offer(fitness, trial_number, run_id)
    if displaced_run_id:
        _delete_mlflow_run(displaced_run_id)
        # Best-effort tag — the displaced run is in a different MLflow run
        # context, so we annotate the new (current) run for traceability.
        import mlflow  # noqa: PLC0415

        mlflow.set_tag("hpo.displaced_run_id", displaced_run_id)

    if should_upload:
        ok = _upload_adapter_and_cleanup(
            adapter_output_dir,
            upload=True,
            cleanup=run_args.cleanup_local_adapters,
        )
        return "true" if ok else "upload_failed"

    # Trial did not earn a top-K slot. Remove the local copy if cleanup is
    # enabled — keeping it would defeat the "don't fill up our HD" goal,
    # and the trial's metrics are already in MLflow even though its
    # weights aren't.
    if run_args.cleanup_local_adapters:
        shutil.rmtree(adapter_output_dir, ignore_errors=True)
    return "skipped_not_top_k"


def _run_single_trial(
    trial: Any,
    *,
    run_args: HPORunArgs,
    fitness_cfg: FitnessConfig,
    prior_losses: list[float],
    top_k: _RunningTopK,
) -> float:
    """Objective function body for one Optuna trial.

    ``top_k`` is mutated in place: when this trial earns a slot, its
    adapter is uploaded to MLflow (S3 in the in-pod config) and any
    displaced trial's run is deleted so the bucket converges to ≤ K
    adapters per study.
    """
    sampled = _suggest_trial_params(
        trial,
        force_diff_aware_loss=run_args.force_diff_aware_loss,
    )
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
    _ensure_experiment_active(
        kwargs.get("mlflow_experiment") or run_args.experiment_name
    )
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
        # Surface the per-trial fraction of zero-gradient micro-batches so the
        # MLflow scoreboard exposes degenerate trials even when their hunk
        # metrics happen to look fine. trainer_state lives next to the saved
        # adapter (RUNE_ADAPTER_DIR/<adapter_id>/trainer_state.json). Logged
        # under the ``train/`` prefix because it is a training-time metric,
        # not an eval-time one — keeping the namespacing honest avoids the
        # MLflow scoreboard treating it as part of the held-out signal.
        all_masked_frac = _all_masked_batch_frac_from_trainer_state(adapter_output_dir)
        mlflow.log_metrics(
            {f"eval/{k}": v for k, v in eval_metrics.items()},
            step=trial.number,
        )
        mlflow.log_metric(
            "train/all_masked_batch_frac_mean", all_masked_frac, step=trial.number
        )
        mlflow.set_tag("hpo.train_all_masked_batch_frac", f"{all_masked_frac:.4f}")
        eval_metrics["all_masked_batch_frac"] = all_masked_frac

        # Compute fitness up-front so we can gate the S3 upload on top-K
        # membership: trials that don't earn a slot never touch the
        # bucket, keeping persisted checkpoint count ≤ keep_top_k per
        # study even on long ``--n-trials`` runs.
        fitness = _compute_fitness(
            eval_metrics["hunk_loss"],
            eval_metrics["hunk_accuracy"],
            eval_metrics["adapter_improvement"],
            prior_losses=prior_losses,
            cfg=fitness_cfg,
        )

        active_run = mlflow.active_run()
        assert active_run is not None
        run_id = active_run.info.run_id
        upload_status = _gate_and_upload_adapter(
            adapter_output_dir,
            run_id=run_id,
            trial_number=trial.number,
            fitness=fitness,
            run_args=run_args,
            top_k=top_k,
        )
        mlflow.set_tag("hpo.adapter_uploaded_to_mlflow", upload_status)
    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")
    logger.info(
        "Trial %d hunk_loss=%.4f hunk_acc=%.3f"
        " adapter_imp=%.3f entropy=%.3f all_masked_frac=%.3f fitness=%.4f",
        trial.number,
        eval_metrics["hunk_loss"],
        eval_metrics["hunk_accuracy"],
        eval_metrics["adapter_improvement"],
        eval_metrics["hunk_entropy"],
        eval_metrics["all_masked_batch_frac"],
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
        upload_adapters_to_mlflow=args.upload_adapters_to_mlflow,
        cleanup_local_adapters=args.cleanup_local_adapters,
        proxy_epochs=args.proxy_epochs,
        encoding_mode=args.encoding_mode,
        force_diff_aware_loss=args.force_diff_aware_loss,
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
    # Bound persisted S3 checkpoints to the same ``--keep-top-k`` budget the
    # post-study local pruner uses; without this every successful trial
    # would push another adapter to the bucket.
    top_k = _RunningTopK(k=run_args.keep_top_k)

    def _objective(trial: optuna.Trial) -> float:
        return _run_single_trial(
            trial,
            run_args=run_args,
            fitness_cfg=fitness_cfg,
            prior_losses=prior_losses,
            top_k=top_k,
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


def _ensure_experiment_active(experiment_name: str) -> None:
    """Restore a soft-deleted MLflow experiment so ``set_experiment`` works.

    MLflow refuses ``set_experiment`` on a soft-deleted experiment with
    *"Cannot set a deleted experiment ... as the active experiment."* —
    typically triggered when an operator hits "Delete" in the UI between
    HPO runs. Restoring the experiment is the documented recovery path
    and preserves prior runs / artifact lineage.

    Implemented as try-set / catch-deleted / restore / retry rather than
    a pre-flight ``get_experiment_by_name`` lookup because that method
    only returns *active* experiments against the REST tracking server
    (and a few other backends) — soft-deleted ones come back as
    ``None``, so a pre-flight check would never see them. The
    catch-restore-retry path is backend-agnostic.
    """
    import mlflow  # noqa: PLC0415
    from mlflow.entities import ViewType  # noqa: PLC0415
    from mlflow.exceptions import MlflowException  # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    try:
        mlflow.set_experiment(experiment_name)
        return
    except MlflowException as exc:
        if "deleted experiment" not in str(exc).lower():
            raise

    # Locate the soft-deleted experiment via search_experiments, which
    # honours ViewType.ALL on every backend. Filter in Python to avoid
    # quoting edge cases in the filter_string DSL.
    client = MlflowClient()
    page_token: str | None = None
    match = None
    while True:
        page = client.search_experiments(view_type=ViewType.ALL, page_token=page_token)
        match = next((e for e in page if e.name == experiment_name), None)
        if match is not None:
            break
        page_token = getattr(page, "token", None)
        if not page_token:
            break

    if match is None:
        raise RuntimeError(
            f"MLflow rejected experiment {experiment_name!r} as deleted but "
            "search_experiments(view_type=ALL) could not locate it."
        )
    client.restore_experiment(match.experiment_id)
    logger.warning(
        "Restored soft-deleted MLflow experiment %r (id=%s) so HPO can write to it.",
        experiment_name,
        match.experiment_id,
    )
    mlflow.set_experiment(experiment_name)


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
    _ensure_experiment_active(experiment_name)
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
