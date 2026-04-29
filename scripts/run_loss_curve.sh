#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
#  Rune — diagnostic loss-curve training run
#
#  Skinny epochs × many epochs over a deterministic random subsample of the
#  mined-pairs dataset. Hyperparameters are drawn from the published Qwen
#  team / DeltaCoder recipes (cosine LR, warmup 0.03, rank 64 / α 32) so the
#  curve is grounded in known-good defaults rather than guesses.
#
#  Per-step metrics (loss, grad_norm, lr) hit MLflow keyed by global_step.
#  A parallel `train/epoch` metric is emitted at the same steps — pick it
#  as the X axis in the MLflow chart panel to view any per-step curve on
#  an epoch axis. End-of-epoch summary points land under `epoch/<key>`.
#
#  Usage:
#    bash scripts/run_loss_curve.sh                 # defaults below
#    bash scripts/run_loss_curve.sh --subsample 800 # extra options forwarded
#                                                   # to scripts/train.sh
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

DATASET="${DATASET:-data/github-pairs/_merged/pairs_all.jsonl}"
SUBSAMPLE="${SUBSAMPLE:-400}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
LR_SCHED="${LR_SCHED:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
RANK="${RANK:-64}"
ALPHA="${ALPHA:-32}"
EXPERIMENT="${EXPERIMENT:-qwen-loss-curve}"

# MLflow + cache speed-ups: same env hygiene as scripts/run_hpo.sh.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export RUNE_PERSIST_BASE_MODEL=1
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" \
        && "$PYTORCH_CUDA_ALLOC_CONF" != *expandable_segments:True* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF},expandable_segments:True"
else
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

[[ -f "$DATASET" ]] || { echo "dataset not found: $DATASET" >&2; exit 1; }

mkdir -p .tmp
TS="$(date +%Y%m%d-%H%M%S)"
ADAPTER_ID="qwen-loss-curve-${TS}"
LOG=".tmp/loss_curve_${TS}.log"

echo "Adapter:    $ADAPTER_ID"
echo "Dataset:    $DATASET (subsample=$SUBSAMPLE, seed=42)"
echo "Schedule:   epochs=$EPOCHS lr=$LR sched=$LR_SCHED warmup=$WARMUP_RATIO"
echo "LoRA:       rank=$RANK alpha=$ALPHA grad_accum=$GRAD_ACCUM"
echo "MLflow:     $MLFLOW_TRACKING_URI  (experiment=$EXPERIMENT)"
echo "Log:        $LOG"
echo

bash scripts/train.sh \
    --dataset "$DATASET" \
    --subsample "$SUBSAMPLE" \
    --adapter-id "$ADAPTER_ID" \
    --warm-start deltacoder \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lr-scheduler "$LR_SCHED" \
    --warmup-ratio "$WARMUP_RATIO" \
    --grad-accum "$GRAD_ACCUM" \
    --rank "$RANK" --alpha "$ALPHA" \
    --diff-aware-loss \
    --experiment-name "$EXPERIMENT" \
    "$@" \
    2>&1 | tee "$LOG"
