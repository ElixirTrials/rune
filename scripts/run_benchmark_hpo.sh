#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
#  Rune — benchmark HPO over Rune pipeline params on failed MBPP problems
#
#  Wraps scripts/optimization/run_benchmark_hpo.py: routes MLflow at the in-pod
#  server (artifacts proxy to S3 via --serve-artifacts), defaults the S3
#  hypernet checkpoint, and tunes the CUDA allocator for a single L4.
#
#  Required: uv on PATH, NVIDIA GPU visible, the MLflow stack up, and AWS
#  credentials (the hypernet checkpoint is on S3).
#  Optional: MLFLOW_TRACKING_URI, HYPERNET_CHECKPOINT, HF_HUB_OFFLINE overrides.
#
#  Usage:
#    scripts/run_benchmark_hpo.sh --smoke              # 1 trial x 2 problems
#    scripts/run_benchmark_hpo.sh                      # full 30-trial study
#    scripts/run_benchmark_hpo.sh --fresh              # wipe prior results, start clean
#    scripts/run_benchmark_hpo.sh --n-trials 10        # any run_benchmark_hpo.py flag
#    HYPERNET_CHECKPOINT=s3://.../ckpt.pt scripts/run_benchmark_hpo.sh
#
#  All arguments (except --fresh) are forwarded verbatim to run_benchmark_hpo.py;
#  see its --help for the full flag list.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

FRESH=0
PASSTHROUGH=()
for a in "$@"; do
    case "$a" in
        -h|--help) sed -n '2,22p' "$0"; exit 0;;
        --fresh)   FRESH=1;;
        *)         PASSTHROUGH+=("$a");;
    esac
done
set -- "${PASSTHROUGH[@]+${PASSTHROUGH[@]}}"

HYPERNET_CHECKPOINT="${HYPERNET_CHECKPOINT:-s3://elixirtrials-949678234935-eu-west-2-artifacts/checkpoints/hypernet_hpo/checkpoint.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# ── prereq checks ──────────────────────────────────────────────────────────
command -v uv >/dev/null || { echo "missing: uv" >&2; exit 127; }
command -v nvidia-smi >/dev/null \
    || { echo "nvidia-smi not found — the HPO drives a 9B model on GPU" >&2; exit 1; }
nvidia-smi -L | grep -q "GPU 0" \
    || { echo "no NVIDIA GPU visible" >&2; exit 1; }

# ── checkpoint: default to the S3 path unless the caller passed one ────────
HAS_CKPT=0
for a in "$@"; do
    case "$a" in
        --hypernet-checkpoint|--hypernet-checkpoint=*) HAS_CKPT=1;;
    esac
done
ARGS=("$@")
[[ $HAS_CKPT -eq 0 ]] && ARGS+=(--hypernet-checkpoint "$HYPERNET_CHECKPOINT")
if [[ "$HYPERNET_CHECKPOINT" == s3://* ]] && ! command -v aws >/dev/null; then
    echo "warning: aws CLI not found — an s3:// checkpoint needs AWS credentials" >&2
fi

# ── MLflow: route through the in-pod server ────────────────────────────────
# The server (infra/docker-compose.yml) runs with --serve-artifacts, so every
# log_artifact streams through it to the S3 --artifacts-destination. A local
# file:// fallback would strand the per-problem JSONL artifacts on disk —
# run_benchmark_hpo.py treats an unreachable server as fatal.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export RUNE_DATABASE_URL="${RUNE_DATABASE_URL:-sqlite:///${HOME}/.rune/rune.db}"
export INFERENCE_PROVIDER="${INFERENCE_PROVIDER:-transformers}"
export TRANSFORMERS_MODEL_NAME="${TRANSFORMERS_MODEL_NAME:-Qwen/Qwen3.5-9B}"
# Skip HF Hub HTTP checks when the model snapshot is already cached.
# Saves ~20 HEAD requests per from_pretrained() call (~1s each).
_HF_MODEL_DIR="${HOME}/.cache/huggingface/hub/models--${TRANSFORMERS_MODEL_NAME//\//$'--'}"
if [[ -z "${HF_HUB_OFFLINE:-}" && -d "$_HF_MODEL_DIR/snapshots" ]]; then
    export HF_HUB_OFFLINE=1
fi

if [[ "$MLFLOW_TRACKING_URI" =~ ^https?:// ]]; then
    if ! curl -fsS --max-time 3 "${MLFLOW_TRACKING_URI%/}/health" >/dev/null; then
        echo "MLflow server not reachable at $MLFLOW_TRACKING_URI" >&2
        echo "Start the stack first:" >&2
        echo "  docker compose -f infra/docker-compose.yml up -d mlflow litestream" >&2
        exit 1
    fi
fi
# AdapterRegistry (used by run_phased_pipeline) defaults to ~/.rune/rune.db;
# SQLAlchemy will not create the parent dir, so do it here.
mkdir -p "${HOME}/.rune"

# ── fresh run: back up and wipe prior Optuna state + results ──────────────
# Extract --output-dir from ARGS if present, else use the Python default.
_ODIR="evaluation_results/benchmark_hpo"
for (( i=0; i<${#ARGS[@]}; i++ )); do
    if [[ "${ARGS[$i]}" == "--output-dir" && $(( i+1 )) -lt ${#ARGS[@]} ]]; then
        _ODIR="${ARGS[$((i+1))]}"
        break
    fi
done
if [[ $FRESH -eq 1 && -d "$_ODIR" ]]; then
    BACKUP="/tmp/benchmark_hpo_backup_$(date +%Y%m%d-%H%M%S)"
    echo "Backing up prior results to $BACKUP"
    mv "$_ODIR" "$BACKUP"
    mkdir -p "$_ODIR"
fi

# ── CUDA allocator: reduce VRAM fragmentation across pipeline runs ─────────
# Must be set before any torch import. Set unconditionally for the
# expandable_segments behaviour; preserve any other pre-set options.
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" \
      && "$PYTORCH_CUDA_ALLOC_CONF" != *expandable_segments:True* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF},expandable_segments:True"
else
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

mkdir -p .claude/runs
LOG=".claude/runs/benchmark-hpo-$(date +%Y%m%d-%H%M%S).log"

echo "Checkpoint:   $HYPERNET_CHECKPOINT"
echo "MLflow URI:   $MLFLOW_TRACKING_URI"
echo "Inference:    $INFERENCE_PROVIDER ($TRANSFORMERS_MODEL_NAME)"
echo "Alloc conf:   $PYTORCH_CUDA_ALLOC_CONF"
[[ $FRESH -eq 1 ]] && echo "Fresh run:    yes (prior results backed up)"
echo "Log:          $LOG"
echo

# ── run ────────────────────────────────────────────────────────────────────
uv run python scripts/optimization/run_benchmark_hpo.py "${ARGS[@]}" \
    2>&1 | tee "$LOG"
