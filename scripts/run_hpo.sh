#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
#  Rune — QLoRA HPO study over mined GitHub pairs
#
#  Wraps scripts/optimization/run_training_hpo.py with sane defaults for a
#  single L4 (22 GB). Prints the MLflow URI before kicking off so you can
#  watch trials live.
#
#  Required: uv on PATH, NVIDIA GPU visible, dataset JSONL on disk.
#  Optional: MLFLOW_TRACKING_URI for a remote MLflow server (else
#  sqlite:///./mlflow.db).
#
#  Usage:
#    scripts/run_hpo.sh                                 # 30 trials, all repos
#    scripts/run_hpo.sh --dataset data/github-pairs/fastapi_fastapi.jsonl
#    scripts/run_hpo.sh --n-trials 10 --subsample 200   # quick study
#    scripts/run_hpo.sh --smoke                         # 2 trials × 1 step
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATASET="data/github-pairs/_merged/pairs_all.jsonl"
N_TRIALS=30
SUBSAMPLE=500
KEEP_TOP_K=3
EXPERIMENT="rune-qlora-hpo"
OUTPUT_ROOT="./hpo_artifacts"
MODEL="qwen3.5-9b"
SMOKE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)         DATASET="$2"; shift 2;;
        --n-trials)        N_TRIALS="$2"; shift 2;;
        --subsample)       SUBSAMPLE="$2"; shift 2;;
        --keep-top-k)      KEEP_TOP_K="$2"; shift 2;;
        --experiment-name) EXPERIMENT="$2"; shift 2;;
        --output-root)     OUTPUT_ROOT="$2"; shift 2;;
        --model)           MODEL="$2"; EXTRA_ARGS+=(--model "$2"); shift 2;;
        --smoke)           SMOKE=1; shift;;
        -h|--help)         sed -n '2,18p' "$0"; exit 0;;
        *)                 EXTRA_ARGS+=("$1"); shift;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# ── prereq checks ──────────────────────────────────────────────────────────
command -v uv >/dev/null || { echo "missing: uv" >&2; exit 127; }
command -v nvidia-smi >/dev/null \
    || { echo "nvidia-smi not found — HPO needs a GPU" >&2; exit 1; }
nvidia-smi -L | grep -q "GPU 0" \
    || { echo "no NVIDIA GPU visible" >&2; exit 1; }

# ── data sync (idempotent) ─────────────────────────────────────────────────
# Mirror training data from the team S3 bucket into ./data. `aws s3 sync`
# compares size and mtime, so reruns are no-ops when nothing changed.
S3_DATA_URI="s3://elixirtrials-949678234935-eu-west-2-artifacts/training-data/"
command -v aws >/dev/null \
    || { echo "missing: aws CLI (needed to sync $S3_DATA_URI)" >&2; exit 127; }
mkdir -p data
echo "Syncing $S3_DATA_URI → data/"
aws s3 sync "$S3_DATA_URI" data/

[[ -f "$DATASET" ]] || { echo "dataset not found: $DATASET" >&2; exit 1; }

# ── persistence: route MLflow + AdapterRegistry through the docker stack ───
# HPO runs on the host but writes go to the in-pod MLflow server (which
# Litestream backs up to S3) and to the bind-mounted rune.db that Litestream
# also watches. If the user opted out via env, respect their override.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export RUNE_DATABASE_URL="${RUNE_DATABASE_URL:-sqlite:///${HOME}/.rune/rune.db}"

# Pre-flight only when the URI looks HTTP — sqlite:// fallbacks (used by
# users who explicitly want local-only) skip the curl check.
if [[ "$MLFLOW_TRACKING_URI" =~ ^https?:// ]]; then
    if ! curl -fsS --max-time 2 "${MLFLOW_TRACKING_URI%/}/health" >/dev/null; then
        echo "MLflow server not reachable at $MLFLOW_TRACKING_URI" >&2
        echo "Start the stack first:  docker compose -f infra/docker-compose.yml up -d mlflow litestream" >&2
        exit 1
    fi
fi

# AdapterRegistry uses ~/.rune/rune.db by default; SQLAlchemy won't create the
# parent dir, so do it here.
mkdir -p "${HOME}/.rune"
mkdir -p .tmp "$OUTPUT_ROOT"

# ── HPO speed-ups ──────────────────────────────────────────────────────────
# RUNE_PERSIST_BASE_MODEL=1 enables the in-process NF4 base-model cache
# (libs/model-training/src/model_training/trainer.py:_get_or_load_base) so the
# 9B base loads once per study instead of per trial. Heldout eval reuses the
# same cached base via PeftModel + unload() at trial end.
export RUNE_PERSIST_BASE_MODEL=1
# HF cache probe: if the base model is fully cached locally we run offline
# (no per-trial HEAD checks → faster, no flaky-network failures mid-study);
# if the cache is cold we stay online so the trainer's first call to
# from_pretrained can populate it. ``snapshot_download(local_files_only=True)``
# is the canonical "is this usable offline?" check — it succeeds iff every
# file referenced by the snapshot is on disk.
#
# Honours an explicit pre-set ``HF_HUB_OFFLINE`` from the environment so a
# user who *wants* to force one mode doesn't get overridden.
if [[ -z "${HF_HUB_OFFLINE:-}" ]]; then
    HF_PROBE_RC=0
    HF_MODEL_ID=$(uv run --no-sync python - "$MODEL" <<'PY' 2>/dev/null
import sys
try:
    from model_training.model_configs import ModelRegistry
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError
except Exception:
    sys.exit(2)
try:
    model_id = ModelRegistry.default().get(sys.argv[1]).model_id
except KeyError:
    sys.exit(2)
print(model_id)
try:
    snapshot_download(model_id, local_files_only=True)
except (LocalEntryNotFoundError, FileNotFoundError, OSError):
    sys.exit(1)
sys.exit(0)
PY
    ) || HF_PROBE_RC=$?

    case "$HF_PROBE_RC" in
        0)
            export HF_HUB_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
            echo "HF cache hit for ${HF_MODEL_ID:-$MODEL} — running OFFLINE"
            ;;
        1)
            export HF_HUB_OFFLINE=0
            export TRANSFORMERS_OFFLINE=0
            echo "HF cache miss for ${HF_MODEL_ID:-$MODEL} — first trial will fetch from Hub"
            ;;
        *)
            export HF_HUB_OFFLINE=0
            export TRANSFORMERS_OFFLINE=0
            echo "HF cache probe failed (could not resolve model '$MODEL') — staying ONLINE"
            ;;
    esac
else
    # Operator override: keep TRANSFORMERS_OFFLINE consistent so we don't
    # repeat the original bug where the two flags disagreed.
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
    echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE pre-set by operator — leaving as-is"
fi
# Reduce VRAM fragmentation across HPO trials. Must be set before any torch
# import — PyTorch reads PYTORCH_CUDA_ALLOC_CONF once at import time, so
# os.environ.setdefault inside Python is fragile (RCA-2 Cause 4). We set
# this UNCONDITIONALLY (overriding any pre-existing value) because RCA-2's
# concern is specifically about expandable_segments being present. If the
# user has set a different value (e.g. max_split_size_mb), append to it.
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" && "$PYTORCH_CUDA_ALLOC_CONF" != *expandable_segments:True* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF},expandable_segments:True"
else
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

TS="$(date +%Y%m%d-%H%M%S)"
LOG=".tmp/hpo_${TS}.log"

# ── run ────────────────────────────────────────────────────────────────────
echo "Dataset:        $DATASET"
echo "Trials:         $N_TRIALS  (subsample=$SUBSAMPLE per trial)"
echo "Keep top-k:     $KEEP_TOP_K"
echo "Experiment:     $EXPERIMENT"
echo "Output root:    $OUTPUT_ROOT"
echo "MLflow URI:     ${MLFLOW_TRACKING_URI}"
echo "Log:            $LOG"
echo "Persist base:   ${RUNE_PERSIST_BASE_MODEL} (HF_HUB_OFFLINE=${HF_HUB_OFFLINE})"
echo "Alloc conf:     ${PYTORCH_CUDA_ALLOC_CONF}"
echo

ARGS=(
    --dataset "$DATASET"
    --output-root "$OUTPUT_ROOT"
    --keep-top-k "$KEEP_TOP_K"
    --experiment-name "$EXPERIMENT"
)
if [[ $SMOKE -eq 1 ]]; then
    ARGS+=(--smoke)
else
    ARGS+=(--n-trials "$N_TRIALS" --subsample "$SUBSAMPLE")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    ARGS+=("${EXTRA_ARGS[@]}")
fi

uv run python scripts/optimization/run_training_hpo.py "${ARGS[@]}" \
    2>&1 | tee "$LOG"
