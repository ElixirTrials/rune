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
#    scripts/run_hpo.sh --dataset data/pairs/fastapi_fastapi.jsonl
#    scripts/run_hpo.sh --n-trials 10 --subsample 200   # quick study
#    scripts/run_hpo.sh --smoke                         # 2 trials × 1 step
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATASET="data/pairs_all.jsonl"
N_TRIALS=30
SUBSAMPLE=500
KEEP_TOP_K=3
EXPERIMENT="rune-qlora-hpo"
OUTPUT_ROOT="./hpo_artifacts"
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
# Skip Hugging Face Hub HEAD checks per trial — the model is already cached
# locally after the first download. Saves ~1-2 s per trial × N trials and
# avoids flaky-network failure modes mid-study.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
