#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
#  Rune — end-to-end HyperLoRA training pipeline
#
#  Automatically detects GPU VRAM and configures both stages accordingly:
#
#    Stage 1: precompute_teacher_logits.py  (one-shot, skipped if cache exists)
#    Stage 2: train_hypernet_hpo.py         (training loop)
#
#  VRAM tiers:
#    <=24 GB  (L4, 3090, 4090)   — NF4, all memory savings enabled
#    25-48 GB (A10G, A6000, A40) — NF4, no offloads, full loss
#    >=49 GB  (A100, H100)       — bf16, full loss, no grad checkpointing
#
#  Usage:
#    scripts/run_hypernet.sh                      # auto-detect everything
#    scripts/run_hypernet.sh --smoke              # quick CI test
#    scripts/run_hypernet.sh --vram-tier high     # force tier override
#    scripts/run_hypernet.sh --num-steps 2000     # pass-through to training
#    scripts/run_hypernet.sh --skip-precompute    # skip stage 1
#    scripts/run_hypernet.sh --s3-uri s3://bucket/prefix/  # logits direct to S3
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CURRENT_STAGE="init"

_on_error() {
    local exit_code=$?
    echo "" >&2
    echo "!! FAILED during: ${CURRENT_STAGE} (exit code ${exit_code})" >&2
    echo "!! Line ${BASH_LINENO[0]} in ${BASH_SOURCE[1]:-$0}" >&2
    exit "$exit_code"
}
trap _on_error ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── defaults ──────────────────────────────────────────────────────────────
TEACHER_ADAPTER="hpo_artifacts/best_diffloss_v1"
DATASET="data/mined/all_unrolled.jsonl"
LOGITS_DIR="data/teacher_logits"
CHECKPOINT_DIR="checkpoints/hypernet_hpo"
BASE_MODEL="Qwen/Qwen3.5-9B"
NUM_STEPS=500
EXPERIMENT="hypernet-hpo"
SMOKE=0
FRESH=0               # --fresh: ignore cached logits + checkpoints, start clean
SKIP_PRECOMPUTE=0
VRAM_TIER=""          # auto, low, mid, high
S3_URI=""             # --s3-uri: stream logits directly to S3
SAGEMAKER=0           # --sagemaker: launch on SageMaker instead of local
FLEET=0               # --fleet: SageMaker fleet mode (spot fallback)
DRY_RUN=0             # --dry-run: print config without launching
SM_EXTRA_ARGS=()
EXTRA_TRAIN_ARGS=()

# ── arg parse ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --teacher-adapter)   TEACHER_ADAPTER="$2"; shift 2;;
        --dataset)           DATASET="$2"; shift 2;;
        --logits-dir)        LOGITS_DIR="$2"; shift 2;;
        --checkpoint-dir)    CHECKPOINT_DIR="$2"; shift 2;;
        --base-model)        BASE_MODEL="$2"; shift 2;;
        --num-steps)         NUM_STEPS="$2"; shift 2;;
        --experiment-name)   EXPERIMENT="$2"; shift 2;;
        --vram-tier)         VRAM_TIER="$2"; shift 2;;
        --smoke)             SMOKE=1; shift;;
        --fresh)             FRESH=1; shift;;
        --skip-precompute)   SKIP_PRECOMPUTE=1; shift;;
        --s3-uri)            S3_URI="$2"; shift 2;;
        --sagemaker)         SAGEMAKER=1; shift;;
        --fleet)             SAGEMAKER=1; FLEET=1; shift;;
        --dry-run)           DRY_RUN=1; shift;;
        --instance-type)     SM_EXTRA_ARGS+=(--instance-type "$2"); shift 2;;
        -h|--help)           sed -n '2,16p' "$0"; exit 0;;
        *)                   EXTRA_TRAIN_ARGS+=("$1"); shift;;
    esac
done

# ── SageMaker dispatch ────────────────────────────────────────────────────
if (( SAGEMAKER )); then
    command -v uv >/dev/null || { echo "missing: uv" >&2; exit 127; }
    SM_CMD=(
        uv run python scripts/launch_sagemaker.py
        --num-steps "$NUM_STEPS"
        --experiment-name "$EXPERIMENT"
        --base-model "$BASE_MODEL"
    )
    if (( FLEET )); then SM_CMD+=(--fleet); fi
    if (( DRY_RUN )); then SM_CMD+=(--dry-run); fi
    if (( SMOKE )); then SM_CMD+=(--smoke); fi
    if [[ -n "$VRAM_TIER" ]]; then SM_CMD+=(--vram-tier "$VRAM_TIER"); fi
    SM_CMD+=("${SM_EXTRA_ARGS[@]}")
    echo "── Launching on SageMaker ─────────────────────────"
    echo "  ${SM_CMD[*]}"
    echo ""
    exec "${SM_CMD[@]}"
fi

CURRENT_STAGE="prereq checks"
# ── prereq checks ────────────────────────────────────────────────────────
command -v uv  >/dev/null || { echo "missing: uv"  >&2; exit 127; }
command -v nvidia-smi >/dev/null || { echo "missing: nvidia-smi" >&2; exit 1; }
nvidia-smi -L | grep -q "GPU 0" || { echo "no GPU visible" >&2; exit 1; }

CURRENT_STAGE="VRAM detection"
# ── VRAM detection ────────────────────────────────────────────────────────
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
           | head -1 | tr -d '[:space:]')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

if [[ -z "$VRAM_TIER" ]]; then
    if   (( VRAM_MIB >= 49000 )); then VRAM_TIER="high"
    elif (( VRAM_MIB >= 25000 )); then VRAM_TIER="mid"
    else                                VRAM_TIER="low"
    fi
fi

echo "────────────────────────────────────────────────────"
echo "  GPU:  ${GPU_NAME} (${VRAM_MIB} MiB)"
echo "  Tier: ${VRAM_TIER}"
echo "────────────────────────────────────────────────────"

# ── tier → flags ──────────────────────────────────────────────────────────
# Each tier builds an array of CLI flags for the two stages.

PRECOMPUTE_PRECISION="nf4"
TRAIN_FLAGS=()

case "$VRAM_TIER" in
    low)
        # <=24 GB: all memory savings, precompute in NF4
        PRECOMPUTE_PRECISION="nf4"
        TRAIN_FLAGS=(
            --base-model-precision nf4
            --gradient-checkpointing
            --optimizer-type adamw-8bit
            --offload-teacher-logits
            --chunk-loss
        )
        ;;
    mid)
        # 25-48 GB: NF4, but drop offloads and use full loss
        PRECOMPUTE_PRECISION="nf4"
        TRAIN_FLAGS=(
            --base-model-precision nf4
            --gradient-checkpointing
            --optimizer-type adamw
            --no-offload-teacher-logits
            --no-chunk-loss
        )
        ;;
    high)
        # >=49 GB: bf16, everything on GPU, full loss
        PRECOMPUTE_PRECISION="bf16"
        TRAIN_FLAGS=(
            --high-vram
        )
        ;;
    *)
        echo "unknown --vram-tier: $VRAM_TIER (expected low/mid/high)" >&2
        exit 1
        ;;
esac

# ── smoke overrides ───────────────────────────────────────────────────────
if (( SMOKE )); then
    NUM_STEPS=5
    EXTRA_TRAIN_ARGS+=(--smoke-test)
    LOGITS_DIR="data/teacher_logits_smoke"
fi

# Namespace checkpoints by experiment to prevent cross-arm contamination
CHECKPOINT_DIR="${CHECKPOINT_DIR}/${EXPERIMENT}"

# ── MLflow ────────────────────────────────────────────────────────────────
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

if [[ "$MLFLOW_TRACKING_URI" =~ ^https?:// ]]; then
    if ! curl -fsS --max-time 2 "${MLFLOW_TRACKING_URI%/}/health" >/dev/null 2>&1; then
        echo "⚠  MLflow not reachable at $MLFLOW_TRACKING_URI — metrics will be local-only"
        export MLFLOW_TRACKING_URI=""
    fi
fi

# ── --fresh: wipe cached logits and checkpoints ─────────────────────────
if (( FRESH )); then
    echo "── --fresh: cleaning cached state ────────────────"
    [[ -d "$LOGITS_DIR" ]] && rm -rf "$LOGITS_DIR" && echo "  removed ${LOGITS_DIR}"
    [[ -d "$CHECKPOINT_DIR" ]] && rm -rf "$CHECKPOINT_DIR" && echo "  removed ${CHECKPOINT_DIR}"
fi

# ── Stage 1: precompute teacher logits ────────────────────────────────────
MANIFEST="${LOGITS_DIR}/manifest.json"

# Detect stale smoke-test cache (≤20 records) and auto-clean
if [[ -z "$S3_URI" ]] && [[ -f "$MANIFEST" ]] && ! (( SMOKE )); then
    N_VALID=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['n_valid'])" 2>/dev/null || echo "?")
    if [[ "$N_VALID" =~ ^[0-9]+$ ]] && (( N_VALID <= 20 )); then
        echo "── Stage 1: STALE CACHE (only ${N_VALID} records — likely from smoke test)"
        echo "  Deleting ${LOGITS_DIR} and regenerating…"
        rm -rf "$LOGITS_DIR"
    fi
fi

if (( SKIP_PRECOMPUTE )); then
    echo "── Stage 1: SKIPPED (--skip-precompute) ──────────"
elif [[ -z "$S3_URI" ]] && [[ -f "$MANIFEST" ]] && ! (( SMOKE )); then
    N_VALID=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['n_valid'])" 2>/dev/null || echo "?")
    echo "── Stage 1: CACHED (${N_VALID} records in ${LOGITS_DIR}) ──"
elif [[ -n "$S3_URI" ]] && ! (( SMOKE )) && aws s3 ls "${S3_URI%/}/manifest.json" >/dev/null 2>&1; then
    N_VALID=$(aws s3 cp "${S3_URI%/}/manifest.json" - 2>/dev/null \
              | python3 -c "import json,sys; print(json.load(sys.stdin)['n_valid'])" 2>/dev/null || echo "?")
    echo "── Stage 1: CACHED (${N_VALID} records in ${S3_URI}) ──"
else
    PRECOMPUTE_DEST="${S3_URI:-$LOGITS_DIR}"
    echo "── Stage 1: precompute teacher logits ────────────"
    echo "  precision: ${PRECOMPUTE_PRECISION}"
    echo "  adapter:   ${TEACHER_ADAPTER}"
    echo "  output:    ${PRECOMPUTE_DEST}"
    echo ""

    PRECOMPUTE_CMD=(
        uv run python scripts/precompute_teacher_logits.py
        --teacher-adapter "$TEACHER_ADAPTER"
        --dataset "$DATASET"
        --base-model "$BASE_MODEL"
        --base-model-precision "$PRECOMPUTE_PRECISION"
    )
    if [[ -n "$S3_URI" ]]; then
        PRECOMPUTE_CMD+=(--s3-uri "$S3_URI")
    else
        PRECOMPUTE_CMD+=(--output-dir "$LOGITS_DIR")
    fi
    if (( SMOKE )); then
        PRECOMPUTE_CMD+=(--smoke-test)
    fi

    CURRENT_STAGE="Stage 1: precompute teacher logits (${PRECOMPUTE_DEST})"
    echo "  ${PRECOMPUTE_CMD[*]}"
    echo ""
    "${PRECOMPUTE_CMD[@]}"
    echo ""
    echo "── Stage 1: DONE ─────────────────────────────────"
fi

echo ""

# ── Stage 2: train hypernetwork ───────────────────────────────────────────
TEACHER_LOGITS_SRC="${S3_URI:-$LOGITS_DIR}"
echo "── Stage 2: train hypernetwork ───────────────────"
echo "  steps:     ${NUM_STEPS}"
echo "  tier:      ${VRAM_TIER}"
echo "  logits:    ${TEACHER_LOGITS_SRC}"
echo "  ckpt:      ${CHECKPOINT_DIR}"
echo ""

TRAIN_CMD=(
    uv run python scripts/train_hypernet_hpo.py
    --teacher-adapter "$TEACHER_ADAPTER"
    --dataset "$DATASET"
    --base-model "$BASE_MODEL"
    --teacher-logits-dir "$TEACHER_LOGITS_SRC"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --num-steps "$NUM_STEPS"
    --experiment-name "$EXPERIMENT"
    "${TRAIN_FLAGS[@]}"
    "${EXTRA_TRAIN_ARGS[@]}"
)

CURRENT_STAGE="Stage 2: train hypernetwork (${NUM_STEPS} steps, tier=${VRAM_TIER})"
echo "  ${TRAIN_CMD[*]}"
echo ""
"${TRAIN_CMD[@]}"

echo ""
echo "── Stage 2: DONE ─────────────────────────────────"
echo ""
echo "Checkpoint: ${CHECKPOINT_DIR}/checkpoint.pt"
