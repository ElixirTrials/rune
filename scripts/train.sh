#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
#  Rune — unified QLoRA training wrapper
#
#  Thin shell over model_training.trainer_cli: sets up env, forwards flags.
#  All actual argparse/validation lives in Python so --help and error
#  messages stay consistent across invocations.
#
#  Usage examples:
#    bash scripts/train.sh --dataset data/pairs/repo.jsonl \
#         --adapter-id my-adapter --warm-start deltacoder
#
#    bash scripts/train.sh --dataset data/pairs/repo.jsonl \
#         --adapter-id smoke --dry-run
#
#    MLFLOW_TRACKING_URI=http://localhost:5000 \
#         bash scripts/train.sh --session-id sess-001 --adapter-id from-sess
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Locate the project root reliably regardless of where the user invoked us from.
cd "${REPO_ROOT}"

# Surface a clear error if uv is missing before Python even starts.
if ! command -v uv >/dev/null 2>&1; then
    echo "train.sh: 'uv' not found on PATH. Install with" \
         "'curl -LsSf https://astral.sh/uv/install.sh | sh'" \
         "or see https://docs.astral.sh/uv/." >&2
    exit 127
fi

# Make MLflow and HF caches / GPU visibility controllable from the parent shell
# without duplicating them here. Python reads these directly; we only export
# for clarity when present.
: "${MLFLOW_TRACKING_URI:=}"
: "${HF_HOME:=}"
: "${CUDA_VISIBLE_DEVICES:=}"

# Expose the workspace-member src directories on PYTHONPATH so Python can
# import model_training / shared / adapter_registry without requiring a
# prior `uv pip install -e` step (workspace packages are not installed by
# default `uv sync`). Prepended so local edits win over any installed copy.
WS_SRC_DIRS=(
    "${REPO_ROOT}/libs/model-training/src"
    "${REPO_ROOT}/libs/shared/src"
    "${REPO_ROOT}/libs/adapter-registry/src"
    "${REPO_ROOT}/libs/inference/src"
    "${REPO_ROOT}/libs/evaluation/src"
)
WS_PP="$(IFS=:; echo "${WS_SRC_DIRS[*]}")"
export PYTHONPATH="${WS_PP}${PYTHONPATH:+:${PYTHONPATH}}"

# Forward all args verbatim to the Python entrypoint.
exec uv run python -m model_training.trainer_cli "$@"
