#!/usr/bin/env bash
# Run both held-out evaluators on a fine-tuned adapter and print a single summary.
#
# Implements the advisor's two evaluation gaps:
#   1. Held-out generalisation (per-token CE + tok_acc) — eval_heldout.py
#   2. Patch quality (syntactic validity + hunk IoU) — eval_patch_quality.py
#
# Usage:
#   bash scripts/_diag/eval_full.sh <ADAPTER_PATH> [N_ROWS]
#
# Example:
#   bash scripts/_diag/eval_full.sh ./hpo_artifacts/run_xxx/adapter 50
set -euo pipefail

ADAPTER="${1:-}"
N_ROWS="${2:-50}"
HELDOUT="${HELDOUT:-data/_ab/pairs_heldout_100.jsonl}"

if [[ -z "$ADAPTER" ]]; then
    echo "usage: $0 <adapter_path> [n_rows]" >&2
    exit 2
fi

if [[ ! -d "$ADAPTER" ]]; then
    echo "error: adapter dir not found: $ADAPTER" >&2
    exit 1
fi

echo "============================================================"
echo "EVALUATION: $ADAPTER on $HELDOUT (n=$N_ROWS)"
echo "============================================================"
echo

echo "--- 1. Held-out CE / tok_acc (base / +deltacoder / +fine-tuned) ---"
uv run python scripts/_diag/eval_heldout.py \
    --heldout "$HELDOUT" \
    --n-rows "$N_ROWS" \
    --adapter "$ADAPTER"
echo

echo "--- 2. Patch quality (syntactic validity / hunk IoU / char-similarity) ---"
# Smaller N for the patch evaluator — generation is slower than forward.
PATCH_N="${PATCH_N:-25}"
uv run python scripts/_diag/eval_patch_quality.py \
    --heldout "$HELDOUT" \
    --n-rows "$PATCH_N" \
    --adapter "$ADAPTER"
echo

echo "============================================================"
echo "Done."
