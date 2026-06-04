#!/usr/bin/env bash
# Eval scaled-corpus ckpts on the FIXED 24 heldout: accessibility (specificity) + k=1 pass@1.
# issue#52 goal-2. REMOVE-BEFORE-MERGE.
set -uo pipefail
cd /workspaces/content
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HELD="/workspaces/rune-gpu/benchmarks/mbpp_recall_heldout.jsonl"
SPEC=tools/_specificity_probe.py
CAP=tools/_recall_capacity_probe.py
WARM="/workspaces/rune-gpu/third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"

eval_ckpt() {
  local tag="$1" ckpt="$2"
  echo "=== EVAL ${tag} start $(date -u +%H:%M:%S) ==="
  tools/run_guarded.sh "/tmp/goal2/spec_${tag}.log" "$SPEC" \
    --ckpt "$ckpt" --corpus "$HELD" --out "/tmp/goal2/spec_${tag}.jsonl"
  echo "=== SPEC ${tag} rc=$? ==="
  tools/run_guarded.sh "/tmp/goal2/cap_${tag}.log" "$CAP" \
    --ckpt "$ckpt" --corpus "$HELD" --k-values 1 --out "/tmp/goal2/cap_${tag}.jsonl"
  echo "=== CAP ${tag} rc=$? $(date -u +%H:%M:%S) ==="
}

eval_ckpt warm "$WARM"
eval_ckpt n80  /tmp/goal2/ckpt/c3_n80.pt
eval_ckpt n160 /tmp/goal2/ckpt/c3_n160.pt
echo "=== GOAL2 EVAL ALL DONE $(date -u +%H:%M:%S) ==="
