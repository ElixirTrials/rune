#!/usr/bin/env bash
# Drive the recall-capacity probe across all 3 arms sequentially under the RAM/disk watchdog.
# REMOVE-BEFORE-MERGE (issue #52 goal-1 scaffolding).
set -uo pipefail
cd /workspaces/content
mkdir -p /tmp/cap
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROBE=tools/_recall_capacity_probe.py
WARM="/workspaces/rune-gpu/third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
CORPUS="/workspaces/rune-gpu/benchmarks/mbpp_recall_heldout.jsonl"
K="1,2,4,8"

run_arm() {
  local name="$1"; shift
  echo "=== ARM ${name} start $(date -u +%H:%M:%S) ==="
  tools/run_guarded.sh "/tmp/cap/${name}.log" "$PROBE" \
    --corpus "$CORPUS" --k-values "$K" --out "/tmp/cap/${name}.jsonl" "$@"
  echo "=== ARM ${name} done rc=$? $(date -u +%H:%M:%S) ==="
  echo "--- SUMMARY ${name} ---"
  grep -A30 "=== SUMMARY ===" "/tmp/cap/${name}.log" | grep -E "pass1|rate|interference|arm|k= " || true
}

run_arm scale0 --scale0
run_arm warm   --ckpt "$WARM"
run_arm c3     --ckpt /tmp/phase1/ckpt/c3_t07_lp2_lg1.pt
echo "=== ALL ARMS COMPLETE $(date -u +%H:%M:%S) ==="
