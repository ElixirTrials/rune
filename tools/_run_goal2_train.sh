#!/usr/bin/env bash
# Train c3-objective hypernet on scaled corpora (N=80,160) for issue#52 goal-2. REMOVE-BEFORE-MERGE.
set -uo pipefail
cd /workspaces/content
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENTRY=tools/_distill_entry.py

train_one() {
  local n="$1"
  echo "=== TRAIN N=${n} start $(date -u +%H:%M:%S) ==="
  tools/run_guarded.sh "/tmp/goal2/train_n${n}.log" "$ENTRY" \
    --config "/tmp/goal2/c3_n${n}.yaml" --max-steps 48
  local rc=$?
  echo "=== TRAIN N=${n} done rc=${rc} $(date -u +%H:%M:%S) ==="
  local dir="/tmp/goal2/ckpt/c3_n${n}"
  local src=""
  for cand in "${dir}/checkpoint_step48.pt" "${dir}/checkpoint.pt" "${dir}/checkpoint_best.pt"; do
    [ -f "$cand" ] && src="$cand" && break
  done
  if [ -n "$src" ]; then
    cp "$src" "/tmp/goal2/ckpt/c3_n${n}.pt"
    echo "=== CKPT N=${n} -> /tmp/goal2/ckpt/c3_n${n}.pt (from $(basename "$src"), $(du -h "$src" | cut -f1)) ==="
  else
    echo "=== CKPT N=${n} MISSING — ls ${dir}: $(ls "$dir" 2>/dev/null || echo none) ==="
  fi
}

train_one 80
train_one 160
echo "=== GOAL2 TRAIN ALL DONE $(date -u +%H:%M:%S) ==="
ls -la /tmp/goal2/ckpt/*.pt 2>/dev/null
