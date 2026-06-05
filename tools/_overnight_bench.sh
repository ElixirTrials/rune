#!/usr/bin/env bash
# Overnight full benchmarking (REMOVE-BEFORE-MERGE). Arms run SERIALLY (GPU is
# serial) under the RAM watchdog; each appends a one-line result to SUMMARY and
# continues if one fails.
#   base  = single-shot, no engine, no adapter (the true "model alone") via the
#           capability-ceiling tool -- FAST and clean (the full-engine base was
#           pathologically slow on over-decomposed tasks).
#   escalate = the engine (zero-shot base first, adapter on repair).
# Published Qwen3-4B LiveCodeBench v6 = 35.1 is the base reference for LCB.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=/tmp/goal3/overnight; mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
C3=/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt
MBPP=benchmarks/mbpp160_tasks.json
echo "=== overnight bench started ===" >> "$SUMMARY"
wait_for () { while pgrep -f "$1" >/dev/null; do sleep 20; done; }
report () { python3 -c "import json;d=json.load(open('$1'));print(f'$2: {d[\"passed_tasks\"]}/{d[\"total_tasks\"]} = {d[\"pass_at_1\"]:.3f}')" >> "$SUMMARY" 2>>"$SUMMARY"; }

# --- MBPP-160 base (single-shot, no engine, no adapter) ---
echo ">>> MBPP-160 base (single-shot ceiling)" | tee -a "$SUMMARY"
tools/run_guarded.sh "$OUT/mbpp_base.log" tools/_capability_ceiling.py \
  --tasks "$MBPP" --out "$OUT/mbpp_base.json" --checkpoint "$C3"
wait_for "_capability_ceiling.*$OUT/mbpp_base.json"
report "$OUT/mbpp_base.json" "MBPP-160 base (single-shot)"

# --- MBPP-160 escalate (engine) ---
echo ">>> MBPP-160 escalate (engine)" | tee -a "$SUMMARY"
tools/run_guarded.sh "$OUT/mbpp_escalate.log" tools/_goal3_multiturn_probe.py run \
  --arm c3 --tasks "$MBPP" --sessions "$OUT/mbpp_escalate_sessions" \
  --out "$OUT/mbpp_escalate.json" --seed 0 --max-iters 12 \
  --prompt-mode escalate --adapter-scaling 0.627
wait_for "_goal3_multiturn_probe.*mbpp_escalate.json"
report "$OUT/mbpp_escalate.json" "MBPP-160 escalate (engine)"

# --- LiveCodeBench v6 functional (49) escalate -> official grade vs published 35.1 ---
echo ">>> LCB v6 functional escalate (vs published 35.1)" | tee -a "$SUMMARY"
tools/run_guarded.sh "$OUT/lcb_escalate.log" tools/_lcb_run.py \
  --arm c3 --prompt-mode escalate --adapter-scaling 0.627 \
  --out "$OUT/lcb_escalate.json" --max-iters 12 --functional-only
wait_for "_lcb_run.*lcb_escalate.json"
PYTHONPATH=/tmp/LiveCodeBench /tmp/lcbenv/bin/python tools/_lcb_grade.py \
  --gens "$OUT/lcb_escalate.json" >> "$SUMMARY" 2>>"$OUT/lcb_escalate_grade.log"

echo "=== overnight bench DONE ===" | tee -a "$SUMMARY"
