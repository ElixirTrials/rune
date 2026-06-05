#!/usr/bin/env bash
# Overnight full benchmarking (REMOVE-BEFORE-MERGE). Runs arms SERIALLY (GPU is
# serial) under the RAM watchdog; each arm logs + appends a one-line result to
# the summary. Continues to the next arm if one fails. Published Qwen3-4B
# LiveCodeBench v6 = 35.1 is the base reference, so escalate is prioritized there.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=/tmp/goal3/overnight
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
CKPT=/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt
MBPP=benchmarks/mbpp160_tasks.json
echo "=== overnight bench started $(cat /proc/uptime | cut -d. -f1)s uptime ===" >> "$SUMMARY"

run_mbpp () {  # arm out mode scaling
  local arm=$1 out=$2 mode=$3 scal=$4
  echo ">>> MBPP-160 $arm ($mode @${scal})" | tee -a "$SUMMARY"
  tools/run_guarded.sh "$OUT/mbpp_${arm}.log" \
    tools/_goal3_multiturn_probe.py run \
    --arm "$( [ "$scal" = 0.0 ] && echo scale0 || echo c3 )" --tasks "$MBPP" \
    --sessions "$OUT/mbpp_${arm}_sessions" --out "$out" \
    --seed 0 --max-iters 12 --prompt-mode "$mode" --adapter-scaling "$scal"
  # the guarded wrapper backgrounds; wait for completion
  while pgrep -f "_goal3_multiturn_probe.*$out" >/dev/null; do sleep 20; done
  python3 -c "import json;d=json.load(open('$out'));print(f'MBPP-160 $arm: pass@1 {d[\"passed_tasks\"]}/{d[\"total_tasks\"]} = {d[\"pass_at_1\"]:.3f}')" >> "$SUMMARY" 2>>"$SUMMARY"
}

run_lcb () {  # arm mode scaling
  local arm=$1 mode=$2 scal=$3 gens="$OUT/lcb_${arm}.json"
  # functional-only: the engine is function-oriented; LCB stdin problems have no
  # entry_point and are not meaningfully supported (separate work).
  echo ">>> LCB v6 functional (49) $arm ($mode @${scal})" | tee -a "$SUMMARY"
  tools/run_guarded.sh "$OUT/lcb_${arm}.log" \
    tools/_lcb_run.py --arm "$( [ "$scal" = 0.0 ] && echo scale0 || echo c3 )" \
    --prompt-mode "$mode" --adapter-scaling "$scal" --out "$gens" --max-iters 12 \
    --functional-only
  while pgrep -f "_lcb_run.*$gens" >/dev/null; do sleep 20; done
  # official grade in the isolated lcbenv
  PYTHONPATH=/tmp/LiveCodeBench /tmp/lcbenv/bin/python tools/_lcb_grade.py --gens "$gens" \
    >> "$SUMMARY" 2>>"$OUT/lcb_${arm}_grade.log"
}

# Priority order (most decision-relevant first):
run_mbpp base     "$OUT/mbpp_base.json"     full     0.0
run_mbpp escalate "$OUT/mbpp_escalate.json" escalate 0.627
run_lcb  escalate escalate 0.627
run_lcb  base     full     0.0

echo "=== overnight bench DONE ===" | tee -a "$SUMMARY"
echo "AGENT_LOOP_WAKE_overnight_done {\"prompt\":\"Overnight bench done. Read /tmp/goal3/overnight/SUMMARY.txt and record findings (MBPP base vs escalate, LCB escalate vs published 35.1) in PR #55.\"}"
