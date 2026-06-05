#!/usr/bin/env bash
# Sequential adapter_scaling sweep on hard 8 (OOM rule: one arm at a time).
set -uo pipefail
cd "$(dirname "$0")/.."
SCALINGS=(0.1 0.25 0.4 0.627 0.8 1.0)
for s in "${SCALINGS[@]}"; do
  echo "=== scaling sweep: adapter_scaling=$s ==="
  tools/run_guarded.sh "/tmp/goal3/scale_${s}.log" \
    tools/_goal3_multiturn_probe.py run \
    --arm c3 --tasks benchmarks/goal3_multistep_all8.json \
    --sessions "/tmp/goal3/scale_${s}_sessions" \
    --out "/tmp/goal3/scale_${s}.json" \
    --seed 0 --max-iters 12 --prompt-mode episodic --adapter-scaling "$s"
done
echo "AGENT_LOOP_WAKE_scaling_complete {\"prompt\":\"Scaling sweep done. Compare /tmp/goal3/scale_*.json vs oracle_fix 4/8; pick best scaling for LCB.\"}"
