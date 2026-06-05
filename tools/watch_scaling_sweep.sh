#!/usr/bin/env bash
# Watch scaling sweep master log for completion or failure signals.
set -uo pipefail
LOG="${1:?master log required}"
FAIL_PAT='WATCHDOG:|Traceback|CUDA out of memory|Killed|No space left on device'

while true; do
  if [ -f "$LOG" ]; then
    if rg -q "$FAIL_PAT" "$LOG" 2>/dev/null; then
      snippet=$(rg -m1 "$FAIL_PAT" "$LOG" 2>/dev/null | head -c 300 || true)
      printf 'AGENT_LOOP_WAKE_scaling_failures {"prompt":"Scaling sweep failure in %s: %s"}\n' "$LOG" "$snippet"
      exit 0
    fi
    if rg -q 'AGENT_LOOP_WAKE_scaling_complete' "$LOG" 2>/dev/null; then
      printf 'AGENT_LOOP_WAKE_scaling_complete {"prompt":"Scaling sweep done. Compare /tmp/goal3/scale_*.json vs oracle_fix 4/8; pick best for LCB."}\n'
      exit 0
    fi
  fi
  sleep 60
done
