#!/usr/bin/env bash
# Watch a guarded probe run: emit AGENT_LOOP_WAKE sentinels on failure signals or completion.
# Usage: tools/watch_probe_run.sh <mode> <logfile> <outfile>
#   mode: failures | complete
set -uo pipefail

MODE="${1:?mode required: failures|complete}"
LOG="${2:?logfile required}"
OUT="${3:?outfile required}"

FAIL_PAT='exhausted all|WATCHDOG:|Traceback|Error:|All repairable subtasks exhausted|budget_remaining.*0|Killed|CUDA out of memory|No space left on device'

case "$MODE" in
  failures)
    while true; do
      if [ -f "$LOG" ]; then
        if rg -q "$FAIL_PAT" "$LOG" 2>/dev/null; then
          snippet=$(rg -m1 "$FAIL_PAT" "$LOG" 2>/dev/null | head -c 300 || true)
          printf 'AGENT_LOOP_WAKE_oracle_failures {"prompt":"Probe run failure/budget signal in %s. Snippet: %s. Inspect log and sessions; report to scratchpad."}\n' \
            "$LOG" "$snippet"
          exit 0
        fi
      fi
      if [ -f "$OUT" ]; then
        exit 0
      fi
      sleep 30
    done
    ;;
  complete)
    while true; do
      if [ -f "$OUT" ] && [ -s "$OUT" ]; then
        result=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"pass@1={d.get('pass_at_1',0):.3f} ({d.get('passed_tasks',0)}/{d.get('total_tasks',0)})\")" "$OUT" 2>/dev/null || echo "done")
        printf 'AGENT_LOOP_WAKE_oracle_complete {"prompt":"Oracle-fix hard run complete: %s. Score vs 3/8 baseline, append scratchpad, proceed to scaling sweep (#1). Out: %s Log: %s"}\n' \
          "$result" "$OUT" "$LOG"
        exit 0
      fi
      if [ -f "$LOG" ] && rg -q 'pass@1=' "$LOG" 2>/dev/null; then
        line=$(rg -m1 'pass@1=' "$LOG" 2>/dev/null || true)
        printf 'AGENT_LOOP_WAKE_oracle_complete {"prompt":"Oracle-fix hard run complete (log line): %s. Score vs 3/8 baseline, append scratchpad, proceed to scaling sweep (#1). Log: %s"}\n' \
          "$line" "$LOG"
        exit 0
      fi
      sleep 60
    done
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
