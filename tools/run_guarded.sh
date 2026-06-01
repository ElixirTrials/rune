#!/usr/bin/env bash
# RAM watchdog: runs a python job under uv, kills it before the ~15GB VM OOMs.
# Usage: tools/run_guarded.sh <logfile> <python-script> [args...]
set -uo pipefail
LOG="${1:?logfile required}"; shift
SCRIPT="${1:?python script required}"; shift
THRESHOLD_KB="${RUNE_RAM_KILL_KB:-13500000}"   # ~13.5 GB RSS+cache ceiling

uv run python "$SCRIPT" "$@" >"$LOG" 2>&1 &
PID=$!
echo "guarded pid=$PID script=$SCRIPT log=$LOG threshold_kb=$THRESHOLD_KB"
while kill -0 "$PID" 2>/dev/null; do
  AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$AVAIL_KB" -lt $((16000000 - THRESHOLD_KB)) ]; then
    echo "WATCHDOG: MemAvailable ${AVAIL_KB}kB too low — killing $PID" | tee -a "$LOG"
    kill -9 "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null
    exit 137
  fi
  sleep 2
done
wait "$PID"; exit $?
