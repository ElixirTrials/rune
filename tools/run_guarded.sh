#!/usr/bin/env bash
# RAM + DISK watchdog: runs a python job under uv, kills it before the ~15GB VM OOMs
# or the disk fills (which also crashes the instance). Registers a pidfile so the
# standalone instance_guard.sh can also see/kill it.
# Usage: tools/run_guarded.sh <logfile> <python-script> [args...]
set -uo pipefail
LOG="${1:?logfile required}"; shift
SCRIPT="${1:?python script required}"; shift

THRESHOLD_KB="${RUNE_RAM_KILL_KB:-13500000}"     # RSS+cache ceiling; kill when MemAvailable < (16G - this)
DISK_MIN_KB="${RUNE_DISK_MIN_KB:-3000000}"       # ~3 GB free floor on the checkpoint filesystems
DISK_PATHS="${RUNE_DISK_PATHS:-/ /tmp /workspaces/rune-gpu}"
PIDDIR="${RUNE_GUARD_PIDDIR:-/tmp/rune-guard}"
mkdir -p "$PIDDIR"

uv run python "$SCRIPT" "$@" >"$LOG" 2>&1 &
PID=$!
PIDFILE="$PIDDIR/$PID.pid"
echo "$SCRIPT" >"$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
echo "guarded pid=$PID script=$SCRIPT log=$LOG ram_thresh_kb=$THRESHOLD_KB disk_min_kb=$DISK_MIN_KB"

min_disk_free_kb() {
  local lo=999999999 p kb
  for p in $DISK_PATHS; do
    [ -e "$p" ] || continue
    kb=$(df -Pk "$p" 2>/dev/null | awk 'NR==2 {print $4}')
    [ -n "$kb" ] && [ "$kb" -lt "$lo" ] && lo=$kb
  done
  echo "$lo"
}

while kill -0 "$PID" 2>/dev/null; do
  AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$AVAIL_KB" -lt $((16000000 - THRESHOLD_KB)) ]; then
    echo "WATCHDOG: MemAvailable ${AVAIL_KB}kB too low — killing $PID" | tee -a "$LOG"
    kill -9 "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; exit 137
  fi
  DISK_KB=$(min_disk_free_kb)
  if [ "$DISK_KB" -lt "$DISK_MIN_KB" ]; then
    echo "WATCHDOG: disk free ${DISK_KB}kB below floor ${DISK_MIN_KB}kB — killing $PID (offload checkpoints to S3)" | tee -a "$LOG"
    kill -9 "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; exit 138
  fi
  sleep 2
done
wait "$PID"; exit $?
