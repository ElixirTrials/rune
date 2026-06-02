#!/usr/bin/env bash
# Standalone instance guard: always-on safety net so the VM never crashes from OOM or a
# full disk. Polls MemAvailable + free disk every few seconds and logs a heartbeat. On a
# breach it kills the registered guarded jobs (PIDs in $PIDDIR, written by run_guarded.sh),
# newest first, to relieve pressure WITHOUT touching the Claude session or MCP servers.
# As a last resort (no guarded job to kill) it only logs LOUDLY — it never kills random
# processes, since that is more dangerous than the breach itself.
# Usage: nohup tools/instance_guard.sh & disown   (or run via Bash run_in_background)
set -uo pipefail

RAM_MIN_KB="${RUNE_RAM_MIN_KB:-1500000}"     # kill guarded jobs if MemAvailable < ~1.5 GB
DISK_MIN_KB="${RUNE_DISK_MIN_KB:-2500000}"   # kill guarded jobs if free disk < ~2.5 GB
DISK_PATHS="${RUNE_DISK_PATHS:-/ /tmp /workspaces/rune-gpu}"
PIDDIR="${RUNE_GUARD_PIDDIR:-/tmp/rune-guard}"
LOG="${RUNE_GUARD_LOG:-/tmp/rune-guard/instance_guard.log}"
INTERVAL="${RUNE_GUARD_INTERVAL:-5}"
mkdir -p "$PIDDIR"

log() { echo "$(date -u +%H:%M:%S) $*" >>"$LOG"; }

min_disk_free_kb() {
  local lo=999999999 p kb
  for p in $DISK_PATHS; do
    [ -e "$p" ] || continue
    kb=$(df -Pk "$p" 2>/dev/null | awk 'NR==2 {print $4}')
    [ -n "$kb" ] && [ "$kb" -lt "$lo" ] && lo=$kb
  done
  echo "$lo"
}

kill_guarded() {  # reason
  local killed=0 f pid
  # newest pidfile first
  for f in $(ls -t "$PIDDIR"/*.pid 2>/dev/null); do
    pid=$(basename "$f" .pid)
    if kill -0 "$pid" 2>/dev/null; then
      log "BREACH ($1): killing guarded pgroup pid=$pid ($(cat "$f" 2>/dev/null))"
      kill -9 -"$pid" "$pid" 2>/dev/null; killed=1  # whole process group, not just leader
      break  # relieve one job, re-measure next tick
    fi
    rm -f "$f"
  done
  [ "$killed" = 0 ] && log "BREACH ($1) but NO guarded job to kill — VM at risk; freeing disk / reducing load needed"
}

log "instance_guard start pid=$$ ram_min_kb=$RAM_MIN_KB disk_min_kb=$DISK_MIN_KB paths='$DISK_PATHS'"
TICK=0
while true; do
  AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  DISK_KB=$(min_disk_free_kb)
  if [ "$AVAIL_KB" -lt "$RAM_MIN_KB" ]; then kill_guarded "RAM ${AVAIL_KB}kB"; fi
  if [ "$DISK_KB" -lt "$DISK_MIN_KB" ]; then kill_guarded "DISK ${DISK_KB}kB"; fi
  TICK=$((TICK+1))
  # heartbeat every ~60s
  if [ $((TICK % 12)) -eq 0 ]; then log "ok ram_avail=${AVAIL_KB}kB disk_free=${DISK_KB}kB"; fi
  sleep "$INTERVAL"
done
