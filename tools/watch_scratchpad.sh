#!/usr/bin/env bash
# Watch instructions/scratchpad.md for content changes (inode-safe: monitor directory).
# Logs MD5 + timestamp to instructions/.scratchpad_watch.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="${ROOT}/instructions/scratchpad.md"
LOG="${ROOT}/instructions/.scratchpad_watch.log"
last=""
hash_file() {
  if [[ -f "$SCRATCH" ]]; then
    md5sum "$SCRATCH" | awk '{print $1}'
  else
    echo "missing"
  fi
}
emit() {
  local h ts
  h="$(hash_file)"
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  if [[ "$h" != "$last" ]]; then
    echo "${ts} md5=${h} bytes=$(wc -c <"$SCRATCH" 2>/dev/null || echo 0)" >>"$LOG"
    last="$h"
    echo "[scratchpad-watch] ${ts} changed md5=${h}"
  fi
}
mkdir -p "${ROOT}/instructions"
: >>"$LOG"
last="$(hash_file)"
echo "[scratchpad-watch] started $(date -u +%Y-%m-%dT%H:%M:%SZ) initial md5=${last}" | tee -a "$LOG"
if ! command -v inotifywait >/dev/null 2>&1; then
  echo "[scratchpad-watch] inotifywait missing; polling every 30s" | tee -a "$LOG"
  while true; do
    emit
    sleep 30
  done
fi
inotifywait -m -e close_write,move,create,delete --format '%e' "${ROOT}/instructions" 2>/dev/null |
  while read -r _; do
    if [[ -f "$SCRATCH" ]]; then
      emit
    fi
  done
