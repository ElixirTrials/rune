#!/usr/bin/env bash
# Watch overnight bench orchestrator: post PR #55 comments on arm completion,
# failure signals, or full run done. Idempotent via state file.
# Usage: tools/watch_overnight_bench.sh [pr_number]
set -uo pipefail
OUT=/tmp/goal3/overnight
SUMMARY="$OUT/SUMMARY.txt"
STATE="$OUT/monitor_state.txt"
MONLOG="$OUT/monitor.log"
PR="${1:-55}"
FAIL_PAT='WATCHDOG:|Traceback|CUDA out of memory|Killed|No space left on device|Error:|exhausted all.*Killed'

mkdir -p "$OUT"
touch "$STATE" "$MONLOG"

post_pr() {
  local tag="$1"
  local body="$2"
  if is_done "pr:$tag"; then
    return 0
  fi
  if gh pr comment "$PR" --body "$body" >>"$MONLOG" 2>&1; then
    mark_done "pr:$tag"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] posted pr:$tag" >>"$MONLOG"
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] FAILED pr:$tag" >>"$MONLOG"
  fi
}

mark_done() { echo "$1" >>"$STATE"; }
is_done() { grep -qxF "$1" "$STATE" 2>/dev/null; }

score_line() {
  local json="$1" label="$2"
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(f'{sys.argv[2]}: **{d[\"passed_tasks\"]}/{d[\"total_tasks\"]} = {d[\"pass_at_1\"]:.3f}**')
" "$json" "$label" 2>/dev/null || echo "$label: (score parse failed)"
}

check_failures() {
  local arm="$1" log="$2"
  [ -f "$log" ] || return 0
  if is_done "fail:$arm"; then
    return 0
  fi
  if rg -q "$FAIL_PAT" "$log" 2>/dev/null; then
    local snippet
    snippet=$(rg -m1 "$FAIL_PAT" "$log" 2>/dev/null | head -c 400 || true)
    post_pr "fail-$arm" "$(cat <<EOF
### Overnight bench — **failure signal** ($arm)

Log: \`$log\`

\`\`\`
$snippet
\`\`\`

Orchestrator may continue to the next arm. GPU: \`$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo n/a)\`
EOF
)"
    mark_done "fail:$arm"
  fi
}

while true; do
  # --- MBPP base (single-shot ceiling) ---
  if [ -f "$OUT/mbpp_base.json" ] && [ -s "$OUT/mbpp_base.json" ] && ! is_done "arm:mbpp_base"; then
    mark_done "arm:mbpp_base"
    post_pr "mbpp_base" "$(cat <<EOF
### Overnight bench — **MBPP-160 base complete** (single-shot, no engine)

$(score_line "$OUT/mbpp_base.json" "MBPP-160 base")

- Out: \`$OUT/mbpp_base.json\`
- Log: \`$OUT/mbpp_base.log\`
- Next: MBPP-160 escalate (engine, c3 @0.627)
EOF
)"
  fi

  check_failures "mbpp_base" "$OUT/mbpp_base.log"

  # --- MBPP escalate (engine) ---
  if [ -f "$OUT/mbpp_escalate.json" ] && [ -s "$OUT/mbpp_escalate.json" ] && ! is_done "arm:mbpp_escalate"; then
    mark_done "arm:mbpp_escalate"
    post_pr "mbpp_escalate" "$(cat <<EOF
### Overnight bench — **MBPP-160 escalate complete** (engine, c3 @0.627)

$(score_line "$OUT/mbpp_base.json" "MBPP-160 base (single-shot)")
$(score_line "$OUT/mbpp_escalate.json" "MBPP-160 escalate (engine)")

- Out: \`$OUT/mbpp_escalate.json\`
- Log: \`$OUT/mbpp_escalate.log\`
- Next: LCB v6 functional escalate (49 tasks, vs published **35.1**)
EOF
)"
  fi

  check_failures "mbpp_escalate" "$OUT/mbpp_escalate.log"

  # --- LCB escalate ---
  if [ -f "$OUT/lcb_escalate.json" ] && [ -s "$OUT/lcb_escalate.json" ] && ! is_done "arm:lcb_run"; then
    mark_done "arm:lcb_run"
    post_pr "lcb_run" "$(cat <<EOF
### Overnight bench — **LCB v6 functional run complete** (escalate, c3 @0.627)

Generation finished; official grade pending.

- Out: \`$OUT/lcb_escalate.json\`
- Log: \`$OUT/lcb_escalate.log\`
EOF
)"
  fi

  check_failures "lcb_escalate" "$OUT/lcb_escalate.log"

  # --- Full orchestrator done (includes LCB grade line in SUMMARY) ---
  if [ -f "$SUMMARY" ] && rg -q 'overnight bench DONE' "$SUMMARY" 2>/dev/null && ! is_done "arm:done"; then
    mark_done "arm:done"
    post_pr "done" "$(cat <<EOF
### Overnight bench — **DONE**

\`\`\`
$(cat "$SUMMARY")
\`\`\`

**Compare:** MBPP base (single-shot) vs escalate (engine); LCB functional escalate vs published Qwen3-4B **35.1**.

Artifacts: \`$OUT/\`
EOF
)"
    printf 'AGENT_LOOP_WAKE_overnight_done {"prompt":"Overnight bench done. Read %s and record findings in PR #%s."}\n' "$SUMMARY" "$PR"
    exit 0
  fi

  # --- Orchestrator died without finishing ---
  if ! pgrep -f '_overnight_bench.sh' >/dev/null && ! is_done "arm:done"; then
    if [ -f "$SUMMARY" ] && ! rg -q 'overnight bench DONE' "$SUMMARY" 2>/dev/null; then
      if ! is_done "orch:died"; then
        mark_done "orch:died"
        post_pr "orch_died" "$(cat <<EOF
### Overnight bench — **orchestrator exited early**

SUMMARY so far:
\`\`\`
$(cat "$SUMMARY" 2>/dev/null || echo "(empty)")
\`\`\`

No running \`_overnight_bench.sh\` process. Check \`$OUT/*.log\` and GPU state.
EOF
)"
        exit 1
      fi
    fi
  fi

  sleep 60
done
