#!/usr/bin/env bash
# Post-fix MBPP-160 head-to-head: base single-shot (no engine/adapter) vs
# escalate engine (c3 @0.627). Sequential under the RAM watchdog. Pre-fix
# baselines preserved as mbpp_{base,escalate}_prefix.json (0.762 / 0.662).
set -uo pipefail
cd /workspaces/content
OUT=/tmp/goal3/overnight
mkdir -p "$OUT"
C3=/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt
MBPP=benchmarks/mbpp160_tasks.json

echo "[driver] === post-fix bench start $(date -Iseconds) ==="

echo "[driver] base (single-shot ceiling) start $(date -Iseconds)"
tools/run_guarded.sh "$OUT/mbpp_base.log" tools/_capability_ceiling.py \
  --tasks "$MBPP" --out "$OUT/mbpp_base.json" --checkpoint "$C3"
echo "[driver] base done rc=$? $(date -Iseconds)"

echo "[driver] escalate (engine c3 @0.627) start $(date -Iseconds)"
tools/run_guarded.sh "$OUT/mbpp_escalate.log" tools/_goal3_multiturn_probe.py run \
  --arm c3 --tasks "$MBPP" \
  --sessions "$OUT/mbpp_escalate_sessions_postfix" \
  --out "$OUT/mbpp_escalate.json" --seed 0 --max-iters 12 \
  --prompt-mode escalate --adapter-scaling 0.627
echo "[driver] escalate done rc=$? $(date -Iseconds)"

echo "[driver] === RESULTS ==="
python3 - <<'PY'
import json
from pathlib import Path
O = Path("/tmp/goal3/overnight")
def show(name, path):
    try:
        d = json.loads(Path(path).read_text())
        print(f"{name}: {d['passed_tasks']}/{d['total_tasks']} = {d['pass_at_1']:.3f}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
show("base_prefix   (pre-fix)", O/"mbpp_base_prefix.json")
show("escalate_prefix(pre-fix)", O/"mbpp_escalate_prefix.json")
show("base   (post-fix)", O/"mbpp_base.json")
show("escalate(post-fix)", O/"mbpp_escalate.json")
PY
echo "[driver] === post-fix bench DONE $(date -Iseconds) ==="
