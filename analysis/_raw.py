import json
from pathlib import Path
base = Path("/tmp/goal3/hard_sessions/multistep")
for task in ["int_to_roman","merge_intervals"]:
    recs=[json.loads(l) for l in (base/task/"session.jsonl").read_text().splitlines() if l.strip()]
    code_steps=[r for r in recs if r["action"] in ("code","repair")]
    r=code_steps[0]
    out=r["output"] or ""
    print("="*70)
    print(task, "step", r["step"], r["action"])
    print("  output has literal backslash-n:", "\\n" in out)
    print("  output has real newline:", "\n" in out)
    print("  repr(first 160):", repr(out[:160]))
