import json
from pathlib import Path
base = Path("/tmp/goal3/hard_sessions_fix/multistep")
for task in ["calculate","decode_string","int_to_roman","merge_intervals"]:
    recs=[json.loads(l) for l in (base/task/"session.jsonl").read_text().splitlines() if l.strip()]
    code_steps=[r for r in recs if r["action"] in ("code","repair")]
    n_litn = sum(1 for r in code_steps if "\\n" in (r["output"] or ""))
    # any diagnose reporting a line-1 syntax error?
    diag_l1 = [r["output"][:120] for r in recs if r["action"]=="diagnose" and ("line 1" in (r["output"] or "").lower() or "line continuation" in (r["output"] or "").lower())]
    actions=[r["action"] for r in recs]
    print(f"{task:16} steps={len(recs)} code/repair-with-literal-\\n={n_litn}  diag-line1={len(diag_l1)}")
