import json
from pathlib import Path
base = Path("/tmp/goal3/oracle_sessions/multistep")
for task in ["calculate","decode_string","int_to_roman","merge_intervals"]:
    recs=[json.loads(l) for l in (base/task/"session.jsonl").read_text().splitlines() if l.strip()]
    seq=[r["action"] for r in recs]
    repaired = "repair" in seq
    # first code step feedback (did oracle produce a real failure signal?)
    code_fb=[r for r in recs if r["action"] in ("code","repair")]
    print(f"\n{task}: steps={len(recs)} seq={seq} repair_engaged={repaired}")
    for r in code_fb:
        fb=r.get("feedback")
        if fb:
            print(f"   {r['action']} exit={fb['exit_code']} stderr={fb['stderr'][:160]!r}")
