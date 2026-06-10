import json
from pathlib import Path
base = Path("/tmp/goal3/hard_sessions_fix/multistep")
for task in ["calculate","decode_string"]:
    print("="*80,"\nTASK:",task)
    recs=[json.loads(l) for l in (base/task/"session.jsonl").read_text().splitlines() if l.strip()]
    for r in recs:
        fb=r.get("feedback")
        fbs = f"exit={fb['exit_code']} stderr={fb['stderr'][:120]!r}" if fb else "None"
        print(f"  step {r['step']} {r['action']:10} target={r['target']!r}")
        print(f"     feedback: {fbs}")
        if r["action"] in ("code","repair"):
            print("     code:", (r["output"] or "")[:240].replace("\n","\\n"))
