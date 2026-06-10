import json
from pathlib import Path
base = Path("/tmp/goal3/hard_sessions/multistep")
for task in ["int_to_roman","calculate","decode_string"]:
    recs=[json.loads(l) for l in (base/task/"session.jsonl").read_text().splitlines() if l.strip()]
    for r in recs:
        if r["action"]=="diagnose":
            print("="*80,"\n",task,"step",r["step"],"DIAGNOSE PROMPT:")
            print(r["prompt"])
            print("  -> diag output:", r["output"][:200])
            break
