import json
from pathlib import Path
recs=[json.loads(l) for l in (Path("/tmp/goal3/oracle_sessions/multistep/decode_string/session.jsonl")).read_text().splitlines() if l.strip()]
for r in recs:
    if r["action"]=="diagnose":
        print("DIAGNOSE PROMPT (error conveyed):")
        print(r["prompt"][:600])
        print("\nDIAGNOSE OUTPUT:", r["output"][:300])
