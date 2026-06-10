import json, sys
from pathlib import Path
base = Path("/tmp/goal3/hard_sessions/multistep")
def feedback_str(fb):
    if not fb: return None
    # show the parts that convey the failure
    return {k: (v[:300] if isinstance(v,str) else v) for k,v in fb.items()}
for task in ["calculate","decode_string","int_to_roman"]:
    print("\n"+"="*90)
    print("TASK:", task)
    recs=[json.loads(l) for l in (base/task/"session.jsonl").read_text().splitlines() if l.strip()]
    print("steps:", [(r["step"], r["action"], r["target"]) for r in recs])
    for r in recs:
        if r["action"] in ("diagnose","repair"):
            print("\n--- step",r["step"],r["action"],"target=",r["target"],"---")
            if r["action"]=="diagnose":
                print("  FEEDBACK(written into Review Feedback):", json.dumps(feedback_str(r["feedback"]))[:500])
                print("  diag output:", (r["output"] or "")[:300].replace("\n","\\n"))
            else:
                tj=r["trajectory"] or ""
                # show the failure-conveyance sections of the adapter conditioning
                print("  TRAJECTORY (adapter conditioning) len=",len(tj))
                print("  >>>", tj[:900].replace("\n","\\n"))
                print("  repair output:", (r["output"] or "")[:300].replace("\n","\\n"))
