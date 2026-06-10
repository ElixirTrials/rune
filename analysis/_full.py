import json
from pathlib import Path
from rune.engine.continuation import strip_self_tests
base = Path("/tmp/goal3/hard_sessions_fix/multistep")
tasks = {t["task_id"].split("/")[-1]: t for t in json.load(open("benchmarks/goal3_multistep_tasks.json"))}
for task in ["calculate","decode_string"]:
    recs=[json.loads(l) for l in (base/task/"session.jsonl").read_text().splitlines() if l.strip()]
    code=strip_self_tests([r for r in recs if r["action"] in ("code","repair")][-1]["output"])
    tc=tasks[task]["test_code"]
    ns={}
    exec(compile(code,"<c>","exec"),ns)
    print("="*60,"\n",task,"-- full held-out tests, per assert:")
    for line in tc.splitlines():
        line=line.strip()
        if not line: continue
        try:
            exec(line, ns); print("  PASS:", line)
        except Exception as e:
            print("  FAIL:", line, "->", type(e).__name__, str(e)[:60]); break
