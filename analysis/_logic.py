import json
from pathlib import Path
from rune.engine.continuation import strip_self_tests
base = Path("/tmp/goal3/hard_sessions_fix/multistep")
tests = {
 "calculate": 'assert calculate("2+3*4")==14\nassert calculate("(2+3)*4")==20\nassert calculate("10-2*3")==4',
 "decode_string": 'assert decode_string("3[a]2[bc]")=="aaabcbc"\nassert decode_string("3[a2[c]]")=="accaccacc"',
}
for task in ["calculate","decode_string"]:
    recs=[json.loads(l) for l in (base/task/"session.jsonl").read_text().splitlines() if l.strip()]
    code=[r for r in recs if r["action"] in ("code","repair")][-1]["output"]
    code=strip_self_tests(code)
    print("="*70,"\n",task,"-- module-load only (engine's in-loop check):")
    ns={}
    try:
        exec(compile(code,"<c>","exec"),ns); print("  module load: OK (exit 0) -> engine marks PASSED")
    except Exception as e:
        print("  module load FAILED:", type(e).__name__, e)
    print("  vs public+held-out tests (scoring):")
    try:
        exec(tests[task], ns); print("  TESTS PASS")
    except AssertionError as e:
        print("  AssertionError ->", repr(str(e)) or "(wrong output)")
    except Exception as e:
        print("  ", type(e).__name__, e)
