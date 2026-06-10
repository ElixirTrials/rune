import json, ast
from pathlib import Path
from rune.engine.continuation import strip_self_tests
named={t['task_id'].split('/')[-1]:t for t in json.load(open('benchmarks/goal3_multistep_tasks.json'))}
opaque={t['task_id'].split('/')[-1]:t for t in json.load(open('benchmarks/goal3_multistep_opaque.json'))}
allt={**named,**opaque}
base=Path('/tmp/goal3/epi_hard_c3e_sessions/multistep')
classes={'calculate':'ORACLE-INSUFFICIENT','int_to_roman_opaque':'ORACLE-INSUFFICIENT',
         'calculate_opaque':'REPAIR-EXHAUSTED','decode_string_opaque':'REPAIR-EXHAUSTED',
         'merge_intervals_opaque':'REPAIR-EXHAUSTED'}
for name in ['calculate','int_to_roman_opaque','calculate_opaque','decode_string_opaque','merge_intervals_opaque']:
    recs=[json.loads(l) for l in (base/name/'session.jsonl').read_text().splitlines() if l.strip()]
    task=allt[name]
    code=strip_self_tests([r for r in recs if r['action'] in ('code','repair','integrate')][-1]['output'])
    ns={}
    try: exec(code,ns)
    except Exception as e:
        print(f"{name}: code load error {type(e).__name__}"); continue
    print(f"\n=== {name}  [{classes[name]}] ===")
    fails=0
    for line in task['test_code'].splitlines():
        line=line.strip()
        if not line.startswith('assert'): continue
        t=ast.parse(line).body[0].test
        call=ast.unparse(t.left); exp=ast.unparse(t.comparators[0])
        try:
            actual=eval(call, dict(ns))
            ok = actual == eval(exp)
            if not ok:
                print(f"  FAIL: {call} = {actual!r}   (expected {exp})")
                fails+=1
                if fails>=2: break
        except Exception as e:
            print(f"  CRASH: {call} -> {type(e).__name__}: {str(e)[:50]}")
            fails+=1
            if fails>=2: break
