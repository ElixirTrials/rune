import json, re, ast
from pathlib import Path
from rune.engine.continuation import strip_self_tests
named={t['task_id'].split('/')[-1]:t for t in json.load(open('benchmarks/goal3_multistep_tasks.json'))}
opaque={t['task_id'].split('/')[-1]:t for t in json.load(open('benchmarks/goal3_multistep_opaque.json'))}
allt={**named,**opaque}
base=Path('/tmp/goal3/epi_hard_c3e_sessions/multistep')
for name in ['calculate','calculate_opaque','decode_string_opaque','int_to_roman_opaque','merge_intervals_opaque']:
    recs=[json.loads(l) for l in (base/name/'session.jsonl').read_text().splitlines() if l.strip()]
    task=allt[name]; ep=task['entry_point']
    actions=[r['action'] for r in recs]
    repair_engaged = 'repair' in actions
    # acceptance_check authored by decompose
    dec=[r for r in recs if r['action']=='decompose'][0]
    accs=re.findall(r'"acceptance_check": "((?:[^"\\]|\\.)*)"', dec['output'])
    # in-loop: did the final code pass the acceptance_check (oracle)?
    code=strip_self_tests([r for r in recs if r['action'] in ('code','repair','integrate')][-1]['output'])
    # first held-out failure: input/expected/actual
    ns={}
    loaderr=None
    try: exec(code,ns)
    except Exception as e: loaderr=f'{type(e).__name__}:{e}'
    firstfail=None
    if not loaderr:
        for line in task['test_code'].splitlines():
            line=line.strip()
            if not line.startswith('assert'): continue
            try: exec(line,ns)
            except Exception as e:
                # extract the call and expected
                firstfail=line; break
    # classify
    n_repairs=actions.count('repair')
    if not repair_engaged:
        cause="ORACLE-INSUFFICIENT: passed its acceptance_check attempt-1, engine stopped, never repaired the held-out bug"
    else:
        cause=f"REPAIR-EXHAUSTED ({n_repairs} repairs): failed its own check, repair couldn't fix in budget"
    print(f"\n=== {name} (ep={ep}) ===")
    print(f"  steps={len(recs)} actions={actions}")
    print(f"  acceptance_check(s) authored: {accs}")
    print(f"  first held-out FAIL: {firstfail}")
    print(f"  >>> ROOT CAUSE: {cause}")
