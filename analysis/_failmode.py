import json, re
from pathlib import Path
from rune.engine.continuation import strip_self_tests
named={t['task_id'].split('/')[-1]:t for t in json.load(open('benchmarks/goal3_multistep_tasks.json'))}
opaque={t['task_id'].split('/')[-1]:t for t in json.load(open('benchmarks/goal3_multistep_opaque.json'))}
allt={**named,**opaque}
base=Path('/tmp/goal3/epi_hard_c3e_sessions/multistep')
fails=['calculate','calculate_opaque','decode_string_opaque','int_to_roman_opaque','merge_intervals_opaque']
for name in fails:
    f=base/name/'session.jsonl'
    if not f.exists(): print(name,'NO SESSION'); continue
    recs=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    task=allt[name]; ep=task['entry_point']
    dec=[r for r in recs if r['action']=='decompose']
    subs=re.findall(r'"name": "([^"]+)"', dec[0]['output']) if dec else []
    # final code
    codes=[r for r in recs if r['action'] in ('code','repair','integrate')]
    final=strip_self_tests(codes[-1]['output'] or '') if codes else ''
    defs=re.findall(r'def (\w+)', final)
    print(f"\n=== {name} (entry_point={ep}) ===")
    print(f"  decompose subtasks={subs}")
    print(f"  final code defines={defs}  | defines_entry_point={ep in defs}")
    # held-out per-assert
    ns={}
    try: exec(final,ns)
    except Exception as e: print(f"  CODE WON'T LOAD: {type(e).__name__}: {str(e)[:60]}"); continue
    tc=task['test_code']; res=[]
    for line in tc.splitlines():
        line=line.strip()
        if not line: continue
        try: exec(line,ns); res.append('P')
        except AssertionError: res.append('A'); break
        except Exception as e: res.append(f'{type(e).__name__}'); break
    print(f"  held-out asserts -> {res}  (A=AssertionError, else exc; first failure stops)")
