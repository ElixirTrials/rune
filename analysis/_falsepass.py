import json, os, sys
sys.path.insert(0, "/workspaces/content/tools"); sys.path.insert(0, "/workspaces/content/src"); sys.path.insert(0, "/tmp/LiveCodeBench")
import _lcb_grade as G
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
G.apply_lcb_harness_patches()
rows={json.loads(l)["question_id"]:json.loads(l) for l in open(G.LCB_JSONL)}
gens=json.load(open("/tmp/goal3/overnight/lcb_postfix_combined.json"))
samples=[G._sample(rows[g["question_id"]]) for g in gens]
gl=[[G.normalize_lcb_submission(g["code_list"][0], G._entry_point(rows[g["question_id"]]), _starter_code=rows[g["question_id"]].get("starter_code",""))] for g in gens]
_m,res,_=codegen_metrics(samples,gl,k_list=[1],num_process_evaluate=8,timeout=6)
official={}
for i,g in enumerate(gens):
    r=res.get(i) if isinstance(res,dict) else res[i]
    try: official[g["question_id"]]=all(x is True or x==1 for x in r[0])
    except Exception: official[g["question_id"]]=False
# in-loop pass from session metadata
def inloop(qid):
    for d in ("lcb_fix3_sessions","lcb_full_postfix_sessions"):
        mp=f"/tmp/goal3/overnight/{d}/{qid}/metadata.json"
        if os.path.exists(mp): return bool(json.load(open(mp)).get("pass_at_1"))
    return None
qids=[g["question_id"] for g in gens]
il={q:inloop(q) for q in qids}
inloop_pass={q for q in qids if il[q]}
off_pass={q for q in qids if official[q]}
falsepass=inloop_pass - off_pass
inloop_fail={q for q in qids if il[q] is False}
print("in-loop pass:", len(inloop_pass))
print("official pass:", len(off_pass))
print("FALSE-PASS (in-loop OK, official FAIL):", len(falsepass), sorted(falsepass))
print("in-loop FAIL (model can't pass even the 2-4 examples):", len(inloop_fail))
