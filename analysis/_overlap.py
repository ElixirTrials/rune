import json
import sys

sys.path.insert(0, "/workspaces/content/tools")
sys.path.insert(0, "/workspaces/content/src")
sys.path.insert(0, "/tmp/LiveCodeBench")

import _lcb_grade as G  # noqa: E402
from lcb_runner.evaluation.compute_code_generation_metrics import (  # noqa: E402
    codegen_metrics,
)

G.apply_lcb_harness_patches()
rows = {json.loads(line)["question_id"]: json.loads(line) for line in open(G.LCB_JSONL)}


def passing(path):
    gens = json.load(open(path))
    samples = [G._sample(rows[g["question_id"]]) for g in gens]
    gen_list = [
        [
            G.normalize_lcb_submission(
                g["code_list"][0],
                G._entry_point(rows[g["question_id"]]),
                _starter_code=rows[g["question_id"]].get("starter_code", ""),
            )
        ]
        for g in gens
    ]
    _m, results, _ = codegen_metrics(
        samples, gen_list, k_list=[1], num_process_evaluate=8, timeout=6
    )
    ok = set()
    for i, g in enumerate(gens):
        r = results.get(i) if isinstance(results, dict) else results[i]
        try:
            tests = r[0]
            passed = all(x is True or x == 1 for x in tests)
        except Exception:
            passed = False
        if passed:
            ok.add(g["question_id"])
    return ok


base = passing("/tmp/goal3/overnight/lcb_base_zeroshot.json")
rune = passing("/tmp/goal3/overnight/lcb_full_postfix.json")
print("BASE_PASS", len(base), sorted(base))
print("RUNE_PASS", len(rune), sorted(rune))
print("BOTH", len(base & rune), sorted(base & rune))
print("BASE_ONLY (rune lost)", len(base - rune), sorted(base - rune))
print("RUNE_ONLY (rune recovered)", len(rune - base), sorted(rune - base))
