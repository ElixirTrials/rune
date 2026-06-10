"""Offline probe: would a differential-vs-brute-force oracle improve LCB pass@1?

REMOVE-BEFORE-MERGE. Generates a brute-force REFERENCE per task with the BASE
model (no adapter = capability ceiling, optimal for reference correctness),
validates it against the public examples, then differential-tests the already-
shipped candidate (lcb_postfix_combined.json) against the reference on boundary +
random inputs. Reports, over the in-loop-passing cohort:
  - false-positive rate on the 10 official-PASS tasks (must be ~0), and
  - detection rate on the 11 false-pass tasks (in-loop OK, official FAIL).

Adapter sweep: --ref-scaling lets us confirm base (0.0) gives the most-valid refs.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import json
from pathlib import Path
from typing import Any

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"

OFFICIAL_PASS = {
    "3709", "3723", "3736", "3750", "3768", "3773", "3778", "3809", "3817", "3832",
}
FALSE_PASS = {
    "3701", "3705", "3717", "3743", "3754", "3760", "3771", "3777", "3786", "3791",
    "3793",
}

_REF_PROMPT = """Write a SIMPLE, obviously-correct BRUTE-FORCE implementation of \
`{entry}` for the task below. Prioritize correctness above all else: enumerate / \
simulate directly, ignore time and memory limits. Handle edge cases (empty inputs, \
single elements, zeros, negatives, duplicates, boundaries) correctly.

{spec}

Define the function named exactly `{entry}`. Output only the function in one \
```python block."""


def _boundary_inputs(public_calls: list[list[Any]]) -> list[list[Any]]:
    """Edge-case variants derived from the public example argument shapes."""
    out: list[list[Any]] = []
    for args in public_calls:
        out.append(list(args))  # the example itself
        for i, v in enumerate(args):
            def _swap(nv: Any, _i: int = i, _a: list[Any] = args) -> list[Any]:
                c = list(_a)
                c[_i] = nv
                return c

            if isinstance(v, list):
                out += [_swap([]), _swap(v[:1]), _swap(v + v), _swap(sorted(v)),
                        _swap(list(reversed(v))), _swap(v * 3)]
                if v and all(isinstance(x, int) for x in v):
                    out += [_swap([-x for x in v]), _swap([0] * len(v)),
                            _swap([v[0]] * len(v))]
            elif isinstance(v, str):
                out += [_swap(""), _swap(v[:1]), _swap(v + v), _swap(v[::-1])]
            elif isinstance(v, int) and not isinstance(v, bool):
                out += [_swap(0), _swap(1), _swap(-v if v else -1), _swap(v + 1)]
    # de-dup
    seen, uniq = set(), []
    for a in out:
        k = repr(a)
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq[:40]


_DIFF_TEMPLATE = """\
from typing import *
import collections, math, heapq, bisect, itertools, functools, re
from collections import defaultdict, deque, Counter, OrderedDict
import base64
_cand_src = base64.b64decode({cand_b64!r}).decode()
_ref_srcs = [base64.b64decode(b).decode() for b in {ref_b64s!r}]
_cn = {{}}
exec(compile(_cand_src, '<cand>', 'exec'), _cn)
cand = _cn[{entry!r}]
refs = []
for _i, _s in enumerate(_ref_srcs):
    _rn = {{}}
    try:
        exec(compile(_s, '<ref%d>' % _i, 'exec'), _rn)
        refs.append(_rn[{entry!r}])
    except Exception:
        pass
inputs = {inputs!r}
disagree = None
for inp in inputs:
    # consensus over references that ran on this input
    vals = []
    for r in refs:
        try:
            vals.append(repr(r(*inp)))
        except Exception:
            pass
    if len(vals) < 2:
        continue  # not enough references agree-able -> not a trusted oracle point
    from collections import Counter as _C
    top, n = _C(vals).most_common(1)[0]
    if n < 2:
        continue  # no >=2 reference majority -> skip
    try:
        c = repr(cand(*inp))
    except Exception as e:
        disagree = (inp, 'CANDIDATE_CRASH:' + type(e).__name__, top); break
    if c != top:
        disagree = (inp, c, top); break
print('DISAGREE' if disagree else 'AGREE', repr(disagree) if disagree else '')
"""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-scaling", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="/tmp/goal3/overnight/oracle_probe.json")
    args = ap.parse_args()

    from rune.bench.lcb import build_public_assert_checks, extract_entry_function
    from rune.config import load_rune_config
    from rune.engine.oracle import parse_public_call_arglists, with_probe_imports
    from rune.engine.parse import extract_code_block
    from rune.model.wrapper import ModelWrapper
    from rune.sandbox.executor import run_in_sandbox

    rows = {json.loads(line)["question_id"]: json.loads(line)
            for line in Path(LCB).read_text().splitlines()}
    cands = {g["question_id"]: g["code_list"][0]
             for g in json.loads(Path(COMBINED).read_text())}

    cfg = load_rune_config(None).override(
        checkpoint_path="/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt",
        adapter_scaling=0.0,
        model_judge=False,
    )
    model = ModelWrapper.from_config(cfg)

    targets = sorted(OFFICIAL_PASS | FALSE_PASS)
    results = []
    for qid in targets:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:4000]
        public = build_public_assert_checks(row)
        calls = parse_public_call_arglists(public, entry) if (public and entry) else []
        # --- generate K brute-force references (BASE: no adapter applied) ---
        from rune.engine.oracle import build_subtask_probe
        valid_refs: list[str] = []
        for k in range(args.k):
            gen = await model.generate(
                prompt=_REF_PROMPT.format(entry=entry, spec=spec),
                max_tokens=1024, temperature=0.2 if k == 0 else 0.6, thinking_budget=0,
            )
            rc = extract_entry_function(extract_code_block(gen.text), entry)
            if not rc.strip() or not public:
                continue
            probe, fired = build_subtask_probe(rc, public)
            if fired and run_in_sandbox(probe, timeout=5).exit_code == 0:
                valid_refs.append(rc)
        ref_valid = len(valid_refs) >= 2  # need a consensus quorum
        # --- consensus differential test ---
        flagged, detail = False, ""
        cand_code = extract_entry_function(cands.get(qid, ""), entry)
        if ref_valid and cand_code.strip() and calls:
            inputs = _boundary_inputs(calls)
            script = _DIFF_TEMPLATE.format(
                cand_b64=base64.b64encode(cand_code.encode()).decode(),
                ref_b64s=[base64.b64encode(r.encode()).decode() for r in valid_refs],
                entry=entry, inputs=inputs,
            )
            res = run_in_sandbox(with_probe_imports(script), timeout=20)
            out = (res.stdout or "").strip()
            flagged = out.startswith("DISAGREE")
            detail = out[:160]
        cohort = "PASS" if qid in OFFICIAL_PASS else "FALSE_PASS"
        results.append({"qid": qid, "entry": entry, "cohort": cohort,
                        "ref_valid": ref_valid, "flagged": flagged, "detail": detail})
        print(f"{qid} {cohort:10s} ref_valid={ref_valid!s:5s} flagged={flagged!s:5s} "
              f"{detail}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    fp = [r for r in results if r["cohort"] == "PASS" and r["flagged"]]
    det = [r for r in results if r["cohort"] == "FALSE_PASS" and r["flagged"]]
    rv = sum(r["ref_valid"] for r in results)
    print(f"\n=== ref_scaling={args.ref_scaling} ===")
    print(f"reference valid: {rv}/{len(results)}")
    print(f"FALSE-POSITIVE (PASS tasks flagged): {len(fp)}/{len(OFFICIAL_PASS)} "
          f"{[r['qid'] for r in fp]}")
    print(f"DETECTION (false-pass tasks flagged): {len(det)}/{len(FALSE_PASS)} "
          f"{[r['qid'] for r in det]}")


if __name__ == "__main__":
    asyncio.run(main())
