"""Experiment: PERFECT oracle — does the model solve the task with perfect critique?

REMOVE-BEFORE-MERGE. Isolates oracle-bottleneck vs model-capability. For each
failing task we run a repair loop where the ORACLE is ground-truth (the hidden
test cases — legitimate as a DIAGNOSTIC, not a shippable oracle): grade the
candidate per-case, find the first failure (wrong / TLE / crash), feed a perfect
critique (input + correct answer + failure type) to the BASE model, regenerate,
repeat. Reports how many failing tasks the model solves WITH perfect feedback.

Base model (no adapter) is the capability ceiling for repair.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"

FALSE_PASS = [
    "3701",
    "3705",
    "3717",
    "3743",
    "3754",
    "3760",
    "3771",
    "3777",
    "3786",
    "3791",
    "3793",
]


def _decode_private(s: str) -> list:
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode()))))


def _test_cases(row: dict) -> list[tuple[list[Any], Any]]:
    """Decoded (args, expected) pairs from public + private test cases."""
    import ast  # noqa: PLC0415

    tc = json.loads(row["public_test_cases"]) + _decode_private(
        row["private_test_cases"]
    )
    out: list[tuple[list[Any], Any]] = []
    for t in tc:
        try:
            args = [ast.literal_eval(a) for a in t["input"].split("\n") if a.strip()]
            exp = ast.literal_eval(t["output"])
        except (ValueError, SyntaxError):
            continue
        out.append((args, exp))
    return out


_GRADE = """\
from typing import *
import collections, math, heapq, bisect, itertools, functools, re, signal
from collections import defaultdict, deque, Counter, OrderedDict
import base64
exec(compile(base64.b64decode({code_b64!r}).decode(), '<cand>', 'exec'), globals())
cand = globals()[{entry!r}]
cases = {cases!r}
def _h(s, f):
    raise TimeoutError()
signal.signal(signal.SIGALRM, _h)
first = None
for i, (args, exp) in enumerate(cases):
    signal.alarm(6)
    try:
        got = cand(*args)
        signal.alarm(0)
        if got != exp:
            first = (i, 'WRONG', repr(args)[:300], repr(got)[:200], repr(exp)[:200])
            break
    except TimeoutError:
        first = (i, 'TLE', repr(args)[:120], '', ''); break
    except Exception as e:
        signal.alarm(0)
        first = (i, 'CRASH:' + type(e).__name__, repr(args)[:300], '', repr(exp)[:200])
        break
print('SOLVED' if first is None else 'FAIL', repr(first) if first else '')
"""

_REPAIR = """\
You wrote this `{entry}` but it is INCORRECT on a hidden test.

Task:
{spec}

Your current code:
```python
{code}
```

PERFECT CRITIQUE — {crit}

Return a corrected `{entry}` that fixes this and still passes the examples.
Output only the function in one ```python block."""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", default="/tmp/goal3/overnight/perfect_oracle.json")
    args = ap.parse_args()

    from rune.bench.lcb import extract_entry_function  # noqa: PLC0415
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.oracle import with_probe_imports  # noqa: PLC0415
    from rune.engine.parse import extract_code_block  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.sandbox.executor import run_in_sandbox  # noqa: PLC0415

    rows = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in Path(LCB).read_text().splitlines()
    }
    cands = {
        g["question_id"]: g["code_list"][0]
        for g in json.loads(Path(COMBINED).read_text())
    }

    cfg = load_rune_config(None).override(
        checkpoint_path="/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt",
        adapter_scaling=0.0,
        model_judge=False,
    )
    model = ModelWrapper.from_config(cfg)

    def grade(code: str, entry: str, cases: list) -> tuple[bool, Any]:
        nc = extract_entry_function(code, entry)
        if not nc.strip():
            return False, (None, "EMPTY", "", "", "")
        script = _GRADE.format(
            code_b64=base64.b64encode(nc.encode()).decode(), entry=entry, cases=cases
        )
        res = run_in_sandbox(with_probe_imports(script), timeout=90)
        out = (res.stdout or "").strip()
        if out.startswith("SOLVED"):
            return True, None
        try:
            payload = eval(out.split(" ", 1)[1]) if " " in out else None  # noqa: S307
        except Exception:
            payload = ("?", "UNKNOWN", out[:120], "", "")
        return False, payload

    results = []
    for qid in FALSE_PASS:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:4000]
        cases = _test_cases(row)
        code = cands.get(qid, "")
        history = []
        solved, solved_round = False, -1
        for rnd in range(args.rounds):
            ok, fail = grade(code, entry, cases)
            if ok:
                solved, solved_round = True, rnd
                break
            _i, kind, inp, got, want = fail
            history.append(kind)
            if kind == "TLE":
                crit = (
                    f"On a large input your function TIMES OUT. Input (truncated): "
                    f"{inp}. Keep the SAME correct behavior but use a faster "
                    f"algorithm."
                )
            elif kind.startswith("CRASH"):
                crit = (
                    f"On input {inp} your function raises {kind}. The correct "
                    f"answer is {want}. Fix it."
                )
            else:
                crit = (
                    f"On input {inp} your function returns {got}, but the correct "
                    f"answer is {want}. Fix the logic."
                )
            gen = await model.generate(
                prompt=_REPAIR.format(
                    entry=entry,
                    spec=spec,
                    code=extract_entry_function(code, entry),
                    crit=crit,
                ),
                max_tokens=1024,
                temperature=0.3,
                thinking_budget=0,
            )
            code = extract_code_block(gen.text) or code
        results.append(
            {
                "qid": qid,
                "entry": entry,
                "solved": solved,
                "round": solved_round,
                "history": history,
                "final_code": extract_entry_function(code, entry)[:1200],
            }
        )
        print(
            f"{qid} {entry:24s} solved={solved!s:5s} round={solved_round} "
            f"history={history}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(results, indent=2))
    n = sum(r["solved"] for r in results)
    print(
        f"\n=== PERFECT ORACLE: model solved {n}/{len(FALSE_PASS)} false-pass tasks "
        f"with perfect critique (rounds<= {args.rounds}) ==="
    )
    print("solved:", [r["qid"] for r in results if r["solved"]])


if __name__ == "__main__":
    asyncio.run(main())
