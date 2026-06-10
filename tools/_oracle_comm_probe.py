"""Audit: is the perfect critique actually reaching the model intact?

Verbose per-round dump (prompt + raw generation + extraction + changed?) for a
few false-pass tasks, to rule out a model<->oracle COMMUNICATION failure as the
cause of the 0/11 perfect-oracle result. Critique input is NOT truncated here.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"
TASKS = ["3705", "3786", "3760"]  # 2 WRONG + 1 TLE


def _decode_private(s: str) -> list:
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode()))))


def _cases(row: dict) -> list[tuple[list[Any], Any]]:
    tc = json.loads(row["public_test_cases"]) + _decode_private(
        row["private_test_cases"]
    )
    out = []
    for t in tc:
        try:
            a = [ast.literal_eval(x) for x in t["input"].split("\n") if x.strip()]
            out.append((a, ast.literal_eval(t["output"])))
        except (ValueError, SyntaxError):
            continue
    return out


_GRADE = """\
from typing import *
import collections, math, heapq, bisect, itertools, functools, re, signal
from collections import defaultdict, deque, Counter, OrderedDict
import base64
exec(compile(base64.b64decode({code_b64!r}).decode(), '<c>', 'exec'), globals())
cand = globals()[{entry!r}]
cases = {cases!r}
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
first = None
for i, (args, exp) in enumerate(cases):
    signal.alarm(6)
    try:
        got = cand(*args); signal.alarm(0)
        if got != exp:
            first = (i, 'WRONG', repr(args), repr(got), repr(exp)); break
    except TimeoutError:
        first = (i, 'TLE', repr(args)[:200], '', ''); break
    except Exception as e:
        signal.alarm(0)
        first = (i, 'CRASH:' + type(e).__name__, repr(args), '', repr(exp)); break
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
    from rune.bench.lcb import extract_entry_function  # noqa: PLC0415
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.oracle import with_probe_imports  # noqa: PLC0415
    from rune.engine.parse import extract_code_block  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.sandbox.executor import run_in_sandbox  # noqa: PLC0415

    rows = {
        json.loads(ln)["question_id"]: json.loads(ln)
        for ln in Path(LCB).read_text().splitlines()
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

    def grade(code: str, entry: str, cases: list) -> Any:
        nc = extract_entry_function(code, entry)
        if not nc.strip():
            return (None, "EMPTY", "", "", "")
        script = _GRADE.format(
            code_b64=base64.b64encode(nc.encode()).decode(), entry=entry, cases=cases
        )
        out = (
            run_in_sandbox(with_probe_imports(script), timeout=90).stdout or ""
        ).strip()
        if out.startswith("SOLVED"):
            return None
        return eval(out.split(" ", 1)[1])  # noqa: S307

    for qid in TASKS:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:3500]
        cases = _cases(row)
        code = cands.get(qid, "")
        print(
            f"\n{'#' * 80}\n# {qid} {entry}  ({len(cases)} test cases)\n{'#' * 80}",
            flush=True,
        )
        for rnd in range(2):
            fail = grade(code, entry, cases)
            if fail is None:
                print(f"[round {rnd}] SOLVED", flush=True)
                break
            _i, kind, inp, got, want = fail
            print(f"\n--- round {rnd} : failure={kind} ---", flush=True)
            print(f"[CRITIQUE] input(len={len(inp)})={inp[:1200]}", flush=True)
            print(f"[CRITIQUE] got={got[:300]}  want={want[:300]}", flush=True)
            if kind == "TLE":
                crit = f"On a large input your function TIMES OUT. Input: {inp}. Keep the SAME correct behavior but use a faster algorithm."  # noqa: E501
            elif kind.startswith("CRASH"):
                crit = f"On input {inp} your function raises {kind}. The correct answer is {want}. Fix it."  # noqa: E501
            else:
                crit = f"On input {inp} your function returns {got}, but the correct answer is {want}. Fix the logic."  # noqa: E501
            prompt = _REPAIR.format(
                entry=entry,
                spec=spec,
                code=extract_entry_function(code, entry),
                crit=crit,
            )
            print(f"[PROMPT len={len(prompt)}]\n{prompt[:2500]}\n[/PROMPT]", flush=True)
            gen = await model.generate(
                prompt=prompt, max_tokens=1024, temperature=0.3, thinking_budget=0
            )
            print(
                f"[RAW GENERATION len={len(gen.text)}]\n{gen.text[:2500]}\n[/GENERATION]",  # noqa: E501
                flush=True,
            )
            new = extract_code_block(gen.text) or code
            ext = extract_entry_function(new, entry)
            changed = ext.strip() != extract_entry_function(code, entry).strip()
            print(
                f"[EXTRACTED len={len(ext)} changed={changed}]\n{ext[:1200]}",
                flush=True,
            )
            code = new


if __name__ == "__main__":
    asyncio.run(main())
