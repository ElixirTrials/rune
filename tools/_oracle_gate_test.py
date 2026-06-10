"""Make-or-break gate: flag the true bug (3705) WITHOUT false-flagging the correct,
currently-passing task (3817). Ground-truth-free.

REMOVE-BEFORE-MERGE. The repair-firing fix needs an in-loop oracle that fires on
hidden bugs but never on correct code (a regression on the 10 passing tasks kills
the +1 margin). 3817 is the known landmine (a correct task the differential
oracle false-flagged before). This tests the CONSERVATIVE gate in isolation:

  references : K diversely-generated brute-force solutions, each kept ONLY if it
               passes ALL public examples.
  inputs     : public-derived boundary inputs + model-PROPOSED adversarial inputs.
  flag rule  : UNANIMOUS -- >=3 valid refs all return the same value V, and the
               candidate returns something != V. (Unanimity, not majority, to kill
               the model-correlated-wrong-reference FP.)

PASS = 3817 NOT flagged AND 3705 flagged. If it can't, the approach can't ship.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import signal
from pathlib import Path
from typing import Any

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"
C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
TASKS = ["3817", "3705"]   # must-NOT-flag, should-flag
K_REFS = 6
MIN_AGREE = 3

_vc = importlib.util.spec_from_file_location(
    "vc", "/workspaces/content/tools/_verify_critique.py")
vc = importlib.util.module_from_spec(_vc)
_vc.loader.exec_module(vc)

_PRE = ("from typing import *\nimport collections, math, heapq, bisect, itertools, "
        "functools, re\nfrom collections import defaultdict, deque, Counter, OrderedDict\n")

_REF_PROMPTS = [
    "Write a SIMPLE, obviously-correct BRUTE-FORCE implementation of `{entry}`. "
    "Prioritize correctness over speed; enumerate/simulate directly; handle edge "
    "cases (empty, single, zeros, negatives, duplicates, boundaries).",
    "Implement `{entry}` by DIRECTLY SIMULATING the definition step by step, the "
    "naive way, ignoring all time limits. Correctness only.",
    "Write a reference `{entry}` from scratch by exhaustive enumeration of all "
    "possibilities described in the statement. Do not optimize.",
]


def _load(code: str, entry: str):
    ns: dict[str, Any] = {}
    exec(_PRE, ns)  # noqa: S102
    exec(code, ns)  # noqa: S102
    return ns[entry]


def _call(fn, args, t=4):
    def _to(_s, _f):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _to)
    signal.alarm(t)
    try:
        return ("ok", fn(*args))
    except TimeoutError:
        return ("tle", None)
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__)
    finally:
        signal.alarm(0)


def _boundary(public_calls: list[list[Any]]) -> list[list[Any]]:
    out: list[list[Any]] = []
    for args in public_calls:
        out.append(list(args))
        for i, v in enumerate(args):
            def swap(nv, _i=i, _a=args):
                c = list(_a)
                c[_i] = nv
                return c
            if isinstance(v, list):
                out += [swap([]), swap(v[:1]), swap(v + v), swap(sorted(v)),
                        swap(list(reversed(v)))]
                if v and all(isinstance(x, int) for x in v):
                    out += [swap([0] * len(v)), swap([v[0]] * len(v)),
                            swap([-x for x in v])]
            elif isinstance(v, int) and not isinstance(v, bool):
                out += [swap(0), swap(1), swap(v + 1), swap(max(0, v - 1))]
            elif isinstance(v, str):
                out += [swap(""), swap(v[:1]), swap(v + v)]
    seen, uniq = set(), []
    for a in out:
        k = repr(a)
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq[:50]


async def main() -> None:
    from rune.bench.lcb import (
        build_public_assert_checks,
        extract_entry_function,
    )
    from rune.config import load_rune_config
    from rune.engine.oracle import parse_public_call_arglists
    from rune.engine.parse import extract_code_block
    from rune.model.wrapper import ModelWrapper

    rows = {json.loads(x)["question_id"]: json.loads(x)
            for x in Path(LCB).read_text().splitlines()}
    cands = {g["question_id"]: g["code_list"][0]
             for g in json.loads(Path(COMBINED).read_text())}

    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT, adapter_scaling=0.0, model_judge=False)  # base refs
    model = ModelWrapper.from_config(cfg)

    for qid in TASKS:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:3500]
        public = build_public_assert_checks(row)
        pub_calls = parse_public_call_arglists(public, entry) if (public and entry) else []
        all_cases = vc._cases(row)
        pub_cases = []
        for t in json.loads(row["public_test_cases"]):
            try:
                pub_cases.append(([ast.literal_eval(x) for x in t["input"].split("\n")
                                   if x.strip()], ast.literal_eval(t["output"])))
            except (ValueError, SyntaxError):
                continue
        cand = _load(extract_entry_function(cands[qid], entry), entry)

        # --- generate K diverse references, keep those passing ALL public ---
        refs = []
        for k in range(K_REFS):
            prompt = (_REF_PROMPTS[k % len(_REF_PROMPTS)].format(entry=entry)
                      + f"\n\n{spec}\n\nDefine `{entry}` exactly. Output only the "
                      f"function in one ```python block.")
            gen = await model.generate(prompt=prompt, max_tokens=1024,
                                       temperature=0.2 + 0.12 * k, thinking_budget=0)
            rc = extract_entry_function(extract_code_block(gen.text) or "", entry)
            if not rc.strip():
                continue
            try:
                rfn = _load(rc, entry)
            except Exception:  # noqa: BLE001
                continue
            if all(_call(rfn, a)[1] == e for a, e in pub_cases):  # passes all public
                refs.append(rfn)
        # --- model-proposed adversarial inputs ---
        proposed: list[list[Any]] = []
        jp = (f"Here is a candidate solution for `{entry}`:\n```python\n"
              f"{extract_entry_function(cands[qid], entry)}\n```\n\n{spec}\n\n"
              f"List up to 6 specific argument tuples (valid per the constraints) "
              f"that are tricky edge cases where this code is most likely WRONG. "
              f"Output each as a Python list of the call arguments, one per line, "
              f"no prose.")
        gen = await model.generate(prompt=jp, max_tokens=512, temperature=0.5,
                                   thinking_budget=0)
        for line in gen.text.splitlines():
            line = line.strip().strip("`").lstrip("-").strip()
            if line.startswith("[") and line.endswith("]"):
                try:
                    v = ast.literal_eval(line)
                    if isinstance(v, list):
                        proposed.append(v)
                except (ValueError, SyntaxError):
                    pass

        inputs = _boundary(pub_calls) + proposed
        # --- unanimous differential gate ---
        flagged = None
        for X in inputs:
            vals = []
            for r in refs:
                st, v = _call(r, X)
                if st == "ok":
                    vals.append(repr(v))
            if len(vals) < MIN_AGREE or len(set(vals)) != 1:
                continue  # not enough refs ran, or refs disagree -> not trusted
            st, cv = _call(cand, X)
            if st == "ok" and repr(cv) != vals[0]:
                flagged = (X, repr(cv), vals[0], len(vals))
                break

        verdict = "FLAG" if flagged else "no-flag"
        want = "should-flag" if qid == "3705" else "must-NOT-flag"
        ok = (flagged is not None) == (qid == "3705")
        print(f"\n{qid} {entry}: refs_valid={len(refs)}/{K_REFS} "
              f"proposed_inputs={len(proposed)} boundary={len(_boundary(pub_calls))}")
        print(f"  verdict={verdict}  ({want})  -> {'PASS' if ok else 'FAIL'}")
        if flagged:
            X, cv, rv, na = flagged
            print(f"  disagreement on {vc._summarize(X)[:120]}: cand={cv[:60]} "
                  f"vs {na} unanimous refs={rv[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
