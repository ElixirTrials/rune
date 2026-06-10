"""Precision/recall of a ground-truth-free differential oracle, with FIXED
high-recall input generation.

REMOVE-BEFORE-MERGE. The first gate missed 3705 for two fixable reasons: the
model-proposed-input parser returned 0 inputs, and boundary mutation only changed
ONE argument at a time (never 'all-equal nums AND k=len'). This fixes both:
  * robust JSON parse of model-proposed adversarial inputs,
  * COORDINATED multi-argument mutations (all-equal/empty/extreme lists crossed
    with k in {1, len, len//2, ...}),
and keeps the precision rule: flag ONLY when >=3 diversely-generated references
(each passing all public examples) UNANIMOUSLY agree on a value != candidate.

Measures, over OFFICIAL_PASS (precision: must flag NONE) and FALSE_PASS (recall):
  precision-side false positives, and recall-side detections.
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
K_REFS = 5
MIN_AGREE = 3

OFFICIAL_PASS = [
    "3709",
    "3723",
    "3736",
    "3750",
    "3768",
    "3773",
    "3778",
    "3809",
    "3817",
    "3832",
]
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

_vc = importlib.util.spec_from_file_location(
    "vc", "/workspaces/content/tools/_verify_critique.py"
)
vc = importlib.util.module_from_spec(_vc)
_vc.loader.exec_module(vc)

_PRE = (
    "from typing import *\nimport collections, math, heapq, bisect, itertools, "
    "functools, re\nfrom collections import defaultdict, deque, Counter, OrderedDict\n"
)

_REF_PROMPTS = [
    "Write a SIMPLE, obviously-correct BRUTE-FORCE `{entry}`. Correctness over "
    "speed; enumerate/simulate directly; handle edge cases.",
    "Implement `{entry}` by DIRECTLY SIMULATING the definition the naive way, "
    "ignoring time limits. Correctness only.",
    "Write a reference `{entry}` by exhaustive enumeration. Do not optimize.",
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


def _coordinated(public_calls: list[list[Any]]) -> list[list[Any]]:
    """Multi-arg mutations: cross list-shapes with int values simultaneously."""
    out: list[list[Any]] = []
    for args in public_calls:
        out.append(list(args))
        list_idx = [i for i, v in enumerate(args) if isinstance(v, list)]
        int_idx = [
            i
            for i, v in enumerate(args)
            if isinstance(v, int) and not isinstance(v, bool)
        ]
        # list shape variants
        for li in list_idx:
            base = args[li]
            L = max(len(base), 1)
            shapes = {
                "all_equal": [base[0] if base else 0] * L,
                "all_zero": [0] * L,
                "single": base[:1] or [0],
                "empty": [],
                "sorted": sorted(base) if base else [],
                "reversed": list(reversed(base)),
                "dup2": (base + base) if base else [0, 0],
            }
            for shp in shapes.values():
                c = list(args)
                c[li] = shp
                # coordinate every int arg with this list's length
                for ii in int_idx:
                    for kv in {1, len(shp), max(1, len(shp) // 2), len(shp) or 1}:
                        c2 = list(c)
                        c2[ii] = kv
                        out.append(c2)
                out.append(c)
    # de-dup
    seen, uniq = set(), []
    for a in out:
        try:
            k = repr(a)
        except Exception:  # noqa: BLE001
            continue
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq[:120]


def _parse_model_inputs(text: str) -> list[list[Any]]:
    """Robustly parse a JSON/py array of argument-lists from model output."""
    s = text.strip()
    # strip code fences
    if "```" in s:
        parts = s.split("```")
        for raw in parts:
            p = raw.strip()
            if p.startswith("["):
                s = p
                break
    # find outermost [...]
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j <= i:
        return []
    blob = s[i : j + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(blob)
            if isinstance(v, list):
                return [list(x) for x in v if isinstance(x, list | tuple)]
        except Exception:  # noqa: BLE001
            continue
    # per-line fallback
    out = []
    for raw in blob.splitlines():
        line = raw.strip().rstrip(",")
        if line.startswith("[") and line.endswith("]"):
            try:
                v = ast.literal_eval(line)
                if isinstance(v, list):
                    out.append(list(v))
            except Exception:  # noqa: BLE001
                pass
    return out


async def main() -> None:
    from rune.bench.lcb import (  # noqa: PLC0415
        build_public_assert_checks,
        extract_entry_function,
    )
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.oracle import parse_public_call_arglists  # noqa: PLC0415
    from rune.engine.parse import extract_code_block  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    rows = {
        json.loads(x)["question_id"]: json.loads(x)
        for x in Path(LCB).read_text().splitlines()
    }
    cands = {
        g["question_id"]: g["code_list"][0]
        for g in json.loads(Path(COMBINED).read_text())
    }
    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT, adapter_scaling=0.0, model_judge=False
    )
    model = ModelWrapper.from_config(cfg)

    async def evaluate(qid: str) -> dict:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:3500]
        public = build_public_assert_checks(row)
        pub_calls = (
            parse_public_call_arglists(public, entry) if (public and entry) else []
        )
        pub_cases = []
        for t in json.loads(row["public_test_cases"]):
            try:
                pub_cases.append(
                    (
                        [
                            ast.literal_eval(x)
                            for x in t["input"].split("\n")
                            if x.strip()
                        ],
                        ast.literal_eval(t["output"]),
                    )
                )
            except (ValueError, SyntaxError):
                continue
        cand = _load(extract_entry_function(cands[qid], entry), entry)

        refs = []
        for k in range(K_REFS):
            p = (
                _REF_PROMPTS[k % len(_REF_PROMPTS)].format(entry=entry)
                + f"\n\n{spec}\n\nDefine `{entry}` exactly. Output only the function "
                f"in one ```python block."
            )
            gen = await model.generate(
                prompt=p, max_tokens=1024, temperature=0.2 + 0.12 * k, thinking_budget=0
            )
            rc = extract_entry_function(extract_code_block(gen.text) or "", entry)
            if not rc.strip():
                continue
            try:
                rfn = _load(rc, entry)
            except Exception:  # noqa: BLE001
                continue
            if all(_call(rfn, a)[1] == e for a, e in pub_cases):
                refs.append(rfn)

        jp = (
            f"Candidate `{entry}`:\n```python\n"
            f"{extract_entry_function(cands[qid], entry)}\n```\n\n{spec}\n\n"
            f"Propose adversarial inputs targeting likely bugs (empty, single, "
            f"all-equal, k=len, duplicates, extremes). Output ONLY a JSON array "
            f"where each element is the argument list for one call, e.g. "
            f"[[[0,0,0],3],[[5],1]]. No prose, no markdown."
        )
        gen = await model.generate(
            prompt=jp, max_tokens=512, temperature=0.5, thinking_budget=0
        )
        proposed = _parse_model_inputs(gen.text)
        inputs = _coordinated(pub_calls) + proposed

        flagged = None
        for X in inputs:
            vals = []
            for r in refs:
                st, v = _call(r, X)
                if st == "ok":
                    vals.append(repr(v))
            if len(vals) < MIN_AGREE or len(set(vals)) != 1:
                continue
            st, cv = _call(cand, X)
            if st == "ok" and repr(cv) != vals[0]:
                flagged = (X, repr(cv), vals[0], len(vals))
                break
        return {
            "qid": qid,
            "entry": entry,
            "refs": len(refs),
            "proposed": len(proposed),
            "inputs": len(inputs),
            "flagged": flagged,
        }

    print("cohort  qid   entry                         refs prop inp  verdict")
    fp = tp = 0
    for cohort, ids in (("PASS", OFFICIAL_PASS), ("FALSE", FALSE_PASS)):
        for qid in ids:
            r = await evaluate(qid)
            fl = "FLAG" if r["flagged"] else "-"
            if cohort == "PASS" and r["flagged"]:
                fp += 1
            if cohort == "FALSE" and r["flagged"]:
                tp += 1
            print(
                f"{cohort:6s}  {qid}  {r['entry'][:28]:28s} {r['refs']:3d} "
                f"{r['proposed']:4d} {r['inputs']:4d}  {fl}",
                flush=True,
            )
            if r["flagged"]:
                X, cv, rv, na = r["flagged"]
                print(
                    f"          on {vc._summarize(X)[:90]}: cand={cv[:40]} "
                    f"vs {na} refs={rv[:40]}",
                    flush=True,
                )
    print(
        f"\n=== PRECISION: {fp}/{len(OFFICIAL_PASS)} passing tasks FALSE-FLAGGED "
        f"(want 0) ==="
    )
    print(f"=== RECALL: {tp}/{len(FALSE_PASS)} false-pass tasks detected ===")


if __name__ == "__main__":
    asyncio.run(main())
