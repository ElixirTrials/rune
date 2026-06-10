"""Verify the perfect-oracle critique is passed PROPERLY (CPU-only, no model).

REMOVE-BEFORE-MERGE. The first cut of the perfect-oracle probe built malformed
critiques (truncated inputs, empty got/expected on TLEs, mislabeled failure
class, spec duplicated in the adapter channel). Before re-running any GPU probe,
this renders -- WITHOUT the model -- exactly what critique each false-pass task
would receive, and asserts it is well-formed:
  * WRONG  -> non-empty observed + expected; large inputs SUMMARIZED, not truncated.
  * TLE    -> a performance critique naming the input sizes (no bogus expected).
  * CRASH  -> exception + expected.
  * undecodable -> flagged (no critique can be built).
"""

from __future__ import annotations

import ast
import base64
import json
import pickle
import signal
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


def _cases(row: dict) -> list:
    raw = json.loads(row["public_test_cases"]) + _decode_private(
        row["private_test_cases"]
    )
    out = []
    for t in raw:
        try:
            a = [ast.literal_eval(x) for x in t["input"].split("\n") if x.strip()]
            out.append((a, ast.literal_eval(t["output"])))
        except (ValueError, SyntaxError):
            continue
    return out


def _summarize(v: Any) -> str:
    """Faithful-but-compact repr: large lists/strings show head+tail+length, so the
    critique never truncates mid-structure into invalid syntax."""
    if isinstance(v, list) and len(v) > 16:
        head = ", ".join(repr(x) for x in v[:6])
        tail = ", ".join(repr(x) for x in v[-3:])
        return f"[{head}, ... <{len(v)} items> ..., {tail}]"
    if isinstance(v, str) and len(v) > 80:
        return f"{v[:48]!r}...<len {len(v)}>...{v[-12:]!r}"
    return repr(v)


def _args_str(args: list) -> str:
    return ", ".join(_summarize(a) for a in args)


def build_critique(entry: str, kind: str, args: list, got: Any, want: Any) -> str:
    """The corrective signal. One well-formed block per failure type."""
    if kind == "TLE":
        shapes = ", ".join(
            f"len={len(a)}" if isinstance(a, list | str) else repr(a) for a in args
        )
        return (
            f"failure_class: too_slow\n"
            f"observed: {entry}(...) TIMES OUT on an input with sizes ({shapes}); "
            f"constraints allow sizes up to 1e5\n"
            f"fix_directive: keep the SAME correct behavior but lower the time "
            f"complexity."
        )
    if kind.startswith("CRASH"):
        return (
            f"failure_class: runtime_error\n"
            f"observed: {entry}({_args_str(args)}) raises {kind}\n"
            f"expected: {_summarize(want)}\n"
            f"fix_directive: return the expected value without crashing."
        )
    return (
        f"failure_class: wrong_answer\n"
        f"observed: {entry}({_args_str(args)}) -> {_summarize(got)}\n"
        f"expected: {_summarize(want)}\n"
        f"fix_directive: fix the algorithm so observed output matches expected."
    )


def first_failure(fn: Any, cases: list) -> tuple[str, list, Any, Any] | None:
    def _to(_s: int, _f: Any) -> None:
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _to)
    for args, exp in cases:
        signal.alarm(6)
        try:
            got = fn(*args)
            signal.alarm(0)
            if got != exp:
                return ("WRONG", args, got, exp)
        except TimeoutError:
            return ("TLE", args, None, None)
        except Exception as e:  # noqa: BLE001
            signal.alarm(0)
            return ("CRASH:" + type(e).__name__, args, None, exp)
    return None


def main() -> None:
    import sys  # noqa: PLC0415

    sys.path.insert(0, "/workspaces/content/src")
    from rune.bench.lcb import extract_entry_function  # noqa: PLC0415

    rows = {
        json.loads(x)["question_id"]: json.loads(x)
        for x in Path(LCB).read_text().splitlines()
    }
    cands = {
        g["question_id"]: g["code_list"][0]
        for g in json.loads(Path(COMBINED).read_text())
    }

    ok_count, bad = 0, []
    for qid in FALSE_PASS:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        wrong = extract_entry_function(cands.get(qid, ""), entry)
        cases = _cases(row)
        ns: dict[str, Any] = {}
        try:
            exec(wrong, ns)  # noqa: S102
            fn = ns[entry]
        except Exception as e:  # noqa: BLE001
            print(f"{qid} {entry}: LOAD_ERR {type(e).__name__}")
            bad.append(qid)
            continue
        fail = first_failure(fn, cases)
        print(f"\n{'=' * 72}\n{qid}  {entry}")
        if fail is None:
            print(
                "  NO failing case in decoded set -> CANNOT build a critique "
                "(decoder gap; not a clean perfect-oracle subject)"
            )
            bad.append(qid)
            continue
        kind, args, got, want = fail
        crit = build_critique(entry, kind, args, got, want)
        # well-formedness checks
        issues = []
        if "-> \n" in crit or "-> $" in crit + "$":
            issues.append("empty observed value")
        if kind == "WRONG" and (
            "expected: \n" in crit or crit.rstrip().endswith("expected:")
        ):
            issues.append("empty expected value")
        if (
            "<" in _args_str(args)
            and "items>" not in _args_str(args)
            and len(_args_str(args)) >= 200
        ):
            issues.append("possible truncation")
        status = "OK" if not issues else "ISSUES: " + ", ".join(issues)
        if not issues:
            ok_count += 1
        else:
            bad.append(qid)
        print(f"  failure={kind}  -> critique [{status}]:")
        for ln in crit.splitlines():
            print(f"    | {ln}")

    print(
        f"\n{'=' * 72}\nSUMMARY: {ok_count}/{len(FALSE_PASS)} well-formed critiques; "
        f"cannot-test/decoder-gap or issues: {bad}"
    )


if __name__ == "__main__":
    main()
