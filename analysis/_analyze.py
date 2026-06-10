"""Compare prefix vs postfix LCB run. Run with project venv (CPU-only ok).
  UV_NO_SYNC=1 uv run --no-sync python /tmp/goal3/overnight/_analyze.py \
    --perqid /tmp/goal3/overnight/perqid_postfix.json \
    --gens /tmp/goal3/overnight/lcb_escalate_postfix.json \
    --sessions /tmp/goal3/overnight/lcb_postfix_sessions \
    --prior /tmp/goal3/overnight/lcb_failure_analysis.json
Reads the per-qid official grade + new sessions and reproduces a comparable
class label for still-failing tasks, then prints the analysis tables.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path


def _last_integrate(events: list[dict]) -> dict | None:
    integ = [e for e in events if e.get("action") == "integrate"]
    return integ[-1] if integ else None


def _feedback_ok(fb_raw: object) -> bool | None:
    """Parse the session feedback (a stringified dict) -> exit_code==0."""
    if not fb_raw:
        return None
    try:
        fb = ast.literal_eval(fb_raw) if isinstance(fb_raw, str) else fb_raw
    except Exception:
        return None
    if isinstance(fb, dict) and "exit_code" in fb:
        return fb["exit_code"] == 0
    return None


def _shipped_code(events: list[dict], gens_code: str) -> str:
    """Prefer the code the gens file shipped; fall back to last integrate/code output."""
    if gens_code and gens_code.strip():
        return gens_code
    integ = _last_integrate(events)
    if integ and integ.get("output", "").strip():
        return integ["output"]
    code_evs = [e for e in events if e.get("action") == "code" and e.get("output", "").strip()]
    return code_evs[-1]["output"] if code_evs else ""


def classify(qid: str, official: str, ship: str, integ_ok: bool | None, fn: str) -> str:
    """Reproduce comparable labels.
    pass            -> official correct (not a failure)
    correct_abstain -> shipped empty/no-fn AND official not pass (engine declined)
    artifact_fn_name_missing -> shipped non-empty but missing `def <fn>`
    model_tle       -> official tle
    engine_oracle_false_pass -> shipped non-empty, fn present, visible integrate passed,
                               but official wrong/runtime (hidden tests fail)
    engine_visible_fail -> shipped non-empty but visible integrate did NOT pass
    """
    if official == "pass":
        return "pass"
    has_code = bool(ship and ship.strip())
    has_fn = bool(fn) and (f"def {fn}" in ship)
    if not has_code:
        return "correct_abstain"
    if fn and not has_fn:
        return "artifact_fn_name_missing"
    if official == "tle":
        return "model_tle"
    if integ_ok is False:
        return "engine_visible_fail"
    # visible passed (or unknown) but hidden failed
    return "engine_oracle_false_pass"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perqid", required=True)
    ap.add_argument("--gens", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--prior", required=True)
    args = ap.parse_args()

    perqid = json.loads(Path(args.perqid).read_text())
    gens = {g["question_id"]: g for g in json.loads(Path(args.gens).read_text())}
    prior = {p["qid"]: p for p in json.loads(Path(args.prior).read_text())}
    sess_root = Path(args.sessions)

    qids = list(prior.keys())
    rows: list[dict] = []
    for qid in qids:
        p = prior[qid]
        fn = p.get("fn", "")
        pg = perqid.get(qid, {"status": "MISSING"})
        official = pg["status"]
        sdir = sess_root / qid
        events: list[dict] = []
        if (sdir / "session.jsonl").is_file():
            events = [json.loads(l) for l in (sdir / "session.jsonl").read_text().splitlines()]
        gcode = gens.get(qid, {}).get("code_list", [""])[0] if qid in gens else ""
        ship = _shipped_code(events, gcode)
        integ = _last_integrate(events)
        integ_ok = _feedback_ok(integ.get("feedback")) if integ else None
        cls = classify(qid, official, ship, integ_ok, fn)
        rows.append(
            {
                "qid": qid,
                "fn": fn,
                "prior_official": p["official"],
                "prior_class": p["class"],
                "post_official": official,
                "post_n": f'{pg.get("n_pass","?")}/{pg.get("n_tests","?")}',
                "post_class": cls,
                "shipped_empty": not (ship and ship.strip()),
                "integrate_visible_pass": integ_ok,
                "n_steps": len(events),
            }
        )

    n = len(rows)
    n_pass = sum(1 for r in rows if r["post_official"] == "pass")
    print(f"=== POST-FIX OFFICIAL on the 49 prior-failures: {n_pass}/{n} ({n_pass/n:.3f}) ===")
    print(f"    prior official on same 49: 0/{n} (0.000)")
    print()

    print("=== WRONG -> CORRECT (newly passing official) ===")
    flips = [r for r in rows if r["prior_official"] != "pass" and r["post_official"] == "pass"]
    if not flips:
        print("    (none)")
    for r in sorted(flips, key=lambda x: x["qid"]):
        print(f"    {r['qid']}  {r['fn']:<32} prior={r['prior_class']:<26} -> pass {r['post_n']}")
    print()

    print("=== CLASS DISTRIBUTION: prior vs post ===")
    prior_dist = Counter(r["prior_class"] for r in rows)
    post_dist = Counter(r["post_class"] for r in rows)
    keys = sorted(set(prior_dist) | set(post_dist))
    print(f"    {'class':<28} {'prior':>6} {'post':>6}")
    for k in keys:
        print(f"    {k:<28} {prior_dist.get(k,0):>6} {post_dist.get(k,0):>6}")
    print()

    print("=== FALSE-PASF COHORT TRANSITIONS (prior engine_oracle_false_pass) ===")
    fp = [r for r in rows if r["prior_class"] == "engine_oracle_false_pass"]
    trans = Counter(r["post_class"] for r in fp)
    print(f"    prior false-pass cohort size: {len(fp)}")
    for k, v in sorted(trans.items()):
        print(f"      now {k:<28} {v}")
    print()

    print("=== FULL PER-TASK TABLE ===")
    hdr = f"{'qid':<6}{'fn':<30}{'prior_class':<26}{'post_off':<9}{'post_n':<8}{'post_class':<26}{'vis_pass':<9}"
    print(hdr)
    for r in sorted(rows, key=lambda x: (x["post_official"] != "pass", x["qid"])):
        print(
            f"{r['qid']:<6}{(r['fn'] or '')[:29]:<30}{r['prior_class']:<26}"
            f"{r['post_official']:<9}{r['post_n']:<8}{r['post_class']:<26}{str(r['integrate_visible_pass']):<9}"
        )

    Path("/tmp/goal3/overnight/analysis_rows.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote /tmp/goal3/overnight/analysis_rows.json")


if __name__ == "__main__":
    main()
