"""Goal-2 corpus-scaling analysis (issue #52) — REMOVE-BEFORE-MERGE.

Does scaling disjoint TRAINING tasks 40->80->160 keep raising held-out (fixed 24) recall?
Two metrics on the SAME eval set:
  (A) accessibility: absent/body m_zero (mean gold logprob, matched - base) via _specificity_probe.
  (B) functional: k=1 pass@1 (name-cued, spec absent) via _recall_capacity_probe, vs scale=0 floor.

Reads n40 from the Phase-1 artifacts (reuse) and n80/n160/warm from /tmp/goal2.
Run: uv run python tools/_goal2_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SPEC = {
    "warm": "/tmp/goal2/spec_warm.jsonl",
    "n40": "/tmp/phase1/heldout_c3_t07_lp2_lg1.jsonl",
    "n80": "/tmp/goal2/spec_n80.jsonl",
    "n160": "/tmp/goal2/spec_n160.jsonl",
}
CAP = {
    "scale0": "/tmp/cap/scale0.jsonl",
    "warm": "/tmp/cap/warm.jsonl",
    "n40": "/tmp/cap/c3.jsonl",
    "n80": "/tmp/goal2/cap_n80.jsonl",
    "n160": "/tmp/goal2/cap_n160.jsonl",
}


def rows(p: str) -> list[dict]:
    fp = Path(p)
    if not fp.exists():
        return []
    return [json.loads(ln) for ln in fp.read_text().splitlines() if ln.strip()]


def boot_mean_ci(xs: list[float], iters: int = 10000, seed: int = 0) -> tuple:
    import random  # noqa: PLC0415

    if not xs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(xs)
    pt = sum(xs) / n
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (round(pt, 4), round(ms[int(0.025 * iters)], 4), round(ms[int(0.975 * iters)], 4))


def boot_diff_ci(pairs: list[tuple[int, int]], iters: int = 10000, seed: int = 0) -> tuple:
    import random  # noqa: PLC0415

    if not pairs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(pairs)
    pt = sum(a - b for a, b in pairs) / n
    ms = sorted(sum((p := pairs[rng.randrange(n)])[0] - p[1] for _ in range(n)) / n for _ in range(iters))  # noqa: E501,F841
    return (round(pt, 3), round(ms[int(0.025 * iters)], 3), round(ms[int(0.975 * iters)], 3))


def main() -> int:
    print("=== (A) accessibility: absent/body m_zero (matched - base), fixed 24 heldout ===")
    print("  arm  |   mean  [   95% CI       ]  n   lp_m")
    for tag, p in SPEC.items():
        rs = [r for r in rows(p) if r.get("regime") == "absent" and r.get("span") == "body"]
        mz = [r["m_zero"] for r in rs]
        lpm = [r["lp_m"] for r in rs]
        pt, lo, hi = boot_mean_ci(mz)
        lp = round(sum(lpm) / len(lpm), 3) if lpm else None
        print(f"  {tag:>4s} | {pt:+.4f} [{lo:+.4f},{hi:+.4f}] {len(mz):2d}  {lp}")

    print("\n=== (B) k=1 pass@1 (name-cued, spec absent), fixed 24 heldout ===")
    s0 = {r["task_id"]: int(r["pass"]) for r in rows(CAP["scale0"]) if r["k"] == 1}
    print("  arm  | pass@1   rate   delta_vs_scale0 [95% CI]")
    for tag, p in CAP.items():
        rs = [r for r in rows(p) if r.get("k") == 1]
        if not rs:
            print(f"  {tag:>4s} | (no data)")
            continue
        pa = sum(r["pass"] for r in rs)
        n = len(rs)
        d = {r["task_id"]: int(r["pass"]) for r in rs}
        common = sorted(set(d) & set(s0))
        pairs = [(d[t], s0[t]) for t in common]
        if tag == "scale0":
            print(f"  {tag:>4s} | {pa:2d}/{n:2d}   {pa / n:.2f}   (floor)")
        else:
            dpt, dlo, dhi = boot_diff_ci(pairs)
            flag = "  <-- excl 0" if (dlo > 0 or dhi < 0) else ""
            print(f"  {tag:>4s} | {pa:2d}/{n:2d}   {pa / n:.2f}   {dpt:+.3f} [{dlo:+.3f},{dhi:+.3f}]{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
