"""Analyse recall-capacity probe outputs (issue #52 goal-1) — REMOVE-BEFORE-MERGE.

Reads /tmp/cap/{scale0,warm,c3}.jsonl and reports, per k:
  - pass@1 per arm (the capacity decay curve)
  - c3 - scale0 paired delta with a bootstrap CI (paired by task_id at fixed k = the clean control)
  - per-within-block position pass rate (recency effect) for k>1
  - cross-task interference count (emitted def != queried name but == another studied name)

Run: uv run python tools/_capacity_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CAP = Path("/tmp/cap")
ARMS = ["scale0", "warm", "c3"]


def load(arm: str) -> list[dict]:
    p = CAP / f"{arm}.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def boot_ci(pairs: list[tuple[int, int]], iters: int = 10000, seed: int = 0) -> tuple:
    """Bootstrap CI of mean(a - b) over paired (a,b) per-task pass indicators."""
    import random  # noqa: PLC0415

    if not pairs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(pairs)
    point = sum(a - b for a, b in pairs) / n
    means = []
    for _ in range(iters):
        s = sum((p := pairs[rng.randrange(n)])[0] - p[1] for _ in range(n)) / n  # noqa: F841
        means.append(s)
    means.sort()
    return (round(point, 3), round(means[int(0.025 * iters)], 3), round(means[int(0.975 * iters)], 3))


def main() -> int:
    data = {arm: load(arm) for arm in ARMS}
    ks = sorted({r["k"] for arm in ARMS for r in data[arm]})
    if not ks:
        print("no data in /tmp/cap — run tools/_run_capacity_arms.sh first", flush=True)
        return 1

    def rate(arm: str, k: int) -> tuple[int, int]:
        rows = [r for r in data[arm] if r["k"] == k]
        return sum(r["pass"] for r in rows), len(rows)

    print("=== capacity decay: pass@1 per arm per k ===")
    hdr = "  k  | " + " | ".join(f"{a:>12s}" for a in ARMS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in ks:
        cells = []
        for a in ARMS:
            p, n = rate(a, k)
            cells.append(f"{p:2d}/{n:2d} {p / n if n else 0:.2f}" if n else "   -   ")
        print(f"  {k:2d} | " + " | ".join(f"{c:>12s}" for c in cells))

    def paired(arm_a: str, arm_b: str) -> None:
        print(f"\n=== {arm_a} - {arm_b} paired delta (per task at fixed k) ===")
        for k in ks:
            da = {r["task_id"]: int(r["pass"]) for r in data[arm_a] if r["k"] == k}
            db = {r["task_id"]: int(r["pass"]) for r in data[arm_b] if r["k"] == k}
            common = sorted(set(da) & set(db))
            pairs = [(da[t], db[t]) for t in common]
            pt, lo, hi = boot_ci(pairs)
            flag = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
            print(f"  k={k:2d}: delta={pt:+.3f} CI[{lo:+.3f},{hi:+.3f}] n={len(pairs)}{flag}")

    paired("c3", "scale0")
    paired("c3", "warm")
    paired("warm", "scale0")

    print("\n=== per-position pass rate (recency; k>1) — c3 ===")
    for k in [k for k in ks if k > 1]:
        rows = [r for r in data["c3"] if r["k"] == k]
        byp: dict[int, list[int]] = {}
        for r in rows:
            byp.setdefault(r["pos"], []).append(int(r["pass"]))
        cells = " ".join(f"pos{p}={sum(v)}/{len(v)}" for p, v in sorted(byp.items()))
        print(f"  k={k:2d}: {cells}")

    print("\n=== interference (emitted == another studied fn's name) ===")
    for a in ARMS:
        for k in [k for k in ks if k > 1]:
            rows = [r for r in data[a] if r["k"] == k]
            inter = 0
            byblock: dict[int, set] = {}
            for r in rows:
                byblock.setdefault(r["block"], set()).add(r["entry_point"])
            for r in rows:
                names = byblock[r["block"]] - {r["entry_point"]}
                if r.get("emitted_def") in names:
                    inter += 1
            if rows:
                print(f"  {a:>8s} k={k:2d}: {inter}/{len(rows)}")

    # prompt-budget invariance check (the thesis surface).
    pt = [r["prompt_tokens"] for r in data["c3"]]
    st = [r["study_tokens"] for r in data["c3"] if r.get("study_tokens") is not None]
    if pt:
        print(f"\n=== budget: prompt_tokens min/max = {min(pt)}/{max(pt)} (flat) ; "
              f"study_tokens min/max = {min(st)}/{max(st)} (grows with k)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
