"""T0 paired significance from a _feedback_swap_eval.py --out dump (CPU-only).

Predeclared (FROZEN 2026-06-02): paired bootstrap 95% CI + sign test on the
per-episode paired difference d_i = arm2.matched_swap - arm1.matched_swap
(trained - warm-start) over the shared eligible set, plus a row-level scatter
summary (heavy-tailed margins -> not a t-test alone).

Go/no-go (calibration-ladder units; NIAH +7.7 NOT used):
  WIN  : bootstrap 95% CI on mean(d) excludes 0 AND sign-test p<0.05 AND arm2 mean
         matched_swap clears >= rung-1 body recall (+0.14).
  NULL : CI includes 0 OR sign test n.s.  (T0 closes significance, not the product
         decision — that is E1/E2.)

  uv run python tools/_feedback_swap_stats.py /tmp/t0_dump.jsonl
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

RUNG1_BODY = 0.14  # calibration ladder rung 1 (qwen body recall)


def sign_test_two_sided(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    # two-sided exact binomial p at p0=0.5
    tail = sum(math.comb(n, j) for j in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def boot_ci(d: np.ndarray, reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    means = d[rng.integers(0, n, size=(reps, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/t0_dump.jsonl"
    rows = [json.loads(line) for line in open(path)]
    scored = [r for r in rows if r.get("eligible") and "arm1" in r and "arm2" in r]
    print(f"[data] {len(rows)} rows, {len(scored)} eligible with both arms\n")
    if not scored:
        print("no paired rows — is --ckpt2 set?")
        return 1

    a1 = np.array([r["arm1"]["matched_swap"] for r in scored])
    a2 = np.array([r["arm2"]["matched_swap"] for r in scored])
    d = a2 - a1

    print(f"  arm1 (warm-start): matched_swap mean={a1.mean():+.4f} frac(>0)={(a1>0).mean():.2f}")
    print(f"  arm2 (trained)   : matched_swap mean={a2.mean():+.4f} frac(>0)={(a2>0).mean():.2f}")
    print(f"  arm2 matched_zero mean={np.mean([r['arm2']['matched_zero'] for r in scored]):+.4f}")
    print(f"\n  PAIRED d = arm2 - arm1: mean={d.mean():+.4f}  n={len(d)}")

    lo, hi = boot_ci(d)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    p = sign_test_two_sided(pos, neg)
    print(f"  bootstrap 95% CI (10k): [{lo:+.4f}, {hi:+.4f}]   excludes 0: {not (lo <= 0 <= hi)}")
    print(f"  sign test: +{pos}/-{neg} (ties {len(d)-pos-neg})  two-sided p={p:.4f}")

    # row-level scatter: heavy-tail check
    order = np.argsort(-np.abs(d))
    top = order[:5]
    contrib = np.abs(d[top]).sum() / np.abs(d).sum() if np.abs(d).sum() > 0 else 0.0
    print(f"\n  scatter: top-5 |d| rows contribute {contrib:.0%} of total |d| (broad if low)")
    for j in top:
        r = scored[j]
        print(f"    row {r['row_idx']:>3} ({r['task_id']}): arm1={a1[j]:+.3f} arm2={a2[j]:+.3f} d={d[j]:+.3f}")

    ci_excl = not (lo <= 0 <= hi)
    win = ci_excl and p < 0.05 and a2.mean() >= RUNG1_BODY
    print(f"\n=== VERDICT: {'WIN' if win else 'NULL/NO-GO'} ===")
    print(f"  CI excludes 0: {ci_excl} | sign p<0.05: {p < 0.05} | "
          f"arm2 mean >= rung-1 body (+{RUNG1_BODY}): {a2.mean() >= RUNG1_BODY} (mean={a2.mean():+.4f})")
    print("  (T0 closes the significance question only; the lever decision is E1/E2.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
