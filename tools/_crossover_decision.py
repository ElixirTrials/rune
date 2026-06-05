"""Go/no-go analysis for the issue #52 body-contrastive cross-over.

Consumes two frozen-probe dumps (warm-start, trained), produced by
  tools/_specificity_probe.py --out <dump.jsonl>
and computes the predeclared decision numbers:

  PRIMARY  (absent, body span): per-episode body m-mismatch for warm-start vs trained,
           the paired trained-minus-warmstart delta, a sign test, and a paired bootstrap
           CI on the per-episode deltas (margins are heavy-tailed -> not a t-test).
  GATE B   (acceptance test): warm-start body m-mismatch must land near the historical
           +0.137 and signature near +3.8..+4.09, else the reconstruction is suspect.
  RETENTION: signature span (absent) must not be traded away by the trained ckpt.

Bar (decision threshold, not truth boundary):
  trained body m-mismatch  +0.137 -> >= +1.0          = reachable (scale to pilot)
  smaller-but-broad: matched>mismatch AND sign test +  = reachable (iterate pilot)
  stays in the +0.14 band (parity passed)              = rethink conditioning path

Run: uv run python tools/_crossover_decision.py /tmp/probe_warmstart.jsonl /tmp/probe_trained.jsonl
Deterministic bootstrap (fixed seed) so the verdict is reproducible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def load(path: str, regime: str, span: str) -> dict[str, dict]:
    """task_id -> row, for the given (regime, span)."""
    out: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["regime"] == regime and r["span"] == span:
            out[r["task_id"]] = r
    return out


def sign_test_p(n_pos: int, n: int) -> float:
    """Two-sided exact binomial sign test p-value (p=0.5)."""
    from math import comb  # noqa: PLC0415

    if n == 0:
        return 1.0
    k = max(n_pos, n - n_pos)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_ci(deltas: list[float], iters: int = 10000) -> tuple[float, float]:
    g = torch.Generator().manual_seed(0)
    t = torch.tensor(deltas, dtype=torch.float64)
    n = len(t)
    means = []
    for _ in range(iters):
        idx = torch.randint(0, n, (n,), generator=g)
        means.append(float(t[idx].mean()))
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return lo, hi


def summarize(tag: str, rows: dict[str, dict]) -> float:
    mm = [r["m_mismatch"] for r in rows.values()]
    mean = sum(mm) / len(mm)
    frac = sum(1 for x in mm if x > 0) / len(mm)
    print(f"  {tag}: mean m-mismatch={mean:+.4f}  frac(>0)={frac:.2f}  n={len(mm)}")
    return mean


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _crossover_decision.py <warmstart_dump> <trained_dump>")
        return 2
    ws_path, tr_path = sys.argv[1], sys.argv[2]

    print("=== PRIMARY: absent / body span ===")
    ws_body = load(ws_path, "absent", "body")
    tr_body = load(tr_path, "absent", "body")
    ws_mean = summarize("warm-start", ws_body)
    tr_mean = summarize("trained   ", tr_body)

    common = sorted(set(ws_body) & set(tr_body))
    deltas = [tr_body[t]["m_mismatch"] - ws_body[t]["m_mismatch"] for t in common]
    print("\n  per-episode body m-mismatch (warm-start -> trained, delta):")
    for t in common:
        w, x = ws_body[t]["m_mismatch"], tr_body[t]["m_mismatch"]
        print(f"    {t:8s} {w:+.4f} -> {x:+.4f}   delta={x - w:+.4f}")
    n_pos = sum(1 for d in deltas if d > 0)
    p = sign_test_p(n_pos, len(deltas))
    lo, hi = bootstrap_ci(deltas)
    dmean = sum(deltas) / len(deltas)
    print(
        f"\n  paired delta mean={dmean:+.4f}  sign test +{n_pos}/-{len(deltas) - n_pos} "
        f"p={p:.3f}  bootstrap95%CI=[{lo:+.4f},{hi:+.4f}]"
    )

    print("\n=== GATE B (acceptance): warm-start must match history ===")
    print(f"  warm-start body m-mismatch = {ws_mean:+.4f}  (historical ~+0.137)")
    ws_sig = load(ws_path, "absent", "sig")
    if ws_sig:
        ws_sig_mean = sum(r["m_mismatch"] for r in ws_sig.values()) / len(ws_sig)
        print(f"  warm-start signature m-mismatch = {ws_sig_mean:+.4f}  (historical +3.8..+4.09)")

    print("\n=== RETENTION: signature not traded away (absent/sig) ===")
    tr_sig = load(tr_path, "absent", "sig")
    if ws_sig and tr_sig:
        ws_s = sum(r["m_mismatch"] for r in ws_sig.values()) / len(ws_sig)
        tr_s = sum(r["m_mismatch"] for r in tr_sig.values()) / len(tr_sig)
        print(f"  signature m-mismatch: warm-start={ws_s:+.4f} -> trained={tr_s:+.4f}")

    print("\n=== VERDICT ===")
    if tr_mean >= 1.0:
        v = "GO (reachable): trained body m-mismatch >= +1.0"
    elif tr_mean > ws_mean and n_pos > len(deltas) - n_pos and p < 0.10:
        v = "WEAK-GO (reachable, iterate pilot): matched>mismatch broadly, sign test +"
    else:
        v = "NO-GO (stays in noise band): rethink conditioning path (NOT proof rank/FT can't work)"
    print(f"  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
