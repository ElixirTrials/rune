"""C2.1 fresh-pool re-estimate of the Phase-1 +0.105 objective effect (issue #52).

De-biases the E-phase1 headline (Δlp_matched +0.105, 17/24 positive, sign-test
p=0.064 — `docs/issue52-experimentation-log.md` E-phase1) by re-estimating the
SAME statistic for the SAME frozen checkpoint on a pool never touched by the
c1-c4 grid-selection event: the 120 tasks of `benchmarks/mbpp_recall_train_160.jsonl`
that are not in `mbpp_recall_train.jsonl` (c3's train40), not in
`mbpp_recall_heldout.jsonl` (the heldout24 the grid was selected on), and not in
the 10 cross-over pilot ids (`configs/issue52_mbpp_body_crossover.jsonl`).
Pool derivation + feasibility: `docs/publication/c01_corpus_lookup.md`.

Methodology (matched to E-phase1 exactly; instrument byte-identical):
- Instrument: `tools/_specificity_probe.py` (the frozen E1 probe restored verbatim
  from git `205fa3d`), invoked per checkpoint with `--ckpt --corpus --out`, just as
  `tools/_phase1_orchestrate.py` did on the heldout24.
- Per-task statistic: Δlp_matched = mean gold logprob of the committed reference
  BODY span (tokens after the `def <entry>(...)` line, MAX_ANS_TOK=96) under the
  matched adapter, ABSENT prompt regime, c3 minus warm-start. Identical to the
  original Δ m-zero (the base lp_z term cancels in the pairing).
- Across-task sign test: exact two-sided binomial on #positive (zeros dropped),
  as in E-phase1 (binom(17,24) -> p=0.064).
- 95% CI: percentile bootstrap of the mean of per-task deltas. DEVIATION: the
  original bootstrap's resample count/seed were session-scratchpad state and are
  not recoverable; this runner pre-registers 10000 resamples, seed 0.

No training, no trajectory generation, no new corpora: every pool row is a
committed benchmark row; both checkpoints are frozen and sha256-verified before
any forward pass.

CPU-only paths (safe while a GPU campaign runs):
  uv run --no-sync python tools/_c21_fresh_pool_run.py --build-pool
  uv run --no-sync python tools/_c21_fresh_pool_run.py --stats-only \
      --c3-dump <jsonl> --warm-dump <jsonl> [--no-mlflow]

Full GPU run (later; ~1-2 GPU-hr, 2 sequential model loads):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-sync python tools/_c21_fresh_pool_run.py \
      --experiment issue52-phase1 --workdir /tmp/c21
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

RUNE = Path("/workspaces/rune-gpu")
TRAIN40 = RUNE / "benchmarks/mbpp_recall_train.jsonl"
TRAIN160 = RUNE / "benchmarks/mbpp_recall_train_160.jsonl"
HELDOUT24 = RUNE / "benchmarks/mbpp_recall_heldout.jsonl"
CROSSOVER10 = RUNE / "configs/issue52_mbpp_body_crossover.jsonl"
POOL_DEFAULT = RUNE / "benchmarks/mbpp_recall_fresh120.jsonl"
PROBE = RUNE / "tools/_specificity_probe.py"

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
C3_SHA256 = "53e24af243a38dfbfad82f7293635bfc592922dd2058fefbbfa10714b5457a3f"
WARM_CKPT = str(
    RUNE / "third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)
WARM_SHA256 = "6438b46c828dd3b5f88f21add0f7f5cacc7994d47bf15eda266786a506044591"

# Committed-file provenance pins (docs/publication/hashes.txt). The pool builder
# refuses to derive from drifted inputs.
TRAIN40_SHA256 = "e60f0dd85fad51142513b7487b680b7486f821b5245212164b1be812b9f860cd"
TRAIN160_SHA256 = "5711834e1ae90ffac77ea60da18ef67824f217a1e2f1858b09be99f164f3c085"
HELDOUT24_SHA256 = "cae274bf1aed31c80d42da82b366d9af727129bbf6aa124afed9643abe762a8f"

EXPERIMENT_DEFAULT = "issue52-phase1"
TRACKING_URI = "http://localhost:5000"


def log(msg: str) -> None:
    print(f"[c21] {msg}", flush=True)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _task_ids(path: Path) -> list[str]:
    return [
        json.loads(ln)["task_id"] for ln in path.read_text().splitlines() if ln.strip()
    ]


def build_pool(out: Path) -> str:
    """Derive the fresh 120-task pool deterministically from committed files.

    Pool = raw train_160 lines (byte-identical, original order) whose task_id is
    not in train40. Verifies input hashes, count == 120, and zero overlap with
    train40 / heldout24 / the 10 cross-over pilot ids. Returns the pool sha256.
    """
    for path, want in ((TRAIN40, TRAIN40_SHA256), (TRAIN160, TRAIN160_SHA256),
                       (HELDOUT24, HELDOUT24_SHA256)):
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"input drifted: {path} sha256 {got} != pinned {want}")

    train40 = set(_task_ids(TRAIN40))
    heldout24 = set(_task_ids(HELDOUT24))
    crossover = set(_task_ids(CROSSOVER10))
    lines = [ln for ln in TRAIN160.read_text().splitlines() if ln.strip()]
    kept = [ln for ln in lines if json.loads(ln)["task_id"] not in train40]
    pool_ids = [json.loads(ln)["task_id"] for ln in kept]

    if len(kept) != 120:
        raise SystemExit(f"pool count {len(kept)} != 120")
    if len(set(pool_ids)) != 120:
        raise SystemExit("duplicate task_ids in pool")
    for name, other in (("train40", train40), ("heldout24", heldout24),
                        ("crossover10", crossover)):
        overlap = set(pool_ids) & other
        if overlap:
            raise SystemExit(f"pool overlaps {name}: {sorted(overlap)}")

    out.write_text("\n".join(kept) + "\n")
    digest = sha256_file(out)
    log(f"pool -> {out}  n=120  sha256={digest}")
    log("overlap checks passed: train40=0 heldout24=0 crossover10=0")
    return digest


def load_dump(path: Path, regime: str, span: str) -> dict[str, dict]:
    """Probe dump rows for one (regime, span), keyed by task_id."""
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return {
        r["task_id"]: r for r in rows if r["regime"] == regime and r["span"] == span
    }


def sign_test_p(n_pos: int, n: int) -> float:
    """Exact two-sided binomial sign test at p=0.5 (E-phase1 convention:
    binom(17,24) -> 0.064). Caller drops zeros."""
    from scipy.stats import binomtest  # noqa: PLC0415

    return float(binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue)


def bootstrap_ci(
    deltas: list[float], resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=np.float64)
    means = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_stats(
    c3: dict[str, dict], warm: dict[str, dict], key: str, resamples: int, seed: int
) -> dict:
    """Paired c3-minus-warm stats over shared task_ids for dump field `key`."""
    tids = sorted(set(c3) & set(warm))
    deltas = [c3[t][key] - warm[t][key] for t in tids]
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_eff = n_pos + n_neg  # sign test drops exact zeros
    lo, hi = bootstrap_ci(deltas, resamples, seed)
    return {
        "n_pairs": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "sign_test_n": n_eff,
        "sign_test_p": sign_test_p(n_pos, n_eff),
        "ci_lo": lo,
        "ci_hi": hi,
        "c3_mean": sum(c3[t][key] for t in tids) / len(tids),
        "warm_mean": sum(warm[t][key] for t in tids) / len(tids),
    }


def run_probe(ckpt: str, pool: Path, out: Path, logfile: Path) -> None:
    """Invoke the frozen probe exactly as _phase1_orchestrate.py did (per-ckpt
    subprocess; sequential model loads so GPU memory is released between arms)."""
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [
        sys.executable, str(PROBE),
        "--ckpt", ckpt,
        "--corpus", str(pool),
        "--out", str(out),
    ]
    log(f"probe: {' '.join(cmd)} (log -> {logfile})")
    with open(logfile, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                            check=False).returncode
    if rc != 0 or not out.exists():
        raise SystemExit(f"probe failed rc={rc} (see {logfile})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-pool", action="store_true",
                    help="derive benchmarks/mbpp_recall_fresh120.jsonl and exit (CPU-only)")
    ap.add_argument("--pool", type=Path, default=POOL_DEFAULT)
    ap.add_argument("--c3-ckpt", default=C3_CKPT)
    ap.add_argument("--warm-ckpt", default=WARM_CKPT)
    ap.add_argument("--c3-sha256", default=C3_SHA256)
    ap.add_argument("--warm-sha256", default=WARM_SHA256)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/c21"))
    ap.add_argument("--stats-only", action="store_true",
                    help="skip probing; compute stats from --c3-dump/--warm-dump (CPU-only)")
    ap.add_argument("--c3-dump", type=Path, default=None)
    ap.add_argument("--warm-dump", type=Path, default=None)
    ap.add_argument("--experiment", default=EXPERIMENT_DEFAULT)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--bootstrap-resamples", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--allow-partial", action="store_true",
                    help="tolerate probe-excluded tasks (default: hard-fail if any "
                         "pool task is missing from either dump)")
    a = ap.parse_args()

    if a.build_pool:
        build_pool(a.pool)
        return 0

    if not a.pool.exists():
        raise SystemExit(f"pool missing: {a.pool} (run --build-pool first)")
    pool_sha = sha256_file(a.pool)
    pool_ids = set(_task_ids(a.pool))
    log(f"pool {a.pool} n={len(pool_ids)} sha256={pool_sha}")

    a.workdir.mkdir(parents=True, exist_ok=True)
    c3_dump = a.c3_dump or a.workdir / "c3_fresh120.jsonl"
    warm_dump = a.warm_dump or a.workdir / "warm_fresh120.jsonl"

    if not a.stats_only:
        for name, ckpt, want in (("c3", a.c3_ckpt, a.c3_sha256),
                                 ("warm", a.warm_ckpt, a.warm_sha256)):
            got = sha256_file(ckpt)
            if got != want:
                raise SystemExit(f"{name} ckpt sha256 {got} != expected {want} ({ckpt})")
            log(f"{name} ckpt verified: {ckpt} sha256={got}")
        run_probe(a.c3_ckpt, a.pool, c3_dump, a.workdir / "probe_c3.log")
        run_probe(a.warm_ckpt, a.pool, warm_dump, a.workdir / "probe_warm.log")

    # Headline: absent/body lp_m (Δlp_matched, the +0.105 statistic).
    c3_ab = load_dump(c3_dump, "absent", "body")
    warm_ab = load_dump(warm_dump, "absent", "body")
    missing = (pool_ids - set(c3_ab)) | (pool_ids - set(warm_ab))
    if missing and not a.allow_partial:
        raise SystemExit(
            f"{len(missing)} pool tasks missing from dumps (probe exclusions?): "
            f"{sorted(missing)[:10]} ... rerun or pass --allow-partial"
        )

    headline = paired_stats(c3_ab, warm_ab, "lp_m",
                            a.bootstrap_resamples, a.bootstrap_seed)
    mzero = paired_stats(c3_ab, warm_ab, "m_zero",
                         a.bootstrap_resamples, a.bootstrap_seed)
    secondary = {}
    for reg, span in (("absent", "sig"), ("present", "body"), ("absent", "full")):
        c3_s = load_dump(c3_dump, reg, span)
        warm_s = load_dump(warm_dump, reg, span)
        if c3_s and warm_s:
            secondary[f"{reg}_{span}"] = paired_stats(
                c3_s, warm_s, "lp_m", a.bootstrap_resamples, a.bootstrap_seed
            )

    summary = {
        "task": "C2.1 fresh-pool re-estimate",
        "pool": str(a.pool), "pool_sha256": pool_sha, "pool_n": len(pool_ids),
        "c3_ckpt": a.c3_ckpt, "c3_sha256": a.c3_sha256,
        "warm_ckpt": a.warm_ckpt, "warm_sha256": a.warm_sha256,
        "statistic": "delta_lp_matched (absent/body reference-span mean gold "
                     "logprob, matched adapter, c3 minus warm-start)",
        "sign_test": "exact two-sided binomial, zeros dropped (E-phase1 convention)",
        "bootstrap": {"method": "percentile-95", "resamples": a.bootstrap_resamples,
                      "seed": a.bootstrap_seed},
        "original_estimate": {"mean": 0.105, "n_pos": 17, "n": 24, "p": 0.064,
                              "ci": [0.033, 0.182], "pool": "heldout24 (selection set)"},
        "headline_delta_lp_matched": headline,
        "crosscheck_delta_m_zero": mzero,
        "secondary": secondary,
    }
    (a.workdir / "c21_summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({"headline": headline}, indent=2))
    gate = "p<0.05 -> de-biased number, strip caveat" if headline["sign_test_p"] < 0.05 \
        else "p>=0.05 -> prose-downgrade signal to article side"
    log(f"pre-registered gate: sign_test_p={headline['sign_test_p']:.4f} -> {gate}")

    if not a.no_mlflow:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(a.experiment)
        with mlflow.start_run(run_name="c21-fresh120-reestimate"):
            mlflow.set_tags({"issue": "52", "task": "C2.1",
                             "instrument": "tools/_specificity_probe.py (frozen E1 probe, git 205fa3d)",
                             "plan": "docs/publication/publication_task_plan.md C2.1",
                             "pool_derivation": "docs/publication/c01_corpus_lookup.md"})
            mlflow.log_params({
                "pool_path": str(a.pool), "pool_sha256": pool_sha,
                "pool_n": len(pool_ids),
                "c3_ckpt": a.c3_ckpt, "c3_sha256": a.c3_sha256,
                "warm_ckpt": a.warm_ckpt, "warm_sha256": a.warm_sha256,
                "regime": "absent", "span": "body", "statistic": "delta_lp_matched",
                "sign_test": "exact_binomial_two_sided",
                "bootstrap_resamples": a.bootstrap_resamples,
                "bootstrap_seed": a.bootstrap_seed,
                "original_run_id": "fe72f9ddd69c4f7b8bd86b6b12372d47",
            })
            flat: dict[str, float] = {}
            for prefix, st in (("", headline), ("mzero_", mzero),
                               *((f"{k}_", v) for k, v in secondary.items())):
                for m in ("n_pairs", "mean_delta", "n_pos", "sign_test_n",
                          "sign_test_p", "ci_lo", "ci_hi", "c3_mean", "warm_mean"):
                    flat[f"{prefix}{m}"] = float(st[m])
            mlflow.log_metrics(flat)
            for f in (c3_dump, warm_dump, a.workdir / "c21_summary.json"):
                if f.exists():
                    mlflow.log_artifact(str(f))
            log(f"mlflow: experiment={a.experiment} run logged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
