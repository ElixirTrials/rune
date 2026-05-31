"""Lightweight corpus + split QC (issue #49) — DeepChecks-equivalent, no heavy dep.

DeepChecks is import-incompatible with our scikit-learn 1.8 (removed `max_error`
scorer), and downgrading sklearn risks the training env, so this delivers the same
DATA-layer value with libs we already have:
  - NEAR-DUPLICATE LEAKAGE: TF-IDF char-ngram max cosine of each val/test context to
    the train set (catches near-dup leakage a key-based family split cannot).
  - PROPERTY DRIFT: context/answer/edit-size distributions train vs val vs test.
  - INTEGRITY: exact-duplicate and empty/short contexts in train.

Complements the family split + `_corpus_stats` + teacher audit. NOT a training
monitor or model-quality gate. Run ad-hoc (CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _rows(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _ctx(r: dict) -> str:
    return str(r.get("activation_text") or r.get("context") or "")


def _ans(r: dict) -> str:
    tt = str(r.get("teacher_text") or "")
    at = str(r.get("activation_text") or "")
    return tt[len(at):] if tt.startswith(at) else str(r.get("answer") or tt)


def _dist(vals: list[int]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {"median": s[len(s) // 2], "p10": s[len(s) // 10],
            "p90": s[min(len(s) - 1, 9 * len(s) // 10)], "max": s[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/rune-corpus")
    ap.add_argument("--prefix", default="external_codereview")
    ap.add_argument("--out", default="docs/superpowers/artifacts/corpus_split_qc.json")
    ap.add_argument("--leak-threshold", type=float, default=0.9)
    a = ap.parse_args()

    sp = {s: _rows(f"{a.dir}/{a.prefix}.{s}.jsonl") for s in ("train", "val", "test")}
    out: dict = {"sizes": {k: len(v) for k, v in sp.items()}}

    # property drift
    out["property_drift"] = {}
    for s, rows in sp.items():
        out["property_drift"][s] = {
            "context_len": _dist([len(_ctx(r)) for r in rows]),
            "answer_len": _dist([len(_ans(r)) for r in rows]),
            "edit_size": _dist([len(str(r.get("post_code", ""))) for r in rows]),
        }

    # integrity (train)
    ctrain = [_ctx(r) for r in sp["train"]]
    out["integrity_train"] = {
        "exact_dup_contexts": len(ctrain) - len(set(ctrain)),
        "empty_or_short": sum(1 for c in ctrain if len(c.strip()) < 20),
    }

    # near-duplicate leakage: TF-IDF char 3-5 grams; max cosine val/test -> train
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000)
    Xtrain = vec.fit_transform(ctrain)
    out["leakage"] = {}
    for s in ("val", "test"):
        ctx = [_ctx(r) for r in sp[s]]
        Xs = vec.transform(ctx)
        # chunked max-cosine to train
        max_sim = np.zeros(Xs.shape[0])
        step = 256
        for i in range(0, Xtrain.shape[0], step):
            sims = cosine_similarity(Xs, Xtrain[i:i + step])
            max_sim = np.maximum(max_sim, sims.max(axis=1))
        leaked = int((max_sim >= a.leak_threshold).sum())
        out["leakage"][s] = {
            "n": len(ctx),
            "max_cosine_median": float(np.median(max_sim)),
            "max_cosine_p90": float(np.percentile(max_sim, 90)),
            f"near_dup_ge_{a.leak_threshold}": leaked,
            "near_dup_frac": leaked / len(ctx) if ctx else 0.0,
        }
        # Write a near-dup-filtered "clean" split for honest held-out eval: drop rows
        # whose context is a near-duplicate of any train context.
        clean = [r for r, m in zip(sp[s], max_sim, strict=True) if m < a.leak_threshold]
        clean_path = f"{a.dir}/{a.prefix}.{s}.clean.jsonl"
        with open(clean_path, "w") as fh:
            for r in clean:
                fh.write(json.dumps(r) + "\n")
        out["leakage"][s]["clean_path"] = clean_path
        out["leakage"][s]["clean_n"] = len(clean)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # verdict: low near-dup leakage + comparable property medians
    leak_ok = all(v["near_dup_frac"] < 0.05 for v in out["leakage"].values())
    print(f"\nLEAKAGE_OK (near-dup<5%): {leak_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
