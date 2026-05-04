"""Audit quality scores across the mined trajectory corpus.

Scores all episodes with the default QualityWeightConfig and prints:
- Score histogram (10 buckets)
- Per-factor breakdown by repo
- Spot-check: 3 highest and 3 lowest non-ep0 episodes
- Edge cases: factor disagreements

Usage:
    uv run python scripts/audit_quality_scores.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from model_training.d2l_models import Trajectory
from model_training.d2l_quality import (
    QualityWeightConfig,
    classify_causal_link,
    is_url_only,
    score_episode_quality,
)

MINED_DIR = Path("data/mined")
CFG = QualityWeightConfig()


def _score_corpus() -> list[dict]:
    records: list[dict] = []
    for tf in sorted(MINED_DIR.glob("*.trajectories.jsonl")):
        repo = tf.stem.replace(".trajectories", "")
        with tf.open() as f:
            for line in f:
                traj = Trajectory(**json.loads(line))
                for ep in traj.episodes:
                    is_ep0 = ep.round == 0
                    q = score_episode_quality(
                        feedback_body=ep.feedback.body,
                        action_diff=ep.action_diff,
                        is_ep0=is_ep0,
                        config=CFG,
                    )
                    causal = (
                        "ep0"
                        if is_ep0
                        else classify_causal_link(ep.feedback.body, ep.action_diff)
                    )
                    fb_len = len(ep.feedback.body.strip())
                    records.append(
                        {
                            "repo": repo,
                            "task_id": traj.task_id,
                            "round": ep.round,
                            "quality_score": q,
                            "causal": causal,
                            "feedback_len": fb_len,
                            "diff_len": len(ep.action_diff),
                            "url_only": is_url_only(ep.feedback.body),
                            "feedback_snippet": ep.feedback.body[:120],
                            "diff_snippet": ep.action_diff[:120],
                        }
                    )
    return records


def _histogram(scores: list[float], n_bins: int = 10) -> None:
    lo, hi = 0.0, 1.0
    step = (hi - lo) / n_bins
    bins = Counter[str]()
    for s in scores:
        idx = min(int((s - lo) / step), n_bins - 1)
        label = f"[{lo + idx * step:.2f}, {lo + (idx + 1) * step:.2f})"
        bins[label] += 1
    print("\n=== Score Histogram ===")
    for label in sorted(bins):
        bar = "#" * (bins[label] * 60 // max(bins.values()))
        print(f"  {label}  {bins[label]:>5}  {bar}")


def _per_factor(records: list[dict]) -> None:
    print("\n=== Per-Factor Breakdown ===")
    non_ep0 = [r for r in records if r["round"] > 0]
    total = len(non_ep0) or 1

    causal_counts = Counter(r["causal"] for r in non_ep0)
    print(f"\nCausal link (non-ep0, n={len(non_ep0)}):")
    for k in ["entity_overlap", "no_overlap", "url_only"]:
        c = causal_counts.get(k, 0)
        print(f"  {k:<20} {c:>5}  ({100 * c / total:.1f}%)")

    fb_buckets = Counter[str]()
    for r in non_ep0:
        if r["url_only"]:
            fb_buckets["url_only"] += 1
        elif r["feedback_len"] >= CFG.feedback_rich_chars:
            fb_buckets["rich (>=100)"] += 1
        elif r["feedback_len"] >= CFG.feedback_moderate_chars:
            fb_buckets["moderate (20-99)"] += 1
        else:
            fb_buckets["short (<20)"] += 1
    print(f"\nFeedback length (non-ep0):")
    for k in ["rich (>=100)", "moderate (20-99)", "short (<20)", "url_only"]:
        c = fb_buckets.get(k, 0)
        print(f"  {k:<20} {c:>5}  ({100 * c / total:.1f}%)")

    prop_fired = sum(
        1
        for r in non_ep0
        if r["feedback_len"] < CFG.proportionality_short_chars
        and r["diff_len"] > CFG.proportionality_diff_chars
    )
    print(f"\nProportionality penalty fired: {prop_fired} ({100 * prop_fired / total:.1f}%)")


def _per_repo(records: list[dict]) -> None:
    print("\n=== Per-Repo Summary ===")
    repos = sorted({r["repo"] for r in records})
    print(f"  {'Repo':<40} {'N':>5} {'Mean':>6} {'Min':>6} {'Max':>6}")
    for repo in repos:
        subset = [r["quality_score"] for r in records if r["repo"] == repo]
        if not subset:
            continue
        print(
            f"  {repo:<40} {len(subset):>5} {sum(subset)/len(subset):>6.3f} "
            f"{min(subset):>6.3f} {max(subset):>6.3f}"
        )


def _spot_check(records: list[dict]) -> None:
    non_ep0 = [r for r in records if r["round"] > 0]
    non_ep0.sort(key=lambda r: r["quality_score"])
    print("\n=== Spot Check: 3 Lowest Non-Ep0 ===")
    for r in non_ep0[:3]:
        print(f"  score={r['quality_score']:.3f}  causal={r['causal']}  fb_len={r['feedback_len']}  diff_len={r['diff_len']}")
        print(f"    feedback: {r['feedback_snippet']}")
        print(f"    diff:     {r['diff_snippet']}")
        print()

    print("=== Spot Check: 3 Highest Non-Ep0 ===")
    for r in non_ep0[-3:]:
        print(f"  score={r['quality_score']:.3f}  causal={r['causal']}  fb_len={r['feedback_len']}  diff_len={r['diff_len']}")
        print(f"    feedback: {r['feedback_snippet']}")
        print(f"    diff:     {r['diff_snippet']}")
        print()


def _edge_cases(records: list[dict]) -> None:
    non_ep0 = [r for r in records if r["round"] > 0]
    print("=== Edge Cases ===")

    overlap_but_url = [
        r for r in non_ep0 if r["causal"] == "entity_overlap" and r["url_only"]
    ]
    print(f"  Entity overlap + URL-only: {len(overlap_but_url)}")

    rich_but_penalty = [
        r
        for r in non_ep0
        if r["feedback_len"] >= CFG.feedback_rich_chars
        and r["feedback_len"] < CFG.proportionality_short_chars
    ]
    print(f"  Rich feedback + proportionality penalty: {len(rich_but_penalty)}")

    no_overlap_rich = [
        r
        for r in non_ep0
        if r["causal"] == "no_overlap" and r["feedback_len"] >= CFG.feedback_rich_chars
    ]
    print(f"  No entity overlap but rich feedback (>= 100 chars): {len(no_overlap_rich)}")
    if no_overlap_rich:
        print("    (These may be false-negative causal — reviewer uses concepts, not identifiers)")
        for r in no_overlap_rich[:3]:
            print(f"      {r['repo']} r{r['round']}: {r['feedback_snippet'][:80]}")


def main() -> None:
    if not MINED_DIR.exists():
        print(f"ERROR: {MINED_DIR} not found", file=sys.stderr)
        sys.exit(1)

    records = _score_corpus()
    print(f"Scored {len(records)} episodes across {len({r['repo'] for r in records})} repos")

    scores = [r["quality_score"] for r in records]
    _histogram(scores)
    _per_factor(records)
    _per_repo(records)
    _spot_check(records)
    _edge_cases(records)


if __name__ == "__main__":
    main()
