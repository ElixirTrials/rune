"""Why does head -500 fail to train while random-500 learns? Compare distributions."""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

DATA = Path("data/github-pairs/_merged/pairs_all.jsonl")
rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
print(f"Total: {len(rows)}")

random.seed(42)
random_sample = random.sample(rows, 500)
head_sample = rows[:500]


def stats(name: str, sample: list) -> None:
    print(f"\n=== {name} (N={len(sample)}) ===")
    # repo distribution
    repos: Counter[str] = Counter()
    for r in sample:
        tid = r.get("task_id", "")
        if tid.startswith("pr_"):
            owner_repo = tid[3:].rsplit("_", 1)[0]
            repos[owner_repo] += 1
    print(f"unique repos: {len(repos)}")
    print("top 5 repos:")
    for repo, n in repos.most_common(5):
        print(f"  {n:3d}× {repo}")
    print(f"top-1 share: {repos.most_common(1)[0][1] / len(sample):.1%}")

    # has_review_feedback
    has_review = sum(
        1 for r in sample if "## Review Feedback" in (r.get("activation_text") or "")
    )
    has_current = sum(
        1 for r in sample if "## Current Code\n" in (r.get("activation_text") or "")
    )
    print("\nstructural:")
    print(
        f"  has_review_feedback: {has_review}/{len(sample)} ({has_review / len(sample):.1%})"
    )
    print(
        f"  has_current_code:    {has_current}/{len(sample)} ({has_current / len(sample):.1%})"
    )

    # length distribution
    act_lens = [len(r.get("activation_text", "")) for r in sample]
    teach_lens = [len(r.get("teacher_text", "")) for r in sample]
    print("\nlengths (chars):")
    print(
        f"  activation_text  median={statistics.median(act_lens):.0f}  p90={sorted(act_lens)[int(0.9 * len(sample))]:.0f}  max={max(act_lens)}"
    )
    print(
        f"  teacher_text     median={statistics.median(teach_lens):.0f}  p90={sorted(teach_lens)[int(0.9 * len(sample))]:.0f}  max={max(teach_lens)}"
    )

    # step_index from metadata
    step_indices: Counter[object] = Counter()
    for r in sample:
        si = (r.get("metadata") or {}).get("step_index", "?")
        step_indices[si] += 1
    print("\nstep_index distribution:")
    for si in sorted(
        step_indices, key=lambda x: (x is None, x if isinstance(x, int) else 999)
    ):
        print(f"  step_index={si}: {step_indices[si]}")

    # outcome from metadata
    outcomes: Counter[str] = Counter()
    for r in sample:
        oc = (r.get("metadata") or {}).get("outcome", "?")
        outcomes[oc] += 1
    print(f"\noutcome distribution: {dict(outcomes)}")

    # language
    langs: Counter[str] = Counter()
    for r in sample:
        lang = (r.get("metadata") or {}).get("language", "?")
        langs[lang] += 1
    print(f"\nlanguage distribution: {dict(langs.most_common(5))}")


stats("HEAD 500 (deterministic)", head_sample)
stats("RANDOM 500 (seed=42)", random_sample)
