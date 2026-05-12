"""Deep-dive: are the JSONL records actually well-formed and trainable?

Pre and post are EXTRACTED at runtime from activation_text + teacher_text.
This script mirrors that extraction and reports diff statistics.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "libs/model-training/src")
sys.path.insert(0, "libs/shared/src")

from model_training.d2l_data import _extract_post_revision, _extract_pre_revision
from model_training.diff_loss import _compute_hunk_ranges

DATA = Path("data/github-pairs/_merged/pairs_all.jsonl")
N_SAMPLE = 200
SEED = 42

random.seed(SEED)
rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
sample = random.sample(rows, N_SAMPLE)
print(f"Loaded {len(rows)} records, sampled {N_SAMPLE} (seed={SEED})")

stats: list[dict[str, Any]] = []
for rec in sample:
    activation = rec.get("activation_text") or ""
    teacher = rec.get("teacher_text") or ""
    if not activation or not teacher:
        stats.append({"skip": "empty"})
        continue

    try:
        pre = _extract_pre_revision(activation)
        post = _extract_post_revision(activation, teacher)
    except Exception as e:
        stats.append({"skip": f"extract_err: {e}"})
        continue

    if not pre and not post:
        stats.append({"skip": "empty pre+post"})
        continue

    hunks = _compute_hunk_ranges(pre, post) if (pre and post) else []
    hunk_chars = sum(e - s for s, e in hunks)
    hunk_frac = hunk_chars / max(1, len(post))

    stats.append(
        {
            "pre_chars": len(pre),
            "post_chars": len(post),
            "pre_lines": len(pre.splitlines()),
            "post_lines": len(post.splitlines()),
            "identical": pre == post and bool(pre),
            "n_hunks": len(hunks),
            "hunk_char_frac": hunk_frac,
            "pre_empty": not pre,
            "post_empty": not post,
            "task_id": rec.get("task_id", "?")[:40],
        }
    )

valid = [s for s in stats if "skip" not in s]
print(f"\n=== {len(valid)}/{N_SAMPLE} records had extractable pre/post ===\n")

# Empties
print(f"pre empty:    {sum(s['pre_empty'] for s in valid)}/{len(valid)}")
print(f"post empty:   {sum(s['post_empty'] for s in valid)}/{len(valid)}")
print(
    f"identical pre==post (and non-empty): {sum(s['identical'] for s in valid)}/{len(valid)}"
)

# Length distribution
both = [s for s in valid if not s["pre_empty"] and not s["post_empty"]]
print(f"\nBoth pre+post non-empty: {len(both)}/{len(valid)}\n")
if both:
    print(
        f"pre_chars  median={statistics.median(s['pre_chars'] for s in both):.0f}  "
        f"p90={sorted(s['pre_chars'] for s in both)[int(0.9 * len(both))]:.0f}  "
        f"max={max(s['pre_chars'] for s in both)}"
    )
    print(
        f"post_chars median={statistics.median(s['post_chars'] for s in both):.0f}  "
        f"p90={sorted(s['post_chars'] for s in both)[int(0.9 * len(both))]:.0f}  "
        f"max={max(s['post_chars'] for s in both)}"
    )

    print("\nn_hunks per record:")
    print(f"  mean={statistics.mean(s['n_hunks'] for s in both):.2f}")
    print(f"  median={statistics.median(s['n_hunks'] for s in both)}")
    print(f"  max={max(s['n_hunks'] for s in both)}")
    print(f"  zero hunks: {sum(1 for s in both if s['n_hunks'] == 0)}/{len(both)}")

    print("\nhunk_char_frac (fraction of post chars inside ANY hunk):")
    fracs = [s["hunk_char_frac"] for s in both]
    print(f"  mean={statistics.mean(fracs):.3f}")
    print(f"  median={statistics.median(fracs):.3f}")
    print(
        f"  p10={sorted(fracs)[int(0.1 * len(fracs))]:.3f}  p25={sorted(fracs)[int(0.25 * len(fracs))]:.3f}"
    )
    print(
        f"  p75={sorted(fracs)[int(0.75 * len(fracs))]:.3f}  p90={sorted(fracs)[int(0.9 * len(fracs))]:.3f}"
    )
    print(
        f"  records hunk_frac >= 0.95: {sum(1 for f in fracs if f >= 0.95)}/{len(fracs)}"
    )
    print(
        f"  records hunk_frac >= 0.99: {sum(1 for f in fracs if f >= 0.99)}/{len(fracs)}"
    )
    print(
        f"  records hunk_frac == 1.0:  {sum(1 for f in fracs if f >= 0.999)}/{len(fracs)}"
    )

    # Show 3 lowest and 3 highest hunk_frac
    both.sort(key=lambda x: x["hunk_char_frac"])
    print("\n=== 3 RECORDS WITH LOWEST hunk_char_frac ===")
    for s in both[:3]:
        print(
            f"  task={s['task_id']} hunks={s['n_hunks']} char_frac={s['hunk_char_frac']:.3f} pre={s['pre_chars']}c post={s['post_chars']}c lines={s['pre_lines']}->{s['post_lines']}"
        )

    print("\n=== 3 RECORDS WITH HIGHEST hunk_char_frac ===")
    for s in both[-3:]:
        print(
            f"  task={s['task_id']} hunks={s['n_hunks']} char_frac={s['hunk_char_frac']:.3f} pre={s['pre_chars']}c post={s['post_chars']}c lines={s['pre_lines']}->{s['post_lines']}"
        )
