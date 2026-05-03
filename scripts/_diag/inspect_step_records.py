"""Show a step_index=1 record's pre and post in full to confirm the diff-of-diffs hypothesis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "libs/model-training/src")
sys.path.insert(0, "libs/shared/src")
from model_training.d2l_data import _extract_post_revision, _extract_pre_revision
from model_training.diff_loss import _compute_hunk_ranges

DATA = Path("data/github-pairs/_merged/pairs_all.jsonl")
rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]

# Find one step_index=1 record with both pre and post
for rec in rows:
    if (rec.get('metadata') or {}).get('step_index') != 1:
        continue
    activation = rec.get('activation_text','')
    teacher = rec.get('teacher_text','')
    pre = _extract_pre_revision(activation)
    post = _extract_post_revision(activation, teacher)
    if not (pre and post and pre != post):
        continue

    print(f"task_id: {rec['task_id']}")
    print(f"step_index: {(rec.get('metadata') or {}).get('step_index')}")
    print(f"language: {(rec.get('metadata') or {}).get('language')}")
    print(f"pre_chars: {len(pre)}, post_chars: {len(post)}")
    h = _compute_hunk_ranges(pre, post)
    hc = sum(e-s for s,e in h)
    print(f"hunks: {len(h)}, hunk_char_frac: {hc/len(post):.3f}")
    print()
    print("===== PRE (first 1500 chars) =====")
    print(pre[:1500])
    print()
    print("===== POST (first 1500 chars) =====")
    print(post[:1500])
    print()
    # Now look at first few hunk ranges
    print("===== FIRST 3 HUNKS (post substrings) =====")
    for i, (s, e) in enumerate(h[:3]):
        print(f"hunk {i}: chars [{s},{e}) = {repr(post[s:e][:200])}")
    break
