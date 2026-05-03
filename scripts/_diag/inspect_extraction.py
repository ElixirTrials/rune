"""Why is _extract_pre_revision returning empty so often?"""
from __future__ import annotations
import json, sys, random
from pathlib import Path

sys.path.insert(0, "libs/model-training/src")
sys.path.insert(0, "libs/shared/src")

from model_training.d2l_data import _extract_pre_revision, _extract_post_revision

DATA = Path("data/github-pairs/_merged/pairs_all.jsonl")
random.seed(42)
rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
sample = random.sample(rows, 200)

# What sections are in activation_text?
from collections import Counter
section_headers = Counter()
no_current_code = 0
empty_pre_count = 0
examples_no_current_code = []

for rec in sample:
    activation = rec.get('activation_text') or ''
    teacher = rec.get('teacher_text') or ''

    # Find all "## " section headers
    headers = [line for line in activation.splitlines() if line.startswith("## ")]
    for h in headers:
        section_headers[h.strip()] += 1

    has_current_code = "## Current Code\n" in activation
    if not has_current_code:
        no_current_code += 1
        if len(examples_no_current_code) < 3:
            examples_no_current_code.append(rec.get('task_id'))

    pre = _extract_pre_revision(activation)
    if not pre:
        empty_pre_count += 1

print(f"=== SECTION HEADERS in activation_text (top 15) ===")
for h, n in section_headers.most_common(15):
    print(f"  {n:3d}× {h}")

print(f"\n=== '## Current Code' marker presence ===")
print(f"  records WITHOUT '## Current Code\\n': {no_current_code}/{len(sample)}")
print(f"  records with empty pre after extraction: {empty_pre_count}/{len(sample)}")
print(f"  examples without marker: {examples_no_current_code}")

# Show the activation_text of one no-current-code record
print(f"\n=== ACTIVATION_TEXT for one record without '## Current Code' ===")
for rec in sample:
    if "## Current Code\n" not in (rec.get('activation_text') or ''):
        print(f"task_id: {rec.get('task_id')}")
        print(f"activation_text (first 800 chars):")
        print(rec['activation_text'][:800])
        print(f"\n---\nteacher_text (first 800 chars):")
        print(rec['teacher_text'][:800])
        break

# Also dig into why hunk_frac is so high — show one high-frac case in detail
print(f"\n=== HIGH hunk_frac case in detail ===")
for rec in sample:
    activation = rec.get('activation_text') or ''
    teacher = rec.get('teacher_text') or ''
    if "## Current Code\n" not in activation:
        continue
    pre = _extract_pre_revision(activation)
    post = _extract_post_revision(activation, teacher)
    if not (pre and post):
        continue
    if pre == post:
        continue
    from model_training.diff_loss import _compute_hunk_ranges
    h = _compute_hunk_ranges(pre, post)
    hc = sum(e - s for s, e in h)
    if hc / max(1, len(post)) > 0.95 and len(post) > 500:
        print(f"task_id: {rec.get('task_id')}")
        print(f"pre_chars={len(pre)}, post_chars={len(post)}, hunks={len(h)}, hunk_char_frac={hc/len(post):.3f}")
        print(f"\nPRE (first 400 chars):")
        print(pre[:400])
        print(f"\nPOST (first 400 chars):")
        print(post[:400])
        # Quick line-level overlap
        pre_lines = set(pre.splitlines())
        post_lines = set(post.splitlines())
        common = pre_lines & post_lines
        print(f"\nUnique lines: pre={len(pre_lines)}, post={len(post_lines)}, common={len(common)}")
        print(f"First 5 common lines: {list(common)[:5]}")
        break
