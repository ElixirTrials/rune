"""Discriminating test: is the repair CHANNEL live, or is it a capability ceiling?

REMOVE-BEFORE-MERGE. The 3705 audit showed the base model re-emitting its input
code BYTE-IDENTICALLY (len=424) after a perfect critique. Byte-identical output
under sampling (temp 0.3, rep_penalty 1.1) means the critique moved the output
distribution by ~zero -- the signature of a DEAD channel, not a model that tried.

This isolates the two hypotheses by varying ONLY the critique text and measuring
whether the generation changes:
  H1 capability ceiling -> output CHANGES for a trivially-actionable instruction.
  H2 communication failure -> output is INVARIANT to critique content.

A drastic, unambiguous instruction ("replace the body with `return sum(nums)%k`")
is the decider: if the model won't even do THAT, the critique isn't reaching it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"
QID = "3705"

_REPAIR = """\
You wrote this `{entry}` but it is INCORRECT on a hidden test.

Task:
{spec}

Your current code:
```python
{code}
```

{crit}

Return a corrected `{entry}` that fixes this and still passes the examples.
Output only the function in one ```python block."""

# (label, critique-line, what a LIVE channel should produce)
VARIANTS = [
    ("V0_real",
     "PERFECT CRITIQUE — On input [[0]*50, 50] your function returns -1, but the "
     "correct answer is 0. Fix the logic.",
     "the real bug fix (hard)"),
    ("V1_no_critique", "", "any attempt"),
    ("V2_trivial_addline",
     "INSTRUCTION — Keep the logic EXACTLY the same, but add `debug = True` as the "
     "very first line inside the function body. Make only that change.",
     "code with `debug = True` added"),
    ("V3_drastic_replace",
     "INSTRUCTION — The entire body is wrong. Replace the WHOLE function body with "
     "exactly this single line: `return sum(nums) % k`. Do not keep the old logic.",
     "body == `return sum(nums) % k`"),
    ("V4_rename",
     "INSTRUCTION — Keep the logic identical but RENAME the function from "
     "`largestInteger` to `solve`.",
     "function named `solve`"),
]


async def main() -> None:
    from rune.bench.lcb import extract_entry_function
    from rune.config import load_rune_config
    from rune.engine.parse import extract_code_block
    from rune.model.wrapper import ModelWrapper

    rows = {json.loads(x)["question_id"]: json.loads(x)
            for x in Path(LCB).read_text().splitlines()}
    cands = {g["question_id"]: g["code_list"][0]
             for g in json.loads(Path(COMBINED).read_text())}
    row = rows[QID]
    meta = json.loads(row["metadata"]) if row.get("metadata") else {}
    entry = meta.get("func_name") or ""
    spec = row.get("question_content", "")[:3500]
    orig = extract_entry_function(cands[QID], entry)

    cfg = load_rune_config(None).override(
        checkpoint_path="/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt",
        adapter_scaling=0.0, model_judge=False)
    model = ModelWrapper.from_config(cfg)

    print(f"# {QID} {entry}  (base, scaling=0)\n# ORIGINAL ({len(orig)} chars):\n{orig}\n")
    for temp in (0.3, 0.0):
        print(f"\n{'='*70}\n=== temperature={temp} ===\n{'='*70}")
        for label, crit, expect in VARIANTS:
            prompt = _REPAIR.format(entry=entry, spec=spec, code=orig, crit=crit)
            gen = await model.generate(
                prompt=prompt, max_tokens=1024, temperature=temp, thinking_budget=0)
            raw = gen.text
            new = extract_entry_function(extract_code_block(raw) or "", entry)
            identical = new.strip() == orig.strip()
            empty = not new.strip()
            tag = ("BYTE-IDENTICAL" if identical else
                   "EMPTY/UNEXTRACTABLE" if empty else "CHANGED")
            print(f"\n--- {label}  (live-channel expects: {expect}) -> {tag}")
            print(f"    raw_gen_len={len(raw)} extracted_len={len(new)}")
            if not identical and not empty:
                # show what changed (first 3 differing-ish lines)
                print("    NEW CODE:")
                for ln in new.strip().splitlines()[:8]:
                    print(f"      {ln}")


if __name__ == "__main__":
    asyncio.run(main())
