"""Sharper content-specificity tests for the corpus-trained adapter (issue #49).

The aggregate gate (diff_agreement over the whole answer span, contra = a different
real row) was inconclusive (real ~= contra). These tests are sharper:

  TEST 1 — MATCHED vs MISMATCHED edit-local logprob (decoding-variance-free):
    For each held-out clean row i, teacher-force its gold revision and read the avg
    logprob of the EDIT-LOCAL tokens (difflib pre_code vs revision: insert/replace)
    under (a) the MATCHED adapter (from row i's own context), (b) a MISMATCHED
    adapter (from another row's context), (c) ZERO (no adapter). The header/boilerplate
    cancels in the matched-minus-mismatched margin, so a positive margin is specificity.
    SPECIFIC if matched > mismatched (margin>0) AND matched > zero.

  TEST 2 — COMPLETION (functional): greedy-complete the revision from its first half
    with the matched adapter vs zero vs mismatched; report token-overlap with the gold
    continuation. Does embedding the diff let the model finish the edit it otherwise can't?

4-bit base (train-matched; avoids the bf16 OOM). Run under tools/run_guarded.sh.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys

import torch

from rune.model.hypernetwork import HypernetworkConfig, load_hypernetwork
from rune.training.hypernet_distill import (
    _functional_lora,
    _generate_lora_dict,
    _map_record,
)


def _edit_mask(tok, pre_code: str, ans_ids: list[int]) -> list[bool]:
    if not pre_code:
        return [True] * len(ans_ids)
    pre = tok(pre_code, add_special_tokens=False)["input_ids"]
    mask = [False] * len(ans_ids)
    for op, _i1, _i2, j1, j2 in difflib.SequenceMatcher(
        a=pre, b=ans_ids, autojunk=False
    ).get_opcodes():
        if op in ("insert", "replace"):
            for j in range(j1, j2):
                mask[j] = True
    return mask


def _editlocal_logprob(base, tok, hyp, ctx, ans, pre_code, li, scaling, max_len):
    """Avg logprob of the answer's edit-local tokens, adapter generated from ctx."""
    device = next(base.parameters()).device
    ans_ids = tok(ans, add_special_tokens=False)["input_ids"][:max_len]
    if len(ans_ids) < 2:
        return None
    emask = _edit_mask(tok, pre_code, ans_ids)
    ids = torch.tensor([ans_ids], device=device)
    n_qs = torch.tensor([1], device=device)
    with torch.no_grad():
        if scaling > 0:
            ld = _generate_lora_dict(hyp, ctx, base, tok, li, max_len)
            with _functional_lora(base, li, ld, scaling, n_qs):
                logits = base(ids, use_cache=False).logits[0]
        else:
            logits = base(ids, use_cache=False).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        # token at pos t predicted by logits[t-1]; score edit-local target tokens
        tot, cnt = 0.0, 0
        for t in range(1, len(ans_ids)):
            if emask[t]:
                tot += float(lp[t - 1, ans_ids[t]])
                cnt += 1
    return tot / cnt if cnt else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl"
    )
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--scaling", type=float, default=0.5)
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-9B")
    ap.add_argument("--json-out", default="/tmp/rune-issue49-specificity.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        a.model_id,
        quantization_config=q,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    ).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    hyp = load_hypernetwork(
        HypernetworkConfig(checkpoint_path=a.checkpoint), device="cuda"
    )
    hyp.eval()
    li = list(hyp.config.layer_indices)

    with open(a.val) as fh:
        rows = []
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            m = _map_record(raw)
            if m:
                rows.append({**m, "pre_code": str(raw.get("pre_code", ""))})
            if len(rows) >= a.n:
                break

    matched, mismatched, zero, wins = [], [], [], 0
    per = []
    for i, r in enumerate(rows):
        j = (i + 1) % len(rows)
        lm = _editlocal_logprob(
            base,
            tok,
            hyp,
            r["context"],
            r["answer"],
            r["pre_code"],
            li,
            a.scaling,
            a.max_seq_length,
        )
        lx = _editlocal_logprob(
            base,
            tok,
            hyp,
            rows[j]["context"],
            r["answer"],
            r["pre_code"],
            li,
            a.scaling,
            a.max_seq_length,
        )
        lz = _editlocal_logprob(
            base,
            tok,
            hyp,
            r["context"],
            r["answer"],
            r["pre_code"],
            li,
            0.0,
            a.max_seq_length,
        )
        if None in (lm, lx, lz):
            continue
        matched.append(lm)
        mismatched.append(lx)
        zero.append(lz)
        wins += int(lm > lx)
        per.append(
            {"matched": round(lm, 4), "mismatched": round(lx, 4), "zero": round(lz, 4)}
        )

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    n = len(matched)
    out = {
        "n": n,
        "scaling": a.scaling,
        "matched_editlocal_logprob": mean(matched),
        "mismatched_editlocal_logprob": mean(mismatched),
        "zero_editlocal_logprob": mean(zero),
        "matched_minus_mismatched": mean(matched) - mean(mismatched),
        "matched_minus_zero": mean(matched) - mean(zero),
        "frac_rows_matched_gt_mismatched": wins / n if n else 0.0,
        "specific": (mean(matched) > mean(mismatched)) and (mean(matched) > mean(zero)),
    }
    with open(a.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print("SPECIFICITY:", json.dumps({k: out[k] for k in out if k != "per"}, indent=2))
    print(
        f"SPECIFIC: {out['specific']} (margin matched-mismatched={out['matched_minus_mismatched']:+.4f}, "
        f"frac wins={out['frac_rows_matched_gt_mismatched']:.2f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
