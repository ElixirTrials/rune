"""Teacher-quality / corpus-readiness audit for issue #49 (advisor + reviewer #5).

THE real precondition + cheap kill-criterion before committing to corpus training.
D2L distills frozen-base-with-context behavior. If base+activation_text does NOT
beat base-alone on the revision span, there is no signal to distill and the real
corpus reproduces the teacher-approx-base weak-gradient collapse -> no hypernet
training can lift pass@1.

Checkpoint-AGNOSTIC: teacher = base + activation_text in-prompt (NO hypernet, NO
adapter); base = base alone. For each sampled row we teacher-force the gold
revision tokens and measure, over the answer span:
  - teacher_acc / base_acc: top-1 == gold token
  - teacher_nll / base_nll
  - DIFF-TOKEN FRACTION: frac of answer tokens where teacher is right AND base is
    wrong  <-- the distillable signal; the decisive number.
Reported both WHOLE-SPAN and EDIT-LOCAL (answer tokens inside insert/replace blocks
of difflib(pre_code_tokens, answer_tokens) — the tokens the review actually changed,
per reviewer #3), and stratified by edit size, with quality_score correlation.

Near-zero edit-local diff-token fraction => STOP / rethink (oracle teacher, reframe).
Healthy => green light for corpus training.

Run under tools/run_guarded.sh. GPU-only.
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rune.config import load_rune_config
from rune.training.hypernet_distill import _map_record, _prepare_ids

PREFIX_TASK = "## Task"  # rows start with this


def _edit_mask(tok: Any, pre_code: str, ans_ids: list[int]) -> list[bool]:
    """True at answer-token positions inside insert/replace blocks vs pre_code.

    Approximates the review's edit: tokens present in the revision but not carried
    over unchanged from the prior code. Whole-span fallback (all True) if no pre_code.
    """
    if not pre_code:
        return [True] * len(ans_ids)
    pre_ids = tok(pre_code, add_special_tokens=False)["input_ids"]
    mask = [False] * len(ans_ids)
    sm = difflib.SequenceMatcher(a=pre_ids, b=ans_ids, autojunk=False)
    for op, _i1, _i2, j1, j2 in sm.get_opcodes():
        if op in ("insert", "replace"):
            for j in range(j1, j2):
                mask[j] = True
    return mask


def _row_metrics(
    base: Any, tok: Any, context: str, answer: str, pre_code: str, max_length: int
) -> dict[str, Any] | None:
    device = next(base.parameters()).device
    full_ids, ans_ids = _prepare_ids(tok, context, answer, max_length)
    if len(ans_ids) < 2:
        return None
    edit_mask = _edit_mask(tok, pre_code, ans_ids)
    full = torch.tensor([full_ids], device=device)
    ans_only = torch.tensor([ans_ids], device=device)
    with torch.no_grad():
        teacher = base(full, use_cache=False).logits[0, -len(ans_ids) :].float()
        base_l = base(ans_only, use_cache=False).logits[0].float()
    # causal shift: logits[:-1] predict ans_ids[1:]
    gold = torch.tensor(ans_ids[1:], device=device)
    t_pred, b_pred = teacher[:-1], base_l[:-1]
    t_top1, b_top1 = t_pred.argmax(-1), b_pred.argmax(-1)
    t_correct = t_top1 == gold
    b_correct = b_top1 == gold
    diff = t_correct & (~b_correct)
    mask = torch.tensor(edit_mask[1:], device=device, dtype=torch.bool)
    t_nll = torch.nn.functional.cross_entropy(t_pred, gold).item()
    b_nll = torch.nn.functional.cross_entropy(b_pred, gold).item()

    def frac(x: Any, m: Any | None = None) -> float:
        if m is None:
            return float(x.float().mean())
        d = int(m.sum())
        return float((x & m).sum() / d) if d else 0.0

    return {
        "n_ans_tokens": len(ans_ids),
        "n_edit_tokens": int(mask.sum()),
        "teacher_acc": frac(t_correct),
        "base_acc": frac(b_correct),
        "teacher_nll": t_nll,
        "base_nll": b_nll,
        "nll_improvement": b_nll - t_nll,
        "diff_token_frac_whole": frac(diff),
        "diff_token_frac_edit": frac(diff, mask),
    }


def _agg(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument(
        "--corpus",
        type=str,
        default="/tmp/rune-corpus/external_codereview.unrolled.jsonl",
    )
    ap.add_argument(
        "--json-out", type=str, default="/tmp/rune-issue49-teacher-quality.json"
    )
    ap.add_argument("--model-id", type=str, default=load_rune_config().model_id)
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    base.eval()
    tok = AutoTokenizer.from_pretrained(args.model_id)

    # load + map rows, capture edit size (post_code len) + quality_score
    with open(args.corpus) as fh:
        raw = [json.loads(line) for line in fh if line.strip()]
    items = []
    for r in raw:
        m = _map_record(r)
        if m is None:
            continue
        items.append(
            {
                "context": m["context"],
                "answer": m["answer"],
                "pre_code": str(r.get("pre_code", "")),
                "edit_size": len(str(r.get("post_code", ""))),
                "quality": (r.get("metadata", {}) or {}).get(
                    "quality_score", r.get("quality_score")
                ),
            }
        )
    # leakage check (reviewer): context must NOT contain the ## Revision answer,
    # else base+context lift is a formatting/leakage artifact not trajectory memory.
    leakage_rows = sum(1 for it in items if "## Revision" in it["context"])
    items.sort(key=lambda x: x["edit_size"])
    # stratified sample: take every k-th so small+large both represented
    if len(items) > args.n:
        step = len(items) / args.n
        items = [items[int(i * step)] for i in range(args.n)]
    median_edit = items[len(items) // 2]["edit_size"] if items else 0

    rows = []
    for it in items:
        m = _row_metrics(
            base, tok, it["context"], it["answer"], it["pre_code"], args.max_length
        )
        if m is None:
            continue
        m["edit_size"] = it["edit_size"]
        m["quality"] = it["quality"]
        m["stratum"] = "large" if it["edit_size"] >= median_edit else "small"
        rows.append(m)

    keys = [
        "teacher_acc",
        "base_acc",
        "teacher_nll",
        "base_nll",
        "nll_improvement",
        "diff_token_frac_whole",
        "diff_token_frac_edit",
    ]
    overall = _agg(rows, keys)
    by_stratum = {
        s: _agg([r for r in rows if r["stratum"] == s], keys)
        for s in ("small", "large")
    }
    # quality correlation with edit-local diff-token fraction (Pearson, cheap)
    q = [
        (r["quality"], r["diff_token_frac_edit"])
        for r in rows
        if isinstance(r["quality"], (int, float))
    ]
    qcorr = None
    if len(q) > 2:
        qs = [a for a, _ in q]
        ds = [b for _, b in q]
        try:
            qcorr = statistics.correlation(qs, ds)
        except Exception:
            qcorr = None

    out = {
        "n_rows": len(rows),
        "leakage_rows_context_has_revision": leakage_rows,
        "median_edit_size": median_edit,
        "overall": overall,
        "by_stratum": by_stratum,
        "quality_vs_diffedit_corr": qcorr,
        # green light if edit-local distillable signal is clearly non-trivial
        "verdict_green": overall["diff_token_frac_edit"] >= 0.10
        and overall["nll_improvement"] > 0.0,
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print("LEAKAGE_ROWS (context has ## Revision):", leakage_rows)
    print("OVERALL:", json.dumps(overall))
    print("BY_STRATUM:", json.dumps(by_stratum))
    print(
        f"VERDICT_GREEN: {out['verdict_green']} "
        f"(edit diff-token frac={overall['diff_token_frac_edit']:.3f}, "
        f"nll_improvement={overall['nll_improvement']:.3f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
