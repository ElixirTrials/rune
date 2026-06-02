"""Lever-B step 1: the IN-CONTEXT CEILING gate for the one-step `avoid` task.

Before building the adapter/feedback-swap apparatus, check the avoid task is even
well-posed (advisor + D2 reviewer): does the critique, delivered IN THE PROMPT,
shift the accepted-over-rejected preference? If not, a flat *adapter* result is
uninterpretable ("fact unstorable" vs "task unsolvable").

Base model ONLY (no hypernet, no adapter) — fast. Episodes from the external
code-review corpus: each REPLACE hunk gives a rejected pre-side region and an
accepted post-side region for the same edit slot; the critique is the review
feedback. Neutral scaffold = the accepted file's lines BEFORE the hunk (common to
both candidates; leaks neither the rejected region nor the feedback).

Difference-in-differences (cancels accepted/rejected intrinsic-likelihood bias,
which appears in both conditions — the candidates are different token strings, so
only the cross-condition DiD is apples-to-apples):
    pref(cond) = mean_lp(accepted | prompt_cond) - mean_lp(rejected | prompt_cond)
    gate_DiD   = pref(critique-in-prompt) - pref(no-critique)
GATE PASS = mean gate_DiD > 0 with a clear majority frac(>0): the critique is
usable when handed to the model directly -> the avoid task is well-posed -> build
the adapter apparatus. Flat/negative -> ill-posed; stop and report.

Run in RUNE's venv: uv run python tools/_avoid_ceiling_probe.py --bf16
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys

import torch

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")
sys.path.insert(0, f"{RUNE}/tools/d2l_control")

import scoring_core  # noqa: E402
from episodes import extract_review_feedback  # noqa: E402

BASE = "Qwen/Qwen3-4B-Instruct-2507"
CORPUS = "/tmp/rune-corpus/external_codereview.val.clean.jsonl"
N_EPISODES = 30
MAX_HUNK_LINES = 6  # keep candidate regions short -> cleaner scoring
CTX_LINES = 20  # recent accepted-file lines before the hunk as scaffold
MAX_CAND_TOK = 64


def build_episodes(n: int, single_hunk: bool = True, scan_cap: int = 100000) -> list[dict]:
    """Corpus rows -> {context, accepted, rejected, feedback}.

    single_hunk (a-priori structural filter, advisor): keep only diffs whose ONLY
    change opcode is a single replace. Removes the wrong-hunk confound — when the
    diff has one edit, a directive critique must concern THAT edit, so the
    accepted-vs-rejected scoring targets the same hunk the critique addresses.
    Chosen blind to DiD sign (no result-dependent selection).
    """
    eps: list[dict] = []
    scanned = 0
    with open(CORPUS) as f:
        for line in f:
            scanned += 1
            if scanned > scan_cap:
                break
            r = json.loads(line)
            at = str(r.get("activation_text", ""))
            pre = str(r.get("pre_code", ""))
            post = str(r.get("post_code", ""))
            fb = extract_review_feedback(at)
            if not fb:
                continue
            pl = pre.splitlines()
            ql = post.splitlines()
            sm = difflib.SequenceMatcher(None, pl, ql)
            ops = sm.get_opcodes()
            changes = [op for op in ops if op[0] != "equal"]
            replaces = [
                (i1, i2, j1, j2)
                for tag, i1, i2, j1, j2 in changes
                if tag == "replace" and i2 > i1 and j2 > j1
            ]
            if single_hunk:
                # exactly one change opcode, and it is the replace
                if len(changes) != 1 or len(replaces) != 1:
                    continue
            if not replaces:
                continue
            i1, i2, j1, j2 = replaces[0]
            if j1 < 1:  # need a non-empty prefix scaffold
                continue
            if (i2 - i1) > MAX_HUNK_LINES or (j2 - j1) > MAX_HUNK_LINES:
                continue
            eps.append(
                {
                    "task_id": str(r.get("task_id", "")),
                    "context": "\n".join(ql[max(0, j1 - CTX_LINES) : j1]),
                    "accepted": "\n".join(ql[j1:j2]),
                    "rejected": "\n".join(pl[i1:i2]),
                    "feedback": fb,
                }
            )
            if len(eps) >= n:
                break
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--n", type=int, default=N_EPISODES)
    a = ap.parse_args()

    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    eps = build_episodes(a.n)
    print(f"built {len(eps)} avoid episodes (replace-hunk + feedback)", flush=True)
    if len(eps) < 5:
        print("too few episodes")
        return 1

    load_kw = dict(
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    )
    if not a.bf16:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    print(f"loading base ({'bf16' if a.bf16 else '4bit'})...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(a.model_id, **load_kw).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    device = next(base.parameters()).device

    def score(prompt: str, cand: str) -> float:
        p = tok(prompt, add_special_tokens=False).input_ids
        c = tok(cand, add_special_tokens=False).input_ids[:MAX_CAND_TOK]
        if len(c) < 1 or len(p) < 1:
            return float("nan")
        ids = torch.tensor([p + c], device=device)
        with torch.no_grad():
            lg = base(ids, use_cache=False).logits[0]
        return scoring_core.mean_gold_logprob(lg, ids[0], len(p), len(c))

    rows = []
    for e in eps:
        ctx = e["context"]
        crit_ctx = ctx + "\n# Reviewer requests: " + e["feedback"].strip() + "\n"
        # no-critique condition
        lp_acc_nc = score(ctx + "\n", e["accepted"])
        lp_rej_nc = score(ctx + "\n", e["rejected"])
        # critique-in-prompt condition
        lp_acc_c = score(crit_ctx, e["accepted"])
        lp_rej_c = score(crit_ctx, e["rejected"])
        if any(x != x for x in (lp_acc_nc, lp_rej_nc, lp_acc_c, lp_rej_c)):  # NaN
            continue
        pref_nc = lp_acc_nc - lp_rej_nc
        pref_c = lp_acc_c - lp_rej_c
        did = pref_c - pref_nc
        rows.append((e["task_id"], pref_nc, pref_c, did))
        print(
            f"  {e['task_id'][:40]:40s} pref_nocrit={pref_nc:+.3f}"
            f"  pref_crit={pref_c:+.3f}  DiD={did:+.3f}",
            flush=True,
        )

    n = len(rows)
    mean_nc = sum(r[1] for r in rows) / n
    mean_c = sum(r[2] for r in rows) / n
    mean_did = sum(r[3] for r in rows) / n
    frac = sum(1 for r in rows if r[3] > 0) / n
    print(
        f"\n=== CEILING GATE (n={n}) ===\n"
        f"  mean pref_nocrit = {mean_nc:+.4f}  (base accepted-vs-rejected, no critique)\n"
        f"  mean pref_crit   = {mean_c:+.4f}  (critique IN PROMPT)\n"
        f"  mean gate_DiD    = {mean_did:+.4f}  frac(DiD>0)={frac:.2f}\n"
        f"  VERDICT: {'PASS - task well-posed, build adapter apparatus' if (mean_did > 0 and frac >= 0.6) else 'FAIL/WEAK - avoid task ill-posed; flat adapter result would be uninterpretable'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
