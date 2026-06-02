"""Temp-sampled avoid pairs — merged YIELD + in-context CEILING probe (advisor).

Goal: build clean (reject, hidden-critique, accept) triples for an avoid-UTILITY
test, avoiding the redundancy trap. The reject must PASS the visible example
assert but FAIL a HIDDEN held-out case; the critique = that hidden failure. Then
the fact the adapter would carry is NOT in the scoring prompt (unlike goal-3
PRESENT / the external_codereview ceiling).

Leakage rule (advisor): held-out tests used OFFLINE to LABEL candidates and form
the critique is legitimate corpus construction. Held-out never enters the model's
prompt or an engine signal.

One base+adapter run on the frozen 10:
  1. per task: sample N candidates at temperature from the adapter@contract path.
  2. keep syntactically valid (ast-parse); strip self-tests.
  3. classify vs example assert (visible) AND held-out tests (offline oracle):
       pass-both          -> ACCEPT
       pass-ex / fail-hid -> GOOD reject (wrong on a hidden case only)
       fail-example       -> leaked reject, DISCARD (critique would be prompt-visible)
  4. YIELD = tasks with >=1 accept AND >=1 good reject.
  5. CEILING on good pairs (base-only, adapter OFF): accept-vs-reject preference
     DiD with the HIDDEN-failure critique in prompt vs not. DiD cancels the
     accept/reject intrinsic-likelihood bias.

Build the adapter/feedback-swap apparatus ONLY if BOTH yield>0 and ceiling clears.
Pre-registered: pass-ex/fail-hidden is rare on easy MBPP -> likely near-zero ->
signal = need harder tasks (a scope call). Probe returns a yes/no, not a number.

Run: uv run python tools/_temp_sample_avoid_probe.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import torch

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")
import scoring_core  # noqa: E402

TASKS_FILE = f"{RUNE}/benchmarks/mbpp_phase0_iter.json"
CKPT = (
    f"{RUNE}/third_party/doc-to-lora/trained_d2l/"
    "qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)
BASE = "Qwen/Qwen3-4B-Instruct-2507"
_ASSERT_RE = re.compile(r">>>\s*(assert .+)")
MAX_ANS_TOK = 160


def example_assert(desc: str) -> str:
    m = _ASSERT_RE.search(desc)
    return m.group(1).strip() if m else ""


def extract_code(text: str, entry_point: str) -> str | None:
    """Best-effort: pull a syntactically-valid module defining entry_point."""
    fence = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
    body = fence.group(1) if fence else text
    idx = body.find("def ")
    if idx < 0:
        return None
    cand = body[idx:]
    try:
        tree = ast.parse(cand)
    except (SyntaxError, ValueError):
        # trim trailing prose lines until it parses or runs out
        lines = cand.splitlines()
        while lines:
            lines.pop()
            try:
                tree = ast.parse("\n".join(lines))
                cand = "\n".join(lines)
                break
            except (SyntaxError, ValueError):
                continue
        else:
            return None
    has_fn = any(
        isinstance(n, ast.FunctionDef) and n.name == entry_point
        for n in ast.walk(tree)
    )
    return cand if has_fn else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--tasks-file", type=str, default=TASKS_FILE)
    ap.add_argument("--exclude", type=str, default="", help="comma task_ids to skip")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", type=str, default="", help="dump harvested pairs JSON")
    a = ap.parse_args()

    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.engine.continuation import strip_self_tests  # noqa: PLC0415
    from rune.engine.graph import render_training_format_trajectory  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.sandbox.executor import run_in_sandbox  # noqa: PLC0415

    cfg = PipelineConfig(model_id=a.model_id, checkpoint_path=a.ckpt)
    print(f"[load] {a.model_id}", flush=True)
    wrapper = ModelWrapper.from_config(cfg)
    peft_model = wrapper._base_model
    tok = wrapper._tokenizer
    peft_model.eval()
    device = next(peft_model.parameters()).device

    tasks = json.loads(Path(a.tasks_file).read_text())
    skip = {x for x in a.exclude.split(",") if x}
    tasks = [t for t in tasks if t["task_id"] not in skip]
    if a.limit:
        tasks = tasks[: a.limit]
    print(f"[tasks] {len(tasks)} from {a.tasks_file} (excluded {len(skip)})", flush=True)

    def passes(code: str, asserts: str) -> bool:
        prog = strip_self_tests(code) + "\n" + asserts + "\n"
        return run_in_sandbox(prog).exit_code == 0

    def failing_held_out(code: str, test_code: str) -> str:
        """Return the first held-out assert the code fails (the hidden critique)."""
        for line in test_code.splitlines():
            if line.strip().startswith("assert") and not passes(code, line):
                return line.strip()
        return ""

    pairs: dict[str, dict] = {}  # task_id -> {accept, reject, critique, desc}
    print("\n=== YIELD (per task) ===", flush=True)
    for t in tasks:
        desc, ep, tc = t["description"], t["entry_point"], t["test_code"]
        ex = example_assert(desc)
        traj = render_training_format_trajectory(task=desc)
        sd = wrapper.generate_adapter(traj, offload_base=False).state_dict
        wrapper.hotswap_adapter(sd)
        prompt = (
            f"Write the Python function. {desc}\nReturn only the code, no explanation."
        )
        chat = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            chat, add_special_tokens=False, add_generation_prompt=True,
            return_tensors="pt",
        )
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
        with torch.no_grad():
            out = peft_model.generate(
                input_ids=ids, max_new_tokens=a.max_new, do_sample=True,
                temperature=a.temp, top_p=0.95, num_return_sequences=a.n,
            )
        cands = [
            extract_code(tok.decode(o[ids.shape[1]:], skip_special_tokens=True), ep)
            for o in out
        ]
        cands = [c for c in cands if c]
        accepts, good_rejects, leaked = [], [], 0
        for c in cands:
            pe = passes(c, ex)
            ph = passes(c, tc)
            if ph:
                accepts.append(c)
            elif pe:  # passes visible example, fails hidden -> GOOD reject
                good_rejects.append(c)
            else:
                leaked += 1
        print(
            f"  {t['task_id']:8s} valid={len(cands):2d}/{a.n} accept={len(accepts)}"
            f" good_reject={len(good_rejects)} leaked={leaked}",
            flush=True,
        )
        if accepts and good_rejects:
            rej = good_rejects[0]
            pairs[t["task_id"]] = {
                "accept": accepts[0],
                "reject": rej,
                "critique": failing_held_out(rej, tc),
                "desc": desc,
            }

    n_pair = len(pairs)
    print(f"\nYIELD: {n_pair}/{len(tasks)} tasks have an accept + good-reject pair", flush=True)
    if a.out and pairs:
        Path(a.out).write_text(json.dumps(pairs, indent=2))
        print(f"[dump] {n_pair} pairs -> {a.out}", flush=True)
    if n_pair == 0:
        print(
            "VERDICT: ZERO good pairs on frozen 10 (pre-registered) -> easy MBPP cannot "
            "support a clean avoid-utility test; SCOPE CALL: harder tasks. Ceiling skipped.",
            flush=True,
        )
        return 0

    # ---- in-context ceiling on the good pairs (base-only, adapter OFF) ----
    def mean_lp(prompt: str, code: str) -> float:
        p = tok(prompt, add_special_tokens=False).input_ids
        c = tok(code, add_special_tokens=False).input_ids[:MAX_ANS_TOK]
        seq = torch.tensor([p + c], device=device)
        with torch.no_grad(), peft_model.disable_adapter():
            lg = peft_model(seq, use_cache=False).logits[0]
        return scoring_core.mean_gold_logprob(lg, seq[0], len(p), len(c))

    print("\n=== CEILING on good pairs (base-only, hidden-failure critique) ===", flush=True)
    rows = []
    for tid, p in pairs.items():
        base_ctx = f"# Task:\n{p['desc']}\n\n# Solution:\n"
        crit_ctx = (
            f"# Task:\n{p['desc']}\n\n"
            f"# A prior attempt failed this hidden case: {p['critique']}\n\n# Solution:\n"
        )
        pref_nc = mean_lp(base_ctx, p["accept"]) - mean_lp(base_ctx, p["reject"])
        pref_c = mean_lp(crit_ctx, p["accept"]) - mean_lp(crit_ctx, p["reject"])
        did = pref_c - pref_nc
        rows.append((tid, did))
        print(f"  {tid:8s} crit={p['critique'][:50]!r} DiD={did:+.3f}", flush=True)
    mean_did = sum(r[1] for r in rows) / len(rows)
    frac = sum(1 for r in rows if r[1] > 0) / len(rows)
    print(
        f"\nCEILING (n={len(rows)}): mean DiD={mean_did:+.4f} frac(>0)={frac:.2f}\n"
        f"VERDICT: {'PASS - pairs well-posed, build adapter/feedback-swap apparatus' if (mean_did > 0 and frac >= 0.6) else 'WEAK - even oracle critique barely moves pref; reconsider before apparatus'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
