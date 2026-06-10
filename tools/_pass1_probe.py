"""Gate-1 pass@1 probe (issue #52 pilot 2) — functional correctness, REAL MBPP tests.

Tests whether adapter-conditioned generation produces CORRECT runnable code, scored against
the canonical 3-test MBPP suites (load_mbpp_tasks, NOT the single gameable docstring assert).
Two regimes per episode:
  PRESENT  (task spec IN the prompt)  = STABILITY gate: trained must not regress vs warm-start.
  ABSENT   ("write the fn you studied", spec NOT in prompt) = CAPABILITY: recall-from-memory.

FRAMING (advisor): n=10, binary pass@1 is COARSE -> report DESCRIPTIVELY + per-episode paired
(which episodes flip; do they align with the lp_matched risers?), NOT an independent
significance test. This is raw greedy generate + sandbox asserts, NOT the xgrammar-constrained
path — it is a generation-stability floor, not xgrammar pass@1.

Run on warm-start AND the trained ckpt:
  uv run python tools/_pass1_probe.py --ckpt <ckpt> --out /tmp/pass1_X.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from rune.config import load_rune_config

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")

BASE = load_rune_config().model_id
CKPT = f"{RUNE}/third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
TASKS_FILE = f"{RUNE}/benchmarks/mbpp_phase0_iter.json"
PRESENT = "Write the following Python function.\n\n{desc}\n\nReturn only the code."
ABSENT = "Write the Python function you have just studied. Return only the code."
_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _FENCE.search(text)
    return m.group(1) if m else text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="held-out jsonl (task_id/description/entry_point/test_code); "
        "overrides the frozen 10",
    )
    ap.add_argument(
        "--scale0",
        action="store_true",
        help="base only, NO adapter (the scale=0 floor); skips hypernet load",
    )
    a = ap.parse_args()

    from ctx_to_lora.modeling.lora_merger import combine_lora  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from rune.bench.mbpp import load_mbpp_tasks  # noqa: PLC0415
    from rune.engine.continuation import strip_self_tests  # noqa: PLC0415
    from rune.engine.graph import render_training_format_trajectory  # noqa: PLC0415
    from rune.model.adapter_contract import effective_scaling  # noqa: PLC0415
    from rune.model.hypernetwork import (  # noqa: PLC0415
        HypernetworkConfig,
        extract_activations_with_model,
        load_hypernetwork,
    )
    from rune.sandbox.executor import run_in_sandbox  # noqa: PLC0415
    from rune.training.hypernet_distill import _functional_lora  # noqa: PLC0415

    if a.corpus:
        # held-out corpus carries test_code inline (no load_mbpp_tasks needed).
        episodes = [
            json.loads(ln)
            for ln in Path(a.corpus).read_text().splitlines()
            if ln.strip()
        ]
        tests = {e["task_id"]: e["test_code"] for e in episodes}
    else:
        episodes = json.loads(Path(TASKS_FILE).read_text())
        ids = {e["task_id"] for e in episodes}
        # REAL MBPP 3-test suites keyed by task_id (not the single docstring assert).
        tasks = {t.task_id: t for t in load_mbpp_tasks(ids=ids)}
        tests = {tid: t.test_code for tid, t in tasks.items()}
        if ids - set(tests):
            print(
                f"[WARN] no MBPP test suite for: {sorted(ids - set(tests))}", flush=True
            )

    tag = "scale=0 (base)" if a.scale0 else a.ckpt.split("/")[-1]
    print(f"loading base{'' if a.scale0 else ' + hypernet'} ({tag}) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        a.model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    ).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    device = next(base.parameters()).device
    n_qs = torch.tensor([1], device="cuda")
    hyp = scaling = li = n_chunks = head_bias = hd = ht = None
    if not a.scale0:
        hyp = load_hypernetwork(
            HypernetworkConfig(checkpoint_path=a.ckpt), device="cuda"
        )
        hyp.eval()
        scaling = effective_scaling(hyp)
        li = [int(x) for x in hyp.config.layer_indices]
        n_chunks = torch.tensor([1], device="cuda")
        head_bias = (
            hyp.get_head_bias() if getattr(hyp.config, "use_bias", False) else None
        )
        hd, ht = next(hyp.parameters()).device, next(hyp.parameters()).dtype

    def adapter_for(desc: str):
        feats, am = extract_activations_with_model(
            render_training_format_trajectory(task=desc),
            base,
            tok,
            li,
            a.max_seq_length,
        )
        with torch.no_grad():
            ld, _ = hyp.generate_weights(feats.to(device=hd, dtype=ht), am.to(hd), None)
        return combine_lora(ld, n_chunks, lora_bias=head_bias)

    def gen(adapter, prompt: str) -> str:
        import contextlib  # noqa: PLC0415

        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_special_tokens=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        pids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
        # scale=0 -> no adapter context (plain base); else apply the functional LoRA.
        ctx = (
            contextlib.nullcontext()
            if adapter is None
            else _functional_lora(base, li, adapter, scaling, n_qs)
        )
        with torch.no_grad(), ctx:
            out = base.generate(
                pids,
                max_new_tokens=a.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][pids.shape[1] :], skip_special_tokens=True)

    def passes(code: str, test_code: str) -> bool:
        full = strip_self_tests(extract_code(code)) + "\n\n" + test_code
        try:
            return run_in_sandbox(full, timeout=15).exit_code == 0
        except Exception:  # noqa: BLE001
            return False

    rows = []
    pres_pass = abs_pass = n = 0
    print(
        "\n=== pass@1 (REAL MBPP tests) — present=stability, absent=capability ===",
        flush=True,
    )
    for e in episodes:
        tid = e["task_id"]
        if tid not in tests:
            continue
        n += 1
        tc = tests[tid]
        ad = None if a.scale0 else adapter_for(e["description"])
        p_ok = passes(gen(ad, PRESENT.format(desc=e["description"])), tc)
        a_ok = passes(gen(ad, ABSENT), tc)
        pres_pass += p_ok
        abs_pass += a_ok
        rows.append({"task_id": tid, "present_pass": p_ok, "absent_pass": a_ok})
        print(
            f"  {tid:8s} present={'PASS' if p_ok else 'fail'}  "
            f"absent={'PASS' if a_ok else 'fail'}",
            flush=True,
        )

    print(f"\n  PRESENT pass@1 (stability) = {pres_pass}/{n}", flush=True)
    print(f"  ABSENT  pass@1 (capability) = {abs_pass}/{n}", flush=True)
    print(
        f"  n={n}, binary: descriptive only — pair per-episode vs baseline, not a sig test.",
        flush=True,
    )
    if a.out:
        Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"  [dump] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
