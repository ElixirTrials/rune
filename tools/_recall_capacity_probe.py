"""Recall-CAPACITY probe (issue #52 goal-1, the decisive experiment) — REMOVE-BEFORE-MERGE.

THE LINCHPIN: can ONE adapter hold MULTIPLE tasks in memory and still recall EACH on demand?
The warm start is doc-to-lora (one doc -> one adapter); multi-task conditioning may not survive.
This probe is the failure-bearing signal the handoff wants before any Phase-2 spend.

Protocol (extends `_pass1_probe` ABSENT regime to multi-task accumulation; the engine is the WRONG
vehicle because it puts prior state in the PROMPT, making scale=0 a dirty control — see scratchpad
2026-06-04 09:05):
  - Partition the corpus into disjoint blocks of k tasks (k in --k-values).
  - Condition ONE adapter on the concatenation of the k task descriptions ("study" -> adapter memory,
    NOT the prompt).
  - Query each of the k tasks NAME-CUED with the spec ABSENT:
        "Write the Python function named `{entry_point}` that you have just studied."
    Score pass@1 against the REAL MBPP 3-test suite.
  - The query prompt is name-only -> ~FLAT token length regardless of k, while the adapter
    conditioning grows. Flat-prompt + growing-memory IS the adapter-as-memory thesis.

Win condition (pre-registered, reflections 2026-06-04): adapter (c3 / warm) pass@1 >> scale=0 with
the SAME name-cued prompt. If scale=0 solves it from the name alone, the name leaks the spec and the
adapter is not doing the work. Watch per-k decay (capacity) and cross-task interference (the emitted
`def NAME(` != the queried name = wrong-function retrieval, the recitation failure mode).

Run:
  uv run python tools/_recall_capacity_probe.py --ckpt <c3.pt> --corpus <heldout.jsonl> \
      --k-values 1,2,4,8 --out /tmp/cap_c3.jsonl
  ... --scale0   (the floor)            ... --ckpt <warm.bin>   (warm-start arm)
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
WARM = (
    f"{RUNE}/third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)
DEFAULT_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
NAME_CUE = (
    "Write the Python function named `{name}` that you have just studied. Return only the code."
)
_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_DEFNAME = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)


def extract_code(text: str) -> str:
    m = _FENCE.search(text)
    return m.group(1) if m else text


def chunks(seq: list, k: int) -> list[list]:
    """Disjoint blocks of size k; trailing partial block dropped (keeps per-k n clean)."""
    return [seq[i : i + k] for i in range(0, len(seq) - k + 1, k)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--corpus", type=str, default=f"{RUNE}/benchmarks/mbpp_recall_heldout.jsonl")
    ap.add_argument("--k-values", type=str, default="1,2,4,8")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--scale0", action="store_true",
                    help="base only, NO adapter — the floor; name cue still given")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    k_values = [int(x) for x in a.k_values.split(",") if x.strip()]

    from ctx_to_lora.modeling.lora_merger import combine_lora  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

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

    episodes = [json.loads(ln) for ln in Path(a.corpus).read_text().splitlines() if ln.strip()]
    tests = {e["task_id"]: e["test_code"] for e in episodes}

    tag = "scale=0 (base)" if a.scale0 else a.ckpt.split("/")[-1]
    print(f"loading base{'' if a.scale0 else ' + hypernet'} ({tag}) "
          f"| corpus={Path(a.corpus).name} n={len(episodes)} | k={k_values}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        a.model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    ).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    device = next(base.parameters()).device
    n_qs = torch.tensor([1], device="cuda")
    hyp = scaling = li = n_chunks = head_bias = hd = ht = None
    if not a.scale0:
        hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=a.ckpt), device="cuda")
        hyp.eval()
        scaling = effective_scaling(hyp)
        li = [int(x) for x in hyp.config.layer_indices]
        n_chunks = torch.tensor([1], device="cuda")
        head_bias = hyp.get_head_bias() if getattr(hyp.config, "use_bias", False) else None
        hd, ht = next(hyp.parameters()).device, next(hyp.parameters()).dtype

    def adapter_for_block(descs: list[str]):
        """Condition ONE adapter on the concatenation of k task descriptions."""
        study = "\n\n".join(descs)
        feats, am = extract_activations_with_model(
            render_training_format_trajectory(task=study), base, tok, li, a.max_seq_length
        )
        with torch.no_grad():
            ld, _ = hyp.generate_weights(feats.to(device=hd, dtype=ht), am.to(hd), None)
        study_tokens = int(am.sum().item()) if hasattr(am, "sum") else None
        return combine_lora(ld, n_chunks, lora_bias=head_bias), study_tokens

    def gen(adapter, prompt: str) -> tuple[str, int]:
        import contextlib  # noqa: PLC0415

        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_special_tokens=False,
            add_generation_prompt=True, return_tensors="pt",
        )
        pids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
        ctx = (
            contextlib.nullcontext()
            if adapter is None
            else _functional_lora(base, li, adapter, scaling, n_qs)
        )
        with torch.no_grad(), ctx:
            out = base.generate(
                pids, max_new_tokens=a.max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][pids.shape[1]:], skip_special_tokens=True), int(pids.shape[1])

    def passes(code: str, test_code: str) -> bool:
        full = strip_self_tests(extract_code(code)) + "\n\n" + test_code
        try:
            return run_in_sandbox(full, timeout=15).exit_code == 0
        except Exception:  # noqa: BLE001
            return False

    rows = []
    summary: dict[int, dict[str, float]] = {}
    print("\n=== recall capacity: name-cued, spec ABSENT, per k ===", flush=True)
    for k in k_values:
        blocks = chunks(episodes, k)
        kp = kn = 0
        interference = 0
        for bi, block in enumerate(blocks):
            descs = [e["description"] for e in block]
            study_tokens = None
            if a.scale0:
                adapter = None
            else:
                adapter, study_tokens = adapter_for_block(descs)
            for pos, e in enumerate(block):
                tid, name = e["task_id"], e["entry_point"]
                if tid not in tests:
                    continue
                kn += 1
                code, ptoks = gen(adapter, NAME_CUE.format(name=name))
                ok = passes(code, tests[tid])
                kp += ok
                emitted = _DEFNAME.search(extract_code(code))
                emitted_name = emitted.group(1) if emitted else None
                # cross-task interference: emitted a DIFFERENT studied function's name.
                other_names = {x["entry_point"] for x in block} - {name}
                if emitted_name in other_names:
                    interference += 1
                rows.append({
                    "k": k, "block": bi, "pos": pos, "task_id": tid, "entry_point": name,
                    "pass": bool(ok), "emitted_def": emitted_name,
                    "prompt_tokens": ptoks, "study_tokens": study_tokens,
                })
        rate = kp / kn if kn else 0.0
        summary[k] = {"pass1": f"{kp}/{kn}", "rate": round(rate, 3),
                      "interference": interference, "blocks": len(blocks)}
        print(f"  k={k:2d}: pass@1={kp}/{kn} ({rate:.3f})  interference={interference}  "
              f"blocks={len(blocks)}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps({"arm": tag, "summary": summary}, indent=2), flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"  [dump] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
