"""Cross-conditioned recitation probe (issue #52 pilot 2 — decisive recitation test).

The accessibility win (matched body lp up) raises a concern (user steer): does recall
drive the base to RECITE the stored body instead of GENERATING to the prompt spec? On
the same-task MBPP corpus adapter-content and prompt-spec never diverge, so present-
regime body-lp can't tell. This probe forces divergence:

  adapter conditioned on episode i  +  episode j's spec IN THE PROMPT (present regime)
  -> generate. HEALTHY = output defines j's entry_point (followed the prompt).
     RECITATION = output defines i's entry_point (adapter overrode the prompt).

Reports spec-follow vs recite. Run on warm-start AND trained: if trained recites MORE
than warm-start, the recall objective induced recitation.

Run (Rune venv, bf16): uv run python tools/_recitation_probe.py --ckpt <ckpt>
"""

from __future__ import annotations

import argparse
import json
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


def derangement(n: int) -> list[int]:
    perm = [(i + 1) % n for i in range(n)]
    if any(perm[i] == i for i in range(n)):
        raise ValueError("invalid derangement")
    return perm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from rune.engine.graph import render_training_format_trajectory  # noqa: PLC0415
    from rune.model.adapter_contract import (  # noqa: PLC0415
        effective_scaling,
    )
    from rune.model.hypernetwork import (  # noqa: PLC0415
        HypernetworkConfig,
        extract_activations_with_model,
        load_hypernetwork,
    )
    from rune.training.hypernet_distill import _functional_lora  # noqa: PLC0415

    tasks = json.loads(Path(TASKS_FILE).read_text())
    n = len(tasks)
    perm = derangement(n)

    print(f"loading base + hypernet ({a.ckpt.split('/')[-1]}) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        a.model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    ).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=a.ckpt), device="cuda")
    hyp.eval()
    scaling = effective_scaling(hyp)
    li = [int(x) for x in hyp.config.layer_indices]
    n_chunks = torch.tensor([1], device="cuda")
    n_qs = torch.tensor([1], device="cuda")
    head_bias = hyp.get_head_bias() if getattr(hyp.config, "use_bias", False) else None
    device = next(base.parameters()).device

    from ctx_to_lora.modeling.lora_merger import combine_lora  # noqa: PLC0415

    def adapter_for(desc: str):
        feats, am = extract_activations_with_model(
            render_training_format_trajectory(task=desc),
            base,
            tok,
            li,
            a.max_seq_length,
        )
        hd, ht = next(hyp.parameters()).device, next(hyp.parameters()).dtype
        feats = feats.to(device=hd, dtype=ht)
        am = am.to(next(hyp.parameters()).device)
        with torch.no_grad():
            ld, _ = hyp.generate_weights(feats, am, None)
        return combine_lora(ld, n_chunks, lora_bias=head_bias)

    follow = recite = neither = both = 0
    print("\n=== cross-conditioned: adapter=i, prompt-spec=j ===", flush=True)
    for i, t in enumerate(tasks):
        j = perm[i]
        adapter_i = adapter_for(t["description"])  # condition on episode i
        prompt = PRESENT.format(desc=tasks[j]["description"])  # j's spec in the prompt
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_special_tokens=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        pids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
        with torch.no_grad(), _functional_lora(base, li, adapter_i, scaling, n_qs):
            out = base.generate(
                pids,
                max_new_tokens=a.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][pids.shape[1] :], skip_special_tokens=True)
        ep_i, ep_j = t["entry_point"], tasks[j]["entry_point"]
        has_i = f"def {ep_i}(" in text or f"{ep_i}(" in text
        has_j = f"def {ep_j}(" in text or f"{ep_j}(" in text
        tag = (
            "BOTH"
            if has_i and has_j
            else "FOLLOW(j)"
            if has_j
            else "RECITE(i)"
            if has_i
            else "neither"
        )
        follow += has_j and not has_i
        recite += has_i and not has_j
        both += has_i and has_j
        neither += not has_i and not has_j
        print(
            f"  adapter={t['task_id']:8s} prompt={tasks[j]['task_id']:8s} "
            f"-> {tag:10s} (wants def {ep_j}; stored def {ep_i})",
            flush=True,
        )

    print(
        f"\n  SPEC-FOLLOW (j only) = {follow}/{n}   RECITE (i only) = {recite}/{n}   "
        f"both = {both}   neither = {neither}",
        flush=True,
    )
    print(
        "  HEALTHY if spec-follow dominates and recite is low (vs warm-start).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
