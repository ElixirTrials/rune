"""Anchor #4 (Rune side): free-form greedy generation at the contract scale.

Mirrors the Sakana code-recall baseline (rune_code_recall.py) but drives RUNE's
ENGINE apply path — ModelWrapper.from_config -> generate_adapter(DOC) ->
hotswap_adapter (real PEFT) -> plain greedy model.generate (NO xgrammar). At
adapter_scaling=1.0 the engine realizes effective scaling = checkpoint lora_alpha
(anchor #3 proved this matches the functional contract logit-for-logit).

Purpose: confirm Rune's OWN stack generates COHERENT free-form text at the
alpha scale (closing the autoregressive-compounding gap that anchor #3's single
forward pass leaves). This is an eyeball coherence check, NOT a pass@1 result and
NOT episode-specific-recall gating (Rune's ctx feature path is the known residual
vs Sakana's +2.2). If free-form gen is coherent here but Rune's xgrammar/MBPP
structured gen breaks, the break is the structured-decode/policy layer, not scale.

LOCAL ONLY — underscore-prefixed, not committed. CPU-importable (GPU imports
deferred). Run on the GPU box:
  uv run python tools/_rune_freeform_gen.py --bf16
"""

from __future__ import annotations

import argparse
import sys

RUNE = "/workspaces/rune-gpu"
CKPT = (
    f"{RUNE}/third_party/doc-to-lora/trained_d2l/"
    "qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)
BASE = "Qwen/Qwen3-4B-Instruct-2507"

# Same snippet + queries as the Sakana code-recall baseline, so coherence (and,
# loosely, recall) is directly comparable across the two stacks.
DOC = (
    "```python\n"
    "RETRY_BUDGET = 7\n\n"
    "def quarkle_resync(payload, attempts):\n"
    "    # reconcile the ledger against the upstream shard\n"
    "    if attempts > RETRY_BUDGET:\n"
    "        return 'ABANDONED'\n"
    "    checksum = (payload * 31 + 17) % 9973\n"
    "    return checksum\n"
    "```\n"
)
QUERIES = [
    "What is the name of the function defined in the code?",
    "What string does the function return when attempts exceed the budget?",
    "Write a Python function that returns the sum of two integers.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--bf16", action="store_true", help="(engine is always bf16)")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    import torch  # noqa: PLC0415

    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    cfg = PipelineConfig(model_id=args.model_id, checkpoint_path=args.ckpt)
    print(f"[load] ModelWrapper.from_config(model_id={args.model_id})", flush=True)
    wrapper = ModelWrapper.from_config(cfg)
    peft_model = wrapper._base_model
    tok = wrapper._tokenizer
    peft_model.eval()
    device = next(peft_model.parameters()).device

    hc = wrapper._hypernet.config
    print(
        f"[contract] use_bias={getattr(hc, 'use_bias', False)} "
        f"r={hc.lora_config.r} lora_alpha={hc.lora_config.lora_alpha} "
        "adapter_scaling=1.0 (effective=lora_alpha)",
        flush=True,
    )

    def gen(ids: torch.Tensor) -> str:
        with torch.no_grad():
            out = peft_model.generate(
                input_ids=ids, max_new_tokens=args.max_new, do_sample=False
            )
        text = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        return str(text).strip()

    # Generate the adapter ONCE from the DOC via the engine path, hot-swap it.
    result = wrapper.generate_adapter(DOC, offload_base=False)

    for q in QUERIES:
        chat = [{"role": "user", "content": q}]
        enc = tok.apply_chat_template(
            chat,
            add_special_tokens=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)

        # base (adapter disabled) for contrast
        with peft_model.disable_adapter():
            base_txt = gen(ids)
        # adapter applied at the contract scale
        wrapper.hotswap_adapter(result.state_dict)
        adapt_txt = gen(ids)

        print(f"\n=== QUERY: {q!r} ===", flush=True)
        print(f"  [base ] {base_txt!r}", flush=True)
        print(f"  [adapt] {adapt_txt!r}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
