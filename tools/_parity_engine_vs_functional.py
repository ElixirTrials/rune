"""Anchor-3 parity: ENGINE PEFT apply path vs FUNCTIONAL contract apply path.

Proves on a REAL checkpoint + REAL base that the two adapter-apply paths produce
the SAME logits for the SAME generated adapter and the SAME input ids:

  ENGINE     — ModelWrapper.from_config -> generate_adapter(text) ->
               hotswap_adapter (real peft.set_peft_model_state_dict) -> base
               PEFT forward. PEFT realizes lora_alpha_peft/r_peft == checkpoint
               lora_alpha (peft_scaling_params), applied un-divided.
  FUNCTIONAL — on the SAME loaded base+hypernet: extract_activations_with_model
               + hypernet.generate_weights (the raw lora_dict) -> assemble_adapter
               (+head bias iff use_bias) -> _functional_lora at effective_scaling
               -> base forward with the PEFT adapter DISABLED.

The reviewer's #1 concern is that engine PEFT scaling is proven only by
arithmetic; this is the real-model logit parity check. The engine side genuinely
goes through ModelWrapper's PEFT hotswap (NOT a hand-rolled apply).

LOCAL ONLY — underscore-prefixed, not part of the committed tree. CPU-importable
(all GPU imports deferred inside main). Do NOT run on the CPU-only worker.

Run on the GPU box (Rune venv):
  uv run python tools/_parity_engine_vs_functional.py --bf16
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

# A single, fixed trajectory doc + a single fixed input prompt. Both paths use
# the SAME doc (-> SAME adapter) and the SAME input ids (-> comparable logits).
DOC = (
    "def add(a, b):\n    return a + b\n\n"
    "# The helper above sums two integers. Tests assert add(2, 3) == 5.\n"
    "# A regression introduced a typo (a - b); the fix restores the sum.\n"
)
PROMPT = "Write a Python function that returns the sum of two integers."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="load base in bf16 (no 4-bit) — matches the validated _pathab_rune.py",
    )
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()

    import torch  # noqa: PLC0415

    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.model.adapter_contract import (  # noqa: PLC0415
        assemble_adapter,
        effective_scaling,
    )
    from rune.model.hypernetwork import (  # noqa: PLC0415
        _to_peft_state_dict,
        extract_activations_with_model,
    )
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.training.hypernet_distill import _functional_lora  # noqa: PLC0415

    # --- build the engine wrapper (loads bf16 base + flash_attention_2 + hypernet,
    # builds PEFT LoraConfig at r_peft/lora_alpha_peft). This is the engine path. ---
    cfg = PipelineConfig(model_id=args.model_id, checkpoint_path=args.ckpt)
    print(f"[load] ModelWrapper.from_config(model_id={args.model_id})", flush=True)
    wrapper = ModelWrapper.from_config(cfg)
    peft_model = wrapper._base_model  # PeftModel
    tok = wrapper._tokenizer
    hyp = wrapper._hypernet
    hyp.eval()
    peft_model.eval()
    device = next(peft_model.parameters()).device

    li = [int(x) for x in hyp.config.layer_indices]
    scaling = effective_scaling(hyp)
    use_bias = bool(getattr(hyp.config, "use_bias", False))
    print(
        f"[contract] layer_indices={li[:4]}... (n={len(li)}) "
        f"effective_scaling={scaling} use_bias={use_bias}",
        flush=True,
    )

    # --- shared input ids (chat-formatted prompt), identical for both paths ---
    chat = [{"role": "user", "content": PROMPT}]
    enc = tok.apply_chat_template(
        chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)

    # === FUNCTIONAL path FIRST — while the PEFT adapter is still zero-init. ===
    # Extraction goes through the PeftModel so its disable_adapter() runs (clean
    # base features). The functional forward needs the UNWRAPPED CausalLM because
    # _functional_lora's get_layers traverses .model.layers, not the PEFT wrapper
    # (PeftModel -> .base_model (LoraModel) -> .model (CausalLM) -> .model.layers).
    raw_base = peft_model.get_base_model()  # underlying CausalLM (no PEFT wrapper)
    hyp_device = next(hyp.parameters()).device
    hyp_dtype = next(hyp.parameters()).dtype

    feats, am = extract_activations_with_model(
        DOC, peft_model, tok, li, args.max_seq_length
    )
    feats = feats.to(device=hyp_device, dtype=hyp_dtype)
    am = am.to(hyp_device)
    with torch.no_grad():
        ld, _ = hyp.generate_weights(feats, am, None)
    n_chunks = torch.tensor([1], device=device)
    n_qs = torch.tensor([1], device=device)
    assembled = assemble_adapter(hyp, ld, n_chunks)

    # disable_adapter() zeroes the (shared) lora.Linear modules under raw_base's
    # forward too, so the functional delta is the ONLY adapter contribution — no
    # implicit reliance on lora_B still being at zero-init.
    with (
        torch.no_grad(),
        peft_model.disable_adapter(),
        _functional_lora(raw_base, li, assembled, scaling, n_qs),
    ):
        func_logits = raw_base(input_ids, use_cache=False).logits[0]

    # === ENGINE path — generate adapter from the SAME doc through the wrapper,
    # hot-swap via real peft.set_peft_model_state_dict, forward the SAME ids. ===
    result = wrapper.generate_adapter(DOC, offload_base=False)

    # Self-diagnostic: the two paths run generate_weights TWICE (engine inside
    # generate_adapter, functional inline above). Under eval()+no_grad these are
    # deterministic and should match. Compare the flattened adapter tensors so a
    # logit diff can be attributed to bf16 numerics vs a generation/assembly
    # divergence (which would NOT be a scaling-contract bug).
    target_modules = list(hyp.config.lora_config.target_modules)
    func_flat = _to_peft_state_dict(assembled, li, target_modules)
    eng_flat = result.state_dict
    adapter_max = 0.0
    missing = sorted(set(func_flat) ^ set(eng_flat))
    for k in func_flat:
        if k in eng_flat:
            d = (eng_flat[k].float() - func_flat[k].float()).abs().max()
            adapter_max = max(adapter_max, float(d))
    print(
        f"[adapter] key_diff_count={len(missing)} "
        f"max_abs_adapter_diff={adapter_max:.4e} "
        f"(0 => identical generation; logit diff is pure bf16 numerics)",
        flush=True,
    )
    # Gate the determinism assumption: if the two generate_weights calls diverged,
    # a logit diff would NOT be a clean scaling-contract signal. Fail loudly here
    # so a generation/extraction divergence can't masquerade as (or mask) parity.
    assert not missing, f"adapter key mismatch between paths: {missing[:8]}"
    assert adapter_max < 1e-3, (
        f"adapter generations diverged (max_abs_adapter_diff={adapter_max:.4e})"
    )

    wrapper.hotswap_adapter(result.state_dict)
    with torch.no_grad():
        engine_logits = peft_model(input_ids, use_cache=False).logits[0]

    # --- compare ---
    diff = (engine_logits.float() - func_logits.float()).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    # bf16 logits over a 4B model: a few-thousand-dim hidden state matmul carries
    # bf16 rounding; atol/rtol are loose enough for bf16 but tight enough that an
    # alpha/r (8x) scaling error or a missing bias half would blow past them.
    atol, rtol, mean_cap = 0.5, 0.02, 0.05
    rank = int(hyp.config.lora_config.r)
    r_peft = 2 * rank if use_bias else rank
    allclose_ok = torch.allclose(
        engine_logits.float(), func_logits.float(), atol=atol, rtol=rtol
    )
    # mean-drift backstop: a single near-zero logit may drift up to atol and still
    # pass element-wise, so guard the aggregate too (reviewer hardening).
    ok = bool(allclose_ok) and mean_abs < mean_cap
    print(
        f"\n=== PARITY engine(PEFT hotswap) vs functional(contract) ===\n"
        f"  regime: use_bias={use_bias} r_peft={r_peft} "
        f"effective_scaling={scaling}\n"
        f"  shape={tuple(engine_logits.shape)} dtype={engine_logits.dtype}\n"
        f"  max_abs_diff={max_abs:.4e}  mean_abs_diff={mean_abs:.4e} "
        f"(cap {mean_cap})\n"
        f"  allclose(atol={atol}, rtol={rtol}) = {allclose_ok}  PASS={ok}",
        flush=True,
    )
    # Also report argmax agreement on the last position (the generation-relevant
    # token) — a contract bug typically flips the top token.
    last_engine = int(engine_logits[-1].argmax())
    last_func = int(func_logits[-1].argmax())
    print(
        f"  last-token argmax: engine={last_engine} functional={last_func} "
        f"match={last_engine == last_func}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
