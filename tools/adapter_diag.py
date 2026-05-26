# tools/adapter_diag.py
"""Quick adapter diagnostics — checks why adapters may have zero effect.

Run: uv run python tools/adapter_diag.py [--config benchmarks/bench.yaml]

All generation params and sweep values come from bench.yaml.

Checks (in order of likelihood):
  1. layer_indices non-empty (empty = hotswap is a no-op)
  2. generated state_dict has keys and non-zero magnitudes
  3. PEFT key compatibility (missing_keys / unexpected_keys)
  4. hotswap actually mutates live model weights
  5. generation changes after hotswap vs baseline
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("benchmarks/bench.yaml")


def main() -> None:
    from rune.config import load_config  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Adapter diagnostic checks",
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help=f"Config YAML path (default: {_DEFAULT_CONFIG})",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.checkpoint_path:
        parser.error("Config must set checkpoint_path")

    bench = cfg.bench
    gen_kwargs = {
        "max_tokens": bench["gen_max_tokens"],
        "temperature": bench["gen_temperature"],
    }
    diag_sweep: list[float] = bench["diag_scaling_sweep"]

    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    model = ModelWrapper.from_config(cfg)

    # --- Check 1: layer_indices ---
    li = model._layer_indices
    print("\n=== CHECK 1: layer_indices ===")
    print(f"  len={len(li)}, values={li}")
    if not li:
        print(
            "  ** FAIL: layer_indices is empty — "
            "_to_peft_state_dict produces an empty dict, "
            "hotswap is a no-op."
        )
        return

    # --- Check 2: generated state_dict keys + magnitudes ---
    trajectory = (
        "ROLE: coder\nTASK: Write add(a,b)\nPLAN: Implement add."
    )
    adapter = model.generate_adapter(trajectory)
    sd = adapter.state_dict
    print("\n=== CHECK 2: adapter state_dict ===")
    print(f"  num_keys={len(sd)}")
    if not sd:
        print(
            "  ** FAIL: state_dict is empty — "
            "hypernetwork produced no weights."
        )
        return
    print(f"  first 4 keys: {list(sd.keys())[:4]}")
    import torch  # noqa: PLC0415
    mags = {
        k: v.abs().mean().item()
        for k, v in list(sd.items())[:8]
    }
    print(f"  magnitudes (first 8): {mags}")
    all_zero = all(v < 1e-12 for v in mags.values())
    if all_zero:
        print("  ** FAIL: all adapter weights are near-zero.")

    # --- Check 3: PEFT key compatibility ---
    from peft import set_peft_model_state_dict  # noqa: PLC0415

    result = set_peft_model_state_dict(model._base_model, sd)
    print("\n=== CHECK 3: PEFT key compatibility ===")
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    print(f"  missing_keys ({len(missing)}): {missing[:5]}")
    print(f"  unexpected_keys ({len(unexpected)}): {unexpected[:5]}")
    if unexpected:
        print(
            "  ** FAIL: PEFT rejected these keys — "
            "hotswap silently no-ops."
        )
        print("  Expected key format sample: ", end="")
        for name, _ in model._base_model.named_parameters():
            if "lora_A" in name:
                print(name)
                break

    # --- Check 4: hotswap mutates live weights ---
    print("\n=== CHECK 4: hotswap mutation ===")
    ref_param_name = None
    for name, _p in model._base_model.named_parameters():
        if "lora_A" in name:
            ref_param_name = name
            break
    if ref_param_name is None:
        print("  ** FAIL: no lora_A parameter found in model.")
        return
    ref_param = dict(model._base_model.named_parameters())[
        ref_param_name
    ]
    before = ref_param.data.clone()

    from typing import Any as _Any  # noqa: PLC0415

    def _scale_b_only(
        raw_sd: dict[str, _Any], s: float,
    ) -> dict[str, _Any]:
        return {
            k: v * s if "lora_B" in k else v
            for k, v in raw_sd.items()
        }

    scaling = cfg.adapter_scaling
    scaled_sd = _scale_b_only(sd, scaling)
    model.hotswap_adapter(scaled_sd)

    after = ref_param.data.clone()
    diff = (after - before).abs().mean().item()
    print(f"  param: {ref_param_name}")
    print(f"  before mean: {before.abs().mean().item():.6e}")
    print(f"  after  mean: {after.abs().mean().item():.6e}")
    print(f"  diff   mean: {diff:.6e}")
    if diff < 1e-12:
        print("  ** FAIL: hotswap did not change model weights.")
    else:
        print("  OK: weights changed after hotswap.")

    # --- Check 5: generation effect (B-only linear scaling) ---
    print("\n=== CHECK 5: generation effect (B-only scaling) ===")
    prompt = (
        "Write a Python function add(a, b) that returns a + b."
    )
    sys_prompt = "You are a code generator."

    zero_sd = {k: torch.zeros_like(v) for k, v in sd.items()}
    model.hotswap_adapter(zero_sd)
    baseline = asyncio.run(
        model.generate(
            prompt, system_prompt=sys_prompt, **gen_kwargs,
        )
    )
    print(f"  baseline[:80]: {baseline.text[:80]!r}")

    for s in diag_sweep:
        model.hotswap_adapter(_scale_b_only(sd, s))
        out = asyncio.run(
            model.generate(
                prompt, system_prompt=sys_prompt, **gen_kwargs,
            )
        )
        differs = out.text != baseline.text
        print(
            f"  scale={s:<6} differs={differs}"
            f"  out[:80]: {out.text[:80]!r}"
        )

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
