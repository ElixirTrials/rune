"""Decisive probe: does the hypernetwork adapter's effective scaling drive the
non-termination / pass@1=0 failure?

Base-model probe (diag_inference_probe.py) already showed grammar decoding
terminates cleanly with NO adapter.  This isolates the one variable the failing
runs add: the generated LoRA adapter, swept across effective scaling.

Rune builds the PEFT model at lora_alpha=alpha*rank=128, r=8 -> PEFT scaling
16.0 (8x the reference alpha/r=2.0).  We build once at 16.0, generate ONE
adapter, then hot-swap scale_lora_b(orig_sd, S/16) to realise effective scaling
S, running the same MBPP task each time.

Outcomes:
  - S=16 runs away (truncated, unparseable) and a lower S terminates + passes
    -> scaling bug confirmed; the passing S is the trained scaling.
  - every S>0 garbage -> adapter quality, not scaling.
  - S=16 already clean -> scaling not the cause; escalate (template / 3072).

Run:  uv run python tools/diag_adapter_scaling_probe.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

OUT = Path("/tmp/adapter_scaling_results.jsonl")

TASK = (
    "Write a function to find tuples which have all elements divisible by k "
    "from the given list of tuples.\n\n"
    ">>> assert find_tuples([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) "
    "== [(6, 24, 12)]"
)

# A representative coding trajectory to condition the hypernetwork.
TRAJECTORY = (
    "Task: implement find_tuples(tuples_list, k) returning tuples whose every "
    "element is divisible by k.\n"
    "Plan: iterate the list, keep a tuple when all(x % k == 0).\n"
    "Code: def find_tuples(lst, k): return [t for t in lst if all(x % k == 0 "
    "for x in t)]\n"
)

CHECKPOINT = "/home/vscode/.cache/rune/checkpoints/8e815654733a4579.pt"
BUILD_SCALING = 16.0  # PEFT scaling the model is built at (alpha*rank/r)


def _log(rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1), flush=True)


def _passes(code: str) -> bool:
    """Exec the extracted code and run the docstring assert (trivial, safe)."""
    try:
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102
        fn = ns.get("find_tuples")
        if fn is None:
            return False
        return fn([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) == [(6, 24, 12)]
    except Exception:  # noqa: BLE001
        return False


async def main() -> None:
    import torch  # noqa: PLC0415

    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.engine.parse import CodeResult  # noqa: PLC0415
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    OUT.write_text("")

    cfg = PipelineConfig(checkpoint_path=CHECKPOINT)
    t0 = time.monotonic()
    wrapper = ModelWrapper.from_config(cfg)
    _log({"event": "loaded", "load_s": round(time.monotonic() - t0, 1)})

    # Generate ONE adapter. offload_base=False: base(18GB)+hypernet(0.9GB) fit on
    # the 23GB card, and this box has only 15GB CPU RAM so offloading the 18GB
    # base to CPU would trip the kernel OOM-killer (silent SIGKILL).
    t0 = time.monotonic()
    adapter = wrapper.generate_adapter(TRAJECTORY, offload_base=False)
    orig_sd = adapter.state_dict
    b_keys = [k for k in orig_sd if "lora_B" in k]
    sample_b_norm = float(orig_sd[b_keys[0]].float().norm()) if b_keys else None
    _log(
        {
            "event": "adapter",
            "gen_s": round(time.monotonic() - t0, 1),
            "n_keys": len(orig_sd),
            "n_lora_B": len(b_keys),
            "sample_B_fro_norm": round(sample_b_norm, 3)
            if sample_b_norm is not None
            else None,
        }
    )

    # Free the hypernetwork so the base model has room for the generation sweep.
    wrapper._hypernet.to("cpu")  # noqa: SLF001
    torch.cuda.empty_cache()

    # effective scaling S realised via scale_lora_b(orig_sd, S / BUILD_SCALING).
    # Covers the V1 working regime (~0.15 = 2.0 PEFT x 0.075 adapter_scaling),
    # intermediate, and the current broken V2 value (16.0).
    for eff in (0.0, 0.075, 0.15, 0.3, 0.5, 1.0, 2.0, 16.0):
        factor = eff / BUILD_SCALING
        sd = scale_lora_b(orig_sd, factor)
        wrapper.hotswap_adapter(sd)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        try:
            res = await wrapper.generate(
                TASK + "\n\nReturn a JSON object with a single key 'code' whose "
                "value is the complete Python function as a string.",
                system_prompt="You are a code generator.",
                output_schema=CodeResult,
                max_tokens=512,
                temperature=0.0,
                repetition_penalty=1.0,
                top_p=1.0,
                presence_penalty=1.5,
                thinking_budget=256,
                skip_completion_retry=True,
            )
            wall = time.monotonic() - t0
            parseable = False
            code = ""
            try:
                code = CodeResult.model_validate_json(res.text).code
                parseable = True
            except Exception:  # noqa: BLE001
                parseable = False
            _log(
                {
                    "event": "gen",
                    "eff_scaling": eff,
                    "scale_lora_b_factor": round(factor, 4),
                    "wall_s": round(wall, 1),
                    "tokens_used": res.tokens_used,
                    "tok_per_s": round(res.tokens_used / wall, 2) if wall else None,
                    "truncated_hit_cap": res.truncated,
                    "parseable_json": parseable,
                    "passes_assert": _passes(code) if parseable else False,
                    "text_chars": len(res.text),
                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "text_head": res.text[:280],
                    "text_tail": res.text[-160:],
                }
            )
        except Exception as e:  # noqa: BLE001
            import traceback  # noqa: PLC0415

            _log(
                {
                    "event": "error",
                    "eff_scaling": eff,
                    "error": str(e),
                    "tb": traceback.format_exc()[-800:],
                }
            )

    _log({"event": "done"})


if __name__ == "__main__":
    asyncio.run(main())
