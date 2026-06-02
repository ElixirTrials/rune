"""Gate: env+load — load Qwen3-4B base + qwen_4b_d2l hypernet via Rune's own path."""

from __future__ import annotations

import json
import traceback
from dataclasses import replace

import torch

from rune.config import PipelineConfig
from rune.model.wrapper import ModelWrapper

CKPT = (
    "/workspaces/rune-gpu/third_party/doc-to-lora/trained_d2l/"
    "qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)


def _scaler_b_absmean(hypernet: object) -> dict[str, float]:
    sb = getattr(hypernet, "scaler_B", None)
    if sb is None:
        return {"present": False}
    vals = []
    n = 0
    try:
        items = sb.items()  # ParameterDict / ModuleDict / dict
    except AttributeError:
        items = []
    for _, t in items:
        with torch.no_grad():
            vals.append(float(t.detach().abs().sum().item()))
            n += int(t.numel())
    total = sum(vals)
    return {
        "present": True,
        "n_entries": len(vals),
        "n_elems": n,
        "absmean": (total / n) if n else 0.0,
    }


def main() -> None:
    out: dict[str, object] = {"gate": "env+load"}
    try:
        base = PipelineConfig()
        cfg = replace(
            base,
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            checkpoint_path=CKPT,
            adapter_scaling=0.0,
        )
        wrapper = ModelWrapper.from_config(cfg)
        hypernet = wrapper._hypernet  # noqa: SLF001
        hc = hypernet.config
        lc = hc.lora_config
        out["load_ok"] = True
        out["layer_indices"] = list(hc.layer_indices)
        out["layer_indices_len"] = len(list(hc.layer_indices))
        out["target_modules"] = list(lc.target_modules)
        out["lora_r"] = int(lc.r)
        out["lora_alpha"] = int(getattr(lc, "lora_alpha", lc.r * 2))
        out["scaler_B"] = _scaler_b_absmean(hypernet)
        out["model_id"] = cfg.model_id
    except Exception as e:  # noqa: BLE001
        out["load_ok"] = False
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
    print("GATE_RESULT_JSON " + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
