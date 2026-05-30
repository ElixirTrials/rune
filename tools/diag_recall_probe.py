"""Recall probe: does the adapter ENCODE the conditioning trajectory?

Two levels, both at the corrected PEFT scaling (lora_alpha=alpha):
  (A) WEIGHT level — generate adapters from two very different trajectories and
      measure relative L2 / cosine between their B@A deltas.  If distinct
      trajectories yield near-identical adapters, the hypernetwork is not
      encoding the trajectory (a real failure, independent of decoding).
  (B) LOGIT level — apply each adapter + a zero baseline over a fixed neutral
      prompt and measure KL divergence between next-token distributions.
      Trajectory sensitivity that argmax-on-trivial-task misses shows up here.

Run under /tmp/run_guarded.sh.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

OUT = Path("/tmp/recall_results.jsonl")

TRAJ_A = (
    "ROLE: coder\nTASK: implement binary search over a sorted integer list.\n"
    "PLAN: maintain lo/hi pointers, compare midpoint, halve the range.\n"
    "INTERFACE: def binary_search(items: list[int], target: int) -> int"
)
TRAJ_B = (
    "ROLE: coder\nTASK: compute Fibonacci numbers recursively.\n"
    "PLAN: base cases n<2, else fib(n-1)+fib(n-2).\n"
    "INTERFACE: def fib(n: int) -> int"
)
NEUTRAL_PROMPT = "Write a Python function for the task described."


def _log(rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def main() -> None:
    import torch  # noqa: PLC0415

    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    OUT.write_text("")
    cfg = PipelineConfig(
        checkpoint_path=(
            "s3://elixirtrials-949678234935-eu-west-2-artifacts/"
            "checkpoints/hypernet_hpo/checkpoint.pt"
        )
    )
    t0 = time.monotonic()
    wrapper = ModelWrapper.from_config(cfg)
    _log({"event": "loaded", "load_s": round(time.monotonic() - t0, 1)})

    sd_a = wrapper.generate_adapter(TRAJ_A).state_dict
    sd_b = wrapper.generate_adapter(TRAJ_B).state_dict

    # (A) WEIGHT level: per-layer delta = B @ A; compare A-traj vs B-traj.
    rels: list[float] = []
    coss: list[float] = []
    for k in sd_a:
        if "lora_B" not in k:
            continue
        ka = k.replace("lora_B", "lora_A")
        da = (sd_b[k].float() @ sd_b[ka].float())
        d0 = (sd_a[k].float() @ sd_a[ka].float())
        rel = (da - d0).norm().item() / (d0.norm().item() + 1e-12)
        cos = torch.nn.functional.cosine_similarity(
            d0.flatten(), da.flatten(), dim=0
        ).item()
        rels.append(rel)
        coss.append(cos)
    _log(
        {
            "event": "weight_divergence",
            "n_layers": len(rels),
            "mean_rel_L2_delta": round(sum(rels) / len(rels), 4),
            "mean_cosine_AB": round(sum(coss) / len(coss), 4),
            "note": "rel_L2~0 & cosine~1 => adapters identical => no encoding",
        }
    )

    # (B) LOGIT level: next-token distribution divergence over NEUTRAL_PROMPT.
    tok = wrapper._tokenizer  # noqa: SLF001
    model = wrapper._base_model  # noqa: SLF001
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415

    msgs = [
        {"role": "system", "content": "You are a code generator."},
        {"role": "user", "content": NEUTRAL_PROMPT},
    ]
    ids = tok.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=False
    )
    ids = (ids["input_ids"] if hasattr(ids, "input_ids") else ids).to(model.device)

    def last_logits(sd: dict[str, Any], scaling: float) -> Any:
        wrapper.hotswap_adapter(scale_lora_b(sd, scaling))
        with torch.no_grad():
            out = model(ids)
        return out.logits[0, -1].float()

    zero = {k: torch.zeros_like(v) for k, v in sd_a.items()}
    for scaling in (0.49, 1.0, 2.0):
        lp_base = torch.log_softmax(last_logits(zero, 0.0), dim=-1)
        lp_a = torch.log_softmax(last_logits(sd_a, scaling), dim=-1)
        lp_b = torch.log_softmax(last_logits(sd_b, scaling), dim=-1)
        kl = torch.nn.functional.kl_div
        _log(
            {
                "event": "logit_divergence",
                "adapter_scaling": scaling,
                "eff_scaling_approx": round(2.0 * scaling, 3),
                "KL_A_vs_base": round(kl(lp_base, lp_a, log_target=True,
                                         reduction="sum").item(), 5),
                "KL_B_vs_base": round(kl(lp_base, lp_b, log_target=True,
                                         reduction="sum").item(), 5),
                "KL_A_vs_B": round(kl(lp_b, lp_a, log_target=True,
                                      reduction="sum").item(), 5),
                "argmax_A": int(lp_a.argmax()),
                "argmax_B": int(lp_b.argmax()),
                "argmax_differs": int(lp_a.argmax()) != int(lp_b.argmax()),
            }
        )

    _log({"event": "done"})


if __name__ == "__main__":
    main()
