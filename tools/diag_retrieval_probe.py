"""Retrieval probe — is the trajectory/context actually EMBEDDED in the adapter
and RETRIEVABLE? (doc-to-lora style needle recall, decoupled from code-gen.)

Put UNIQUE facts the base model cannot guess into the conditioning trajectory,
encode it into the adapter, then ask the model to recall them with a LEAN prompt
(facts NOT in the prompt). If the model returns 73921 / frobnicate_payload /
_tally_zorblax, the adapter embedded the context and it is retrievable. Compare
WITH adapter (real) vs NO adapter (zero) vs CONTRADICTORY adapter (different
facts). The base model alone (zero) cannot know these -> any recall is the
adapter's doing.

This is the cleanest "is it embedded / can we retrieve it" test; prior probes
conflated embedding with using-it-to-write-correct-code.

Run under /tmp/run_guarded.sh.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

OUT = Path("/tmp/retrieval_results.jsonl")
PEFT_SCALING = 2.0

# Unique, unguessable facts embedded ONLY in the trajectory.
REAL_TRAJ = (
    "## Task\nImplement the billing module per the spec below.\n\n"
    "## Current Code\nMAGIC_OFFSET = 73921\nRETRY_CEILING = 449\n\n"
    "def frobnicate_payload(items):\n"
    "    return sum(i * i for i in items) + MAGIC_OFFSET\n\n"
    "def _tally_zorblax(x):\n    return x % RETRY_CEILING\n\n"
    "## Review Feedback\nKeep MAGIC_OFFSET = 73921 and the helper name "
    "_tally_zorblax; the public entry point stays frobnicate_payload."
)
CONTRA_TRAJ = (
    "## Task\nImplement the billing module per the spec below.\n\n"
    "## Current Code\nMAGIC_OFFSET = 11111\nRETRY_CEILING = 222\n\n"
    "def gizmo_handler(items):\n"
    "    return sum(items) + MAGIC_OFFSET\n\n"
    "def _scan_blarg(x):\n    return x % RETRY_CEILING\n\n"
    "## Review Feedback\nKeep MAGIC_OFFSET = 11111 and the helper _scan_blarg."
)

# Recall queries — facts are NOT in the prompt. (query_id, prompt, needle)
QUERIES = [
    ("offset_cont", "MAGIC_OFFSET =", "73921"),
    ("offset_qa",
     "From the code in your loaded context, the integer value of MAGIC_OFFSET "
     "is exactly", "73921"),
    ("fn_name",
     "From your loaded context, the public payload-processing function is named",
     "frobnicate_payload"),
    ("helper_name",
     "From your loaded context, the private helper function is named",
     "_tally_zorblax"),
]


def _log(rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


async def main() -> None:
    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415
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

    sd_real = wrapper.generate_adapter(REAL_TRAJ).state_dict
    sd_contra = wrapper.generate_adapter(CONTRA_TRAJ).state_dict
    zero = {k: v * 0.0 for k, v in sd_real.items()}
    _log({"event": "adapters_ready"})

    async def ask(sd: dict[str, Any], eff: float, prompt: str) -> str:
        wrapper.hotswap_adapter(scale_lora_b(sd, eff / PEFT_SCALING))
        r = await wrapper.generate(
            prompt,
            system_prompt="Answer using only the information in your loaded "
            "context. Be terse.",
            output_schema=None,
            max_tokens=40,
            temperature=0.0,
            repetition_penalty=1.0,
            top_p=1.0,
            presence_penalty=0.0,
            thinking_budget=0,
            skip_completion_retry=True,
        )
        return r.text

    for eff in (1.5, 8.0, 12.0, 16.0):
        for qid, prompt, needle in QUERIES:
            rec: dict[str, Any] = {"event": "recall", "eff": eff, "query": qid,
                                   "needle": needle}
            for label, sd in (("real", sd_real), ("zero", zero),
                              ("contra", sd_contra)):
                try:
                    out = await ask(sd, eff, prompt)
                except Exception as exc:  # noqa: BLE001
                    out = f"<err:{exc!r}>"
                rec[f"{label}_hit"] = needle in out
                rec[f"{label}_out"] = out[:80]
            _log(rec)

    _log({"event": "done"})


if __name__ == "__main__":
    asyncio.run(main())
