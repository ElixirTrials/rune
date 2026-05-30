"""Continuation probe — reproduce the VALIDATED adapter-as-memory signal.

Prior probes tested fresh single-shot generation (wrong scenario). The adapter
was trained on continuation/revision (## Task / ## Current Code <diff> / ##
Review Feedback -> ## Revision) and its demonstrated value is COMPLETING code
across the token boundary: real-trajectory > no-adapter > contradictory.

This uses the engine's REAL continuation path (generate_continuation) with a
partial function the model must complete. The user prompt carries NO task
semantics (only the bare signature is in assistant_prefix); the ADAPTER is the
only source of "keep tuples where ALL elements are divisible by K". So:
  - REAL adapter (this task + 'use all(...)') -> correct completion, passes assert
  - ZERO adapter                              -> worse (no semantic guidance)
  - CONTRA adapter (reverse-words task)       -> worst (actively misleads)
across scalings INCLUDING the ~12 continuation regime my earlier probes (capped
at 7.84) never reached. Continuation effective scaling in the engine today is
~1.5 (PEFT 2.0 x adapter_scaling 0.49 x cont_multiplier 1.53); pre-fix it was
~12 (PEFT 16). If REAL>>ZERO only at high scaling, the fix is to restore
continuation scaling (raise cont_multiplier ~8x), decoupled from the first round.

Run under /tmp/run_guarded.sh.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

OUT = Path("/tmp/continuation_results.jsonl")
PEFT_SCALING = 2.0

# Training-format conditioning (## Task / ## Current Code / ## Review Feedback).
FIND_TRAJ = (
    "## Task\nImplement find_tuples(test_list, K): return the tuples whose every "
    "element is divisible by K.\n\n"
    "## Current Code\ndef find_tuples(test_list, K):\n    result = []\n    for "
    "sub in test_list:\n        # TODO: keep sub only if ALL its elements % K == 0\n"
    "\n## Review Feedback\nComplete the loop body: append sub to result only when "
    "all(ele % K == 0 for ele in sub); then return result. Use all(), not any()."
)
CONTRA_TRAJ = (
    "## Task\nImplement reverse_words(s): reverse the order of the words in the "
    "sentence string s.\n\n"
    "## Current Code\ndef reverse_words(s):\n    words = s.split()\n\n"
    "## Review Feedback\nReverse the word list and join with single spaces; "
    "return the result."
)

# The model must CONTINUE this — body truncated right before the condition.
ASSISTANT_PREFIX = (
    "def find_tuples(test_list, K):\n    result = []\n    for sub in test_list:\n"
    "        "
)
CONT_SYSTEM = (
    "Output only Python code. No commentary, no markdown fences. Continue exactly "
    "from where the code left off."
)
USER_PROMPT = "Complete the implementation."  # neutral; no task semantics

ASSERT_SRC = (
    "assert find_tuples([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) "
    "== [(6, 24, 12)]"
)
ASSERT_NEG = (  # the any()-bug returns all tuples -> this would be wrong
    "assert find_tuples([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) "
    "!= [(6, 24, 12), (7, 9, 6), (12, 18, 21)]"
)


def _log(rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def _passes(full_code: str) -> bool:
    ns: dict[str, Any] = {}
    try:
        exec(full_code, ns)  # noqa: S102 - GPU-box probe
        exec(ASSERT_SRC, ns)  # noqa: S102
        exec(ASSERT_NEG, ns)  # noqa: S102
    except Exception:
        return False
    return True


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

    sd_real = wrapper.generate_adapter(FIND_TRAJ).state_dict
    sd_contra = wrapper.generate_adapter(CONTRA_TRAJ).state_dict
    zero = {k: v * 0.0 for k, v in sd_real.items()}
    _log({"event": "adapters_ready"})

    async def cont(sd: dict[str, Any], eff: float) -> str:
        wrapper.hotswap_adapter(scale_lora_b(sd, eff / PEFT_SCALING))
        r = await wrapper.generate_continuation(
            system_prompt=CONT_SYSTEM,
            user_prompt=USER_PROMPT,
            assistant_prefix=ASSISTANT_PREFIX,
            max_tokens=200,
            temperature=0.0,
            repetition_penalty=1.0,
            top_p=1.0,
            presence_penalty=1.5,
        )
        return r.text

    # Continuation effective scaling: current engine ~1.5; validated regime ~12.
    for eff in (1.5, 4.0, 8.0, 12.0, 16.0):
        rec: dict[str, Any] = {"event": "cont", "eff": eff}
        for label, sd in (("real", sd_real), ("zero", zero), ("contra", sd_contra)):
            try:
                chunk = await cont(sd, eff)
            except Exception as exc:  # noqa: BLE001
                rec[f"{label}_err"] = repr(exc)[:160]
                chunk = ""
            full = ASSISTANT_PREFIX + chunk
            rec[f"{label}_uses_all"] = "all(" in chunk
            rec[f"{label}_uses_any_bug"] = "any(" in chunk and "all(" not in chunk
            rec[f"{label}_passes"] = _passes(full)
            rec[f"{label}_cont_head"] = chunk[:120]
        _log(rec)

    _log({"event": "done"})


if __name__ == "__main__":
    asyncio.run(main())
