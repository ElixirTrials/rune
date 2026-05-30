"""Combined adapter validation at the corrected PEFT scaling (lora_alpha=alpha).

Tests the user's three acceptance criteria in ONE model load:
  (1) terminates  — structured CodeResult generation closes the JSON, no cap hit
  (2) passes      — extracted code passes the task assert
  (3) recall      — output is trajectory-sensitive: a contradictory trajectory
                    diverges from the task trajectory (adapter encodes the ctx)

Reuses adapter_probe.py's trajectory definitions. mmap'd checkpoint load +
offload_base=False keep it within the ~15GB CPU RAM box; run under
/tmp/run_guarded.sh so it can never OOM the VM.

Run:  bash /tmp/run_guarded.sh /tmp/adapter_scaling_run.log \
        tools/diag_adapter_scaling_probe.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

OUT = Path("/tmp/adapter_scaling_results.jsonl")

# --- recall conditions (verbatim from adapter_probe.py) ---
ADD_PROMPT = "Write a Python function add(a, b) that returns a + b."
TASK_TRAJ = f"ROLE: coder\nTASK: {ADD_PROMPT}\nPLAN: Implement add function."
CONTRA_TRAJ = (
    f"ROLE: coder\nTASK: {ADD_PROMPT}\n"
    "PLAN: Implement a sorting routine.\n"
    "INTERFACE: def sort_list(items: list[int]) -> list[int]"
)

# --- terminate+pass condition ---
MBPP_PROMPT = (
    "Write a function to find tuples which have all elements divisible by k "
    "from the given list of tuples.\n\n"
    ">>> assert find_tuples([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) "
    "== [(6, 24, 12)]\n\n"
    "Return a JSON object with a single key 'code' whose value is the complete "
    "Python function as a string."
)
MBPP_TRAJ = (
    "ROLE: coder\nTASK: find_tuples(lst, k) -> tuples whose every element is "
    "divisible by k.\nPLAN: filter with all(x % k == 0 for x in t)."
)


def _log(rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def _coherent(text: str) -> bool:
    if not text:
        return False
    printable = sum(c.isprintable() or c in "\n\t" for c in text)
    return printable / len(text) > 0.95


def _passes_find_tuples(code: str) -> bool:
    try:
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102
        fn = ns.get("find_tuples")
        return bool(fn) and fn(
            [(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6
        ) == [(6, 24, 12)]
    except Exception:  # noqa: BLE001
        return False


async def main() -> None:
    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.engine.parse import CodeResult  # noqa: PLC0415
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

    # Pre-generate adapters for each trajectory (reused across scalings).
    adapters = {
        "task": wrapper.generate_adapter(TASK_TRAJ).state_dict,
        "contra": wrapper.generate_adapter(CONTRA_TRAJ).state_dict,
    }
    _log({"event": "adapters_ready", "trajectories": list(adapters)})

    async def gen(sd: dict[str, Any], scaling: float, prompt: str,
                  schema: Any, max_tokens: int) -> Any:
        wrapper.hotswap_adapter(scale_lora_b(sd, scaling))
        return await wrapper.generate(
            prompt,
            system_prompt="You are a code generator.",
            output_schema=schema,
            max_tokens=max_tokens,
            temperature=0.0,
            repetition_penalty=1.0,
            top_p=1.0,
            presence_penalty=1.5,
            thinking_budget=0,
            skip_completion_retry=True,
        )

    # --- (3) RECALL: task vs contradictory trajectory, across scalings ---
    # effective scaling = PEFT(alpha/r=2.0) x adapter_scaling
    for scaling in (0.0, 0.49, 1.0, 2.0):
        task_out = (await gen(adapters["task"], scaling, ADD_PROMPT, None, 160)).text
        contra_out = (
            await gen(adapters["contra"], scaling, ADD_PROMPT, None, 160)
        ).text
        _log(
            {
                "event": "recall",
                "adapter_scaling": scaling,
                "eff_scaling_approx": round(2.0 * scaling, 3),
                "task_vs_contra_differ": task_out != contra_out,
                "task_coherent": _coherent(task_out),
                "contra_coherent": _coherent(contra_out),
                "contra_mentions_sort": "sort" in contra_out.lower(),
                "task_head": task_out[:160],
                "contra_head": contra_out[:160],
            }
        )

    # --- (1)+(2) TERMINATE + PASS: structured MBPP at bench scaling 0.49 ---
    mbpp_sd = wrapper.generate_adapter(MBPP_TRAJ).state_dict
    for scaling in (0.49, 1.0):
        t0 = time.monotonic()
        res = await gen(mbpp_sd, scaling, MBPP_PROMPT, CodeResult, 512)
        wall = time.monotonic() - t0
        parseable, code = False, ""
        try:
            code = CodeResult.model_validate_json(res.text).code
            parseable = True
        except Exception:  # noqa: BLE001
            pass
        _log(
            {
                "event": "terminate_pass",
                "adapter_scaling": scaling,
                "eff_scaling_approx": round(2.0 * scaling, 3),
                "wall_s": round(wall, 1),
                "tokens_used": res.tokens_used,
                "truncated_hit_cap": res.truncated,
                "parseable_json": parseable,
                "passes_assert": _passes_find_tuples(code) if parseable else False,
                "coherent": _coherent(res.text),
                "text_tail": res.text[-160:],
            }
        )

    _log({"event": "done"})


if __name__ == "__main__":
    asyncio.run(main())
