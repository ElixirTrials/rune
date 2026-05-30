"""Decisive experiment: scaling x output-mode x conditioning.

Answers two questions the prior fix left open:
  Q1 (conditioning): is HIGH effective scaling needed for the adapter to encode
     the trajectory?  Prior research (adapter-as-memory-report, 2026-05-28) found
     the memory effect at effective ~7.84-12.  My lora_alpha=alpha fix dropped
     bench effective to ~0.98 (8x below that) and recall looked weak.
  Q2 (termination): can we get pass@1 WITHOUT the xgrammar JSON runaway?  At high
     scaling structured JSON never closed (calib 11165-char runaway).  Does
     FREEFORM generation terminate + pass at the SAME high scaling?

Current code has PEFT scaling = alpha/r = 2.0 (my fix).  Effective scaling is
realised via scale_lora_b(sd, eff/2.0).  We sweep effective scaling across the
validated-high regime that my earlier probes never reached.

Part 1 TERMINATION+PASS:  full prompt (task present), find_tuples task,
  structured (xgrammar CodeResult) vs freeform, across effective scaling.
Part 2 CONDITIONING:  LEAN prompt (NO task text — adapter is the only source),
  two different real-code.j2 trajectories, freeform, across scaling.  If high
  scaling makes the two outputs diverge and each matches its trajectory's task,
  the adapter encodes the trajectory.

Run under /tmp/run_guarded.sh.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

OUT = Path("/tmp/scaling_mode_results.jsonl")
PEFT_SCALING = 2.0  # current lora_alpha=alpha, r=8 -> alpha/r = 2.0

PRACTICES = (
    "PRACTICES: Clean layered architecture, no stubs or placeholders, no dead "
    "code, specific exceptions with context."
)

# --- real code.j2-format trajectories (fresh task: no prior/existing code) ---
FIND_DESC = (
    "Write a function find_tuples(tuples_list, k) to find tuples which have all "
    "elements divisible by k from the given list of tuples."
)
REV_DESC = (
    "Write a function reverse_words(s) that reverses the order of words in the "
    "given sentence string."
)


def _code_traj(desc: str) -> str:
    return (
        "ROLE: coder\n"
        f"PROJECT: {desc[:300]}\n"
        "SUBTASK: _main (1/1)\n"
        f"DESCRIPTION: {desc[:500]}\n\n"
        "PLAN:\n"
        f"{desc[:1200]}\n"
        f"{PRACTICES}\n"
    )


# Full prompt: task present (mimics engine for the find_tuples pass test).
FIND_FULL_PROMPT = (
    "Write a function to find tuples which have all elements divisible by k "
    "from the given list of tuples.\n\n"
    ">>> assert find_tuples([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) "
    "== [(6, 24, 12)]"
)
FIND_FULL_PROMPT_JSON = (
    FIND_FULL_PROMPT
    + "\n\nReturn a JSON object with a single key 'code' whose value is the "
    "complete Python function as a string."
)
# Lean prompt: NO task text — the adapter is the only source of the task.
LEAN_PROMPT = (
    "Implement the subtask described in your loaded context. "
    "Write the complete Python function."
)


def _log(rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def _coherent(t: str) -> bool:
    return bool(t) and sum(c.isprintable() or c in "\n\t" for c in t) / len(t) > 0.95


def _passes_find(code: str) -> bool:
    try:
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102
        fn = ns.get("find_tuples")
        return bool(fn) and fn([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) == [
            (6, 24, 12)
        ]
    except Exception:  # noqa: BLE001
        return False


async def main() -> None:
    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.engine.continuation import extract_partial_code  # noqa: PLC0415
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

    sd_find = wrapper.generate_adapter(_code_traj(FIND_DESC)).state_dict
    sd_rev = wrapper.generate_adapter(_code_traj(REV_DESC)).state_dict
    _log({"event": "adapters_ready"})

    async def gen(
        sd: dict[str, Any], eff: float, prompt: str, schema: Any, max_tokens: int
    ) -> Any:
        wrapper.hotswap_adapter(scale_lora_b(sd, eff / PEFT_SCALING))
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

    EFFS = [0.0, 2.0, 4.0, 7.84, 12.0]

    # ---- Part 1: TERMINATION + PASS, structured vs freeform (Q2) ----
    for eff in EFFS:
        # structured (xgrammar JSON)
        r = await gen(sd_find, eff, FIND_FULL_PROMPT_JSON, CodeResult, 768)
        code = ""
        ok = False
        try:
            code = CodeResult.model_validate_json(r.text).code
            ok = True
        except Exception:  # noqa: BLE001
            code = extract_partial_code(r.text)
        _log(
            {
                "event": "term",
                "mode": "structured",
                "eff": eff,
                "tokens": r.tokens_used,
                "truncated": r.truncated,
                "parseable": ok,
                "passes": _passes_find(code),
                "coherent": _coherent(r.text),
                "tail": r.text[-120:],
            }
        )
        # freeform (no grammar)
        r = await gen(sd_find, eff, FIND_FULL_PROMPT, None, 768)
        code = extract_partial_code(r.text)
        _log(
            {
                "event": "term",
                "mode": "freeform",
                "eff": eff,
                "tokens": r.tokens_used,
                "truncated": r.truncated,
                "passes": _passes_find(code),
                "coherent": _coherent(r.text),
                "tail": r.text[-120:],
            }
        )

    # ---- Part 2: CONDITIONING with LEAN prompt (Q1) ----
    # No task text in the prompt: the adapter is the ONLY task source.
    for eff in EFFS:
        rf = await gen(sd_find, eff, LEAN_PROMPT, None, 512)
        rr = await gen(sd_rev, eff, LEAN_PROMPT, None, 512)
        tf, tr = rf.text.lower(), rr.text.lower()
        _log(
            {
                "event": "cond",
                "eff": eff,
                "outputs_differ": rf.text != rr.text,
                "find_adapter_mentions_div": (
                    "% k" in tf or "divisible" in tf or "all(" in tf
                ),
                "find_adapter_passes": _passes_find(extract_partial_code(rf.text)),
                "rev_adapter_mentions_reverse": (
                    "reverse" in tr or "[::-1]" in tr or "split" in tr
                ),
                "find_head": rf.text[:140],
                "rev_head": rr.text[:140],
            }
        )

    _log({"event": "done"})


if __name__ == "__main__":
    asyncio.run(main())
