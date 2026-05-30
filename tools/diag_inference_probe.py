"""Diagnostic probe: isolate generation speed (P-A) and non-termination (P-B).

Calls the REAL rune.model.inference.generate() on the base Qwen3.5-9B (no
hypernetwork adapter — removes that confound) for one simple MBPP task under
controlled configs, varying ONE knob at a time. Writes each result to
/tmp/probe_results.jsonl immediately so partial data survives a shutdown.

Run:  uv run python tools/diag_inference_probe.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

OUT = Path("/tmp/probe_results.jsonl")

TASK = (
    "Write a function to find tuples which have all elements divisible by k "
    "from the given list of tuples.\n\n"
    ">>> assert find_tuples([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) "
    "== [(6, 24, 12)]"
)


def _log(rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1), flush=True)


async def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rune.engine.parse import CodeResult
    from rune.model import inference

    OUT.write_text("")  # truncate
    model_id = "Qwen/Qwen3.5-9B"

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(
        "cuda"
    )
    model.eval()
    load_s = time.monotonic() - t0
    attn = getattr(model.config, "_attn_implementation", "?")
    _log(
        {
            "event": "loaded",
            "load_s": round(load_s, 1),
            "attn_impl": attn,
            "eos_token": tok.eos_token,
            "eos_id": tok.eos_token_id,
            "use_cache_cfg": getattr(model.config, "use_cache", None),
            "dtype": str(model.dtype),
            "device": str(model.device),
        }
    )

    prompt = (
        f"{TASK}\n\nReturn a JSON object with a single key 'code' whose value is "
        "the complete Python function as a string."
    )

    # (label, output_schema, presence_penalty, thinking_budget, max_tokens)
    configs: list[tuple[str, Any, float, int, int]] = [
        # P-A ceiling: raw speed, no grammar/presence/thinking
        ("raw_nothink_nopen_nogrammar", None, 0.0, 0, 384),
        # P-B isolate: grammar ON, presence OFF, no thinking — does JSON terminate?
        ("grammar_nothink_nopen", CodeResult, 0.0, 0, 384),
        # P-B isolate: grammar ON, presence 1.5 — does presence suppress closing?
        ("grammar_nothink_pen15", CodeResult, 1.5, 0, 384),
        # REAL config: grammar + presence + thinking
        ("grammar_think_pen15_REAL", CodeResult, 1.5, 256, 384),
    ]

    for label, schema, pen, think, maxtok in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        try:
            res = await inference.generate(
                model,
                tok,
                prompt,
                system_prompt="You are a code generator.",
                output_schema=schema,
                max_tokens=maxtok,
                temperature=0.0,  # deterministic — speed/termination not sampling noise
                repetition_penalty=1.0,
                top_p=1.0,
                presence_penalty=pen,
                thinking_budget=think,
                skip_completion_retry=True,  # we want to SEE truncation, not mask it
            )
            wall = time.monotonic() - t0
            think_chars = len(res.thinking)
            text = res.text
            # parse attempt
            parseable = False
            if schema is not None:
                try:
                    CodeResult.model_validate_json(text)
                    parseable = True
                except Exception:
                    parseable = False
            _log(
                {
                    "event": "gen",
                    "config": label,
                    "wall_s": round(wall, 1),
                    "tokens_used": res.tokens_used,
                    "tok_per_s": round(res.tokens_used / wall, 2) if wall else None,
                    "truncated_hit_cap": res.truncated,
                    "thinking_chars": think_chars,
                    "text_chars": len(text),
                    "parseable_json": parseable,
                    "peak_gpu_gb": round(
                        torch.cuda.max_memory_allocated() / 1e9, 2
                    ),
                    "text_head": text[:300],
                    "text_tail": text[-200:],
                    "think_tail": res.thinking[-200:] if think else "",
                }
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            _log(
                {
                    "event": "error",
                    "config": label,
                    "error": str(e),
                    "tb": traceback.format_exc()[-800:],
                }
            )

    _log({"event": "done"})


if __name__ == "__main__":
    asyncio.run(main())
