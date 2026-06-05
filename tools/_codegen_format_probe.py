"""Probe: does freeform code output (no JSON schema) fix the over-escape?

REMOVE-BEFORE-MERGE.

Root cause (confirmed): code actions emit `{"code": "..."}`; the model sometimes
over-escapes newlines (`\\n` -> literal backslash-n), collapsing the code to one line
-> phantom line-1 SyntaxError. Fix cut (advisor): drop output_schema for code actions
(freeform), let Qwen3-Instruct emit a ```python fence, de-fence with markdown-it,
validate with ast. This probe runs the REAL generation path both ways on the three
tasks that broke and checks: does freeform produce compilable, real-newline code?
"""

from __future__ import annotations

import asyncio

from rune.config import load_rune_config
from rune.engine.continuation import extract_partial_code, validate_syntax
from rune.engine.graph import render_training_format_trajectory
from rune.engine.parse import CodeResult, _extract_code_block, render_template
from rune.model.adapter import scale_lora_b
from rune.model.wrapper import ModelWrapper

C3 = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"

TASKS = [
    (
        "int_to_roman",
        '"""\nImplement int_to_roman(num: int) -> str converting an '
        "integer in 1..3999 to its Roman numeral, using subtractive notation "
        '(IV, IX, XL, XC, CD, CM).\n\n>>> assert int_to_roman(9) == "IX"\n"""',
    ),
    (
        "decode_string",
        '"""\nImplement decode_string(s: str) -> str. The encoding rule '
        "is k[encoded], where the substring inside the brackets is repeated exactly k "
        "times. Input may be nested, e.g. '3[a2[c]]'.\n\n"
        '>>> assert decode_string("3[a]2[bc]") == "aaabcbc"\n"""',
    ),
    (
        "calculate",
        '"""\nImplement calculate(expression: str) -> int that evaluates an '
        "arithmetic expression of non-negative integers with +, -, *, / and "
        'parentheses.\n\n>>> assert calculate("2+3*4") == 14\n"""',
    ),
]


def _report(label: str, raw: str, code: str) -> None:
    lit = "\\n" in code and "\n" not in code.split("\\n", 1)[0]
    ok = validate_syntax(code)
    nlines = code.count(chr(10)) + 1
    print(f"  [{label}] raw_repr={raw[:70]!r}")
    print(f"           compiles={ok}  literal-\\n={lit}  lines={nlines}")


def main() -> None:
    cfg = load_rune_config(None).override(checkpoint_path=C3, adapter_scaling=1.0)
    mw = ModelWrapper.from_config(cfg)

    async def gen(prompt: str, schema: type | None) -> str:
        r = await mw.generate(
            prompt=prompt,
            system_prompt="You are a code generator.",
            output_schema=schema,
            max_tokens=768,
            temperature=0.7,
            presence_penalty=cfg.presence_penalty,
            thinking_budget=cfg.thinking_budget,
        )
        return r.text

    for name, spec in TASKS:
        traj = render_training_format_trajectory(spec, "", "")
        ad = mw.generate_adapter(traj).state_dict
        mw.hotswap_adapter(scale_lora_b(ad, 1.0))
        prompt = render_template(
            "prompt_code", subtask_name=name, project_label=name, entry_point=name
        )
        print(f"\n===== {name} =====")
        raw_json = asyncio.run(gen(prompt, CodeResult))
        _report("JSON schema (current)", raw_json, extract_partial_code(raw_json))
        raw_free = asyncio.run(gen(prompt, None))
        _report("freeform (proposed)", raw_free, _extract_code_block(raw_free))


if __name__ == "__main__":
    main()
