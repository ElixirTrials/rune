"""Diagnostic: isolate the single-word degeneration (REMOVE-BEFORE-MERGE).

Faithful reproduction OUTSIDE the rune runner on the EXACT prompts that
degenerated in the engine (mbpp/108, 113, 115), with engine generation settings
and schema=True (the path the engine actually uses). Ablates the thinking phase
(on/off) x adapter (base/c3), N seeds each, to test whether the forced
</think>-terminated thinking phase (wrong for non-thinking Qwen3-Instruct-2507)
causes the degeneration. Reports per-cell degeneration rate + an example + the
thinking-phase tail.
"""

from __future__ import annotations

import ast
import asyncio

from rune.config import load_rune_config
from rune.engine.graph import render_training_format_trajectory
from rune.engine.parse import CodeResult, extract_code_from_raw, render_template
from rune.model.adapter import scale_lora_b
from rune.model.inference import generate
from rune.model.wrapper import ModelWrapper

C3 = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
N = 6

# (spec, entry_point) for the three tasks that degenerated in the engine.
PROMPTS = [
    (
        '"""\nWrite a function to merge three lists into a single sorted list.\n\n'
        ">>> assert merge_sorted_list([25, 24, 15, 4, 5, 29, 110],"
        "[19, 20, 11, 56, 25, 233, 154],[24, 26, 54, 48])=="
        "[4, 5, 11, 15, 19, 20, 24, 24, 25, 25, 26, 29, 48, 54, 56, 110, 154, 233]\n"
        '"""',
        "merge_sorted_list",
    ),
    (
        '"""\nWrite a function to check if a string represents an integer or not.\n\n'
        '>>> assert check_integer("python")==False\n"""',
        "check_integer",
    ),
    (
        '"""\nWrite a function to check whether all dictionaries in a list are '
        'empty or not.\n\n>>> assert empty_dit([{},{},{}])==True\n"""',
        "empty_dit",
    ),
]


def _degenerate(raw: str) -> tuple[bool, str]:
    """True if the schema output is NOT real code (single-word collapse)."""
    code = extract_code_from_raw(raw, CodeResult, fallback_to_raw=True)
    try:
        ast.parse(code)
        ok = "def " in code and len(code) > 30
    except SyntaxError:
        ok = False
    return (not ok), code[:40]


def main() -> None:
    import torch  # noqa: PLC0415

    cfg = load_rune_config(None).override(checkpoint_path=C3, adapter_scaling=1.0)
    mw = ModelWrapper.from_config(cfg)
    bm, tok = mw._base_model, mw._tokenizer

    def set_adapter(which: str, spec: str) -> None:
        traj = render_training_format_trajectory(spec, "", "")
        ad = mw.generate_adapter(traj)
        mw.hotswap_adapter(scale_lora_b(ad.state_dict, 1.0 if which == "c3" else 0.0))

    def cell(adapter: str, thinking: bool) -> None:
        degen = 0
        total = 0
        ex = None
        ex_think = ""
        for spec, entry in PROMPTS:
            prompt = render_template(
                "prompt_code", subtask_name="_main", project_label=spec, entry_point=entry
            )
            for i in range(N):
                set_adapter(adapter, spec)
                torch.manual_seed(i)
                res = asyncio.run(
                    generate(
                        bm,
                        tok,
                        prompt,
                        system_prompt="You are a code generator.",
                        output_schema=CodeResult,
                        thinking_budget=(1024 if thinking else 0),
                        temperature=0.3,
                        top_p=0.9,
                        max_tokens=1024,
                        presence_penalty=1.5,
                    )
                )
                total += 1
                bad, codehead = _degenerate(res.text)
                if bad:
                    degen += 1
                    if ex is None:
                        ex = codehead
                        ex_think = (res.thinking or "")[-130:]
        tag = f"adapter={adapter:4} thinking={str(thinking):5} schema=True"
        print(f"  {tag}: DEGEN {degen}/{total}  ex={ex!r}")
        if thinking and ex_think:
            print(f"      thinking_tail={ex_think!r}")

    print(f"=== faithful degeneration ablation (N={N} x 3 real prompts) ===")
    for adapter in ("base", "c3"):
        for thinking in (True, False):
            cell(adapter, thinking)


if __name__ == "__main__":
    main()
