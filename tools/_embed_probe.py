"""Does the EMBEDDING (not retraining) fix the recitation? (REMOVE-BEFORE-MERGE)

The episode probe showed: putting the failing code in ## Current Code makes the
reproduction-trained c3 RECITE the failure (the fix-vs-failure logprob gap shrinks
from +0.69 task-only to +0.27). Hypothesis (owner): maybe it's just how we embed
it. Test several "adapter episode templates" x prompt instructions and measure the
gap = logprob(FIX) - logprob(FAILURE). A template that DELINEATES the failure as
something to AVOID (not Current Code) + a prompt that says "avoid these failure
modes" should WIDEN the gap (failure less accessible, fix still high).
"""

from __future__ import annotations

from rune.config import load_rune_config
from rune.model.adapter import scale_lora_b
from rune.model.wrapper import ModelWrapper

C3 = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"

SCENARIOS = [
    {
        "spec": '"""\nWrite a python function `add` returning the sum of two '
        'integers.\n\n>>> assert add(2, 3) == 5\n"""',
        "fail": "def add(a, b):\n    return a - b",
        "err": "AssertionError: add(2, 3) returned -1, expected 5",
        "summary": "subtracted instead of adding (used '-' where '+' is required)",
        "fix": "def add(a, b):\n    return a + b",
    },
    {
        "spec": '"""\nWrite a python function `largest` returning the larger of two '
        'integers.\n\n>>> assert largest(2, 7) == 7\n"""',
        "fail": "def largest(a, b):\n    return min(a, b)",
        "err": "AssertionError: largest(2, 7) returned 2, expected 7",
        "summary": "returned the smaller value (used min where max is required)",
        "fix": "def largest(a, b):\n    return max(a, b)",
    },
]


def _templates(s: dict) -> dict[str, str]:
    spec, fail, err, summ = s["spec"], s["fail"], s["err"], s["summary"]
    return {
        # baseline: no failure info
        "task_only": f"## Task\n{spec}",
        # current bad encoding: failure in ## Current Code (recall surface)
        "current_code": (
            f"## Task\n{spec}\n\n## Current Code\n{fail}\n\n## Review Feedback\n{err}"
        ),
        # delineate failure as a thing to avoid (not Current Code)
        "failed_attempts": (
            f"## Task\n{spec}\n\n## Failed Attempts (do NOT repeat)\n{fail}\n"
            f"-- error: {err}"
        ),
        # summarize the failure MODE (no raw failing code to recite)
        "failure_summary": (f"## Task\n{spec}\n\n## Failure Modes to Avoid\n- {summ}"),
        # keep Current Code EMPTY + failure in an avoid section
        "empty_plus_avoid": (
            f"## Task\n{spec}\n\n## Current Code\n\n\n"
            f"## Failed Attempts (do NOT repeat)\n{fail} -- {err}"
        ),
    }


P_DEFAULT = "Output the corrected implementation of the function."
P_AVOID = (
    "Your context lists the failure modes / failed attempts and their errors. "
    "Implement the corrected function and AVOID repeating those failure modes."
)

# (template, prompt) conditions to evaluate
CONDS = [
    ("task_only", P_DEFAULT),
    ("current_code", P_DEFAULT),
    ("current_code", P_AVOID),
    ("failed_attempts", P_AVOID),
    ("failure_summary", P_AVOID),
    ("empty_plus_avoid", P_AVOID),
]
SYSTEM = "You are a code generator."


def _lp(bm: object, tok: object, user: str, target: str, device: object) -> float:
    import torch  # noqa: PLC0415

    def ids(x: object) -> object:
        return x["input_ids"] if hasattr(x, "input_ids") else x

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    plen = ids(
        tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ).shape[1]
    full = ids(
        tok.apply_chat_template(
            [*msgs, {"role": "assistant", "content": target}], return_tensors="pt"
        )
    ).to(device)
    tgt = full[0, plen:]
    with torch.no_grad():
        logits = bm(full).logits.float()
    lp = torch.log_softmax(logits[0, plen - 1 : -1], dim=-1)
    return float(lp.gather(1, tgt.unsqueeze(1)).mean())


def main() -> None:
    cfg = load_rune_config(None).override(checkpoint_path=C3, adapter_scaling=1.0)
    mw = ModelWrapper.from_config(cfg)
    bm, tok = mw._base_model, mw._tokenizer
    device = next(bm.parameters()).device

    for i, s in enumerate(SCENARIOS):
        tpls = _templates(s)
        print(f"\n===== scenario {i}: {s['fix'].splitlines()[0]} =====")
        print(f"{'template':18} {'prompt':8} {'lp(FIX)':>8} {'lp(FAIL)':>9} {'GAP':>7}")
        for tpl_name, prompt in CONDS:
            ad = mw.generate_adapter(tpls[tpl_name]).state_dict
            mw.hotswap_adapter(scale_lora_b(ad, 1.0))
            lpf = _lp(bm, tok, prompt, s["fix"], device)
            lpx = _lp(bm, tok, prompt, s["fail"], device)
            pm = "avoid" if prompt is P_AVOID else "default"
            print(f"{tpl_name:18} {pm:8} {lpf:+8.3f} {lpx:+9.3f} {lpf - lpx:+7.3f}")


if __name__ == "__main__":
    main()
