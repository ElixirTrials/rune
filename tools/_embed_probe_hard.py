"""Embed probe on HARD tasks where base lp(FIX) has headroom (REMOVE-BEFORE-MERGE).

The toy add/largest probe was saturated: base already assigns ~0.9 prob/token to the
fix, so the only thing that can move is lp(FAIL) and every template looks like
recitation. Advisor: re-run the SAME (template, prompt) grid on tasks where the fix is
genuinely non-obvious (multi-line, low base lp), and against the HONEST baseline
(current_code — a real repair turn), not task_only (which can't host a repair turn).

Decisive test: on hard tasks, does delineating the failure as "do NOT repeat" + an
avoid prompt beat current_code/default? And does lp(FIX) actually have room to rise
(adapter helping the fix), or does only lp(FAIL) move (pure recitation)?
"""

from __future__ import annotations

from rune.config import load_rune_config
from rune.model.adapter import scale_lora_b
from rune.model.wrapper import ModelWrapper

C3 = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"

SCENARIOS = [
    {
        "name": "int_to_roman",
        "spec": '"""\nImplement int_to_roman(num: int) -> str converting an integer in '
        "1..3999 to its Roman numeral, using subtractive notation (IV, IX, XL, XC, "
        'CD, CM).\n\n>>> assert int_to_roman(9) == "IX"\n"""',
        "fail": (
            "def int_to_roman(num):\n"
            "    vals = [1000, 500, 100, 50, 10, 5, 1]\n"
            '    syms = ["M", "D", "C", "L", "X", "V", "I"]\n'
            '    res = ""\n'
            "    for v, s in zip(vals, syms):\n"
            "        while num >= v:\n"
            "            res += s\n"
            "            num -= v\n"
            "    return res"
        ),
        "err": "AssertionError: int_to_roman(4) returned 'IIII', expected 'IV'",
        "summary": "omitted the subtractive pairs (IV, IX, XL, XC, CD, CM); "
        "built 4 as 'IIII' by only using additive symbols",
        "fix": (
            "def int_to_roman(num):\n"
            "    vals = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]\n"
            '    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", '
            '"V", "IV", "I"]\n'
            '    res = ""\n'
            "    for v, s in zip(vals, syms):\n"
            "        while num >= v:\n"
            "            res += s\n"
            "            num -= v\n"
            "    return res"
        ),
    },
    {
        "name": "decode_string",
        "spec": '"""\nImplement decode_string(s: str) -> str. The encoding rule is '
        "k[encoded], where the substring inside the brackets is repeated exactly k "
        "times (k is a positive integer). Input may be nested, e.g. '3[a2[c]]'.\n\n"
        '>>> assert decode_string("3[a]2[bc]") == "aaabcbc"\n"""',
        "fail": (
            "def decode_string(s):\n"
            "    stack = []\n"
            '    cur = ""\n'
            "    k = 0\n"
            "    for ch in s:\n"
            "        if ch.isdigit():\n"
            "            k = int(ch)\n"
            "        elif ch == '[':\n"
            "            stack.append((cur, k))\n"
            '            cur, k = "", 0\n'
            "        elif ch == ']':\n"
            "            prev, rep = stack.pop()\n"
            "            cur = prev + cur * rep\n"
            "        else:\n"
            "            cur += ch\n"
            "    return cur"
        ),
        "err": "AssertionError: decode_string(\"10[a]\") returned '', expected "
        "'aaaaaaaaaa'",
        "summary": "parsed the repeat count one digit at a time (k = int(ch) "
        "overwrites), so the 2-digit count 10 became 0",
        "fix": (
            "def decode_string(s):\n"
            "    stack = []\n"
            '    cur = ""\n'
            "    k = 0\n"
            "    for ch in s:\n"
            "        if ch.isdigit():\n"
            "            k = k * 10 + int(ch)\n"
            "        elif ch == '[':\n"
            "            stack.append((cur, k))\n"
            '            cur, k = "", 0\n'
            "        elif ch == ']':\n"
            "            prev, rep = stack.pop()\n"
            "            cur = prev + cur * rep\n"
            "        else:\n"
            "            cur += ch\n"
            "    return cur"
        ),
    },
    {
        "name": "merge_intervals",
        "spec": '"""\nImplement merge_intervals(intervals: list) -> list that merges '
        "all overlapping intervals (each [start, end]) and returns the "
        "non-overlapping intervals sorted by start. Touching intervals merge.\n\n"
        ">>> assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
        '"""',
        "fail": (
            "def merge_intervals(intervals):\n"
            "    res = []\n"
            "    for s, e in intervals:\n"
            "        if res and s <= res[-1][1]:\n"
            "            res[-1][1] = max(res[-1][1], e)\n"
            "        else:\n"
            "            res.append([s, e])\n"
            "    return res"
        ),
        "err": "AssertionError: merge_intervals([[1,4],[0,4]]) returned "
        "[[1,4],[0,4]], expected [[0,4]]",
        "summary": "merged without sorting by start first, so out-of-order "
        "intervals were never compared and stayed unmerged",
        "fix": (
            "def merge_intervals(intervals):\n"
            "    intervals = sorted(intervals)\n"
            "    res = []\n"
            "    for s, e in intervals:\n"
            "        if res and s <= res[-1][1]:\n"
            "            res[-1][1] = max(res[-1][1], e)\n"
            "        else:\n"
            "            res.append([s, e])\n"
            "    return res"
        ),
    },
]


def _templates(s: dict) -> dict[str, str]:
    spec, fail, err, summ = s["spec"], s["fail"], s["err"], s["summary"]
    return {
        "task_only": f"## Task\n{spec}",
        "current_code": (
            f"## Task\n{spec}\n\n## Current Code\n{fail}\n\n## Review Feedback\n{err}"
        ),
        "failed_attempts": (
            f"## Task\n{spec}\n\n## Failed Attempts (do NOT repeat)\n{fail}\n"
            f"-- error: {err}"
        ),
        "failure_summary": (
            f"## Task\n{spec}\n\n## Failure Modes to Avoid\n- {summ}"
        ),
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

    for s in SCENARIOS:
        tpls = _templates(s)
        # Headroom check: base (adapter OFF) lp of FIX and FAIL under the avoid prompt.
        with bm.disable_adapter():
            base_fix = _lp(bm, tok, P_AVOID, s["fix"], device)
            base_fail = _lp(bm, tok, P_AVOID, s["fail"], device)
        print(f"\n===== {s['name']} =====")
        print(
            f"base(adapter off): lp(FIX) {base_fix:+.3f}  lp(FAIL) {base_fail:+.3f}  "
            f"GAP {base_fix - base_fail:+.3f}"
        )
        print(f"{'template':18} {'prompt':8} {'lp(FIX)':>8} {'lp(FAIL)':>9} {'GAP':>7}")
        for tpl_name, prompt in CONDS:
            ad = mw.generate_adapter(tpls[tpl_name]).state_dict
            mw.hotswap_adapter(scale_lora_b(ad, 1.0))
            lpf = _lp(bm, tok, prompt, s["fix"], device)
            lpx = _lp(bm, tok, prompt, s["fail"], device)
            pm = "avoid" if prompt is P_AVOID else "default"
            print(
                f"{tpl_name:18} {pm:8} {lpf:+8.3f} {lpx:+9.3f} {lpf - lpx:+7.3f}"
            )


if __name__ == "__main__":
    main()
