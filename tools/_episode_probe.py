"""Validate the hypernet uses EPISODE info to drive the next step (REMOVE-BEFORE-MERGE).

Before spending GPU on a multi-turn HPO, check the mechanism R2 assumes: when the
adapter is conditioned on the episode (## Task + ## Current Code [the failing
attempt] + ## Review Feedback [the error] + ## Previous Attempts), does it
(1) RESPOND to that info (different prior code/errors -> different adapter), and
(2) USE it usefully (raise the gold-logprob of the CORRECT fix, lower that of the
repeated failure) vs conditioning on the task alone?

Honest prior: c3 was distilled with EMPTY Current Code / Review Feedback, so it
may receive the episode yet not leverage it — which is itself the finding (RL
needed before the episode helps).
"""

from __future__ import annotations

from rune.config import load_rune_config
from rune.engine.graph import render_training_format_trajectory
from rune.model.adapter import scale_lora_b
from rune.model.wrapper import ModelWrapper

C3 = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"

# Repair scenarios: (spec, failing code, error, the CORRECT fix).
SCENARIOS = [
    {
        "spec": '"""\nWrite a python function `add` that returns the sum of two '
        'integers.\n\n>>> assert add(2, 3) == 5\n"""',
        "fail": "def add(a, b):\n    return a - b",
        "err": "AssertionError: add(2, 3) returned -1, expected 5",
        "fail2": "def add(a, b):\n    return a * b",
        "err2": "AssertionError: add(2, 3) returned 6, expected 5",
        "fix": "def add(a, b):\n    return a + b",
    },
    {
        "spec": '"""\nWrite a python function `largest` returning the larger of two '
        'integers.\n\n>>> assert largest(2, 7) == 7\n"""',
        "fail": "def largest(a, b):\n    return min(a, b)",
        "err": "AssertionError: largest(2, 7) returned 2, expected 7",
        "fail2": "def largest(a, b):\n    return a",
        "err2": "AssertionError: largest(2, 7) returned 2, expected 7",
        "fix": "def largest(a, b):\n    return max(a, b)",
    },
]

REPAIR_PROMPT = (
    "Repair the function: the task, your failing Current Code, the Review Feedback "
    "(error), and your Previous Attempts are in your context. Output the corrected "
    "implementation."
)
SYSTEM = "You are a code generator."


def _adapter_max_diff(d1: dict, d2: dict) -> float:
    m = 0.0
    for k, v in d1.items():
        if "lora" in k:
            m = max(m, float((v.float() - d2[k].float()).abs().max()))
    return m


def main() -> None:
    cfg = load_rune_config(None).override(checkpoint_path=C3, adapter_scaling=1.0)
    mw = ModelWrapper.from_config(cfg)
    bm, tok = mw._base_model, mw._tokenizer
    device = next(bm.parameters()).device

    def adapter(traj: str) -> dict:
        return mw.generate_adapter(traj).state_dict

    def gold_logprob(target: str, scale: float, sd: dict | None) -> float:
        if sd is None:
            with bm.disable_adapter():
                return _lp(bm, tok, SYSTEM, REPAIR_PROMPT, target, device)
        mw.hotswap_adapter(scale_lora_b(sd, scale))
        return _lp(bm, tok, SYSTEM, REPAIR_PROMPT, target, device)

    for i, s in enumerate(SCENARIOS):
        spec = s["spec"]
        traj_task = render_training_format_trajectory(spec, "", "")
        traj_ep = render_training_format_trajectory(
            spec, current_code=s["fail"], feedback=s["err"]
        )
        traj_ep2 = render_training_format_trajectory(
            spec, current_code=s["fail2"], feedback=s["err2"]
        )
        traj_hist = render_training_format_trajectory(
            spec,
            current_code=s["fail"],
            feedback=s["err"],
            attempts=[{"code": s["fail2"], "error": s["err2"], "passed": False}],
        )
        a_task, a_ep, a_ep2 = adapter(traj_task), adapter(traj_ep), adapter(traj_ep2)

        print(f"\n===== scenario {i}: {s['fix'].splitlines()[0]} =====")
        print("SENSITIVITY (adapter max|Δ| vs task-only conditioning):")
        print(f"  episode(fail1)  vs task-only: {_adapter_max_diff(a_ep, a_task):.4f}")
        print(f"  episode(fail2)  vs task-only: {_adapter_max_diff(a_ep2, a_task):.4f}")
        print(
            "  episode1 vs episode2 (diff prior code/err):",
            f"{_adapter_max_diff(a_ep, a_ep2):.4f}",
        )

        print("USEFULNESS (mean gold-logprob; higher=more accessible):")
        for tgt_name, tgt in (("FIX", s["fix"]), ("FAILURE(repeat)", s["fail"])):
            lp0 = gold_logprob(tgt, 0.0, a_task)  # scale0 == base (adapter off)
            lpt = gold_logprob(tgt, 1.0, a_task)
            lpe = gold_logprob(tgt, 1.0, a_ep)
            lph = gold_logprob(tgt, 1.0, adapter(traj_hist))
            print(
                f"  {tgt_name:16}: base {lp0:+.3f} | c3@task {lpt:+.3f} | "
                f"c3@episode {lpe:+.3f} | c3@+history {lph:+.3f}"
            )


def _lp(bm, tok, system, user, target, device) -> float:
    import torch  # noqa: PLC0415

    def _ids(x: object) -> object:
        return x["input_ids"] if hasattr(x, "input_ids") else x

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    p_ids = _ids(
        tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ).to(device)
    full = _ids(
        tok.apply_chat_template(
            [*msgs, {"role": "assistant", "content": target}], return_tensors="pt"
        )
    ).to(device)
    plen = p_ids.shape[1]
    tgt = full[0, plen:]
    with torch.no_grad():
        logits = bm(full).logits.float()
    lp = torch.log_softmax(logits[0, plen - 1 : -1], dim=-1)
    return float(lp.gather(1, tgt.unsqueeze(1)).mean())


if __name__ == "__main__":
    main()
