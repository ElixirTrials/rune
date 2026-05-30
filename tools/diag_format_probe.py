"""H1 (format-mismatch) probe: does feeding the hypernetwork the EXACT
TRAINING-FORMAT conditioning text (## Task / ## Current Code / ## Review
Feedback, per v1-final d2l_data.py::unroll_trajectory_to_pairs) make the
generated adapter actually condition on the code — where the engine's OOD
code.j2 format (render_template('code', ...)) produced generic functions?

Design (one model load):
  Two formats x two tasks = 4 adapters:
    A = OOD engine format  (render_template('code', state_to_ctx(...)))
    B = TRAINING format    (hand-built "## Task / ## Current Code / ## Review
                            Feedback", byte-for-byte as unroll_trajectory_to_pairs)
    tasks: find_tuples (keystone) and reverse_words (different-task control)

The discriminator is NOT string presence — both trajectories already embed the
token "find_tuples" (A's EXISTING CODE stub, B's ## Current Code buggy body), so
"find_tuples in output" is satisfied by mere echoing. Instead the prior buggy
diff for find_tuples uses `any(e % K == 0 ...)` (returns ALL tuples -> FAILS the
assert); only a genuinely corrected `all(...)` passes
    find_tuples([(6,24,12),(7,9,6),(12,18,21)],6) == [(6,24,12)]
So assert_pass means the adapter drove the fix, not an echo.

The lean prompt carries NO task text (the adapter is the only task source) and
is identical across all 4 adapters. Conditioning = the find adapter emits a
find_tuples-shaped fix while the SAME-format reverse adapter emits a
reverse_words-shaped function (outputs differ per task).

GO  (H1 supported): B_find passes the assert at some scaling AND B differs
    across tasks (B_find emits find_tuples, B_rev emits reverse_words), WHILE
    A_find never passes and A is generic / does not differ across tasks.
NO-GO (H1 refuted): B_find never passes the assert, or B_find ~= B_rev.

CPU-forbidden: run on the GPU box only, under the RAM watchdog:
    bash /tmp/run_guarded.sh /tmp/format_run.log tools/diag_format_probe.py
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

OUT = Path("/tmp/format_results.jsonl")
PEFT_SCALING = 2.0
EFFECTIVE_SCALINGS = (0.98, 4.0, 7.84)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
FIND_DESC = (
    "Write a function find_tuples(test_list, K) to find tuples which have all "
    "elements divisible by K from the given list of tuples."
)
FIND_PLAN = (
    "Iterate the list of tuples; keep a tuple only when ALL of its elements are "
    "divisible by K using all(x % K == 0 for x in tup). Return the kept tuples."
)
REV_DESC = (
    "Write a function reverse_words(s) that reverses the order of the words in "
    "the given sentence string s."
)
REV_PLAN = (
    "Split s on whitespace into words, reverse the word list, and join with "
    "single spaces. Return the resulting string."
)

# Prior buggy diffs (the "## Current Code" body in the training format / the
# EXISTING CODE stub in the engine format). The find_tuples bug is `any` instead
# of `all`: it keeps a tuple when ANY element is divisible by K, so it returns
# every tuple -> FAILS the assert. Only a corrected `all` passes.
FIND_BUGGY = (
    "def find_tuples(test_list, K):\n"
    "    result = []\n"
    "    for sub in test_list:\n"
    "        if any(ele % K == 0 for ele in sub):\n"
    "            result.append(sub)\n"
    "    return result"
)
REV_BUGGY = (
    "def reverse_words(s):\n"
    "    words = s.split()\n"
    "    return ' '.join(words)"
)

# Real review notes pointing at the any-vs-all / order bug.
FIND_REVIEW = (
    "This keeps a tuple when ANY element is divisible by K, but the task wants "
    "tuples where ALL elements are divisible by K. find_tuples("
    "[(6,24,12),(7,9,6),(12,18,21)],6) returns every tuple instead of just "
    "[(6,24,12)]. Use all(...) over the tuple's elements, not any(...)."
)
REV_REVIEW = (
    "This returns the words in their original order; the task asks to reverse "
    "the order of the words. Reverse the word list before joining."
)

# Task-neutral lean prompt — identical across all 4 adapters, carries no task
# text so the loaded adapter is the only source of the task.
LEAN_PROMPT = (
    "Apply the fix described in your loaded context and output the corrected, "
    "complete Python function. Output only the function body, no explanation."
)

ASSERT_SRC = (
    "assert find_tuples([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) "
    "== [(6, 24, 12)]"
)

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_DEF_RE = re.compile(r"(def\s+\w+\s*\(.*)", re.DOTALL)


def _log(rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def _extract_code(raw: str) -> str:
    """Best-effort: prefer fenced block, else slice from first `def `."""
    from rune.engine.continuation import extract_partial_code  # noqa: PLC0415

    via_json = ""
    try:
        via_json = extract_partial_code(raw)
    except Exception:
        via_json = ""
    candidate = via_json if (via_json and via_json != raw) else raw
    m = _FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1)
    m2 = _DEF_RE.search(candidate)
    if m2:
        candidate = m2.group(1)
    return candidate.strip()


def _passes_assert(raw: str) -> bool:
    """Exec the extracted code in a fresh namespace and run the assert.

    Fully guarded: any failure (no code, syntax error, wrong logic, runtime
    error) returns False rather than raising.
    """
    code = _extract_code(raw)
    if "def find_tuples" not in code:
        return False
    ns: dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102 - sandboxed probe, GPU box only
        exec(ASSERT_SRC, ns)  # noqa: S102
    except Exception:
        return False
    return True


async def main() -> None:
    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.engine.graph import state_to_ctx  # noqa: PLC0415
    from rune.engine.parse import render_template  # noqa: PLC0415
    from rune.engine.policy import _with_target  # noqa: PLC0415
    from rune.engine.state import Subtask  # noqa: PLC0415
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    OUT.write_text("")

    # --- Format A: OOD engine code.j2 (render_template), exactly as engine ----
    def code_traj(desc: str, plan: str, existing: str) -> str:
        sub = Subtask(name="_main", description=desc, depends_on=())
        state: dict[str, Any] = {
            "task": desc,
            "subtasks": [sub],
            "plans": {"_main": plan},
            "code_results": {"_main": existing},  # -> EXISTING CODE (the diff)
            "feedback": {},
            "trajectory": [],
            "diagnosis": {},
        }
        action = _with_target("code", "_main")
        return render_template("code", **state_to_ctx(state, action))

    # --- Format B: TRAINING format, byte-for-byte unroll_trajectory_to_pairs --
    # activation_text = "\n\n".join([
    #     "## Task\n<desc>",
    #     "## Current Code\n<prior_diff>",   (header for review_comment kind)
    #     "## Review Feedback\n<feedback>",
    # ])
    def train_traj(desc: str, buggy: str, review: str) -> str:
        parts = [
            f"## Task\n{desc}",
            f"## Current Code\n{buggy}",
            f"## Review Feedback\n{review}",
        ]
        return "\n\n".join(parts)

    traj = {
        "A_find": code_traj(FIND_DESC, FIND_PLAN, FIND_BUGGY),
        "A_rev": code_traj(REV_DESC, REV_PLAN, REV_BUGGY),
        "B_find": train_traj(FIND_DESC, FIND_BUGGY, FIND_REVIEW),
        "B_rev": train_traj(REV_DESC, REV_BUGGY, REV_REVIEW),
    }
    _log({
        "event": "trajectories",
        "chars": {k: len(v) for k, v in traj.items()},
        "B_find_head": traj["B_find"][:400],
        "A_find_head": traj["A_find"][:400],
    })

    cfg = PipelineConfig(
        checkpoint_path=(
            "s3://elixirtrials-949678234935-eu-west-2-artifacts/"
            "checkpoints/hypernet_hpo/checkpoint.pt"
        )
    )
    t0 = time.monotonic()
    wrapper = ModelWrapper.from_config(cfg)
    _log({"event": "loaded", "load_s": round(time.monotonic() - t0, 1)})

    # One model load, four adapters.
    sds = {k: wrapper.generate_adapter(v).state_dict for k, v in traj.items()}
    _log({"event": "adapters_generated", "keys": list(sds.keys())})

    async def gen(key: str, eff: float) -> str:
        wrapper.hotswap_adapter(scale_lora_b(sds[key], eff / PEFT_SCALING))
        r = await wrapper.generate(
            LEAN_PROMPT,
            system_prompt="You are a code generator.",
            output_schema=None,
            max_tokens=400,
            temperature=0.0,
            repetition_penalty=1.0,
            top_p=1.0,
            presence_penalty=1.5,
            thinking_budget=0,
            skip_completion_retry=True,
        )
        return r.text

    for eff in EFFECTIVE_SCALINGS:
        rec: dict[str, Any] = {"event": "format_cmp", "eff": eff}
        outs: dict[str, str] = {}
        for key in ("A_find", "A_rev", "B_find", "B_rev"):
            try:
                outs[key] = await gen(key, eff)
            except Exception as exc:  # never abort before the done sentinel
                outs[key] = ""
                rec[f"{key}_error"] = repr(exc)[:200]

        a_find, a_rev = outs["A_find"].lower(), outs["A_rev"].lower()
        b_find, b_rev = outs["B_find"].lower(), outs["B_rev"].lower()

        # Primary discriminator: execute the assert.
        rec["A_find_assert_pass"] = _passes_assert(outs["A_find"])
        rec["B_find_assert_pass"] = _passes_assert(outs["B_find"])

        # Conditioning: same-format adapter emits its OWN function per task.
        rec["A_find_writes_find_tuples"] = "find_tuples" in a_find
        rec["A_rev_writes_reverse_words"] = "reverse_words" in a_rev
        rec["A_differs_across_tasks"] = a_find != a_rev
        rec["A_conditions"] = (
            "find_tuples" in a_find and "reverse_words" in a_rev
        )
        rec["B_find_writes_find_tuples"] = "find_tuples" in b_find
        rec["B_rev_writes_reverse_words"] = "reverse_words" in b_rev
        rec["B_differs_across_tasks"] = b_find != b_rev
        rec["B_conditions"] = (
            "find_tuples" in b_find and "reverse_words" in b_rev
        )

        # Secondary color (string presence — confounded, NOT used for GO).
        rec["B_find_mentions_all"] = (
            "all(" in b_find or " all " in b_find
        )
        rec["B_find_extracted_head"] = _extract_code(outs["B_find"])[:200]
        rec["A_find_extracted_head"] = _extract_code(outs["A_find"])[:200]
        rec["B_rev_extracted_head"] = _extract_code(outs["B_rev"])[:160]
        _log(rec)

    _log({"event": "done"})


if __name__ == "__main__":
    asyncio.run(main())
