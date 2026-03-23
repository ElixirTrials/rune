#!/usr/bin/env python3
"""Math evaluation agent using MathSandbox + gpt-oss-120b via HuggingFace Inference Router.
Each problem gets its own isolated MathSandbox (stateful IPython kernel). The
model reasons step-by-step and executes Python via execute_python tool calls.
Final answers are extracted from \\boxed{} in the model's response.

Usage::
    python scripts/math_eval.py
    python scripts/math_eval.py --n 5 --dataset olym_math_easy
    python scripts/math_eval.py --reasoning-effort high --max-tokens 4096

Environment variables:
    HF_TOKEN   HuggingFace inference token
"""
# ruff: noqa: T201, E402
# mypy: ignore-errors
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ── make libs importable without a full install ───────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "libs" / "shared" / "src"))
from shared.mathbox import MathConfig, MathSandbox

# ── colour helpers ─────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
MAGENTA = "\033[35m"
RED     = "\033[31m"
BLUE    = "\033[34m"

def c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET

def banner(title: str, width: int = 62) -> str:
    pad = (width - len(title) - 2) // 2
    return c(f"{'─'*pad} {title} {'─'*(width - pad - len(title) - 2)}", CYAN, BOLD)

def abbrev(text: str, max_len: int = 400) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + c(f" … [{len(text)-max_len} chars] … ", DIM) + text[-half:]

# ── tool schema ────────────────────────────────────────────────────────────────
MATH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code in a stateful IPython kernel. "
                "State (variables, imports, definitions) persists across calls. "
                "Pre-imported: math, numpy, sympy, itertools, collections, mpmath (mp.dps=64). "
                "Always use print() to display results — the kernel captures stdout only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to show results.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

# ── answer extraction ─────────────────────────────────────────────────────────
def extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} value from text, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len(r"\boxed{")
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[start:j])
                    break
                depth -= 1
            j += 1
        i = idx + 1
    return results[-1] if results else None

# ── HF inference client ───────────────────────────────────────────────────────
# Available providers (swap the suffix to change backend):
#   :fireworks-ai  — good balance of speed and availability
#   :cerebras      — fastest cold start
#   :together-ai   — solid fallback
MODEL = "openai/gpt-oss-120b:fireworks-ai"

def make_client():
    from openai import OpenAI
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set.")
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

# ── streaming renderer ────────────────────────────────────────────────────────
def stream_response(client, messages: list[dict], max_tokens: int) -> tuple[str, list]:
    """Stream a chat completion, printing tokens live.

    Returns:
        (full_text, tool_calls_list)
        tool_calls_list is empty if the model produced a final answer.
    """
    print()
    print(c("╭─ model " + "─" * 53, MAGENTA, BOLD))
    print(c("  ", MAGENTA), end="", flush=True)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=MATH_TOOLS,
        tool_choice="auto",
        max_tokens=max_tokens,
        stream=True,
    )

    collected_text = ""
    # tool call accumulators keyed by index
    tool_call_chunks: dict[int, dict] = {}
    finish_reason = None

    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue

        finish_reason = choice.finish_reason or finish_reason
        delta = choice.delta

        # ── stream text content ────────────────────────────────────────────────
        if delta.content:
            collected_text += delta.content
            print(delta.content, end="", flush=True)

        # ── accumulate tool call deltas ────────────────────────────────────────
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_call_chunks:
                    tool_call_chunks[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                slot = tool_call_chunks[idx]
                if tc_delta.id:
                    slot["id"] += tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        slot["function"]["name"] += tc_delta.function.name
                        # Print tool name as soon as we know it
                        if slot["function"]["name"] and not slot["function"]["arguments"]:
                            print()
                            print(
                                c(f"  ⚙  [{slot['function']['name']}]", YELLOW, BOLD),
                                flush=True,
                            )
                    if tc_delta.function.arguments:
                        slot["function"]["arguments"] += tc_delta.function.arguments

    print()  # newline after streamed content

    tool_calls = list(tool_call_chunks.values()) if finish_reason == "tool_calls" else []
    return collected_text, tool_calls


# ── loop-guard helpers ────────────────────────────────────────────────────────

_MAX_CODE_LINES = 100

_CODE_TOO_LONG_MSG = (
    "[CODE TOO LONG — {n} lines] Your code block is too large. This usually "
    "means you are brute-forcing analytically in a loop that will not converge. "
    "Stop. You already have a numerical result. "
    "State your final answer now using \\boxed{{}}. "
    "If you genuinely need more computation, write a compact script (< 50 lines)."
)

_NEAR_INT_TOLERANCE = 1e-4

_NEAR_INT_HINT = (
    "\n[HINT] The value {val:.6f} is within {tol} of the integer {n}. "
    "This strongly suggests the exact answer is {n}. "
    "If your analysis supports this, conclude now with \\boxed{{{n}}}."
)

_FLOAT_RE = re.compile(r"[-+]?\d+\.\d{4,}")


def _check_code_length(code: str) -> str | None:
    """Return a steering message if *code* exceeds the line limit, else None."""
    n = len(code.splitlines())
    if n > _MAX_CODE_LINES:
        return _CODE_TOO_LONG_MSG.format(n=n)
    return None


def _near_integer_hint(result: str) -> str:
    """Scan *result* for floats suspiciously close to an integer.

    Returns an appended hint string if one is found, otherwise "".
    """
    for m in _FLOAT_RE.finditer(result):
        try:
            val = float(m.group())
        except ValueError:
            continue
        nearest = round(val)
        if abs(val - nearest) < _NEAR_INT_TOLERANCE and 0 <= nearest <= 99999:
            return _NEAR_INT_HINT.format(val=val, tol=_NEAR_INT_TOLERANCE, n=nearest)
    return ""


# ── per-problem agent loop ────────────────────────────────────────────────────
def solve_problem(
    problem: str,
    system_prompt: str,
    sandbox: MathSandbox,
    client,
    reasoning_effort: str,
    max_tokens: int,
    max_iterations: int = 15,
) -> tuple[str, str | None, list[dict]]:
    """Drive the generate → tool-call → observe loop for one problem.

    Args:
        problem:          The problem statement (user turn).
        system_prompt:    Rendered math_prompt.j2 content.
        sandbox:          A fresh MathSandbox instance for this problem.
        client:           OpenAI client pointed at HF router.
        reasoning_effort: "low" | "medium" | "high"
        max_tokens:       Token budget per generation call.
        max_iterations:   Max generate→execute cycles before giving up.

    Returns:
        Tuple of (final_response_text, extracted_boxed_answer, message_history).
    """
    # gpt-oss reads reasoning effort from the system prompt
    full_system = f"Reasoning: {reasoning_effort}\n\n{system_prompt}"

    messages: list[dict] = [
        {"role": "system", "content": full_system},
        {"role": "user",   "content": problem},
    ]

    last_results: list[str] = []

    for iteration in range(max_iterations):
        text, tool_calls = stream_response(client, messages, max_tokens)

        # ── no tool calls → final answer ───────────────────────────────────────
        if not tool_calls:
            messages.append({"role": "assistant", "content": text})
            return text, extract_boxed(text), messages

        # ── append assistant message with tool_calls (OpenAI format) ──────────
        # The assistant message must carry the tool_calls list so the API can
        # match tool results back to their call IDs.
        messages.append({
            "role": "assistant",
            "content": text or None,        # may be None when only tool calls emitted
            "tool_calls": [
                {
                    "id":       tc["id"],
                    "type":     "function",
                    "function": tc["function"],
                }
                for tc in tool_calls
            ],
        })

        # ── execute each tool call ─────────────────────────────────────────────
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            code = args.get("code", "")
            lines = code.splitlines()

            print()
            for line in lines[:25]:
                print(c(f"     {line}", BLUE))
            if len(lines) > 25:
                print(c(f"     … ({len(lines) - 25} more lines)", DIM))

            # ── guard: reject oversized code blocks ───────────────────────────
            too_long = _check_code_length(code)
            if too_long:
                result = too_long
                print(c(f"  ↳  [code too long — {len(lines)} lines, blocked]", RED, BOLD))
            elif name == "execute_python":
                print(c(f"  ↳  [{name}] running…", YELLOW))
                result = sandbox.execute(code)
            else:
                result = f"[ERROR] Unknown tool '{name}'. Only execute_python is available."

            # ── guard: nudge model when result is near an integer ─────────────
            hint = _near_integer_hint(result)
            if hint:
                result += hint
                print(c(f"  ↳  [near-integer hint appended]", YELLOW))

            print(c("  ↳  result:", DIM))
            for line in abbrev(result, 500).splitlines():
                print(c(f"     {line}", GREEN))

            # ── tool result message — must include tool_call_id ────────────────
            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],   # ← critical for OpenAI-format APIs
                "content":      result,
            })

            last_results.append(result)
            if len(last_results) >= 3 and len(set(last_results[-3:])) == 1:
                bail = "[stuck] Same result 3 times in a row — stopping loop."
                print(c(f"\n  ⚠  {bail}", RED))
                messages.append({"role": "assistant", "content": bail})
                return bail, None, messages

    final = "[max iterations reached]"
    messages.append({"role": "assistant", "content": final})
    return final, None, messages


# ── system prompt ─────────────────────────────────────────────────────────────
def render_system_prompt(answer_range: str = "0 to 99999") -> str:
    from jinja2 import Environment, FileSystemLoader
    template_dir = REPO_ROOT / "libs" / "shared" / "src" / "shared" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    return env.get_template("math_prompt.j2").render(answer_range=answer_range)


# ── dataset loading ───────────────────────────────────────────────────────────
DATA_ROOT = REPO_ROOT / "libs" / "evaluation" / "src" / "evaluation" / "data"

DATASET_PATHS: dict[str, str] = {
    "daft_math":      "daft_math/train",
    "olym_math_easy": "olym_math/en-easy/test",
    "olym_math_hard": "olym_math/en-hard/test",
}


_ANSWER_RANGE = (0, 99999)


def _integer_in_range(value: object) -> int | None:
    """Return *value* as an int if it is a non-negative integer in [0, 99999].

    Accepts plain ints, float-valued ints (e.g. 42.0), and digit-only strings.
    Returns None for LaTeX expressions, multi-part answers, or out-of-range values.
    """
    lo, hi = _ANSWER_RANGE
    if isinstance(value, int):
        return value if lo <= value <= hi else None
    if isinstance(value, float) and value == int(value):
        v = int(value)
        return v if lo <= v <= hi else None
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if s.lstrip("-").isdigit():
            v = int(s)
            return v if lo <= v <= hi else None
    return None


def load_problems(dataset: str, n: int) -> list[dict]:
    """Load up to *n* problems whose ground-truth answer is an integer in [0, 99999].

    - daft_math   : uses the pre-resolved ``Integer Variant Answer`` (always int).
    - olym_math_* : uses ``answer``; skips rows with LaTeX / non-integer answers.
    """
    from datasets import load_from_disk

    ds = load_from_disk(str(DATA_ROOT / DATASET_PATHS[dataset]))
    rows: list[dict] = []
    skipped = 0

    for i, item in enumerate(ds):
        if len(rows) >= n:
            break

        if dataset == "daft_math":
            raw = item["Integer Variant Answer"]
            gt = _integer_in_range(raw)
            if gt is None:
                skipped += 1
                continue
            rows.append({
                "id": f"daft_{i}",
                "problem": item["Integer Answer Variant Question"],
                "ground_truth": str(gt),
                "original_answer": item["Original Answer"],
                "source": item.get("Source", ""),
            })
        else:
            gt = _integer_in_range(item["answer"])
            if gt is None:
                skipped += 1
                continue
            rows.append({
                "id": item.get("unique_id", f"{dataset}_{i}"),
                "problem": item["problem"],
                "ground_truth": str(gt),
                "subject": item.get("subject", ""),
            })

    if skipped:
        print(c(f"  (skipped {skipped} problems with non-integer or out-of-range answers)", DIM))

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Math eval agent — gpt-oss-120b via HF")
    p.add_argument("--n",                type=int,   default=10)
    p.add_argument("--dataset",          type=str,   default="olym_math_easy",
                   choices=list(DATASET_PATHS.keys()))
    p.add_argument("--reasoning-effort", type=str,   default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--max-tokens",       type=int,   default=4096)
    p.add_argument("--max-iterations",   type=int,   default=15)
    p.add_argument("--provider",         type=str,   default="fireworks-ai",
                   choices=["fireworks-ai", "cerebras", "together-ai", "novita"],
                   help="HF inference provider suffix")
    p.add_argument("--sandbox-timeout",  type=float, default=180.0,
                   help="Per-execution timeout for MathSandbox in seconds (default: 180)")
    p.add_argument("--problem",          type=str,   default=None,
                   metavar="TEXT",
                   help="Run a single ad-hoc problem instead of loading a dataset.")
    p.add_argument("--answer",           type=str,   default="?",
                   metavar="VALUE",
                   help="Ground-truth answer for --problem (for display/scoring only).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global MODEL
    MODEL = f"openai/gpt-oss-120b:{args.provider}"

    client = make_client()
    system_prompt = render_system_prompt()

    print(banner("gpt-oss-120b math eval"))
    print(c(f"  model    : {MODEL}", DIM))
    print(c(f"  reasoning: {args.reasoning_effort}", DIM))
    if args.problem:
        print(c(f"  mode     : ad-hoc problem", DIM))
    else:
        print(c(f"  dataset  : {args.dataset}  n={args.n}", DIM))
    print(c(f"  sandbox  : timeout={args.sandbox_timeout:.0f}s per execution", DIM))
    print()

    # ── load dataset ──────────────────────────────────────────────────────────
    if args.problem:
        problems = [{"id": "adhoc_0", "problem": args.problem, "ground_truth": args.answer}]
    else:
        problems = load_problems(args.dataset, args.n)

    results = []
    for i, item in enumerate(problems[: args.n]):
        problem  = item["problem"]
        expected = item.get("ground_truth", "?")

        print(banner(f"Problem {i+1}/{min(args.n, len(problems))}"))
        print(c(f"  {textwrap.shorten(problem, 200)}", CYAN))
        print(c(f"  expected: {expected}", DIM))

        sandbox = MathSandbox(MathConfig(default_timeout=args.sandbox_timeout))
        try:
            response, boxed, history = solve_problem(
                problem       = problem,
                system_prompt = system_prompt,
                sandbox       = sandbox,
                client        = client,
                reasoning_effort = args.reasoning_effort,
                max_tokens    = args.max_tokens,
                max_iterations = args.max_iterations,
            )
        finally:
            sandbox.close()

        correct = str(boxed).strip() == str(expected).strip() if boxed else False
        status  = c("✓ CORRECT", GREEN, BOLD) if correct else c("✗ WRONG", RED, BOLD)
        print()
        print(f"  {status}  boxed={boxed!r}  expected={expected!r}")
        results.append({"problem": problem, "expected": expected,
                        "boxed": boxed, "correct": correct})

    # ── summary ───────────────────────────────────────────────────────────────
    n_correct = sum(r["correct"] for r in results)
    print()
    print(banner("Results"))
    print(c(f"  {n_correct}/{len(results)} correct", BOLD))


if __name__ == "__main__":
    main()