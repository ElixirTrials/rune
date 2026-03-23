#!/workspace/rune/.chat-venv/bin/python
"""Math evaluation agent using MathSandbox + Qwen3.5 (Transformers, no adapters).

Each problem gets its own isolated MathSandbox (stateful IPython kernel). The
model reasons step-by-step and executes Python via execute_python tool calls.
Final answers are extracted from \\boxed{} in the model's response.

Usage::

    python scripts/math_eval.py
    python scripts/math_eval.py --n 5 --dataset olym_math_easy
    python scripts/math_eval.py --no-thinking --max-new-tokens 16384
    python scripts/math_eval.py --problem "Find 2+2" --answer "4"

Environment variables:
    RUNE_TOOLS_MODEL   HuggingFace model ID or local path
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
from threading import Thread

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

# ── Qwen3.5 tool-call / thinking parsing ──────────────────────────────────────
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1)
        args = {pm.group(1): pm.group(2).strip() for pm in _PARAM_RE.finditer(m.group(2))}
        calls.append({"name": name, "arguments": args})
    return calls


def split_thinking(raw: str) -> tuple[str, str]:
    """Split raw output into (thinking_content, response_body).

    Returns ("", raw) when no </think> marker is present.
    """
    if "</think>" in raw:
        idx = raw.index("</think>")
        return raw[:idx].strip(), raw[idx + len("</think>"):].strip()
    return "", raw.strip()


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


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str):
    """Load tokenizer + model onto GPU(s) with float16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(c(f"  Loading tokenizer from {model_id} …", DIM))
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(c(f"  Loading model weights onto GPU (float16) …", DIM))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    gpus = torch.cuda.device_count()
    for i in range(gpus):
        props = torch.cuda.get_device_properties(i)
        mem_used = torch.cuda.memory_allocated(i) / 1e9
        print(c(f"  GPU {i}: {props.name} — {mem_used:.1f} GB used", GREEN))

    return tokenizer, model


# ── streaming renderer ────────────────────────────────────────────────────────

class _StreamRenderer:
    """State-machine that renders streaming tokens with think/tool formatting."""

    _TAGS = ["<think>", "</think>", "<tool_call>", "</tool_call>"]

    def __init__(self, start_in_think: bool = False) -> None:
        self._buf = ""
        self._tool_xml = ""
        if start_in_think:
            self._state = "think"
            print()
            print(c("  💭 thinking", DIM), flush=True)
            print(c("  ", DIM), end="", flush=True)
        else:
            self._state = "normal"

    def feed(self, token: str) -> None:
        self._buf += token
        self._flush()

    def finish(self) -> None:
        if self._buf:
            if self._state == "think":
                print(c(self._buf, DIM), end="", flush=True)
            else:
                self._emit(self._buf)
            self._buf = ""
        print()

    def _flush(self) -> None:
        while self._buf:
            prev = len(self._buf)
            if self._state == "normal":
                self._flush_normal()
            elif self._state == "think":
                self._flush_think()
            elif self._state == "tool":
                self._flush_tool()
            else:
                break
            if len(self._buf) == prev:
                break

    def _flush_normal(self) -> None:
        b = self._buf
        _OPEN  = ("<think>", "<tool_call>")
        _CLOSE = ("</think>", "</tool_call>")
        _ALL   = _OPEN + _CLOSE
        for tag in _ALL:
            if b.startswith(tag):
                self._buf = b[len(tag):]
                if tag == "<think>":
                    self._state = "think"
                    print()
                    print(c("  💭 thinking", DIM), flush=True)
                    print(c("  ", DIM), end="", flush=True)
                elif tag == "<tool_call>":
                    self._state = "tool"
                    self._tool_xml = tag
                return
        for tag in _ALL:
            for prefix_len in range(1, len(tag)):
                if b.endswith(tag[:prefix_len]):
                    safe = b[:-prefix_len]
                    if safe:
                        self._emit(safe)
                        self._buf = b[-prefix_len:]
                    return
        self._emit(b)
        self._buf = ""

    def _flush_think(self) -> None:
        close = "</think>"
        idx = self._buf.find(close)
        if idx >= 0:
            print(c(self._buf[:idx], DIM), end="", flush=True)
            self._buf = self._buf[idx + len(close):]
            self._state = "normal"
            print()
        else:
            for prefix_len in range(len(close) - 1, 0, -1):
                if self._buf.endswith(close[:prefix_len]):
                    safe = self._buf[:-prefix_len]
                    if safe:
                        print(c(safe, DIM), end="", flush=True)
                        self._buf = self._buf[-prefix_len:]
                    return
            print(c(self._buf, DIM), end="", flush=True)
            self._buf = ""

    def _flush_tool(self) -> None:
        close = "</tool_call>"
        idx = self._buf.find(close)
        if idx >= 0:
            self._tool_xml += self._buf[:idx + len(close)]
            self._buf = self._buf[idx + len(close):]
            self._state = "normal"
            m = re.search(r"<function=([^>]+)>", self._tool_xml)
            name = m.group(1) if m else "tool"
            print()
            print(c(f"  ⚙  [{name}]", YELLOW, BOLD), flush=True)
            self._tool_xml = ""
        else:
            self._tool_xml += self._buf
            self._buf = ""

    def _emit(self, text: str) -> None:
        print(text, end="", flush=True)


# ── generation ────────────────────────────────────────────────────────────────

def generate_response(
    tokenizer,
    model,
    messages: list[dict],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Apply chat template, stream tokens, return raw collected text."""
    import torch
    from transformers import TextIteratorStreamer

    prompt: str = tokenizer.apply_chat_template(
        messages,
        tools=MATH_TOOLS,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [im_end_id, tokenizer.eos_token_id]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=180.0,
    )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=lambda: model.generate(**gen_kwargs), daemon=True)

    print()
    print(c("╭─ model " + "─" * 53, MAGENTA, BOLD))
    print(c("  ", MAGENTA), end="", flush=True)

    renderer = _StreamRenderer(start_in_think=enable_thinking)
    collected: list[str] = []

    with torch.no_grad():
        thread.start()
        for token in streamer:
            collected.append(token)
            renderer.feed(token)
        thread.join()

    renderer.finish()
    return "".join(collected)


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
    tokenizer,
    model,
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    max_iterations: int = 15,
) -> tuple[str, str | None, list[dict]]:
    """Drive the generate → tool-call → observe loop for one problem.

    Args:
        problem: The problem statement (user turn).
        system_prompt: Rendered math_prompt.j2 content.
        sandbox: A fresh MathSandbox instance for this problem.
        tokenizer: Loaded tokenizer.
        model: Loaded model.
        enable_thinking: Whether to use Qwen3.5 thinking mode.
        max_new_tokens: Token budget per generation call.
        temperature: Sampling temperature.
        max_iterations: Max generate→execute cycles before giving up.

    Returns:
        Tuple of (final_response_text, extracted_boxed_answer, message_history).
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]

    last_results: list[str] = []

    for iteration in range(max_iterations):
        raw = generate_response(
            tokenizer, model, messages, enable_thinking, max_new_tokens, temperature
        )
        thinking, body = split_thinking(raw)
        tool_calls = parse_tool_calls(body)

        if not tool_calls:
            clean = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", body).strip()
            messages.append({"role": "assistant", "content": clean})
            # Cascading fallbacks: body → thinking block → full raw text
            answer = (
                extract_boxed(clean)
                or extract_boxed(thinking)
                or extract_boxed(raw)
            )
            if answer is None and "\\boxed" in raw:
                # \boxed present but unparseable — show tail for diagnosis
                snip = repr(raw[-300:])
                print(c(f"\n  ⚠  extract_boxed failed — raw tail: {snip}", RED))
            return clean, answer, messages

        messages.append({"role": "assistant", "content": raw})

        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]
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

            messages.append({"role": "tool", "content": result})

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
    """Render math_prompt.j2 with Jinja2."""
    from jinja2 import Environment, FileSystemLoader

    template_dir = REPO_ROOT / "libs" / "shared" / "src" / "shared" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    return env.get_template("math_prompt.j2").render(answer_range=answer_range)


# ── dataset loading ───────────────────────────────────────────────────────────

DATASET_PATHS: dict[str, str] = {
    "daft_math":      "daft_math/train",
    "olym_math_easy": "olym_math/en-easy/test",
    "olym_math_hard": "olym_math/en-hard/test",
}

DATA_ROOT = REPO_ROOT / "libs" / "evaluation" / "src" / "evaluation" / "data"

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

    Problems are iterated in dataset order; scanning stops once *n* valid rows
    have been collected.
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


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Math eval agent — MathSandbox + Transformers (no adapters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "RUNE_TOOLS_MODEL",
            "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B"
            "/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        ),
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dataset",
        default="daft_math",
        choices=list(DATASET_PATHS.keys()),
        help="Which dataset to evaluate (default: daft_math)",
    )
    parser.add_argument(
        "--n", type=int, default=2,
        help="Number of problems to evaluate (default: 2)",
    )
    parser.add_argument(
        "--problem",
        metavar="TEXT",
        help="Run a single ad-hoc problem instead of loading a dataset.",
    )
    parser.add_argument(
        "--answer",
        metavar="VALUE",
        default="?",
        help="Ground-truth answer for --problem (for display/scoring only).",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable Qwen3.5 thinking mode (faster per token)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=100000,
        help="Max tokens per generation call (default: 32768; thinking mode burns many tokens)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature (default: 0.3; use 0 for greedy)",
    )
    parser.add_argument(
        "--sandbox-timeout", type=float, default=180.0,
        help="Per-execution timeout for MathSandbox in seconds (default: 180)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=DATA_ROOT / "results",
        help="Directory to save per-problem JSON and summary",
    )
    args = parser.parse_args()

    enable_thinking = not args.no_thinking

    print()
    print(banner("Math Eval Agent"))
    print(c(f"  Model   : {args.model}", CYAN))
    if args.problem:
        print(c(f"  Mode    : ad-hoc problem", CYAN))
    else:
        print(c(f"  Dataset : {args.dataset}  (n={args.n})", CYAN))
    print(c(f"  Thinking: {'on' if enable_thinking else 'off'}", CYAN))
    print(c(f"  Sandbox timeout: {args.sandbox_timeout:.0f}s per execution", CYAN))
    print()

    tokenizer, model = load_model(args.model)
    print(c("  ✓ Model ready\n", GREEN, BOLD))

    system_prompt = render_system_prompt()

    if args.problem:
        problems = [{"id": "adhoc_0", "problem": args.problem, "ground_truth": args.answer}]
    else:
        problems = load_problems(args.dataset, args.n)
    print(c(f"  Loaded {len(problems)} problem(s)\n", DIM))

    out_dir = args.out_dir / (args.dataset if not args.problem else "adhoc")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, prob in enumerate(problems):
        print()
        print(banner(f"Problem {idx + 1}/{len(problems)}  [{prob['id']}]"))
        print()
        print(c("  Problem:", CYAN, BOLD))
        for line in textwrap.wrap(prob["problem"], width=78):
            print(f"    {line}")
        print()
        print(c(f"  Ground truth : {prob['ground_truth']}", GREEN, BOLD))
        if prob.get("source"):
            print(c(f"  Source       : {prob['source']}", DIM))
        print()

        with MathSandbox(MathConfig(default_timeout=args.sandbox_timeout)) as sandbox:
            final_response, model_answer, history = solve_problem(
                problem=prob["problem"],
                system_prompt=system_prompt,
                sandbox=sandbox,
                tokenizer=tokenizer,
                model=model,
                enable_thinking=enable_thinking,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

        correct = (
            str(model_answer).strip() == str(prob["ground_truth"]).strip()
            if model_answer is not None
            else False
        )
        verdict = "✓ CORRECT" if correct else "✗ WRONG"
        verdict_color = GREEN if correct else RED

        print()
        print(c("  ┌─ Result " + "─" * 50, CYAN, BOLD))
        print(c(f"  │  Model answer : {model_answer}", MAGENTA, BOLD))
        print(c(f"  │  Ground truth : {prob['ground_truth']}", GREEN))
        print(c(f"  │  Verdict      : {verdict}", verdict_color, BOLD))
        n_turns = sum(1 for m in history if m["role"] == "assistant")
        print(c(f"  │  Turns        : {n_turns}", DIM))
        print(c("  └" + "─" * 58, CYAN, BOLD))

        record: dict = {
            **{k: v for k, v in prob.items()},
            "model": args.model,
            "model_answer": model_answer,
            "model_response": final_response,
            "correct": correct,
            "n_turns": n_turns,
            "history": history,
        }
        results.append(record)

        prob_file = out_dir / f"{prob['id']}.json"
        with open(prob_file, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(c(f"\n  Saved → {prob_file}", DIM))

    # ── summary ───────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_correct = sum(1 for r in results if r["correct"])
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "n_problems": len(results),
        "n_correct": n_correct,
        "accuracy": n_correct / len(results) if results else 0.0,
        "timestamp": ts,
        "results": [
            {k: v for k, v in r.items() if k != "history"}
            for r in results
        ],
    }
    summary_file = out_dir / f"summary_{ts}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(banner("Summary"))
    print(c(f"  Correct  : {n_correct}/{len(results)}", GREEN if n_correct > 0 else RED, BOLD))
    print(c(f"  Accuracy : {summary['accuracy']:.0%}", CYAN))
    print(c(f"  Saved    : {summary_file}", DIM))
    print()


if __name__ == "__main__":
    main()
