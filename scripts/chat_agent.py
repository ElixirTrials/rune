#!/workspace/rune/.venv-tools/bin/python
"""Interactive chat with the Rune code-execution agent.

Uses HuggingFace Transformers directly — no server, no vLLM, starts in ~20s.
Model is loaded once and kept in GPU memory for the session.

Usage::

    python scripts/chat_agent.py
    python scripts/chat_agent.py --model /path/to/model
    python scripts/chat_agent.py --no-thinking    # skip <think> blocks (faster)
    python scripts/chat_agent.py --max-new-tokens 2048

Environment variables:
    RUNE_TOOLS_MODEL   HuggingFace model ID or local path
                       (default: Qwen/Qwen3.5-4B)
"""

# ruff: noqa: T201, E402
# mypy: ignore-errors
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import textwrap
from pathlib import Path

# ── make libs/shared importable without installing into the workspace venv ─────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "libs" / "shared" / "src"))

from shared.executor import SubprocessExecutor
from shared.tools import CODE_EXECUTOR_TOOLS

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

def abbrev(text: str, max_len: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + c(f" … [{len(text)-max_len} chars] … ", DIM) + text[-half:]

# ── Qwen3.5 tool call parsing ─────────────────────────────────────────────────
# Model outputs:
#   <tool_call>
#   <function=FNAME>
#   <parameter=PNAME>
#   VALUE
#   </parameter>
#   </function>
#   </tool_call>

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract all tool calls from a raw model output string.

    Returns a list of ``{"name": str, "arguments": dict}`` dicts.
    """
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1)
        args = {
            pm.group(1): pm.group(2).strip()
            for pm in _PARAM_RE.finditer(m.group(2))
        }
        calls.append({"name": name, "arguments": args})
    return calls


def split_thinking(raw: str) -> tuple[str, str]:
    """Split ``</think>``-delimited thinking content from the response body.

    When the model runs in thinking mode the generation starts at the point
    *after* ``<think>\\n`` (which was pre-filled in the prompt), so the raw
    output looks like::

        I need to figure out...\\n</think>\\n\\nHere is the answer.

    Returns ``(thinking_text, response_text)``.
    """
    if "</think>" in raw:
        idx = raw.index("</think>")
        return raw[:idx].strip(), raw[idx + len("</think>"):].strip()
    return "", raw.strip()


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str):
    """Load tokenizer and model onto GPU(s) with float16.

    Uses float16 instead of bfloat16 because V100 (compute 7.0) does not
    support bfloat16. On Ampere+ (compute 8.0+) you can safely switch to
    torch.bfloat16 for slightly better numerics.
    """
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


# ── stream renderer ───────────────────────────────────────────────────────────

class _StreamRenderer:
    """State-machine that renders streaming tokens with clean formatting.

    Tags handled:
      <think>…</think>      — printed dim/italic, with a "💭 thinking" header
      <tool_call>…</tool_call> — XML suppressed; shows a "⚙  [tool]" badge instead
    Everything else prints normally, character by character.
    """

    _TAGS = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
    _MAX_LOOKAHEAD = max(len(t) for t in _TAGS)

    def __init__(self, start_in_think: bool = False) -> None:
        self._buf = ""          # lookahead / partial-tag buffer
        self._tool_xml = ""     # accumulates raw tool-call XML

        # Qwen3.5 pre-fills "<think>\n" into the generation prompt, so the
        # model's very first token is already INSIDE the think block — we never
        # see an opening <think> tag in the stream.  Start in think state when
        # enable_thinking is True so the content is dimmed correctly.
        if start_in_think:
            self._state = "think"
            print()
            print(c("  💭 thinking", DIM), flush=True)
            print(c("  ", DIM), end="", flush=True)
        else:
            self._state = "normal"

    # ── public ──

    def feed(self, token: str) -> None:
        self._buf += token
        self._flush()

    def finish(self) -> None:
        """Drain whatever is left after the stream ends."""
        if self._buf:
            if self._state == "think":
                print(c(self._buf, DIM), end="", flush=True)
            else:
                self._emit(self._buf)
            self._buf = ""
        print()  # terminal newline

    # ── internals ──

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
                break  # no progress — waiting for more tokens to arrive

    def _flush_normal(self) -> None:
        b = self._buf
        # Tags we either transition on or silently suppress
        _OPEN  = ("<think>", "<tool_call>")
        _CLOSE = ("</think>", "</tool_call>")   # stray close tags — suppress
        _ALL   = _OPEN + _CLOSE

        # Check for a full tag match at the start of the buffer
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
                # </think> and </tool_call> in normal state are stray — drop silently
                return
        # Could we be mid-way through any tag?
        for tag in _ALL:
            for prefix_len in range(1, len(tag)):
                if b.endswith(tag[:prefix_len]):
                    safe = b[: -prefix_len]
                    if safe:
                        self._emit(safe)
                        self._buf = b[-prefix_len:]
                    return  # wait for more tokens
        # Nothing ambiguous — emit and clear
        self._emit(b)
        self._buf = ""

    def _flush_think(self) -> None:
        close = "</think>"
        idx = self._buf.find(close)
        if idx >= 0:
            # Print up to the close tag, then switch back to normal
            print(c(self._buf[:idx], DIM), end="", flush=True)
            self._buf = self._buf[idx + len(close):]
            self._state = "normal"
            print()  # newline after thinking block
        else:
            # Hold back enough characters to detect a partial close tag
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
            self._tool_xml += self._buf[: idx + len(close)]
            self._buf = self._buf[idx + len(close):]
            self._state = "normal"
            # Pretty-print just the tool name — args are shown after execution
            name = _extract_tool_name(self._tool_xml)
            badge = f"⚙  [{name}]" if name else "⚙  [tool call]"
            print()
            print(c(f"  {badge}", YELLOW, BOLD), flush=True)
            self._tool_xml = ""
        else:
            self._tool_xml += self._buf
            self._buf = ""

    def _emit(self, text: str) -> None:
        print(text, end="", flush=True)


def _extract_tool_name(xml: str) -> str:
    """Pull the function name out of a <tool_call> block."""
    m = re.search(r"<function=([^>]+)>", xml)
    return m.group(1) if m else ""


# ── generation ────────────────────────────────────────────────────────────────

def generate_response(
    tokenizer,
    model,
    messages: list[dict],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Apply chat template, stream tokens through the renderer, return raw text."""
    import torch
    from threading import Thread
    from transformers import TextIteratorStreamer

    prompt: str = tokenizer.apply_chat_template(
        messages,
        tools=CODE_EXECUTOR_TOOLS,
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
        timeout=120.0,
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
    print(c("╭─ agent " + "─" * 53, MAGENTA, BOLD))
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


# ── agent turn ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a coding agent with access to an isolated bash sandbox.\n\n"
    "The execute tool runs BASH — any shell command works:\n"
    "  ls, grep, curl, df -h, pip install, python3, cat, sed, etc.\n\n"
    "Key rules:\n"
    "- To run Python code: python3 -c 'print(42)'  "
    "OR  write a .py file then: python3 script.py\n"
    "- To install packages:  pip install pandas numpy -q\n"
    "  (packages installed in one call persist in all later calls)\n"
    "- The working directory persists — files you create stay there\n"
    "- You can chain commands with &&:  "
    "pip install pandas -q && python3 -c 'import pandas; print(\"ok\")'\n\n"
    "If a command fails, read the error and try a DIFFERENT approach. "
    "Never repeat a command that just failed unchanged."
)


async def run_turn(
    messages: list[dict],
    executor: SubprocessExecutor,
    tokenizer,
    model,
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    max_iterations: int,
) -> str:
    """Drive a full generate → tool-call → observe loop for one user turn."""
    last_results: list[str] = []  # track recent results to detect loops

    for iteration in range(max_iterations):
        raw = generate_response(
            tokenizer, model, messages, enable_thinking,
            max_new_tokens, temperature,
        )

        # Streaming already printed the raw tokens live.
        # Parse thinking / tool calls from the collected text.
        _, body = split_thinking(raw)
        tool_calls = parse_tool_calls(body)

        if not tool_calls:
            # Final text response — strip any stray XML markers
            clean = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", body).strip()
            return clean

        # Append the full raw assistant output (template handles <think> blocks)
        messages.append({"role": "assistant", "content": raw})

        # Execute each parsed tool call and print result
        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]

            # Show a preview of what's about to run
            if name == "execute":
                code = args.get("code") or args.get("command") or args.get("script") or ""
                lines = code.splitlines()
                print()
                for line in lines[:20]:
                    print(c(f"     {line}", BLUE))
                if len(lines) > 20:
                    print(c(f"     … ({len(lines) - 20} more lines)", DIM))
            elif name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                preview = content[:120].replace("\n", "↵")
                print(c(f"     → {path}", BLUE))
                print(c(f"       {preview}{'…' if len(content) > 120 else ''}", DIM))
            elif name == "read_file":
                print(c(f"     → {args.get('path', '')}", BLUE))

            print(c(f"  ↳  [{name}] running…", YELLOW))

            result = await executor._dispatch_async(name, args)

            print(c("  ↳  result", DIM))
            for line in abbrev(result, 400).splitlines():
                print(c(f"     {line}", GREEN))

            messages.append({"role": "tool", "content": result})

            # Loop detection: if the last 3 results are identical, bail out
            last_results.append(result)
            if len(last_results) >= 3 and len(set(last_results[-3:])) == 1:
                bail = "[stuck] Same result 3 times in a row — stopping loop."
                print(c(f"\n  ⚠  {bail}", RED))
                return bail

    return "[max iterations reached]"


# ── REPL ──────────────────────────────────────────────────────────────────────

async def repl(tokenizer, model, enable_thinking: bool, max_new_tokens: int, temperature: float) -> None:
    print()
    print(banner("Rune Code Agent  (Transformers)"))
    print(c(f"  Model    : {model.config._name_or_path}", CYAN))
    print(c(f"  Thinking : {'on' if enable_thinking else 'off'}", CYAN))
    print(c(f"  Executor : SubprocessExecutor (isolated tempdir)", CYAN))
    print(c("  Commands : 'reset' for fresh sandbox  |  'exit' or Ctrl-D to quit", DIM))
    print()

    async def new_executor() -> SubprocessExecutor:
        ex = SubprocessExecutor(timeout=60.0)
        await ex.start()
        return ex

    executor = await new_executor()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(c("  ✓ Sandbox ready\n", GREEN))

    try:
        while True:
            try:
                print(c("╭─ you " + "─" * 55, CYAN, BOLD))
                user_input = input(c("╰▶ ", CYAN, BOLD)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                break
            if user_input.lower() == "reset":
                await executor.stop()
                executor = await new_executor()
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                print(c("  ✓ Sandbox and conversation history reset\n", GREEN))
                continue

            messages.append({"role": "user", "content": user_input})

            try:
                answer = await run_turn(
                    messages, executor, tokenizer, model,
                    enable_thinking, max_new_tokens, temperature,
                    max_iterations=20,
                )
            except Exception as e:
                import traceback
                print(c(f"\n  ✗ Error: {e}", RED))
                traceback.print_exc()
                messages.pop()  # remove the user message so state stays clean
                print()
                continue

            # Append final assistant answer to history
            messages.append({"role": "assistant", "content": answer})
            print()

    finally:
        await executor.stop()
        print(c("  Sandbox stopped. Bye!", DIM))


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rune code agent — Transformers mode (no server required)",
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
        "--no-thinking", action="store_true",
        help="Disable Qwen3.5 thinking mode (much faster per token)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=4096,
        help="Max tokens to generate per turn (default: 4096)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7; use 0 for greedy)",
    )
    args = parser.parse_args()

    print(banner("Loading Qwen3.5-4B"))
    print()
    tokenizer, model = load_model(args.model)
    print(c("  ✓ Model ready\n", GREEN, BOLD))

    asyncio.run(repl(
        tokenizer, model,
        enable_thinking=not args.no_thinking,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    ))


if __name__ == "__main__":
    main()
