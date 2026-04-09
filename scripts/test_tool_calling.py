#!/usr/bin/env python3
"""Multi-turn tool-calling smoke test for any OpenAI-compatible vLLM server.

Exercises the full agentic loop — model generates code → MathSandbox executes
it → result fed back → repeat — using the same ``MATH_TOOLS`` schema and
``MathSandbox`` that the production benchmark runner uses.

Usage::

    # Against the running Gemma 4 server (default)
    uv run python scripts/test_tool_calling.py

    # Different server / model
    uv run python scripts/test_tool_calling.py \\
        --base-url http://localhost:8001/v1 \\
        --model Qwen/Qwen3-30B-A3B

    # Quiet (pass/fail only)
    uv run python scripts/test_tool_calling.py --quiet

    # Extend or shrink the max agentic iterations per problem
    uv run python scripts/test_tool_calling.py --max-iter 15

Tests
-----
1. Health — GET /health returns 200.
2. Model discovery — GET /v1/models lists at least one model.
3. Single-turn chat — plain completion, no tools.
4. Single tool call — problem that should call execute_python exactly once.
5. Multi-turn tool loop — problem requiring several tool-call / observe cycles.
6. State persistence — two problems sharing one MathSandbox; second reuses
   a variable defined in the first.
"""
# ruff: noqa: T201
# mypy: ignore-errors
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
import urllib.request
from typing import Any

# ── Colour helpers ────────────────────────────────────────────────────────────

_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_RED   = "\033[31m"
_CYAN  = "\033[36m"
_YEL   = "\033[33m"
_DIM   = "\033[2m"
_RST   = "\033[0m"

def _ok(msg: str)   -> None: print(f"{_GREEN}  ✓  {_RST}{msg}")
def _fail(msg: str) -> None: print(f"{_RED}  ✗  {_RST}{msg}")
def _info(msg: str) -> None: print(f"{_CYAN}  ·  {_RST}{msg}")
def _head(msg: str) -> None: print(f"\n{_BOLD}{msg}{_RST}")
def _dim(msg: str)  -> None: print(f"{_DIM}{msg}{_RST}")


# ── Test cases ────────────────────────────────────────────────────────────────

# Each entry: (label, system_prompt, user_message, expected_substring_in_answer)
# expected_substring is checked case-insensitively against the final model reply.
_PROBLEMS: list[tuple[str, str, str, str]] = [
    (
        "single-turn chat (no tools)",
        "You are a helpful math assistant. Answer briefly.",
        "What is 17 multiplied by 23? Give only the number.",
        "391",
    ),
    (
        "single tool call — simple arithmetic",
        (
            "You are a math assistant with access to a Python sandbox. "
            "Use execute_python to verify your answers. "
            "Always print() results."
        ),
        "Use the Python sandbox to compute 2**10 + 3**5. Print the result.",
        "1267",
    ),
    (
        "multi-turn tool loop — prime factorisation",
        (
            "You are a math assistant with access to a Python sandbox. "
            "Use execute_python whenever computation helps. "
            "Always print() results."
        ),
        (
            "Find the largest prime factor of 600851475143. "
            "Use the Python sandbox to compute it. "
            "State your final answer as: The answer is <number>."
        ),
        "6857",
    ),
    (
        "multi-turn tool loop — combinatorics",
        (
            "You are a math assistant with access to a Python sandbox. "
            "Use execute_python for computation. "
            "Always print() results."
        ),
        (
            "How many ways can you arrange the letters in the word MISSISSIPPI? "
            "Use the sandbox to calculate it. "
            "State your final answer as: The answer is <number>."
        ),
        "34650",
    ),
]

# Problem that tests state persistence across two sandbox calls in *one* session.
_STATE_TEST_PROMPT = (
    "You have a Python sandbox. "
    "First call: define x = 42 and print it. "
    "Second call: compute and print x * x. "
    "Report the final value."
)


# ── Agentic loop ──────────────────────────────────────────────────────────────


def _run_agentic_loop(
    client: Any,
    model_id: str,
    system_prompt: str,
    user_message: str,
    sandbox: Any,
    max_iter: int,
    temperature: float,
    max_tokens: int,
    quiet: bool,
    use_tools: bool = True,
) -> tuple[str, int, list[dict]]:
    """Run the generate → tool-execute → observe loop.

    Returns ``(final_text, total_tokens, conversation_history)``.
    """
    from shared.mathbox import MAX_CODE_LINES, MATH_TOOLS, near_int_hint

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    total_tokens = 0

    for iteration in range(max_iter):
        kwargs: dict[str, Any] = dict(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if use_tools:
            kwargs["tools"] = MATH_TOOLS
            kwargs["tool_choice"] = "auto"

        t0 = time.perf_counter()
        resp = client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0

        choice = resp.choices[0]
        total_tokens += resp.usage.completion_tokens if resp.usage else 0

        # ── structured tool calls (OpenAI protocol) ───────────────────────────
        if choice.message.tool_calls:
            if not quiet:
                _dim(f"    [iter {iteration+1}] model called {len(choice.message.tool_calls)} tool(s)  ({elapsed:.1f}s)")

            # Append the assistant turn (with tool_calls field).
            messages.append(choice.message.model_dump(exclude_unset=False))

            for tc in choice.message.tool_calls:
                try:
                    args: dict[str, Any] = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                code: str = args.get("code", "")

                if not quiet:
                    _dim(f"    [tool] {tc.function.name}:")
                    for line in code.splitlines()[:6]:
                        _dim(f"           {line}")
                    if len(code.splitlines()) > 6:
                        _dim(f"           … ({len(code.splitlines())} lines total)")

                if len(code.splitlines()) > MAX_CODE_LINES:
                    result = f"[ERROR] Code too long ({len(code.splitlines())} lines > {MAX_CODE_LINES} limit)"
                else:
                    result = sandbox.execute(code)

                hint = near_int_hint(result)
                tool_output = result + hint

                if not quiet:
                    _dim(f"    [sandbox] → {tool_output[:120].strip()}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })

            continue

        # ── no tool calls → model is done ────────────────────────────────────
        final_text: str = choice.message.content or ""
        if not quiet:
            _dim(f"    [iter {iteration+1}] final answer  ({elapsed:.1f}s, {total_tokens} tokens)")
        return final_text, total_tokens, messages

    # max iterations hit — return whatever content we have
    final_text = choice.message.content or ""  # type: ignore[possibly-undefined]
    return final_text, total_tokens, messages


# ── Test runner ───────────────────────────────────────────────────────────────


def _discover_model(base_url: str) -> str:
    models_url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/v1/models"
    with urllib.request.urlopen(models_url, timeout=10) as r:
        data = json.loads(r.read())
    return data["data"][0]["id"]


def run_tests(
    base_url: str,
    model_arg: str,
    max_iter: int,
    temperature: float,
    max_tokens: int,
    quiet: bool,
) -> bool:
    from openai import OpenAI
    from shared.mathbox import MathSandbox

    passed = 0
    failed = 0
    results: list[tuple[str, bool, str]] = []

    # ── 1. Health check ───────────────────────────────────────────────────────
    _head("1 / Health check")
    health_url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=10) as r:
            status = r.status
        if status == 200:
            _ok(f"GET {health_url} → 200")
            passed += 1
            results.append(("Health check", True, ""))
        else:
            _fail(f"GET {health_url} → {status}")
            failed += 1
            results.append(("Health check", False, f"HTTP {status}"))
            return False
    except Exception as exc:
        _fail(f"Health check failed: {exc}")
        results.append(("Health check", False, str(exc)))
        return False

    # ── 2. Model discovery ────────────────────────────────────────────────────
    _head("2 / Model discovery")
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    try:
        if model_arg == "auto":
            model_id = _discover_model(base_url)
            _ok(f"Discovered model: {model_id}")
        else:
            model_id = model_arg
            _ok(f"Using explicit model: {model_id}")
        passed += 1
        results.append(("Model discovery", True, ""))
    except Exception as exc:
        _fail(f"Could not discover model: {exc}")
        results.append(("Model discovery", False, str(exc)))
        failed += 1
        return False

    # ── 3–6. Agentic loop tests ───────────────────────────────────────────────
    with MathSandbox() as sandbox:
        for idx, (label, sys_prompt, user_msg, expected) in enumerate(_PROBLEMS, start=3):
            _head(f"{idx} / {label}")
            _info(f"Problem: {user_msg[:100]}{'…' if len(user_msg) > 100 else ''}")
            use_tools = idx > 3  # first problem is plain chat, rest use tools

            try:
                final, n_tok, _ = _run_agentic_loop(
                    client, model_id, sys_prompt, user_msg,
                    sandbox, max_iter, temperature, max_tokens,
                    quiet, use_tools=use_tools,
                )
                snippet = final[:300].replace("\n", " ")
                if not quiet:
                    _dim(f"    Reply: {snippet}{'…' if len(final) > 300 else ''}")

                if expected.lower() in final.lower():
                    _ok(f"Answer contains '{expected}'  ({n_tok} tokens)")
                    passed += 1
                    results.append((label, True, ""))
                else:
                    _fail(f"Expected '{expected}' not found in reply")
                    if not quiet:
                        print(textwrap.indent(final[:600], "    "))
                    failed += 1
                    results.append((label, False, f"expected '{expected}' missing"))

            except Exception as exc:
                _fail(f"Exception: {exc}")
                failed += 1
                results.append((label, False, str(exc)))

        # ── 7. State-persistence test ─────────────────────────────────────────
        _head("7 / State persistence across tool calls (single sandbox session)")
        _info("x = 42 in call 1, then x*x in call 2 — sandbox must remember x")
        sandbox.reset()

        sys_prompt_state = (
            "You are a math assistant with access to a Python sandbox. "
            "Use execute_python. Always print() results."
        )
        try:
            final, n_tok, history = _run_agentic_loop(
                client, model_id, sys_prompt_state, _STATE_TEST_PROMPT,
                sandbox, max_iter, temperature, max_tokens,
                quiet, use_tools=True,
            )
            snippet = final[:300].replace("\n", " ")
            if not quiet:
                _dim(f"    Reply: {snippet}")

            # We want to see 1764 (= 42*42) appear in either the final text or
            # any sandbox output in the conversation history.
            history_str = json.dumps(history)
            found_1764 = "1764" in final or "1764" in history_str

            if found_1764:
                _ok(f"State preserved — 1764 (42²) found  ({n_tok} tokens)")
                passed += 1
                results.append(("State persistence", True, ""))
            else:
                _fail("1764 not found — sandbox state may not have persisted")
                failed += 1
                results.append(("State persistence", False, "1764 not found"))

        except Exception as exc:
            _fail(f"Exception: {exc}")
            failed += 1
            results.append(("State persistence", False, str(exc)))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    _head("Results")
    for label, ok, note in results:
        status = f"{_GREEN}PASS{_RST}" if ok else f"{_RED}FAIL{_RST}"
        suffix = f"  {_DIM}({note}){_RST}" if note else ""
        print(f"  {status}  {label}{suffix}")

    colour = _GREEN if failed == 0 else _RED
    print(f"\n{colour}{_BOLD}  {passed}/{total} passed{_RST}\n")

    return failed == 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-turn tool-calling smoke test for a vLLM / OpenAI-compatible server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Gemma 4 (default — expects server on :8000)
              uv run python scripts/test_tool_calling.py

              # Qwen3 on a different port
              uv run python scripts/test_tool_calling.py \\
                  --base-url http://localhost:8001/v1 \\
                  --model Qwen/Qwen3-30B-A3B

              # Pass/fail output only
              uv run python scripts/test_tool_calling.py --quiet
        """),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: $VLLM_BASE_URL or http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model ID, or 'auto' to discover from GET /v1/models (default: auto)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=12,
        help="Max agentic-loop iterations per problem (default: 12)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 — greedy)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max new tokens per generation step (default: 4096)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-turn detail; show pass/fail only",
    )
    args = parser.parse_args()

    import os
    base_url: str = (
        args.base_url
        or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    )

    print(f"\n{_BOLD}Tool-calling smoke test{_RST}")
    print(f"  server : {base_url}")
    print(f"  model  : {args.model}")
    print(f"  max_iter: {args.max_iter}  temperature: {args.temperature}  max_tokens: {args.max_tokens}")

    ok = run_tests(
        base_url=base_url,
        model_arg=args.model,
        max_iter=args.max_iter,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        quiet=args.quiet,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
