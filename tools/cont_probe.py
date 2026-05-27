# tools/cont_probe.py
"""Single-shot continuation probe for adapter-as-memory experimentation.

One run = one configuration. Dumps all artifacts to runs/cont_probe/<timestamp>/
for manual inspection. No Optuna, no MLflow.

Run:
  uv run python tools/cont_probe.py \
      --scenario mid_fn --scaling 5.0 --traj-window 10 \
      --prompt-template tail --last-lines 3 --max-tokens 256 --no-schema
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Seed code for the two scenarios
# ---------------------------------------------------------------------------

SEED_MID_FN = '''\
class Calculator:
    """Simple calculator with history tracking."""

    def __init__(self) -> None:
        self._history: list[str] = []

    def add(self, a: float, b: float) -> float:
        result = a + b
        self._history.append(f"add({a}, {b}) = {result}")
        return result

    def subtract(self, a: float, b: float) -> float:
        result = a - b
        self._history.append(f"subtract({a}, {b}) = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self._history.append(f"multiply({a}, {b}) = {result}")
        return result

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        result = a / b
        self._history.append(f"divide({a}, {b}) = {result}")
        return result

    def history(self) -> list[str]:
        return list(self._history[-10:])
'''

SEED_CROSS_FN = '''\
"""Utility functions for text processing."""


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip."""
    import re
    return re.sub(r"\\s+", " ", text).strip()


def count_words(text: str) -> int:
    """Return the number of whitespace-delimited words."""
    return len(text.split())


def truncate(text: str, max_len: int = 80, suffix: str = "...") -> str:
    """Truncate text to max_len characters, appending suffix if trimmed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def extract_emails(text: str) -> list[str]:
    """Return all email addresses found in text."""
    import re
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+", text)
'''

# Tasks that describe what the model should produce
TASK_MID_FN = (
    "Write a Python module with a Calculator class that supports "
    "add, subtract, multiply, divide (with ZeroDivisionError handling), "
    "power, and a history method that returns the last 10 operations. "
    "Include type hints and a test class with at least 8 test methods."
)

TASK_CROSS_FN = (
    "Write a Python utility module with functions: normalize_whitespace, "
    "count_words, truncate, extract_emails, slugify, and wrap_text. "
    "Include comprehensive tests for all functions."
)

# ---------------------------------------------------------------------------
# Truncation helpers (where we "cut" the seed code)
# ---------------------------------------------------------------------------


def _truncate_mid_fn() -> tuple[str, str]:
    """Cut the Calculator mid-function (inside divide, after the zero-check).

    Returns (full_seed, accumulated_before).
    """
    lines = SEED_MID_FN.splitlines(keepends=True)
    # Find "if b == 0:" inside divide — cut right after that line
    for i, line in enumerate(lines):
        if "if b == 0:" in line:
            cut = i + 1  # keep the if-line, cut before the raise
            break
    else:
        cut = len(lines) // 2
    accumulated = "".join(lines[:cut])
    return SEED_MID_FN, accumulated


def _truncate_cross_fn() -> tuple[str, str]:
    """Cut the module cleanly between function 2 and function 3.

    Returns (full_seed, accumulated_before).
    """
    lines = SEED_CROSS_FN.splitlines(keepends=True)
    # Find the third "def " — cut right before it
    def_count = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            def_count += 1
            if def_count == 3:
                # back up to include the blank line before the def
                cut = i - 1 if i > 0 and lines[i - 1].strip() == "" else i
                accumulated = "".join(lines[:cut])
                return SEED_CROSS_FN, accumulated
    accumulated = "".join(lines[: len(lines) // 2])
    return SEED_CROSS_FN, accumulated


SCENARIOS: dict[str, tuple[str, Any]] = {
    "mid_fn": (TASK_MID_FN, _truncate_mid_fn),
    "cross_fn": (TASK_CROSS_FN, _truncate_cross_fn),
}

# ---------------------------------------------------------------------------
# Helpers (copied from continuation_scaling_hpo.py to avoid its imports)
# ---------------------------------------------------------------------------


def _last_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if n > 0 else ""


def _first_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[:n])


def _scale_b_only_inplace(sd: dict[str, Any], factor: float) -> dict[str, Any]:
    for k, v in sd.items():
        if "lora_B" in k:
            sd[k] = v * factor
    return sd


def _coherence_at_boundary(before: str, after: str) -> float:
    if not before or not after:
        return 0.0
    last_line = before.rstrip().splitlines()[-1] if before.strip() else ""
    first_line = after.lstrip().splitlines()[0] if after.strip() else ""
    if not last_line or not first_line:
        return 0.0
    indent_before = len(last_line) - len(last_line.lstrip())
    indent_after = len(first_line) - len(first_line.lstrip())
    indent_ok = 1.0 if abs(indent_before - indent_after) <= 8 else 0.0
    printable = sum(1 for c in after if c.isprintable() or c in "\n\t")
    ascii_ratio = printable / max(len(after), 1)
    return indent_ok * 0.5 + ascii_ratio * 0.5


def _regenerated_existing(continuation: str, accumulated_tail: str) -> bool:
    """Check if the continuation re-emits code already in the tail window."""
    first_line = ""
    for line in continuation.splitlines():
        if line.strip():
            first_line = line.strip()
            break
    if not first_line:
        return False
    return first_line in accumulated_tail

# ---------------------------------------------------------------------------
# Trajectory builders (what the hypernetwork sees)
# ---------------------------------------------------------------------------

_TRAJ_CODE_CAP = 4000


def _cap_code(accumulated: str) -> str:
    if len(accumulated) <= _TRAJ_CODE_CAP:
        return accumulated
    return "...\n" + accumulated[-_TRAJ_CODE_CAP:]


def _traj_sliding_window(task: str, accumulated: str, window: int) -> str:
    tail = _last_n_lines(accumulated, window)
    return f"GOAL: {task[:200]}\nRESUME FROM:\n{tail}"


def _traj_minimal(task: str, accumulated: str, window: int) -> str:
    return f"GOAL: {task[:200]}\nCODE SO FAR:\n{_cap_code(accumulated)}"


def _traj_with_counter(task: str, accumulated: str, window: int) -> str:
    return (
        f"CONTINUATION 1/5\n"
        f"GOAL: {task[:200]}\n"
        f"CODE SO FAR:\n{_cap_code(accumulated)}"
    )


def _traj_with_structure(task: str, accumulated: str, window: int) -> str:
    lines = accumulated.splitlines()
    n_lines = len(lines)
    n_defs = sum(1 for ln in lines if ln.strip().startswith("def "))
    n_classes = sum(1 for ln in lines if ln.strip().startswith("class "))
    return (
        f"CONTINUATION 1/5\n"
        f"GOAL: {task[:200]}\n"
        f"STRUCTURE: {n_lines} lines, {n_classes} classes, {n_defs} functions\n"
        f"CODE SO FAR:\n{_cap_code(accumulated)}"
    )


def _traj_code_template(task: str, accumulated: str, window: int) -> str:
    """Matches the code.j2 template format the hypernetwork was trained on."""
    return (
        f"ROLE: coder\n"
        f"PROJECT: {task[:300]}\n"
        f"SUBTASK: continuation (1/1)\n"
        f"DESCRIPTION: Continue generating code from where it left off.\n"
        f"\n"
        f"PLAN:\n"
        f"Continue the implementation. The code below is partially complete.\n"
        f"\n"
        f"EXISTING CODE:\n"
        f"{_cap_code(accumulated)}\n"
        f"PRACTICES: Clean layered architecture, no stubs or placeholders, no dead\n"
        f"code, specific exceptions with context."
    )


TRAJECTORY_FLAVORS: dict[str, Any] = {
    "sliding_window": _traj_sliding_window,
    "minimal_goal_code": _traj_minimal,
    "with_attempt_counter": _traj_with_counter,
    "with_structural_summary": _traj_with_structure,
    "code_template": _traj_code_template,
}

# ---------------------------------------------------------------------------
# Prompt templates (what the model sees in the user turn)
# ---------------------------------------------------------------------------


def _prompt_tail(accumulated: str, first_lines: int, last_lines: int, task: str) -> str:
    tail = _last_n_lines(accumulated, last_lines)
    return f"Continue the code:\n{tail}"


def _prompt_head_tail(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    head = _first_n_lines(accumulated, first_lines) if first_lines > 0 else ""
    tail = _last_n_lines(accumulated, last_lines)
    parts = [p for p in [head, "...", tail] if p]
    return "Continue the code:\n" + "\n".join(parts)


def _prompt_instruction(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    tail = _last_n_lines(accumulated, last_lines)
    return (
        f"Original task: {task[:150]}\n"
        f"Continue from where the code left off:\n{tail}"
    )


def _prompt_minimal(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    return f"Continue writing code for: {task[:150]}"


PROMPT_TEMPLATES: dict[str, Any] = {
    "tail": _prompt_tail,
    "head_tail": _prompt_head_tail,
    "instruction": _prompt_instruction,
    "minimal": _prompt_minimal,
}

# ---------------------------------------------------------------------------
# Generation: plaintext (--no-schema) vs schema (xgrammar)
# ---------------------------------------------------------------------------


def _generate_plaintext(
    base_model: Any,
    tokenizer: Any,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float,
    top_p: float,
    top_k: int,
    no_repeat_ngram_size: int,
) -> tuple[str, int]:
    """Direct generation without grammar constraints. Returns (text, token_count)."""
    import torch  # noqa: PLC0415

    sampling: dict[str, Any] = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    # TODO: enable_thinking=True could let the adapter continue
    # thinking past the context window — test once code continuation
    # is proven.
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", enable_thinking=False,
    )
    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(base_model.device)
    else:
        input_ids = encoded.to(base_model.device)

    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **sampling,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, len(new_tokens)


def _generate_schema(
    base_model: Any,
    tokenizer: Any,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float,
    top_p: float,
    top_k: int,
    no_repeat_ngram_size: int,
) -> tuple[str, int]:
    """Grammar-constrained generation via xgrammar + CodeResult schema."""
    import json as _json  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import xgrammar as xgr  # noqa: PLC0415

    from rune.engine.parse import CodeResult  # noqa: PLC0415

    sampling: dict[str, Any] = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    base_inner = getattr(base_model, "base_model", base_model)
    model_config = getattr(base_inner, "config", None)
    text_cfg = getattr(model_config, "text_config", model_config)
    vocab_size = getattr(text_cfg, "vocab_size", None) or tokenizer.vocab_size

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size,
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    schema_json = _json.dumps(CodeResult.model_json_schema())
    compiled = compiler.compile_json_schema(schema_json, max_whitespace_cnt=16)
    logits_processor = xgr.contrib.hf.LogitsProcessor(compiled)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", enable_thinking=False,
    )
    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(base_model.device)
    else:
        input_ids = encoded.to(base_model.device)

    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            logits_processor=[logits_processor],
            **sampling,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return text, len(new_tokens)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def _diagnose(
    continuation: str,
    accumulated_before: str,
    traj_window: int,
    n_tokens: int,
    max_tokens: int,
    scaling: float,
    no_schema: bool,
    prompt_template: str,
) -> dict[str, Any]:
    tail_window = _last_n_lines(accumulated_before, traj_window)
    coh = _coherence_at_boundary(accumulated_before, continuation)
    regen = _regenerated_existing(continuation, tail_window)

    schema_valid = False
    if not no_schema:
        try:
            from rune.engine.parse import CodeResult  # noqa: PLC0415
            CodeResult.model_validate_json(continuation)
            schema_valid = True
        except Exception:
            pass

    return {
        "coherence": round(coh, 4),
        "regenerated_existing": regen,
        "grammar_completed": not no_schema and schema_valid,
        "schema_valid": schema_valid,
        "stopped_early": n_tokens < max_tokens,
        "tokens_used": n_tokens,
        "scaling": scaling,
        "no_schema": no_schema,
        "prompt_template": prompt_template,
        "traj_window": traj_window,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-shot continuation probe",
    )
    parser.add_argument(
        "--scenario", choices=["mid_fn", "cross_fn"], required=True,
    )
    parser.add_argument("--scaling", type=float, default=5.0)
    parser.add_argument(
        "--prompt-template",
        choices=list(PROMPT_TEMPLATES.keys()),
        default="tail",
    )
    parser.add_argument("--traj-window", type=int, default=10)
    parser.add_argument("--last-lines", type=int, default=3)
    parser.add_argument("--first-lines", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-schema", action="store_true")
    parser.add_argument(
        "--trajectory",
        choices=list(TRAJECTORY_FLAVORS.keys()),
        default="with_attempt_counter",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("benchmarks/bench.yaml"),
    )
    args = parser.parse_args()

    from rune.config import load_config  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    cfg = load_config(args.config)
    if not cfg.checkpoint_path:
        parser.error("Config must set checkpoint_path")

    print("Loading model...", file=sys.stderr, flush=True)
    model = ModelWrapper.from_config(cfg)
    base_model = model._base_model
    tokenizer = model._tokenizer

    task_text, truncate_fn = SCENARIOS[args.scenario]
    full_seed, accumulated_before = truncate_fn()

    traj_fn = TRAJECTORY_FLAVORS[args.trajectory]
    trajectory = traj_fn(task_text, accumulated_before, args.traj_window)

    print("Generating adapter...", file=sys.stderr, flush=True)
    adapter = model.generate_adapter(trajectory, offload_base=False)
    model.hotswap_adapter(
        _scale_b_only_inplace(adapter.state_dict, args.scaling),
    )

    prompt_fn = PROMPT_TEMPLATES[args.prompt_template]
    prompt = prompt_fn(accumulated_before, args.first_lines, args.last_lines, task_text)

    system_prompt = "You are a code generator."

    no_repeat_ngram = cfg.hpo.get("no_repeat_ngram_size", 12)

    print("Generating continuation...", file=sys.stderr, flush=True)
    if args.no_schema:
        continuation, n_tokens = _generate_plaintext(
            base_model, tokenizer, prompt, system_prompt,
            args.max_tokens, cfg.temperature, cfg.repetition_penalty,
            cfg.top_p, cfg.top_k, no_repeat_ngram,
        )
    else:
        continuation, n_tokens = _generate_schema(
            base_model, tokenizer, prompt, system_prompt,
            args.max_tokens, cfg.temperature, cfg.repetition_penalty,
            cfg.top_p, cfg.top_k, no_repeat_ngram,
        )

    diagnosis = _diagnose(
        continuation, accumulated_before, args.traj_window,
        n_tokens, args.max_tokens, args.scaling,
        args.no_schema, args.prompt_template,
    )

    # Dump artifacts
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / "cont_probe" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "00_seed_code.txt").write_text(full_seed)
    (run_dir / "01_accumulated_before.txt").write_text(accumulated_before)
    (run_dir / "02_trajectory.txt").write_text(trajectory)
    (run_dir / "03_prompt.txt").write_text(prompt)
    (run_dir / "04_continuation.txt").write_text(continuation)
    (run_dir / "05_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2) + "\n",
    )

    print(f"\n{'='*60}", flush=True)
    print(f"Run saved to: {run_dir}", flush=True)
    print(f"{'='*60}", flush=True)
    print(json.dumps(diagnosis, indent=2), flush=True)


if __name__ == "__main__":
    main()
