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
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rune.engine.continuation import dedup_code as _dedup_code
from rune.engine.continuation import extract_code as _extract_code

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

SEED_LARGE = '''\
"""Data pipeline: load, validate, transform, and export records."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Record:
    """A single data record with validation metadata."""

    id: str
    timestamp: datetime
    payload: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    valid: bool = True
    errors: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False


class ValidationError(Exception):
    pass


class Pipeline:
    """Configurable ETL pipeline for Record objects."""

    def __init__(self, strict: bool = False) -> None:
        self._strict = strict
        self._records: list[Record] = []
        self._transforms: list[Any] = []
        self._stats: dict[str, int] = {
            "loaded": 0, "valid": 0, "invalid": 0, "exported": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def load_json(self, path: Path) -> list[Record]:
        with open(path) as f:
            raw = json.load(f)
        records = []
        for item in raw:
            rec = Record(
                id=item["id"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
                payload=item.get("payload", {}),
                tags=item.get("tags", []),
            )
            records.append(rec)
        self._records.extend(records)
        self._stats["loaded"] += len(records)
        return records

    def load_csv(self, path: Path, delimiter: str = ",") -> list[Record]:
        records = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                rec = Record(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    payload={k: v for k, v in row.items()
                             if k not in ("id", "timestamp")},
                )
                records.append(rec)
        self._records.extend(records)
        self._stats["loaded"] += len(records)
        return records

    def validate(self, records: list[Record] | None = None) -> list[Record]:
        targets = records if records is not None else self._records
        for rec in targets:
            if not rec.id or not rec.id.strip():
                rec.add_error("Missing id")
            if not re.match(r"^[A-Za-z0-9_-]+$", rec.id):
                rec.add_error(f"Invalid id format: {rec.id}")
            if rec.timestamp > datetime.now():
                rec.add_error("Timestamp in the future")
            if not rec.payload:
                rec.add_error("Empty payload")
        valid = [r for r in targets if r.valid]
        invalid = [r for r in targets if not r.valid]
        self._stats["valid"] += len(valid)
        self._stats["invalid"] += len(invalid)
        if self._strict and invalid:
            raise ValidationError(
                f"{len(invalid)} records failed validation"
            )
        return valid

    def add_transform(self, fn: Any) -> None:
        self._transforms.append(fn)

    def transform(self, records: list[Record] | None = None) -> list[Record]:
        targets = records if records is not None else self._records
        for fn in self._transforms:
            targets = [fn(r) for r in targets]
        return targets

    def filter_by_tags(
        self, records: list[Record], required: set[str],
    ) -> list[Record]:
        return [r for r in records if required.issubset(set(r.tags))]

    def deduplicate(self, records: list[Record]) -> list[Record]:
        seen: set[str] = set()
        result: list[Record] = []
        for rec in records:
            if rec.id not in seen:
                seen.add(rec.id)
                result.append(rec)
        return result

    def sort_by_timestamp(
        self, records: list[Record], reverse: bool = False,
    ) -> list[Record]:
        return sorted(records, key=lambda r: r.timestamp, reverse=reverse)

    def export_json(self, records: list[Record], path: Path) -> int:
        data = []
        for rec in records:
            data.append({
                "id": rec.id,
                "timestamp": rec.timestamp.isoformat(),
                "payload": rec.payload,
                "tags": rec.tags,
                "valid": rec.valid,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._stats["exported"] += len(data)
        return len(data)

    def export_csv(self, records: list[Record], path: Path) -> int:
        if not records:
            return 0
        fieldnames = ["id", "timestamp", "valid"]
        payload_keys = set()
        for rec in records:
            payload_keys.update(rec.payload.keys())
        fieldnames.extend(sorted(payload_keys))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                row: dict[str, Any] = {
                    "id": rec.id,
                    "timestamp": rec.timestamp.isoformat(),
                    "valid": rec.valid,
                }
                row.update(rec.payload)
                writer.writerow(row)
        self._stats["exported"] += len(records)
        return len(records)

    def summary(self) -> str:
        return (
            f"Pipeline: {self._stats[\'loaded\']} loaded, "
            f"{self._stats[\'valid\']} valid, "
            f"{self._stats[\'invalid\']} invalid, "
            f"{self._stats[\'exported\']} exported"
        )


def merge_pipelines(*pipelines: Pipeline) -> Pipeline:
    merged = Pipeline()
    for p in pipelines:
        merged._records.extend(p._records)
        for k in merged._stats:
            merged._stats[k] += p._stats.get(k, 0)
    return merged


def batch_process(
    paths: list[Path], strict: bool = False,
) -> tuple[list[Record], dict[str, int]]:
    pipe = Pipeline(strict=strict)
    all_records: list[Record] = []
    for path in paths:
        if path.suffix == ".json":
            records = pipe.load_json(path)
        elif path.suffix == ".csv":
            records = pipe.load_csv(path)
        else:
            continue
        valid = pipe.validate(records)
        all_records.extend(valid)
    return all_records, pipe.stats
'''

TASK_LARGE = (
    "Write a Python data pipeline module with a Record dataclass and a "
    "Pipeline class supporting load_json, load_csv, validate (with strict "
    "mode), transform, filter_by_tags, deduplicate, sort_by_timestamp, "
    "export_json, export_csv, and a summary method. Include merge_pipelines "
    "and batch_process helper functions."
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


_LARGE_CUT_POINTS = {
    "small": "def filter_by_tags",
    "medium": "def export_json",
    "large": "def merge_pipelines",
    "full": None,
}


def _truncate_large(cut: str = "medium") -> tuple[str, str]:
    """Cut the large pipeline module at a named boundary.

    cut: 'small' (~2000 chars), 'medium' (~3200 chars), 'large' (~4500 chars), 'full' (all ~5200 chars).
    Returns (full_seed, accumulated_before).
    """
    marker = _LARGE_CUT_POINTS.get(cut)
    if marker is None:
        return SEED_LARGE, SEED_LARGE
    lines = SEED_LARGE.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip().startswith(marker):
            cut_idx = i - 1 if i > 0 and lines[i - 1].strip() == "" else i
            accumulated = "".join(lines[:cut_idx])
            return SEED_LARGE, accumulated
    return SEED_LARGE, SEED_LARGE


SCENARIOS: dict[str, tuple[str, Any]] = {
    "mid_fn": (TASK_MID_FN, _truncate_mid_fn),
    "cross_fn": (TASK_CROSS_FN, _truncate_cross_fn),
    "large_small": (TASK_LARGE, lambda: _truncate_large("small")),
    "large_medium": (TASK_LARGE, lambda: _truncate_large("medium")),
    "large_large": (TASK_LARGE, lambda: _truncate_large("large")),
    "large_full": (TASK_LARGE, lambda: _truncate_large("full")),
}

# ---------------------------------------------------------------------------
# Helpers
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

_TRAJ_CODE_CAP = 3500


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


_LAST_THINK_SUMMARY = ""


def _traj_code_template(task: str, accumulated: str, window: int) -> str:
    """Match code.j2 layout, maximize code density for hypernetwork."""
    reasoning = ""
    if _LAST_THINK_SUMMARY:
        reasoning = f"\nPRIOR REASONING:\n{_LAST_THINK_SUMMARY}\n"
    return (
        f"ROLE: coder\n"
        f"PROJECT: {task[:300]}\n"
        f"SUBTASK: continuation (1/1)\n"
        f"DESCRIPTION: Continue generating code from where it left off.\n"
        f"\n"
        f"PLAN:\nContinue implementing remaining functionality.\n"
        f"{reasoning}"
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


def _prompt_bare(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    tail = _last_n_lines(accumulated, last_lines) if last_lines > 0 else ""
    if tail:
        return f"Continue the code:\n{tail}"
    return "Continue writing code."


def _prompt_bare_directed(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    tail = _last_n_lines(accumulated, last_lines) if last_lines > 0 else ""
    if tail:
        return f"Continue implementing the module.\n{tail}"
    return "Continue implementing the module."


def _prompt_structural(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    """Task + what's done + tail. Model infers what's missing."""
    funcs = []
    current_class = None
    for line in accumulated.splitlines():
        stripped = line.strip()
        if stripped.startswith("class "):
            current_class = stripped.split("(")[0].split(":")[0].replace("class ", "").strip()
        elif stripped.startswith("def "):
            name = stripped.split("(")[0].replace("def ", "").strip()
            if current_class and (line.startswith("    ") or line.startswith("\t")):
                funcs.append(f"{current_class}.{name}")
            else:
                funcs.append(name)
                current_class = None

    tail = _last_n_lines(accumulated, last_lines) if last_lines > 0 else ""
    parts = [f"Task: {task[:200]}"]
    if funcs:
        parts.append(f"Done: {', '.join(funcs[-8:])}.")
    parts.append("Write ONLY the next unimplemented method. No redefinitions.")
    if tail:
        parts.append(f"Resume:\n{tail}")
    return "\n".join(parts)


def _prompt_task_only(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    """Task spec + tail. Adapter carries what's done."""
    tail = _last_n_lines(accumulated, last_lines) if last_lines > 0 else ""
    parts = [f"Task: {task[:200]}"]
    parts.append("Write ONLY the next unimplemented method. No redefinitions.")
    if tail:
        parts.append(f"Resume:\n{tail}")
    return "\n".join(parts)


PROMPT_TEMPLATES: dict[str, Any] = {
    "tail": _prompt_tail,
    "head_tail": _prompt_head_tail,
    "instruction": _prompt_instruction,
    "minimal": _prompt_minimal,
    "bare": _prompt_bare,
    "bare_directed": _prompt_bare_directed,
    "structural": _prompt_structural,
    "task_only": _prompt_task_only,
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
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", enable_thinking=False,
        add_generation_prompt=True,
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
        add_generation_prompt=True,
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


def _extract_think(raw: str) -> str:
    """Extract content from <think> blocks."""
    blocks = re.findall(r"<think>(.*?)</think>", raw, re.DOTALL)
    if blocks:
        return "\n".join(b.strip() for b in blocks)
    m = re.search(r"<think>(.*)", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _summarize_think(think_text: str) -> str:
    """Condense think block to key observations (first 3 sentences, max 200 chars)."""
    if not think_text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", think_text.strip())
    summary = " ".join(sentences[:3])
    return summary[:200]



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-shot continuation probe",
    )
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()), required=True,
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
    parser.add_argument("--rounds", type=int, default=1)
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
    _full_seed, accumulated = truncate_fn()

    system_prompt = (
        "Output only Python code. No commentary, no explanations, "
        "no markdown fences. Continue exactly from where the code left off."
    )
    no_repeat_ngram = cfg.hpo.get("no_repeat_ngram_size", 12)
    traj_fn = TRAJECTORY_FLAVORS[args.trajectory]
    prompt_fn = PROMPT_TEMPLATES[args.prompt_template]

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / "cont_probe" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch  # noqa: PLC0415

    empty_rounds = 0
    for rnd in range(args.rounds):
        torch.cuda.empty_cache()
        print(f"\n--- Round {rnd + 1}/{args.rounds} ({len(accumulated)} chars) ---",
              file=sys.stderr, flush=True)

        trajectory = traj_fn(task_text, accumulated, args.traj_window)

        print("Generating adapter...", file=sys.stderr, flush=True)
        adapter = model.generate_adapter(trajectory, offload_base=False)
        model.hotswap_adapter(
            _scale_b_only_inplace(adapter.state_dict, args.scaling),
        )

        prompt = prompt_fn(accumulated, args.first_lines, args.last_lines, task_text)

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
            continuation, accumulated, args.traj_window,
            n_tokens, args.max_tokens, args.scaling,
            args.no_schema, args.prompt_template,
        )

        prefix = f"{rnd:02d}"
        (run_dir / f"{prefix}_accumulated.txt").write_text(accumulated)
        (run_dir / f"{prefix}_trajectory.txt").write_text(trajectory)
        (run_dir / f"{prefix}_prompt.txt").write_text(prompt)
        (run_dir / f"{prefix}_raw_continuation.txt").write_text(continuation)
        (run_dir / f"{prefix}_diagnosis.json").write_text(
            json.dumps(diagnosis, indent=2) + "\n",
        )

        code = _extract_code(continuation)
        if args.rounds > 1:
            code = _dedup_code(code, accumulated)
        (run_dir / f"{prefix}_code.txt").write_text(code)

        global _LAST_THINK_SUMMARY  # noqa: PLW0603
        think_raw = _extract_think(continuation)
        _LAST_THINK_SUMMARY = _summarize_think(think_raw)

        print(f"Round {rnd + 1}: {n_tokens} tokens, "
              f"stopped_early={diagnosis['stopped_early']}", flush=True)
        if _LAST_THINK_SUMMARY:
            print(f"Think summary: {_LAST_THINK_SUMMARY[:100]}", flush=True)
        print(f"Code extracted ({len(code)} chars):", flush=True)
        print(code[:300], flush=True)

        if args.rounds > 1 and code.strip():
            accumulated = accumulated.rstrip() + "\n" + code.strip() + "\n"
            empty_rounds = 0
        elif args.rounds > 1:
            empty_rounds += 1

        if args.rounds > 1 and diagnosis["stopped_early"]:
            print(f"Model stopped early (EOS) — task complete after round {rnd + 1}.",
                  flush=True)
            break

        if args.rounds > 1 and empty_rounds >= 2:
            print(f"No new code for {empty_rounds} consecutive rounds — stopping.",
                  flush=True)
            break

    (run_dir / "final_accumulated.txt").write_text(accumulated)

    print(f"\n{'='*60}", flush=True)
    print(f"Run saved to: {run_dir}", flush=True)
    print(f"{'='*60}", flush=True)
    if args.rounds == 1:
        print(json.dumps(diagnosis, indent=2), flush=True)


if __name__ == "__main__":
    main()
