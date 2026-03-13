"""Data pipeline for KL-divergence context distillation training.

Provides functions for:
- Converting trajectories to distillation records (activation/teacher split)
- Generating needle-in-haystack synthetic datasets for CI smoke testing
- JSONL persistence (save/load round-trip)
- Task-ID-level train/test splitting
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "format_for_distillation",
    "generate_needle_dataset",
    "save_jsonl",
    "load_jsonl",
    "split_by_task_id",
]

# ---------------------------------------------------------------------------
# Needle dataset templates (deterministic, no LLM, suitable for CI)
# ---------------------------------------------------------------------------

_NEEDLE_TEMPLATES: list[dict[str, str]] = [
    {
        "trajectory_template": (
            "def {func_name}({param}: {type_hint}) -> {return_type}:\n"
            '    """Return the processed value."""\n'
            "    return {value}"
        ),
        "query_template": "What is the return type of {func_name}?",
        "answer_template": "{return_type}",
    },
    {
        "trajectory_template": (
            "def {func_name}({param}: {type_hint} = {value}) -> {return_type}:\n"
            '    """Process with default."""\n'
            "    return {param}"
        ),
        "query_template": "What is the default value of {param} in {func_name}?",
        "answer_template": "{value}",
    },
    {
        "trajectory_template": (
            "class {func_name}:\n"
            "    {param}: {type_hint} = {value}\n"
            "\n"
            "    def get(self) -> {return_type}:\n"
            "        return self.{param}"
        ),
        "query_template": "What type is the {param} attribute in class {func_name}?",
        "answer_template": "{type_hint}",
    },
    {
        "trajectory_template": (
            "from {value} import {func_name}\n"
            "\n"
            "def use_{param}(x: {type_hint}) -> {return_type}:\n"
            "    return {func_name}(x)"
        ),
        "query_template": "From which module is {func_name} imported?",
        "answer_template": "{value}",
    },
    {
        "trajectory_template": (
            "def {func_name}({param}: {type_hint}) -> {return_type}:\n"
            "    if not {param}:\n"
            "        raise {value}('Invalid input')\n"
            "    return {param}"
        ),
        "query_template": "What exception does {func_name} raise for invalid input?",
        "answer_template": "{value}",
    },
    {
        "trajectory_template": (
            "{func_name} = {value}\n"
            "\n"
            "def get_{param}() -> {type_hint}:\n"
            "    return {func_name}"
        ),
        "query_template": "What is the value of constant {func_name}?",
        "answer_template": "{value}",
    },
    {
        "trajectory_template": (
            "{func_name} = {type_hint}[{value}]\n"
            "\n"
            "def process({param}: {func_name}) -> {return_type}:\n"
            "    return {param}"
        ),
        "query_template": "What is the type alias {func_name} defined as?",
        "answer_template": "{type_hint}[{value}]",
    },
    {
        "trajectory_template": (
            "def {func_name}(\n"
            "    {param}: {type_hint},\n"
            "    *,\n"
            "    flag: bool = {value},\n"
            ") -> {return_type}:\n"
            "    return {param} if flag else None"
        ),
        "query_template": "What is the default value of flag in {func_name}?",
        "answer_template": "{value}",
    },
    {
        "trajectory_template": (
            "import {value}\n"
            "\n"
            "@{value}.decorator\n"
            "def {func_name}({param}: {type_hint}) -> {return_type}:\n"
            "    return {param}"
        ),
        "query_template": "What decorator is applied to {func_name}?",
        "answer_template": "{value}.decorator",
    },
    {
        "trajectory_template": (
            "class {func_name}({value}):\n"
            "    def {param}(self) -> {type_hint}:\n"
            "        return {return_type}()"
        ),
        "query_template": "What class does {func_name} inherit from?",
        "answer_template": "{value}",
    },
    {
        "trajectory_template": (
            "def {func_name}({param}: list[{type_hint}]) -> {return_type}:\n"
            "    return sorted({param}, key=lambda x: x.{value})"
        ),
        "query_template": "What attribute is used as the sort key in {func_name}?",
        "answer_template": "{value}",
    },
    {
        "trajectory_template": (
            "MAPPING: dict[{type_hint}, {return_type}] = {\n"
            "    {value!r}: {func_name},\n"
            "}\n"
            "\n"
            "def lookup({param}: {type_hint}) -> {return_type}:\n"
            "    return MAPPING[{param}]"
        ),
        "query_template": "What is the key type in MAPPING used by lookup?",
        "answer_template": "{type_hint}",
    },
]

# Slot values for deterministic template filling (indexed by template position)
_SLOT_VALUES: list[dict[str, str]] = [
    {
        "func_name": "compute_score",
        "param": "value",
        "type_hint": "float",
        "return_type": "float",
        "value": "value * 2.0",
    },
    {
        "func_name": "parse_name",
        "param": "text",
        "type_hint": "str",
        "return_type": "str",
        "value": "'unknown'",
    },
    {
        "func_name": "Config",
        "param": "max_retries",
        "type_hint": "int",
        "return_type": "int",
        "value": "3",
    },
    {
        "func_name": "TokenizerBase",
        "param": "encode",
        "type_hint": "str",
        "return_type": "bytes",
        "value": "collections",
    },
    {
        "func_name": "validate_input",
        "param": "data",
        "type_hint": "str",
        "return_type": "str",
        "value": "ValueError",
    },
    {
        "func_name": "MAX_TOKENS",
        "param": "limit",
        "type_hint": "int",
        "return_type": "int",
        "value": "4096",
    },
    {
        "func_name": "TokenList",
        "param": "tokens",
        "type_hint": "list",
        "return_type": "str",
        "value": "str",
    },
    {
        "func_name": "safe_get",
        "param": "item",
        "type_hint": "str",
        "return_type": "str",
        "value": "False",
    },
    {
        "func_name": "register_hook",
        "param": "fn",
        "type_hint": "Callable",
        "return_type": "Callable",
        "value": "functools",
    },
    {
        "func_name": "Transformer",
        "param": "forward",
        "type_hint": "Tensor",
        "return_type": "Tensor",
        "value": "Module",
    },
    {
        "func_name": "sort_records",
        "param": "records",
        "type_hint": "Record",
        "return_type": "list",
        "value": "timestamp",
    },
    {
        "func_name": "route_lookup",
        "param": "key",
        "type_hint": "str",
        "return_type": "Handler",
        "value": "str",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Persist a list of dicts to a JSONL file.

    Args:
        records: List of JSON-serializable dicts to save.
        path: File path for output. Parent directories are created if needed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dicts, one per non-empty line.
    """
    src = Path(path)
    return [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def format_for_distillation(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a trajectory to distillation records with activation/teacher split.

    Each record has:
    - activation_text: trajectory context + task description (NO answer tokens)
    - teacher_text: trajectory context + task description + answer
    - task_id: identifier for train/test splitting

    Only successful trajectories (outcome == 'success') produce records.

    Args:
        trajectory: Trajectory dict with task_id/session_id, task_description,
            steps, and outcome fields.

    Returns:
        List of distillation record dicts, or empty list if no successful outcome.
    """
    if trajectory.get("outcome") != "success":
        return []

    steps: list[dict[str, Any]] = trajectory.get("steps", [])
    task_description: str = trajectory.get("task_description", "")
    task_id: str = trajectory.get("task_id") or trajectory.get("session_id", "")

    # Build trajectory context from step descriptions (no answer tokens)
    trajectory_parts: list[str] = []
    for step in steps:
        desc = step.get("description", "")
        if desc:
            trajectory_parts.append(desc)

    trajectory_text = "\n".join(trajectory_parts)

    # Build activation_text (context only, no answer)
    activation_base = f"{trajectory_text}\n{task_description}".strip()

    records: list[dict[str, str]] = []
    for step in steps:
        if not step.get("tests_passed"):
            continue

        # Extract answer from canonical_solution or generated_code
        answer: str = step.get("canonical_solution") or step.get("generated_code", "")
        if not answer:
            continue

        records.append(
            {
                "task_id": task_id,
                "activation_text": activation_base,
                "teacher_text": f"{activation_base}\n{answer}",
            }
        )

    return records


def generate_needle_dataset(n: int = 20) -> list[dict[str, str]]:
    """Generate needle-in-haystack records for CI smoke testing.

    Records are deterministic (no randomness, no LLM). Each record contains
    a code fact embedded in a function/class context with a query and answer.

    Args:
        n: Number of records to generate. Cycles through templates if n exceeds
           the number of available templates.

    Returns:
        List of n record dicts with activation_text, teacher_text, and task_id.
    """
    records: list[dict[str, str]] = []
    n_templates = len(_NEEDLE_TEMPLATES)
    n_slots = len(_SLOT_VALUES)

    for i in range(n):
        template = _NEEDLE_TEMPLATES[i % n_templates]
        slots = _SLOT_VALUES[i % n_slots]

        try:
            trajectory = template["trajectory_template"].format(**slots)
            query = template["query_template"].format(**slots)
            answer = template["answer_template"].format(**slots)
        except KeyError:
            # If template has a slot not in _SLOT_VALUES, use a safe fallback
            trajectory = (
                f"# code fact {i}\ndef func_{i}(x: int) -> int:\n    return {i}"
            )
            query = f"What does func_{i} return?"
            answer = str(i)

        activation_text = f"{trajectory}\n\nQ: {query}"
        teacher_text = f"{trajectory}\n\nQ: {query}\nA: {answer}"

        records.append(
            {
                "task_id": f"needle_{i}",
                "activation_text": activation_text,
                "teacher_text": teacher_text,
            }
        )

    return records


def split_by_task_id(
    records: list[dict[str, Any]],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records at task-ID boundary with no task_id crossing train/test.

    Augmented records that share a task_id are always assigned to the same
    partition, preventing task-family leakage.

    Args:
        records: List of record dicts, each with a 'task_id' field.
        test_fraction: Fraction of unique task_ids to assign to test set.
            Minimum 1 task_id goes to test even if fraction rounds to 0.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_records, test_records) where task_ids never overlap.
    """
    task_ids = sorted({r["task_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    n_test = max(1, int(len(task_ids) * test_fraction))
    test_ids = set(task_ids[:n_test])
    train = [r for r in records if r["task_id"] not in test_ids]
    test = [r for r in records if r["task_id"] in test_ids]
    return train, test
