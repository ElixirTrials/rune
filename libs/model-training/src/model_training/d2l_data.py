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
from typing import Any, Literal

from model_training.d2l_diff import compress_diff

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a Python code generator. Output only code, no explanation."

__all__ = [
    "format_for_distillation",
    "generate_needle_dataset",
    "generate_trajectory_dataset",
    "augment_trajectories",
    "save_jsonl",
    "load_jsonl",
    "split_by_task_id",
    "pairs_to_chat_messages",
    "unroll_trajectory_to_pairs",
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
        n: Number of records to generate. Cycles through templates if n
            exceeds the number of available templates.

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


def generate_trajectory_dataset(
    source: str = "humaneval",
    max_tasks: int | None = None,
) -> list[dict[str, str]]:
    """Generate trajectory records from a coding task dataset.

    Each record has activation_text (prompt only, no solution) and teacher_text
    (prompt + canonical solution), making it ready for KL-divergence distillation.

    Args:
        source: Dataset source identifier. Currently only "humaneval" is supported.
        max_tasks: Maximum number of tasks to process. If None, all tasks are used.

    Returns:
        List of trajectory record dicts with task_id, activation_text, and
        teacher_text fields.

    Raises:
        ValueError: If an unsupported source is specified.
    """
    from datasets import load_dataset  # noqa: PLC0415

    if source == "humaneval":
        dataset = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    else:
        raise ValueError(
            f"Unsupported source: {source!r}. Only 'humaneval' is supported."
        )

    records: list[dict[str, str]] = []
    tasks = list(dataset) if max_tasks is None else list(dataset)[:max_tasks]
    for task in tasks:
        task_id: str = task["task_id"]
        prompt: str = task["prompt"]
        canonical_solution: str = task["canonical_solution"]

        # activation_text: the function signature + docstring (no solution)
        activation_text = prompt.rstrip()
        # teacher_text: prompt + solution (the full implementation)
        teacher_text = prompt.rstrip() + canonical_solution

        records.append(
            {
                "task_id": task_id,
                "activation_text": activation_text,
                "teacher_text": teacher_text,
            }
        )

    logger.info("Generated %d trajectory records from %r", len(records), source)
    return records


def augment_trajectories(
    trajectories: list[dict[str, Any]],
    n_variants: int = 3,
    model: str = "Qwen/Qwen3.5-9B",
    ollama_base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Produce LLM-augmented variants of trajectory records.

    For each input trajectory, generates up to n_variants augmented records
    using an Ollama LLM. Augmented records always inherit the source task_id
    to preserve split integrity when mixed with originals.

    Augmentation strategies (up to n_variants selected in order):
    1. Paraphrase the task description
    2. Reorder/drop steps in the trajectory
    3. Rename variables throughout the trajectory

    Args:
        trajectories: List of trajectory record dicts with task_id,
            activation_text, and teacher_text fields.
        n_variants: Number of augmented variants to produce per trajectory.
            Maximum 3 (one per augmentation strategy).
        model: Ollama model identifier to use for augmentation.
        ollama_base_url: Ollama base URL. Defaults to "http://localhost:11434".

    Returns:
        List of augmented trajectory dicts. Each record has the same task_id
        as its source trajectory, with LLM-generated activation_text and
        teacher_text.
    """
    import asyncio  # noqa: PLC0415

    from inference.ollama_provider import OllamaProvider  # noqa: PLC0415

    base_url = ollama_base_url or "http://localhost:11434"
    provider = OllamaProvider(base_url=base_url)

    augmentation_prompts = [
        (
            "Paraphrase the following coding task description and trajectory, "
            "keeping the same logic but using different wording:\n\n{text}"
        ),
        (
            "Rewrite the following coding trajectory by reordering and slightly "
            "dropping some intermediate steps, while preserving the overall "
            "outcome:\n\n{text}"
        ),
        (
            "Rewrite the following code trajectory by renaming variables and "
            "function parameters to different names, keeping the same logic:"
            "\n\n{text}"
        ),
    ]

    async def _augment_one(
        trajectory: dict[str, Any],
        prov: Any,
        n: int,
        mdl: str,
    ) -> list[dict[str, Any]]:
        task_id: str = trajectory["task_id"]
        activation_text: str = trajectory.get("activation_text", "")
        teacher_text: str = trajectory.get("teacher_text", "")

        results: list[dict[str, Any]] = []
        strategies = augmentation_prompts[:n]
        for strategy_prompt in strategies:
            augment_prompt = strategy_prompt.format(text=teacher_text)
            result = await prov.generate(augment_prompt, mdl)
            augmented_teacher = result.text

            # Build augmented activation by stripping the solution portion
            # Use teacher text length ratio to approximate activation
            activation_prompt = strategy_prompt.format(text=activation_text)
            act_result = await prov.generate(activation_prompt, mdl)
            augmented_activation = act_result.text

            results.append(
                {
                    "task_id": task_id,  # CRITICAL: inherit source task_id
                    "activation_text": augmented_activation,
                    "teacher_text": augmented_teacher,
                }
            )
        return results

    async def _augment_all(
        trajs: list[dict[str, Any]],
        n: int,
        mdl: str,
    ) -> list[dict[str, Any]]:
        all_tasks = [_augment_one(t, provider, n, mdl) for t in trajs]
        nested = await asyncio.gather(*all_tasks)
        return [record for group in nested for record in group]

    augmented = asyncio.run(_augment_all(trajectories, n_variants, model))
    logger.info(
        "Augmented %d trajectories into %d records (n_variants=%d)",
        len(trajectories),
        len(augmented),
        n_variants,
    )
    return augmented


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


# ---------------------------------------------------------------------------
# SFT chat-message converter (consumed by trainer.py for mined-pair training)
# ---------------------------------------------------------------------------


def _extract_revision(activation_text: str, teacher_text: str) -> str:
    r"""Return the assistant-side text from a mined pair record.

    A pair's ``teacher_text`` is always ``activation_text`` plus a trailing
    section produced by :func:`normalize_mined_pairs` — either
    ``"\n\n## Revision\n..."`` for review cycles or
    ``"\n\n## Implementation\n..."`` for the initial commit pair. We return
    that suffix verbatim so the model learns to output the section header
    (which mirrors what reviewers see in diff tools) alongside the code.

    When ``teacher_text`` is identical to ``activation_text`` (degenerate
    record with no delta) an empty string is returned so the caller skips
    the pair. When ``teacher_text`` does not start with ``activation_text``
    (corrupt record), the full ``teacher_text`` is returned as a best-effort
    fallback rather than dropping the datum silently.
    """
    if teacher_text.startswith(activation_text):
        return teacher_text[len(activation_text) :].lstrip("\n")
    return teacher_text


def _extract_pre_revision(activation_text: str) -> str:
    """Extract the ``## Current Code`` body from an activation_text string.

    The activation_text produced by :func:`normalize_mined_pairs` looks like::

        ## Task
        <description>

        ## Current Code
        <diff>

        ## Review Feedback
        <feedback>

    The ``## Current Code`` section is absent for initial-commit pairs whose
    activation_text only contains ``## Task``.

    Args:
        activation_text: The activation-side prompt for one mined pair.

    Returns:
        The code body under ``## Current Code`` up to the next ``## ``
        heading or end-of-string.  Returns ``""`` when the section is absent.
    """
    marker = "## Current Code\n"
    start = activation_text.find(marker)
    if start == -1:
        return ""
    body_start = start + len(marker)
    # Find the next "## " heading that follows the marker.
    next_heading = activation_text.find("\n## ", body_start)
    if next_heading == -1:
        return activation_text[body_start:].rstrip("\n")
    return activation_text[body_start:next_heading].rstrip("\n")


_CURRENT_CODE_ELIDED = (
    "(omitted — see the previous assistant turn for the up-to-date code)"
)


def _strip_current_code_section(activation_text: str) -> str:
    """Replace the ``## Current Code`` body with an elision sentinel.

    Only used for non-first turns of a multi-turn conversation: the
    previous assistant turn already contains the latest revision, so
    repeating the full file under ``## Current Code`` is pure padding.
    The model still sees the surrounding ``## Task`` / ``## Review
    Feedback`` framing — only the redundant code body is replaced — so
    review feedback stays grounded.

    Returns the activation text unchanged when the marker is absent
    (e.g. an initial-commit pair that slipped into a non-first slot).

    Args:
        activation_text: The original activation text for one mined pair.

    Returns:
        The activation text with the ``## Current Code`` body replaced
        by :data:`_CURRENT_CODE_ELIDED`.
    """
    marker = "## Current Code\n"
    start = activation_text.find(marker)
    if start == -1:
        return activation_text
    body_start = start + len(marker)
    next_heading = activation_text.find("\n## ", body_start)
    if next_heading == -1:
        return activation_text[:body_start] + _CURRENT_CODE_ELIDED + "\n"
    return (
        activation_text[:body_start]
        + _CURRENT_CODE_ELIDED
        + activation_text[next_heading:]
    )


def _extract_post_revision(activation_text: str, teacher_text: str) -> str:
    """Extract the code body from the assistant-side section of a pair.

    Calls :func:`_extract_revision` then strips the leading section header
    line (``## Revision`` or ``## Implementation``) so that ``post_code``
    contains only the code body.

    Args:
        activation_text: The activation-side prompt for one mined pair.
        teacher_text: The full teacher-side text (activation + revision).

    Returns:
        The code body after the section header, or ``""`` when
        :func:`_extract_revision` returns empty (degenerate pair).
    """
    revision = _extract_revision(activation_text, teacher_text)
    if not revision:
        return ""
    # Strip the first line if it looks like a section header.
    first_newline = revision.find("\n")
    if first_newline != -1:
        first_line = revision[:first_newline]
        if first_line.startswith("## "):
            return revision[first_newline:].lstrip("\n")
    return revision


def _pairs_to_single_turn(
    pairs: list[dict[str, Any]], system_prompt: str
) -> tuple[list[list[dict[str, str]]], list[dict[str, list[str]]]]:
    """single_turn helper: one [system, user, assistant] per pair.

    Returns:
        Tuple of (conversations, pre_post_records) where each element of
        pre_post_records carries per-turn ``pre_codes`` / ``post_codes`` lists
        (length 1 for single-turn) so the downstream collator can align
        hunks per assistant turn rather than against a joined blob.
    """
    conversations: list[list[dict[str, str]]] = []
    pre_post_records: list[dict[str, list[str]]] = []
    for pair in pairs:
        user = pair.get("activation_text", "")
        teacher = pair.get("teacher_text", "")
        assistant = _extract_revision(user, teacher)
        if not assistant:
            continue
        conversations.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
        pre_post_records.append(
            {
                "pre_codes": [_extract_pre_revision(user)],
                "post_codes": [_extract_post_revision(user, teacher)],
            }
        )
    return conversations, pre_post_records


def _group_pairs_by_task(
    pairs: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Group pairs by source_task_id, preserving first-appearance order.

    Falls back to ``task_id`` when ``metadata.source_task_id`` is missing.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for pair in pairs:
        meta = pair.get("metadata") or {}
        key = meta.get("source_task_id") or pair.get("task_id", "")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(pair)
    return groups, order


def pairs_to_chat_messages(
    pairs: list[dict[str, Any]],
    *,
    mode: Literal["multi_turn", "single_turn"] = "multi_turn",
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[list[list[dict[str, str]]], list[dict[str, list[str]]]]:
    r"""Convert mined pair records into SFT chat conversations.

    ``multi_turn`` (preferred when pairs share a ``source_task_id``): emits
    one conversation per task grouping — ``[system, user_1, assistant_1,
    user_2, assistant_2, ...]`` — where each (user, assistant) pair is one
    review→revision cycle. This preserves the attempt-error-correction
    structure so the adapter can encode the trajectory rather than only the
    final code.

    ``single_turn``: emits one conversation per pair (no clustering). Useful
    when ``metadata.source_task_id`` is missing or the caller prefers flat
    examples. Each conversation is ``[system, user, assistant]``.

    Pairs are grouped in the order they appear in the input; within a group,
    pairs are sorted by ``metadata.step_index`` to preserve chronological
    order of review cycles. Empty input returns ``([], [])``.

    Args:
        pairs: List of pair records as emitted by ``normalize_mined_pairs``.
            Each record must have ``activation_text`` and ``teacher_text``.
            ``metadata.source_task_id`` and ``metadata.step_index`` are used
            for grouping/ordering when present; fall back to ``task_id``.
        mode: ``"multi_turn"`` clusters pairs by task_id; ``"single_turn"``
            emits one conversation per pair.
        system_prompt: System message for every conversation.

    Returns:
        A tuple ``(conversations, pre_post_records)``.

        ``conversations`` is a list of chat-message lists suitable for
        ``datasets.Dataset.from_list([{"messages": m} for m in convs])`` and
        TRL's ``SFTTrainer`` with ``assistant_only_loss=True``.

        ``pre_post_records`` is a list of ``{"pre_codes": list[str],
        "post_codes": list[str]}`` dicts aligned 1:1 with ``conversations``.
        Each list has one entry per assistant turn — preserved as a list (not
        joined) so the diff-aware collator can align hunks against each
        turn's own pre/post pair instead of a concatenated blob whose token
        layout no longer matches the chat-templated sequence.

        Invalid or missing fields are handled gracefully: missing
        ``activation_text`` / ``teacher_text`` produce empty strings, and
        missing ``metadata`` keys fall back to empty string / 0 defaults.

    Raises:
        None: No exceptions are raised — invalid input fields fall back
            to empty-string / zero defaults.
    """
    if not pairs:
        return [], []

    if mode == "single_turn":
        return _pairs_to_single_turn(pairs, system_prompt)

    # multi_turn: cluster by source_task_id (or task_id), sort by step_index.
    groups, group_order = _group_pairs_by_task(pairs)
    conversations: list[list[dict[str, str]]] = []
    pre_post_records: list[dict[str, list[str]]] = []
    for key in group_order:
        group = sorted(
            groups[key],
            key=lambda p: (p.get("metadata") or {}).get("step_index", 0),
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        turn_pre_codes: list[str] = []
        turn_post_codes: list[str] = []
        accepted_turns = 0
        for pair in group:
            user = pair.get("activation_text", "")
            teacher = pair.get("teacher_text", "")
            assistant = _extract_revision(user, teacher)
            if not assistant:
                continue
            # In non-first turns the previous assistant message already
            # carries the up-to-date code, so re-pasting the full
            # ``## Current Code`` body in every subsequent user turn
            # quadruples the conversation length on long-tail PRs and is
            # the dominant driver of the ``denom=0`` zero-grad batches
            # under ``keep_start`` truncation (RCA-5 H2). Keep ``pre_codes``
            # populated from the original body so the diff collator still
            # sees the correct hunk boundaries.
            user_for_chat = (
                _strip_current_code_section(user) if accepted_turns > 0 else user
            )
            messages.append({"role": "user", "content": user_for_chat})
            messages.append({"role": "assistant", "content": assistant})
            turn_pre_codes.append(_extract_pre_revision(user))
            turn_post_codes.append(_extract_post_revision(user, teacher))
            accepted_turns += 1
        # A valid SFT conversation needs at least one user/assistant turn.
        if len(messages) >= 3:
            conversations.append(messages)
            pre_post_records.append(
                {
                    "pre_codes": turn_pre_codes,
                    "post_codes": turn_post_codes,
                }
            )
    return conversations, pre_post_records


def unroll_trajectory_to_pairs(traj: "Trajectory") -> list[dict[str, Any]]:
    """Unroll one trajectory into per-episode SFT pairs.

    Each pair has:
        prompt:   prior_diff + formatted feedback (task description, review
                  comment, or test/CI/lint summary).
        response: action_diff for that round (the model's target).
        task_id:  the PR's task_id, repeated across all rounds.
        round:    0-indexed round number.
    """
    pairs: list[dict[str, Any]] = []
    for ep in traj.episodes:
        feedback_text = _format_feedback(ep.feedback)
        if ep.prior_diff:
            prompt = (
                "Prior diff:\n"
                f"{ep.prior_diff}\n\n"
                "Feedback:\n"
                f"{feedback_text}"
            )
        else:
            prompt = feedback_text
        pairs.append(
            {
                "task_id": traj.task_id,
                "round": ep.round,
                "prompt": prompt,
                "response": ep.action_diff,
                "feedback_kind": ep.feedback.kind.value,
            }
        )
    return pairs


def _format_feedback(fb: "Feedback") -> str:
    """One human-readable string per Feedback, prioritising the summary."""
    if fb.summary:
        head = fb.summary
    else:
        head = fb.body
    parts = [head]
    if fb.anchor and (fb.anchor.file or fb.anchor.test):
        loc = fb.anchor.test or (
            f"{fb.anchor.file}:{fb.anchor.line}" if fb.anchor.line
            else fb.anchor.file
        )
        parts.append(f"[at {loc}]")
    return " ".join(p for p in parts if p)
