"""Trajectory recording and formatting for coding session distillation.

Provides functions to persist, load, and convert coding session trajectories
into SFT-compatible chat format for LoRA fine-tuning pipelines.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SYSTEM_PROMPT = "You are a Python code generator. Output only code, no explanation."


def _get_trajectory_dir() -> Path:
    """Return the trajectory storage directory, respecting RUNE_TRAJECTORY_DIR env var.

    Reads env var inside function body (not module level) so that monkeypatch.setenv()
    works correctly in tests.
    """
    env_dir = os.environ.get("RUNE_TRAJECTORY_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".rune" / "trajectories"


def record_trajectory(
    session_id: str,
    steps: list[dict[str, Any]],
    outcome: Optional[str] = None,
    *,
    task_description: str = "",
    task_type: str = "",
    adapter_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Persist a coding session trajectory to disk for future distillation.

    Args:
        session_id: Unique identifier for the coding session.
        steps: List of step dicts, each containing attempt results.
        outcome: Final session result ('success', 'exhausted', or None).
        task_description: Natural language description of the coding task.
        task_type: Category of task (e.g. 'function', 'class', 'refactor').
        adapter_ids: LoRA adapter IDs used during the session.

    Returns:
        A dict with 'session_id' and 'file_path' keys.
    """
    trajectory_dir = _get_trajectory_dir()
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    file_path = trajectory_dir / f"{session_id}.json"
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    trajectory: dict[str, Any] = {
        "session_id": session_id,
        "task_description": task_description,
        "task_type": task_type,
        "adapter_ids": adapter_ids if adapter_ids is not None else [],
        "outcome": outcome,
        "timestamp": timestamp,
        "steps": steps,
    }

    file_path.write_text(json.dumps(trajectory, indent=2))

    return {"session_id": session_id, "file_path": str(file_path)}


def load_trajectory(trajectory_id: str) -> dict[str, Any]:
    """Load a stored trajectory by session ID.

    Args:
        trajectory_id: The session ID used as the filename (without .json).

    Returns:
        A dict containing the full trajectory data including steps and metadata.

    Raises:
        FileNotFoundError: If no trajectory file exists for the given ID.
    """
    trajectory_dir = _get_trajectory_dir()
    file_path = trajectory_dir / f"{trajectory_id}.json"
    # Let FileNotFoundError propagate naturally if file does not exist
    return json.loads(file_path.read_text())  # type: ignore[no-any-return]


def format_for_sft(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a trajectory into SFT-compatible chat format.

    Only successful trajectories (outcome == 'success') produce output.
    Extracts the final step where tests_passed is True as the assistant message.

    Args:
        trajectory: A trajectory dict as returned by load_trajectory.

    Returns:
        A list of 3 message dicts ([system, user, assistant]) for successful
        trajectories, or an empty list if the trajectory did not succeed.
    """
    if trajectory.get("outcome") != "success":
        return []

    steps: list[dict[str, Any]] = trajectory.get("steps", [])
    successful_step = next(
        (s for s in reversed(steps) if s.get("tests_passed")),
        None,
    )

    if successful_step is None:
        return []

    task_description: str = trajectory.get("task_description", "")
    generated_code: str = successful_step.get("generated_code", "")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_description},
        {"role": "assistant", "content": generated_code},
    ]


def compute_assistant_masks(
    tokenizer: Any, messages: list[dict[str, str]]
) -> dict[str, list[int]]:
    r"""Tokenize a chat conversation and emit a 0/1 mask over assistant tokens.

    Used when the tokenizer's chat template lacks ``{% generation %}`` markers
    (Qwen3.5-9B as of TRL 1.3.0), so TRL's ``return_assistant_tokens_mask=True``
    path raises ``ValueError`` from ``get_training_chat_template``. We bypass
    that path by pre-tokenizing here and providing ``input_ids`` +
    ``assistant_masks`` directly to SFTTrainer; ``_prepare_dataset`` short-
    circuits all preprocessing when ``input_ids`` is already present
    (sft_trainer.py:1067) and ``DataCollatorForLanguageModeling.torch_call``
    consumes ``assistant_masks`` from the batch (sft_trainer.py:179-180).

    Approach (Qwen-family marker scan): tokenize once and walk the token
    stream looking for ``<|im_start|>{assistant_role_token}`` boundaries
    that close at ``<|im_end|>``. Marks the whole assistant turn —
    role+wrapper+content+thinking-blocks, since training the model to
    reproduce its own thinking is desirable. This avoids the prefix-of-
    full invariant failure that bites stateful templates: Qwen3.5 injects
    ``<think>`` markers only on the *trailing* assistant turn, so
    incremental prefix tokenization breaks. Marker scan reads only the
    final, ground-truth tokenization and is therefore stable.

    Args:
        tokenizer: A HuggingFace tokenizer that uses ``<|im_start|>`` /
            ``<|im_end|>`` role markers (Qwen and ChatML-family).
        messages: A list of ``{"role", "content"}`` dicts.

    Returns:
        ``{"input_ids": [...], "assistant_masks": [...]}`` with both lists
        the same length. ``assistant_masks`` has at least one ``1`` when
        any message has ``role == "assistant"``.

    Raises:
        ValueError: If the tokenizer doesn't expose ``<|im_start|>`` /
            ``<|im_end|>`` token IDs — the marker-scan algorithm depends
            on them.
    """
    # ``return_dict=False`` is critical: with ``return_dict=True`` (the
    # default in newer transformers) the call returns a BatchEncoding whose
    # ``list(...)`` yields the *keys* (``["input_ids", "attention_mask"]``),
    # not the token IDs — which silently corrupts the dataset.
    full_ids: list[int] = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
    )
    mask: list[int] = [0] * len(full_ids)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # ``convert_tokens_to_ids`` returns the unk-id for unknown markers on
    # most tokenizers; treat any non-distinct or None result as "missing".
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if (
        im_start_id is None
        or im_end_id is None
        or im_start_id == unk_id
        or im_end_id == unk_id
    ):
        raise ValueError(
            "compute_assistant_masks: tokenizer lacks <|im_start|> / "
            "<|im_end|> markers; the marker-scan path requires a Qwen- or "
            "ChatML-family tokenizer."
        )

    # The role token sequence that follows <|im_start|>. For Qwen3.5,
    # tokenizer.encode("assistant") → [74455] (single-token role). We
    # encode at runtime so non-Qwen ChatML tokenizers with multi-token
    # role names still work.
    assistant_role_ids: list[int] = list(
        tokenizer.encode("assistant", add_special_tokens=False)
    )
    if not assistant_role_ids:
        return {"input_ids": full_ids, "assistant_masks": mask}

    n = len(full_ids)
    role_len = len(assistant_role_ids)
    i = 0
    while i < n:
        if full_ids[i] != im_start_id:
            i += 1
            continue
        # Match <|im_start|> + assistant role tokens.
        head_end = i + 1 + role_len
        if head_end > n or full_ids[i + 1 : head_end] != assistant_role_ids:
            i += 1
            continue
        # Walk to the closing <|im_end|>; if missing (truncated tail),
        # extend to end-of-sequence.
        j = head_end
        while j < n and full_ids[j] != im_end_id:
            j += 1
        end = j if j < n else n - 1
        for k in range(i, end + 1):
            mask[k] = 1
        i = end + 1

    return {"input_ids": full_ids, "assistant_masks": mask}
