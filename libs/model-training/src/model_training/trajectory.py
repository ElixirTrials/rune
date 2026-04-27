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

    Approach: render prefixes ``messages[:i+1]`` with ``add_generation_prompt
    =False`` and diff token counts to find each turn's boundary. The mask is
    1 on the entire assistant turn (including ``<|im_start|>assistant\\n``
    and ``<|im_end|>`` wrappers) — TRL's reference path with
    ``{% generation %}`` markers wraps only the content, so this is a small
    over-mask of ~5 trivially-predictable wrapper tokens per assistant turn.

    Args:
        tokenizer: A HuggingFace tokenizer with a chat_template.
        messages: A list of ``{"role", "content"}`` dicts.

    Returns:
        ``{"input_ids": [...], "assistant_masks": [...]}`` with both lists
        the same length. ``assistant_masks`` has at least one ``1`` when
        any message has ``role == "assistant"``.

    Raises:
        ValueError: If the prefix-of-full token-id invariant fails (would
            indicate template-side state that we cannot reliably mask).
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

    # Some chat templates (e.g. Qwen3.5) refuse to render conversations
    # without at least one user message. We can't tokenize a system-only
    # prefix in that case; skip until the first user message and rely on
    # the default-zero mask for those leading system tokens (correct, since
    # role=system → mask=0 by definition).
    first_user_idx = next(
        (i for i, m in enumerate(messages) if m.get("role") == "user"), None
    )
    if first_user_idx is None:
        return {"input_ids": full_ids, "assistant_masks": mask}

    cursor = 0
    for i in range(first_user_idx, len(messages)):
        prefix_ids: list[int] = list(
            tokenizer.apply_chat_template(
                messages[: i + 1],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
        )
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "compute_assistant_masks: prefix-of-full invariant violated "
                f"at message {i}; cannot reliably compute assistant_masks."
            )
        if messages[i].get("role") == "assistant":
            for j in range(cursor, len(prefix_ids)):
                mask[j] = 1
        cursor = len(prefix_ids)
    return {"input_ids": full_ids, "assistant_masks": mask}
