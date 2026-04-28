"""Green-phase behavior tests for model_training.trajectory module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from model_training.trajectory import (
    compute_assistant_masks,
    format_for_sft,
    load_trajectory,
    record_trajectory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_STEPS: list[dict[str, Any]] = [
    {
        "attempt": 0,
        "generated_code": "def add(a, b):\n    return a + b",
        "stdout": "",
        "stderr": "AssertionError",
        "exit_code": 1,
        "tests_passed": False,
    },
    {
        "attempt": 1,
        "generated_code": "def add(a, b):\n    return a + b\n\nassert add(1, 2) == 3",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": True,
    },
]


# ---------------------------------------------------------------------------
# record_trajectory tests
# ---------------------------------------------------------------------------


def test_record_trajectory_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_trajectory writes a JSON file to the configured trajectory directory."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    record_trajectory("sess-001", SAMPLE_STEPS, outcome="success")
    expected_file = tmp_path / "sess-001.json"
    assert expected_file.exists(), "Expected JSON file was not created"
    data = json.loads(expected_file.read_text())
    assert data["session_id"] == "sess-001"
    assert data["outcome"] == "success"
    assert "timestamp" in data
    assert len(data["steps"]) == 2


def test_record_trajectory_returns_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_trajectory returns dict with session_id and file_path keys."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    result = record_trajectory("sess-002", [], outcome="exhausted")
    assert result["session_id"] == "sess-002"
    assert "file_path" in result
    assert Path(result["file_path"]).exists()


def test_record_trajectory_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_trajectory creates the directory if it does not exist."""
    subdir = tmp_path / "nested" / "trajectories"
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(subdir))
    assert not subdir.exists(), "Directory should not exist yet"
    record_trajectory("sess-003", [], outcome="success")
    assert subdir.exists(), "Directory should have been created"
    assert (subdir / "sess-003.json").exists()


def test_record_trajectory_includes_task_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_trajectory includes task_description, task_type, and adapter_ids."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    record_trajectory(
        "sess-004",
        SAMPLE_STEPS,
        outcome="success",
        task_description="Write an add function",
        task_type="function",
        adapter_ids=["adapter-001", "adapter-002"],
    )
    data = json.loads((tmp_path / "sess-004.json").read_text())
    assert data["task_description"] == "Write an add function"
    assert data["task_type"] == "function"
    assert data["adapter_ids"] == ["adapter-001", "adapter-002"]


# ---------------------------------------------------------------------------
# load_trajectory tests
# ---------------------------------------------------------------------------


def test_load_trajectory_reads_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_trajectory reads back a previously recorded trajectory (round-trip)."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    record_trajectory(
        "sess-005",
        SAMPLE_STEPS,
        outcome="success",
        task_description="Round-trip test",
        task_type="function",
        adapter_ids=["adapter-abc"],
    )
    loaded = load_trajectory("sess-005")
    assert loaded["session_id"] == "sess-005"
    assert loaded["outcome"] == "success"
    assert loaded["task_description"] == "Round-trip test"
    assert len(loaded["steps"]) == 2


def test_load_trajectory_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_trajectory raises FileNotFoundError for non-existent session_id."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_trajectory("does-not-exist")


# ---------------------------------------------------------------------------
# format_for_sft tests
# ---------------------------------------------------------------------------


def _make_trajectory(
    outcome: str,
    steps: list[dict[str, Any]],
    task_description: str = "Write an add function",
) -> dict[str, Any]:
    """Helper to create a trajectory dict for format_for_sft tests."""
    return {
        "session_id": "sess-test",
        "outcome": outcome,
        "task_description": task_description,
        "steps": steps,
    }


def test_format_for_sft_success() -> None:
    """format_for_sft returns [system, user, assistant] for a successful trajectory."""
    trajectory = _make_trajectory(
        "success", SAMPLE_STEPS, task_description="Write an add function"
    )
    messages = format_for_sft(trajectory)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Write an add function"
    assert messages[2]["role"] == "assistant"
    # assistant content should be from the last step with tests_passed=True
    assert "tests_passed" not in messages[2]["content"]  # it's code, not a dict
    assert "def add" in messages[2]["content"]


def test_format_for_sft_exhausted_returns_empty() -> None:
    """format_for_sft returns empty list for trajectories with outcome != 'success'."""
    trajectory = _make_trajectory("exhausted", SAMPLE_STEPS)
    result = format_for_sft(trajectory)
    assert result == []


def test_format_for_sft_no_successful_step_returns_empty() -> None:
    """format_for_sft returns [] when no step has tests_passed=True."""
    failing_steps: list[dict[str, Any]] = [
        {"attempt": 0, "generated_code": "broken", "tests_passed": False},
        {"attempt": 1, "generated_code": "still broken", "tests_passed": False},
    ]
    trajectory = _make_trajectory("success", failing_steps)
    result = format_for_sft(trajectory)
    assert result == []


# ---------------------------------------------------------------------------
# compute_assistant_masks
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Stand-in ChatML-family tokenizer with <|im_start|>/<|im_end|> markers.

    The synthetic render mirrors the real Qwen3.5 layout so the marker-scan
    in compute_assistant_masks operates on realistic token sequences:

        <|im_start|> {role_id} \\n {content_tokens...} <|im_end|> \\n
    """

    IM_START_ID = 1
    IM_END_ID = 2
    NEWLINE_ID = 3
    UNK_ID = 99
    _ROLE_IDS: dict[str, int] = {"system": 10, "user": 11, "assistant": 12}

    unk_token_id: int = 99

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|im_start|>": self.IM_START_ID,
            "<|im_end|>": self.IM_END_ID,
        }.get(token, self.UNK_ID)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text in self._ROLE_IDS:
            return [self._ROLE_IDS[text]]
        return [ord(c) + 100 for c in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        for msg in messages:
            ids.append(self.IM_START_ID)
            ids.append(self._ROLE_IDS.get(msg["role"], self.UNK_ID))
            ids.append(self.NEWLINE_ID)
            ids.extend(ord(c) + 100 for c in msg["content"])
            ids.append(self.IM_END_ID)
            ids.append(self.NEWLINE_ID)
        return ids


def test_compute_assistant_masks_marks_only_assistant_turn_span() -> None:
    """Assistant turn (im_start..im_end inclusive) → 1; rest → 0."""
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = compute_assistant_masks(tok, messages)

    assert "input_ids" in result
    assert "assistant_masks" in result
    assert len(result["input_ids"]) == len(result["assistant_masks"])
    # System turn = 1+1+1+3+1+1 = 8 tokens, user turn = 1+1+1+5+1+1 = 10.
    # Assistant turn = 1+1+1+2+1+1 = 7 tokens, six of which (im_start through
    # im_end inclusive) are marked; trailing newline is not.
    assert sum(result["assistant_masks"]) == 6
    assert all(m == 0 for m in result["assistant_masks"][: 8 + 10])  # sys + user
    assert all(m == 1 for m in result["assistant_masks"][18:24])  # assistant span


def test_compute_assistant_masks_no_assistant_yields_all_zero() -> None:
    """Conversation with no assistant role → mask is all zeros."""
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ask"},
    ]
    result = compute_assistant_masks(tok, messages)
    assert 1 not in result["assistant_masks"]
    assert sum(result["assistant_masks"]) == 0


def test_compute_assistant_masks_multi_turn_marks_each_assistant_turn() -> None:
    """Both assistant turns in a multi-turn conversation are masked."""
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    result = compute_assistant_masks(tok, messages)
    # Each assistant turn spans 1+1+1+2+1+1=7 tokens, six marked → 12 ones.
    assert sum(result["assistant_masks"]) == 12


def test_compute_assistant_masks_truncated_tail_marks_to_eos() -> None:
    """Missing trailing <|im_end|> → mask extends to end-of-sequence."""

    class TruncatedTokenizer(_FakeTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            return_dict: bool = True,
        ) -> list[int]:
            ids = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                return_dict=return_dict,
            )
            # Drop the trailing <|im_end|>\n of the last assistant turn.
            if messages and messages[-1].get("role") == "assistant" and len(ids) >= 2:
                ids = ids[:-2]
            return ids

    tok = TruncatedTokenizer()
    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    result = compute_assistant_masks(tok, messages)
    # All assistant tokens through end-of-sequence (no im_end terminator).
    assert result["assistant_masks"][-1] == 1


def test_compute_assistant_masks_raises_when_markers_missing() -> None:
    """Tokenizers without <|im_start|>/<|im_end|> markers must raise."""

    class NoMarkerTokenizer:
        unk_token_id = 99

        def convert_tokens_to_ids(self, token: str) -> int:
            return 99  # always unk

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [12]

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            return_dict: bool = True,
        ) -> list[int]:
            return [1, 2, 3]

    with pytest.raises(ValueError, match="lacks <\\|im_start\\|>"):
        compute_assistant_masks(NoMarkerTokenizer(), [{"role": "user", "content": "x"}])


def test_compute_assistant_masks_returns_int_lists_not_batchencoding_keys() -> None:
    """Regression: verify input_ids is list[int], not BatchEncoding keys.

    HuggingFace tokenizers' ``apply_chat_template(tokenize=True)`` defaults
    to ``return_dict=True``, which returns a ``BatchEncoding`` whose
    ``list(...)`` yields the *keys* (``["input_ids", "attention_mask"]``)
    rather than the token IDs. Without ``return_dict=False``, our helper
    silently produced a string-typed dataset that crashed TRL's collator
    with ``ValueError: too many dimensions 'str'``.
    """

    class BatchEncodingLikeTokenizer(_FakeTokenizer):
        """Mimics newer transformers' default return_dict=True behavior."""

        def apply_chat_template(  # type: ignore[override]
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            return_dict: bool = True,
        ) -> Any:
            ids = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
            )
            if return_dict:
                # Mimic BatchEncoding: list(obj) returns the dict keys.
                class _BE:
                    def __init__(self, ids: list[int]) -> None:
                        self._ids = ids

                    def __iter__(self) -> Any:
                        return iter(["input_ids", "attention_mask"])

                    def __len__(self) -> int:
                        return 2

                return _BE(ids)
            return ids

    result = compute_assistant_masks(
        BatchEncodingLikeTokenizer(),
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
    )
    assert all(isinstance(t, int) for t in result["input_ids"]), (
        f"expected list[int], got {[type(t).__name__ for t in result['input_ids']]}"
    )
