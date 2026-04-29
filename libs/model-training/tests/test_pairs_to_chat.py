"""Tests for ``pairs_to_chat_messages`` — mined-pair → SFT chat converter.

Covers:
- Empty input → empty output.
- Single pair, single_turn: one [system, user, assistant] conversation.
- Multiple pairs sharing source_task_id, multi_turn: one clustered
  conversation ordered by step_index.
- Multiple pairs with different source_task_ids, multi_turn: one
  conversation per task.
- Missing metadata.source_task_id falls back to task_id.
- Corrupt record where teacher_text does not start with activation_text
  returns the full teacher_text.
- pre_post_records aligned 1:1 with conversations.
- Multi-turn pre/post concatenation.
- Skipped pairs do not appear in pre_post_records.
- Initial commit pairs have empty pre_code.
"""

from __future__ import annotations

from model_training.d2l_data import pairs_to_chat_messages


def _pair(
    *,
    task_id: str = "pr_repo_1",
    source_task_id: str | None = "pr_repo_1",
    step_index: int = 0,
    activation: str = "## Task\nWrite add()",
    teacher: str | None = None,
) -> dict:
    if teacher is None:
        teacher = f"{activation}\n\n## Implementation\ndef add(a,b): return a+b"
    meta: dict = {"step_index": step_index, "outcome": "merged"}
    if source_task_id is not None:
        meta["source_task_id"] = source_task_id
    return {
        "task_id": task_id,
        "activation_text": activation,
        "teacher_text": teacher,
        "metadata": meta,
    }


def test_empty_input_returns_empty() -> None:
    convs, pre_post = pairs_to_chat_messages([])
    assert (convs, pre_post) == ([], [])
    convs2, pre_post2 = pairs_to_chat_messages([], mode="single_turn")
    assert (convs2, pre_post2) == ([], [])


def test_single_turn_one_pair() -> None:
    convs, pre_post = pairs_to_chat_messages([_pair()], mode="single_turn")
    assert len(convs) == 1
    assert len(pre_post) == len(convs)
    msgs = convs[0]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"].startswith("## Task")
    # Assistant message keeps the "## Implementation" section heading.
    assert "## Implementation" in msgs[2]["content"]
    assert "def add" in msgs[2]["content"]


def test_multi_turn_clusters_by_source_task_id() -> None:
    pairs = [
        _pair(step_index=0, activation="## Task\nWrite add()"),
        _pair(
            step_index=1,
            activation="## Task\nWrite add()\n\n## Review Feedback\nHandle floats",
            teacher=(
                "## Task\nWrite add()\n\n## Review Feedback\nHandle floats"
                "\n\n## Revision\ndef add(a,b): return float(a)+float(b)"
            ),
        ),
    ]
    convs, pre_post = pairs_to_chat_messages(pairs, mode="multi_turn")
    assert len(convs) == 1
    assert len(pre_post) == len(convs)
    msgs = convs[0]
    # system, u1, a1, u2, a2
    assert [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "## Review Feedback" in msgs[3]["content"]
    assert "float(a)" in msgs[4]["content"]


def test_multi_turn_orders_by_step_index_regardless_of_input_order() -> None:
    # Reverse chronological input; converter must sort by step_index.
    reversed_pairs = [
        _pair(
            step_index=2,
            activation="## Task\nX\n\n## Review Feedback\nfeedback2",
            teacher="## Task\nX\n\n## Review Feedback\nfeedback2\n\n## Revision\nfinal",
        ),
        _pair(step_index=0, activation="## Task\nX"),
        _pair(
            step_index=1,
            activation="## Task\nX\n\n## Review Feedback\nfeedback1",
            teacher="## Task\nX\n\n## Review Feedback\nfeedback1\n\n## Revision\nmid",
        ),
    ]
    convs, pre_post = pairs_to_chat_messages(reversed_pairs, mode="multi_turn")
    assert len(convs) == 1
    assert len(pre_post) == len(convs)
    assistants = [m["content"] for m in convs[0] if m["role"] == "assistant"]
    assert len(assistants) == 3
    assert "## Implementation" in assistants[0]  # step 0
    assert "mid" in assistants[1]  # step 1
    assert "final" in assistants[2]  # step 2


def test_multi_turn_distinct_task_ids_produce_separate_conversations() -> None:
    pairs = [
        _pair(task_id="pr_repo_1", source_task_id="pr_repo_1"),
        _pair(task_id="pr_repo_2", source_task_id="pr_repo_2"),
    ]
    convs, pre_post = pairs_to_chat_messages(pairs, mode="multi_turn")
    assert len(convs) == 2
    assert len(pre_post) == len(convs)


def test_multi_turn_falls_back_to_task_id_when_metadata_missing() -> None:
    # metadata exists but has no source_task_id: the converter must still
    # group by task_id so a task-level conversation is produced.
    p1 = _pair(task_id="pr_only_1", source_task_id=None, step_index=0)
    p2 = _pair(
        task_id="pr_only_1",
        source_task_id=None,
        step_index=1,
        activation="## Task\nx\n\n## Review Feedback\nf",
        teacher="## Task\nx\n\n## Review Feedback\nf\n\n## Revision\ny",
    )
    convs, pre_post = pairs_to_chat_messages([p1, p2], mode="multi_turn")
    assert len(convs) == 1
    assert len(pre_post) == len(convs)
    assert len(convs[0]) == 5  # system + 2×(user,assistant)


def test_corrupt_teacher_text_still_emits_assistant_message() -> None:
    # teacher_text does NOT start with activation_text — should fall back
    # to the full teacher_text as the assistant reply rather than drop.
    pair = {
        "task_id": "x",
        "activation_text": "HELLO",
        "teacher_text": "WORLD",
        "metadata": {"source_task_id": "x", "step_index": 0},
    }
    convs, pre_post = pairs_to_chat_messages([pair], mode="single_turn")
    assert len(convs) == 1
    assert len(pre_post) == len(convs)
    assert convs[0][2]["content"] == "WORLD"


def test_multi_turn_skips_pair_with_empty_assistant() -> None:
    # If a pair has teacher_text == activation_text, the extracted tail is
    # empty, so the pair is skipped. A group with only skipped pairs
    # produces no conversation.
    pair = {
        "task_id": "x",
        "activation_text": "same",
        "teacher_text": "same",
        "metadata": {"source_task_id": "x", "step_index": 0},
    }
    convs, pre_post = pairs_to_chat_messages([pair], mode="multi_turn")
    assert convs == []
    assert pre_post == []


# ---------------------------------------------------------------------------
# New tests for pre_post_records
# ---------------------------------------------------------------------------


def test_pre_post_aligned_with_conversations() -> None:
    """Two distinct task_ids → two conversations, two pre_post records."""
    pairs = [
        _pair(task_id="pr_repo_1", source_task_id="pr_repo_1"),
        _pair(task_id="pr_repo_2", source_task_id="pr_repo_2"),
    ]
    convs, pre_post = pairs_to_chat_messages(pairs, mode="multi_turn")
    assert len(pre_post) == len(convs) == 2
    for record in pre_post:
        assert "pre_codes" in record
        assert "post_codes" in record
        assert isinstance(record["pre_codes"], list)
        assert isinstance(record["post_codes"], list)
        assert all(isinstance(s, str) for s in record["pre_codes"])
        assert all(isinstance(s, str) for s in record["post_codes"])


def test_pre_post_multi_turn_per_turn_lists() -> None:
    """Multi-turn: pre/post codes are emitted as per-turn parallel lists."""
    activation_0 = "## Task\nWrite add()"
    teacher_0 = f"{activation_0}\n\n## Implementation\ndef add(a,b): return a+b"

    activation_1 = (
        "## Task\nWrite add()\n\n"
        "## Current Code\ndef add(a,b): return a+b\n\n"
        "## Review Feedback\nHandle floats"
    )
    teacher_1 = f"{activation_1}\n\n## Revision\ndef add(a,b): return float(a)+float(b)"

    pairs = [
        _pair(step_index=0, activation=activation_0, teacher=teacher_0),
        _pair(step_index=1, activation=activation_1, teacher=teacher_1),
    ]
    convs, pre_post = pairs_to_chat_messages(pairs, mode="multi_turn")
    assert len(convs) == 1
    assert len(pre_post) == 1

    record = pre_post[0]
    # Two assistant turns → two entries per list, in step_index order.
    assert record["pre_codes"] == ["", "def add(a,b): return a+b"]
    assert record["post_codes"] == [
        "def add(a,b): return a+b",
        "def add(a,b): return float(a)+float(b)",
    ]


def test_pre_post_skips_match_conversation_skips() -> None:
    """Skipped pair (empty assistant) must not appear in pre_post_records."""
    good_pair = _pair(task_id="pr_a", source_task_id="pr_a", step_index=0)
    skip_pair = {
        "task_id": "pr_b",
        "activation_text": "same",
        "teacher_text": "same",
        "metadata": {"source_task_id": "pr_b", "step_index": 0},
    }
    convs, pre_post = pairs_to_chat_messages([good_pair, skip_pair], mode="multi_turn")
    assert len(convs) == 1
    assert len(pre_post) == 1
    assert "pre_codes" in pre_post[0]
    assert "post_codes" in pre_post[0]
    # Only the good pair → length 1 lists.
    assert len(pre_post[0]["pre_codes"]) == 1
    assert len(pre_post[0]["post_codes"]) == 1


def test_pre_post_single_turn_emits_length_one_lists() -> None:
    """single_turn: pre_codes / post_codes are length-1 lists."""
    activation = "## Task\nImplement fizzbuzz"
    teacher = f"{activation}\n\n## Implementation\ndef fizzbuzz(n): pass"
    pair = _pair(activation=activation, teacher=teacher)

    convs, pre_post = pairs_to_chat_messages([pair], mode="single_turn")
    assert len(convs) == 1
    assert len(pre_post) == 1
    assert pre_post[0]["pre_codes"] == [""]  # initial commit → empty pre
    assert len(pre_post[0]["post_codes"]) == 1
    assert "fizzbuzz" in pre_post[0]["post_codes"][0]
