"""Failing tests for model_training.d2l_data module (TEST-02).

Tests cover 5 required data pipeline behaviors:
1. format_for_distillation produces records with activation_text and teacher_text fields
2. format_for_distillation activation_text has no answer, teacher_text has answer
3. generate_needle_dataset returns records with activation_text, teacher_text, task_id
4. save_jsonl / load_jsonl round-trip returns identical records
5. split_by_task_id produces train and test sets with zero task_id overlap
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(
    task_id: str = "HumanEval/0",
    task_description: str = "Write a function that adds two numbers.",
    outcome: str = "success",
    answer: str = "def add(a, b):\n    return a + b",
) -> dict[str, Any]:
    """Create a minimal trajectory dict for format_for_distillation tests."""
    return {
        "task_id": task_id,
        "session_id": task_id,
        "task_description": task_description,
        "outcome": outcome,
        "steps": [
            {
                "attempt": 0,
                "generated_code": "def add(a, b): pass",
                "tests_passed": False,
                "description": "Initial attempt",
            },
            {
                "attempt": 1,
                "generated_code": answer,
                "tests_passed": True,
                "description": "Corrected implementation",
                "canonical_solution": answer,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Test 1: format_for_distillation produces records with required fields
# ---------------------------------------------------------------------------


def test_format_for_distillation_produces_records_with_required_fields() -> None:
    """format_for_distillation returns records with activation_text and teacher_text."""
    from model_training.d2l_data import format_for_distillation

    trajectory = _make_trajectory()
    records = format_for_distillation(trajectory)

    assert len(records) > 0, "Expected at least one record from a successful trajectory"
    for record in records:
        assert "activation_text" in record, "Record missing 'activation_text' field"
        assert "teacher_text" in record, "Record missing 'teacher_text' field"
        assert "task_id" in record, "Record missing 'task_id' field"


# ---------------------------------------------------------------------------
# Test 2: activation_text has no answer, teacher_text has answer
# ---------------------------------------------------------------------------


def test_format_for_distillation_activation_has_no_answer_teacher_has_answer() -> None:
    """activation_text does NOT contain answer; teacher_text DOES contain answer."""
    from model_training.d2l_data import format_for_distillation

    answer = "def add(a: int, b: int) -> int:\n    return a + b"
    trajectory = _make_trajectory(answer=answer)
    records = format_for_distillation(trajectory)

    assert len(records) > 0
    record = records[0]

    # activation_text must NOT contain the answer code
    assert answer not in record["activation_text"], (
        "activation_text must not contain the answer"
    )
    # teacher_text MUST contain the answer code
    assert answer in record["teacher_text"], "teacher_text must contain the answer"


# ---------------------------------------------------------------------------
# Test 3: generate_needle_dataset returns records with required fields
# ---------------------------------------------------------------------------


def test_generate_needle_dataset_returns_records_with_required_fields() -> None:
    """generate_needle_dataset returns list of dicts with required fields."""
    from model_training.d2l_data import generate_needle_dataset

    records = generate_needle_dataset(n=5)

    assert isinstance(records, list), "Expected a list of records"
    assert len(records) == 5, f"Expected 5 records, got {len(records)}"
    for i, record in enumerate(records):
        assert "activation_text" in record, f"Record {i} missing 'activation_text'"
        assert "teacher_text" in record, f"Record {i} missing 'teacher_text'"
        assert "task_id" in record, f"Record {i} missing 'task_id'"


# ---------------------------------------------------------------------------
# Test 4: save_jsonl / load_jsonl round-trip
# ---------------------------------------------------------------------------


def test_save_jsonl_load_jsonl_round_trip(tmp_path: Path) -> None:
    """save_jsonl then load_jsonl returns identical records."""
    from model_training.d2l_data import load_jsonl, save_jsonl

    records: list[dict[str, Any]] = [
        {"task_id": "HumanEval/0", "activation_text": "A", "teacher_text": "A+ans"},
        {"task_id": "HumanEval/1", "activation_text": "B", "teacher_text": "B+ans"},
    ]

    out_path = tmp_path / "test_data.jsonl"
    save_jsonl(records, out_path)

    loaded = load_jsonl(out_path)
    assert loaded == records, "Round-trip JSONL load did not return identical records"


# ---------------------------------------------------------------------------
# Test 5: split_by_task_id has zero task_id overlap between train and test
# ---------------------------------------------------------------------------


def test_split_by_task_id_zero_task_id_overlap() -> None:
    """split_by_task_id produces train and test with no task_id overlap."""
    from model_training.d2l_data import split_by_task_id

    # Create records with 10 distinct task_ids (2 records each for variety)
    records: list[dict[str, Any]] = []
    for i in range(10):
        tid = f"HumanEval/{i}"
        records.extend(
            [
                {"task_id": tid, "activation_text": f"a{i}", "teacher_text": f"a{i}x"},
                {"task_id": tid, "activation_text": f"b{i}", "teacher_text": f"b{i}x"},
            ]
        )

    train, test = split_by_task_id(records, test_fraction=0.2, seed=42)

    train_ids = {r["task_id"] for r in train}
    test_ids = {r["task_id"] for r in test}

    overlap = train_ids & test_ids
    assert overlap == set(), f"Expected zero task_id overlap, got: {overlap}"
    assert len(test) > 0, "Test set should not be empty"
    assert len(train) > 0, "Train set should not be empty"
    assert len(train) + len(test) == len(records), (
        "All records should be in train or test"
    )
