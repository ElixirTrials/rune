"""Tests for model_training.d2l_data module (TEST-02 + trajectory/augmentation).

Tests cover 5 original data pipeline behaviors plus trajectory generation
and augmentation:
1. format_for_distillation produces records with activation_text and teacher_text fields
2. format_for_distillation activation_text has no answer, teacher_text has answer
3. generate_needle_dataset returns records with activation_text, teacher_text, task_id
4. save_jsonl / load_jsonl round-trip returns identical records
5. split_by_task_id produces train and test sets with zero task_id overlap
6. generate_trajectory_dataset returns records with required fields
7. generate_trajectory_dataset task_id starts with "HumanEval/"
8. augment_trajectories returns n_variants records per input
9. augmented records inherit source task_id
10. augmented + original records pass split_by_task_id with zero task-ID leakage
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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


# ---------------------------------------------------------------------------
# Helpers for trajectory tests
# ---------------------------------------------------------------------------


def _make_humaneval_dataset(n: int = 5) -> list[dict[str, Any]]:
    """Return minimal HumanEval-like task dicts for testing."""
    return [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": f"def func_{i}(x: int) -> int:\n    '''Return x + {i}.'''\n",
            "canonical_solution": f"    return x + {i}\n",
            "test": f"assert func_{i}(0) == {i}",
            "entry_point": f"func_{i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test 6: generate_trajectory_dataset returns records with required fields
# ---------------------------------------------------------------------------


def test_generate_trajectory_dataset_returns_records_with_required_fields() -> None:
    """generate_trajectory_dataset returns list of dicts with task_id, activation_text, teacher_text."""
    from model_training.d2l_data import generate_trajectory_dataset

    fake_dataset = _make_humaneval_dataset(5)
    _inject_fake_datasets_module(fake_dataset)

    records = generate_trajectory_dataset(source="humaneval", max_tasks=5)

    assert isinstance(records, list), "Expected a list"
    assert len(records) == 5, f"Expected 5 records, got {len(records)}"
    for record in records:
        assert "task_id" in record, "Record missing 'task_id'"
        assert "activation_text" in record, "Record missing 'activation_text'"
        assert "teacher_text" in record, "Record missing 'teacher_text'"


# ---------------------------------------------------------------------------
# Test 7: generate_trajectory_dataset task_id starts with "HumanEval/"
# ---------------------------------------------------------------------------


def test_generate_trajectory_dataset_task_id_starts_with_humaneval() -> None:
    """generate_trajectory_dataset records have task_id starting with 'HumanEval/'."""
    from model_training.d2l_data import generate_trajectory_dataset

    fake_dataset = _make_humaneval_dataset(3)
    _inject_fake_datasets_module(fake_dataset)

    records = generate_trajectory_dataset(source="humaneval", max_tasks=3)

    for record in records:
        assert record["task_id"].startswith("HumanEval/"), (
            f"task_id '{record['task_id']}' does not start with 'HumanEval/'"
        )


# ---------------------------------------------------------------------------
# Test 8: augment_trajectories returns n_variants records per input
# ---------------------------------------------------------------------------


import sys


def _inject_fake_datasets_module(fake_data: list[dict[str, Any]]) -> None:
    """Inject a fake datasets module into sys.modules for patching.

    Avoids network calls to Hugging Face Hub in CI.
    """
    if "datasets" not in sys.modules:
        fake_datasets = ModuleType("datasets")
        sys.modules["datasets"] = fake_datasets

    load_mock = MagicMock(return_value=fake_data)
    sys.modules["datasets"].load_dataset = load_mock  # type: ignore[attr-defined]


def _inject_fake_inference_modules(mock_provider_instance: MagicMock) -> None:
    """Inject fake inference.ollama_provider into sys.modules for patching.

    Avoids importing the real inference library (which may require GPU deps).
    The fake module exposes a MockOllamaProvider class that returns the
    provided mock instance when constructed.
    """
    if "inference" not in sys.modules:
        fake_inference = ModuleType("inference")
        sys.modules["inference"] = fake_inference

    if "inference.ollama_provider" not in sys.modules:
        fake_ollama_mod = ModuleType("inference.ollama_provider")
        sys.modules["inference.ollama_provider"] = fake_ollama_mod

    # Make OllamaProvider(base_url=...) return the mock instance
    mock_class = MagicMock(return_value=mock_provider_instance)
    sys.modules["inference.ollama_provider"].OllamaProvider = mock_class  # type: ignore[attr-defined]


def _make_mock_provider() -> MagicMock:
    """Create a mock OllamaProvider whose generate method returns .text."""
    mock_result = MagicMock()
    mock_result.text = "augmented text content"
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=mock_result)
    return provider


def test_augment_trajectories_returns_n_variants_per_input() -> None:
    """augment_trajectories returns n_variants augmented records for each input."""
    from model_training.d2l_data import augment_trajectories

    trajectories = [
        {
            "task_id": "HumanEval/0",
            "activation_text": "def add(a, b): ...",
            "teacher_text": "def add(a, b): return a + b",
        },
        {
            "task_id": "HumanEval/1",
            "activation_text": "def sub(a, b): ...",
            "teacher_text": "def sub(a, b): return a - b",
        },
    ]

    mock_provider = _make_mock_provider()
    _inject_fake_inference_modules(mock_provider)

    augmented = augment_trajectories(trajectories, n_variants=3)

    assert len(augmented) == len(trajectories) * 3, (
        f"Expected {len(trajectories) * 3} augmented records, got {len(augmented)}"
    )


# ---------------------------------------------------------------------------
# Test 9: augmented records inherit source task_id
# ---------------------------------------------------------------------------


def test_augmented_records_inherit_source_task_id() -> None:
    """augment_trajectories records have same task_id as their source trajectory."""
    from model_training.d2l_data import augment_trajectories

    source_task_id = "HumanEval/42"
    trajectories = [
        {
            "task_id": source_task_id,
            "activation_text": "prompt text",
            "teacher_text": "prompt text with solution",
        }
    ]

    mock_provider = _make_mock_provider()
    _inject_fake_inference_modules(mock_provider)

    augmented = augment_trajectories(trajectories, n_variants=3)

    assert len(augmented) == 3
    for record in augmented:
        assert record["task_id"] == source_task_id, (
            f"Augmented record task_id '{record['task_id']}' != source '{source_task_id}'"
        )


# ---------------------------------------------------------------------------
# Test 10: augmented + original records pass split_by_task_id with zero leakage
# ---------------------------------------------------------------------------


def test_augmented_and_original_records_zero_task_id_leakage() -> None:
    """Mixing augmented and original records: split_by_task_id still has zero task_id overlap."""
    from model_training.d2l_data import augment_trajectories, split_by_task_id

    n_tasks = 10
    trajectories = [
        {
            "task_id": f"HumanEval/{i}",
            "activation_text": f"prompt {i}",
            "teacher_text": f"prompt {i} + solution",
        }
        for i in range(n_tasks)
    ]

    mock_provider = _make_mock_provider()
    _inject_fake_inference_modules(mock_provider)

    augmented = augment_trajectories(trajectories, n_variants=3)

    all_records: list[dict[str, Any]] = trajectories + augmented

    train, test = split_by_task_id(all_records, test_fraction=0.2, seed=42)

    train_ids = {r["task_id"] for r in train}
    test_ids = {r["task_id"] for r in test}
    overlap = train_ids & test_ids
    assert overlap == set(), f"Expected zero task_id overlap, got: {overlap}"
    assert len(train) + len(test) == len(all_records)
