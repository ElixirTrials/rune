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

import sys
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
    """generate_trajectory_dataset returns dicts with required fields."""
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
            f"Augmented task_id '{record['task_id']}' != source '{source_task_id}'"
        )


# ---------------------------------------------------------------------------
# Test 10: augmented + original records pass split_by_task_id with zero leakage
# ---------------------------------------------------------------------------


def test_augmented_and_original_records_zero_task_id_leakage() -> None:
    """Augmented + original records: split_by_task_id has zero overlap."""
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


# ---------------------------------------------------------------------------
# Test 11: normalize_mined_trajectory – merged PR becomes success
# ---------------------------------------------------------------------------


def test_normalize_mined_trajectory_merged_pr_becomes_success() -> None:
    """Merged PR trajectory normalizes to outcome='success' with correct prefixes."""
    from model_training.d2l_data import normalize_mined_trajectory

    mined: dict[str, Any] = {
        "task_id": "pr_owner/repo_42",
        "task_description": "Add caching layer to API",
        "outcome": "merged",
        "steps": [
            {
                "type": "commit",
                "description": "initial implementation",
                "content": "def cache(): pass",
            },
            {
                "type": "review",
                "description": "",
                "content": "Needs error handling",
            },
            {
                "type": "commit",
                "description": "address review feedback",
                "content": "def cache():\n    try: ...\n    except: ...",
            },
        ],
    }

    result = normalize_mined_trajectory(mined)

    assert result["outcome"] == "success"
    assert result["task_id"] == "pr_owner/repo_42"
    assert result["session_id"] == "pr_owner/repo_42"
    assert result["task_description"] == "Add caching layer to API"

    steps = result["steps"]
    assert len(steps) == 3

    # First commit step
    assert steps[0]["description"].startswith("[Commit]")
    assert steps[0]["tests_passed"] is False
    assert "canonical_solution" not in steps[0]

    # Review step
    assert steps[1]["description"].startswith("[Review]")
    assert steps[1]["generated_code"] == ""
    assert steps[1]["tests_passed"] is False

    # Last commit step — tests_passed=True and canonical_solution set
    assert steps[2]["description"].startswith("[Commit]")
    assert steps[2]["tests_passed"] is True
    expected_code = "def cache():\n    try: ...\n    except: ..."
    assert steps[2]["canonical_solution"] == expected_code


# ---------------------------------------------------------------------------
# Test 12: normalize_mined_trajectory – closed unmerged PR becomes failure
# ---------------------------------------------------------------------------


def test_normalize_mined_trajectory_closed_unmerged_pr_becomes_failure() -> None:
    """Closed (unmerged) PR trajectory normalizes to outcome='failure'."""
    from model_training.d2l_data import normalize_mined_trajectory

    mined: dict[str, Any] = {
        "task_id": "pr_owner/repo_99",
        "task_description": "Refactor auth module",
        "outcome": "closed",
        "steps": [
            {
                "type": "commit",
                "description": "refactor attempt",
                "content": "class Auth: ...",
            },
        ],
    }

    result = normalize_mined_trajectory(mined)

    assert result["outcome"] == "failure"
    # Last commit step should NOT have tests_passed=True for failure outcome
    assert result["steps"][0]["tests_passed"] is False
    assert "canonical_solution" not in result["steps"][0]


# ---------------------------------------------------------------------------
# Test 13: normalized trajectory produces distillation records
# ---------------------------------------------------------------------------


def test_normalized_mined_trajectory_produces_distillation_records() -> None:
    """Normalized merged PR passes through format_for_distillation correctly."""
    from model_training.d2l_data import (
        format_for_distillation,
        normalize_mined_trajectory,
    )

    mined: dict[str, Any] = {
        "task_id": "pr_owner/repo_5",
        "task_description": "Implement retry logic",
        "outcome": "merged",
        "steps": [
            {
                "type": "commit",
                "description": "add retry decorator",
                "content": "@retry\ndef fetch(): ...",
            },
            {
                "type": "review",
                "description": "",
                "content": "LGTM",
            },
            {
                "type": "commit",
                "description": "final cleanup",
                "content": "@retry(max=3)\ndef fetch(): return get()",
            },
        ],
    }

    normalized = normalize_mined_trajectory(mined)
    records = format_for_distillation(normalized)

    assert len(records) >= 1, "Expected at least 1 distillation record"

    record = records[0]
    assert "task_id" in record
    assert "activation_text" in record
    assert "teacher_text" in record

    # activation_text should contain [Commit] from step descriptions
    assert "[Commit]" in record["activation_text"]

    # teacher_text should extend activation_text
    assert record["teacher_text"].startswith(record["activation_text"])
    assert len(record["teacher_text"]) > len(record["activation_text"])


# ---------------------------------------------------------------------------
# Test 14: normalize_mined_trajectory – closed issue becomes success
# ---------------------------------------------------------------------------


def test_normalize_mined_trajectory_closed_issue_becomes_success() -> None:
    """Closed issue trajectory normalizes to outcome='success' (opposite of PR)."""
    from model_training.d2l_data import normalize_mined_trajectory

    mined: dict[str, Any] = {
        "task_id": "issue_owner/repo_7",
        "task_description": "Fix memory leak in worker pool",
        "outcome": "closed",
        "steps": [
            {
                "type": "commit",
                "description": "fix pool cleanup",
                "content": "pool.shutdown(wait=True)",
            },
        ],
    }

    result = normalize_mined_trajectory(mined)

    assert result["outcome"] == "success", (
        "Closed issue must map to 'success' (unlike closed PR which is 'failure')"
    )


# ---------------------------------------------------------------------------
# Helpers for normalize_mined_pairs tests
# ---------------------------------------------------------------------------


def _make_mined_trajectory(
    task_id: str = "pr_owner/repo_42",
    task_description: str = "Add widget support",
    steps: list[dict[str, str]] | None = None,
    outcome: str = "merged",
) -> dict[str, Any]:
    """Create a minimal mined trajectory dict for normalize_mined_pairs tests."""
    if steps is None:
        steps = [
            {
                "type": "commit",
                "description": "Initial impl",
                "content": "+def widget(): pass",
            },
        ]
    return {
        "task_id": task_id,
        "task_description": task_description,
        "steps": steps,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Tests 15-22: normalize_mined_pairs
# ---------------------------------------------------------------------------


def test_normalize_mined_pairs_single_commit_produces_step0() -> None:
    """A single-commit PR produces one step_0 pair."""
    from model_training.d2l_data import normalize_mined_pairs

    trajectory = _make_mined_trajectory()
    pairs = normalize_mined_pairs(trajectory)

    assert len(pairs) == 1
    assert pairs[0]["task_id"] == "pr_owner/repo_42"
    assert pairs[0]["metadata"]["step_index"] == 0
    assert "Add widget support" in pairs[0]["activation_text"]
    assert "+def widget(): pass" in pairs[0]["teacher_text"]
    assert "## Implementation" in pairs[0]["teacher_text"]


def test_normalize_mined_pairs_review_revision_cycle() -> None:
    """A commit-review-commit trajectory produces step_0 + step_1."""
    from model_training.d2l_data import normalize_mined_pairs

    trajectory = _make_mined_trajectory(
        steps=[
            {"type": "commit", "description": "V1", "content": "+v1 code"},
            {
                "type": "review",
                "description": "Review",
                "content": "Add error handling",
            },
            {
                "type": "commit",
                "description": "V2",
                "content": "+v2 with error handling",
            },
        ]
    )
    pairs = normalize_mined_pairs(trajectory)

    assert len(pairs) == 2
    assert pairs[0]["task_id"] == "pr_owner/repo_42"
    assert pairs[0]["metadata"]["step_index"] == 0
    assert "## Task" in pairs[0]["activation_text"]
    assert "## Implementation" in pairs[0]["teacher_text"]
    assert pairs[1]["task_id"] == "pr_owner/repo_42"
    assert pairs[1]["metadata"]["step_index"] == 1
    assert "## Current Code" in pairs[1]["activation_text"]
    assert "+v1 code" in pairs[1]["activation_text"]
    assert "## Review Feedback" in pairs[1]["activation_text"]
    assert "Add error handling" in pairs[1]["activation_text"]
    assert "## Revision" in pairs[1]["teacher_text"]
    assert "+v2 with error handling" in pairs[1]["teacher_text"]


def test_normalize_mined_pairs_consecutive_commits_uses_last() -> None:
    """Multiple commits before first review: last commit used for step_0."""
    from model_training.d2l_data import normalize_mined_pairs

    trajectory = _make_mined_trajectory(
        steps=[
            {"type": "commit", "description": "Draft", "content": "+draft"},
            {"type": "commit", "description": "Polish", "content": "+polished"},
            {"type": "review", "description": "Review", "content": "Fix typo"},
            {"type": "commit", "description": "Fixed", "content": "+fixed"},
        ]
    )
    pairs = normalize_mined_pairs(trajectory)

    assert len(pairs) == 2
    assert "+polished" in pairs[0]["teacher_text"]
    assert "+draft" not in pairs[0]["teacher_text"]
    assert "+polished" in pairs[1]["activation_text"]


def test_normalize_mined_pairs_consecutive_reviews_concatenated() -> None:
    """Multiple reviews before a commit are concatenated."""
    from model_training.d2l_data import normalize_mined_pairs

    trajectory = _make_mined_trajectory(
        steps=[
            {"type": "commit", "description": "V1", "content": "+v1"},
            {"type": "review", "description": "R1", "content": "Fix A"},
            {"type": "review", "description": "R2", "content": "Also fix B"},
            {"type": "commit", "description": "V2", "content": "+v2"},
        ]
    )
    pairs = normalize_mined_pairs(trajectory)

    assert len(pairs) == 2
    assert "Fix A" in pairs[1]["activation_text"]
    assert "Also fix B" in pairs[1]["activation_text"]


def test_normalize_mined_pairs_trailing_reviews_skipped() -> None:
    """Reviews after the last commit are ignored."""
    from model_training.d2l_data import normalize_mined_pairs

    trajectory = _make_mined_trajectory(
        steps=[
            {"type": "commit", "description": "V1", "content": "+v1"},
            {"type": "review", "description": "Review", "content": "Needs work"},
        ]
    )
    pairs = normalize_mined_pairs(trajectory)

    assert len(pairs) == 1
    assert pairs[0]["task_id"] == "pr_owner/repo_42"
    assert pairs[0]["metadata"]["step_index"] == 0


def test_normalize_mined_pairs_empty_steps_returns_empty() -> None:
    """Empty steps list produces no pairs."""
    from model_training.d2l_data import normalize_mined_pairs

    trajectory = _make_mined_trajectory(steps=[])
    assert normalize_mined_pairs(trajectory) == []


def test_normalize_mined_pairs_metadata_includes_outcome_and_language() -> None:
    """Metadata captures outcome and language from caller."""
    from model_training.d2l_data import normalize_mined_pairs

    merged = _make_mined_trajectory(outcome="merged")
    closed = _make_mined_trajectory(outcome="closed")

    merged_pairs = normalize_mined_pairs(merged, language="python")
    closed_pairs = normalize_mined_pairs(closed, language="go")

    assert merged_pairs[0]["metadata"]["outcome"] == "merged"
    assert merged_pairs[0]["metadata"]["language"] == "python"
    assert closed_pairs[0]["metadata"]["outcome"] == "closed"
    assert closed_pairs[0]["metadata"]["language"] == "go"


def test_normalize_mined_pairs_compatible_with_split_by_task_id() -> None:
    """All steps from the same PR land in the same split."""
    from model_training.d2l_data import normalize_mined_pairs, split_by_task_id

    t1 = _make_mined_trajectory(
        task_id="pr_a/b_1",
        steps=[
            {"type": "commit", "description": "C1", "content": "+c1"},
            {"type": "review", "description": "R1", "content": "Fix"},
            {"type": "commit", "description": "C2", "content": "+c2"},
        ],
    )
    t2 = _make_mined_trajectory(
        task_id="pr_c/d_2",
        steps=[
            {"type": "commit", "description": "C1", "content": "+c1"},
        ],
    )

    all_pairs = normalize_mined_pairs(t1) + normalize_mined_pairs(t2)
    train, test = split_by_task_id(all_pairs, test_fraction=0.5, seed=42)

    # All steps from same PR must be in same split — no source_task_id overlap
    train_sources = {r["metadata"]["source_task_id"] for r in train}
    test_sources = {r["metadata"]["source_task_id"] for r in test}
    assert not train_sources & test_sources


def test_normalize_mined_pairs_multiple_rounds() -> None:
    """A complex PR with 3 review rounds produces 4 pairs."""
    from model_training.d2l_data import normalize_mined_pairs

    trajectory = _make_mined_trajectory(
        steps=[
            {"type": "commit", "description": "V1", "content": "+v1"},
            {"type": "review", "description": "R1", "content": "Fix X"},
            {"type": "commit", "description": "V2", "content": "+v2"},
            {"type": "review", "description": "R2", "content": "Fix Y"},
            {"type": "commit", "description": "V3", "content": "+v3"},
            {"type": "review", "description": "R3", "content": "Fix Z"},
            {"type": "commit", "description": "V4", "content": "+v4"},
        ]
    )
    pairs = normalize_mined_pairs(trajectory)

    assert len(pairs) == 4
    assert pairs[0]["metadata"]["step_index"] == 0
    assert pairs[1]["metadata"]["step_index"] == 1
    assert pairs[2]["metadata"]["step_index"] == 2
    assert pairs[3]["metadata"]["step_index"] == 3
    assert "+v2" in pairs[2]["activation_text"]
    assert "+v1" not in pairs[2]["activation_text"]
