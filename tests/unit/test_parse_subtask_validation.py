"""Unit tests for _parse_subtask_list JSON parsing via DecomposeResult.

Covers:
- Normal (2-8 item) JSON output passes through with original dicts intact
- Name-based depends_on is preserved through Pydantic round-trip
- Too many items (>8) triggers ValidationError fallback
- Single item passes validation
- Empty / non-JSON output triggers fallback
- Hard cap of 8 after dedup in run_phased_pipeline
- validate=False still uses DecomposeResult (max_length=8 enforced)
"""

import json
import sys
from pathlib import Path

# Bootstrap path so rune_runner imports work without a full install
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

from scripts.rune_runner import _parse_subtask_list  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_subtasks(names: list[str]) -> str:
    """Build a valid DecomposeResult JSON string for the given names."""
    subtasks = [{"name": n, "description": f"do the {n} work", "depends_on": []} for n in names]
    return json.dumps({"subtasks": subtasks})


def _json_subtasks_with_deps(names: list[str]) -> str:
    """Build DecomposeResult JSON where items 2+ depend on item 1."""
    subtasks = [{"name": names[0], "description": "first task", "depends_on": []}]
    for name in names[1:]:
        subtasks.append({"name": name, "description": "depends work", "depends_on": [names[0]]})
    return json.dumps({"subtasks": subtasks})


# ---------------------------------------------------------------------------
# Normal pass-through (2-8 items)
# ---------------------------------------------------------------------------


def test_three_items_passthrough() -> None:
    """3 subtasks: validation passes, original dicts returned."""
    output = _json_subtasks(["parse_input", "transform", "write_output"])
    result = _parse_subtask_list(output)
    assert len(result) == 3
    assert result[0]["name"] == "parse_input"


def test_six_items_passthrough() -> None:
    """6 subtasks: at upper-middle of valid range."""
    names = [f"step_{i}" for i in range(6)]
    result = _parse_subtask_list(_json_subtasks(names))
    assert len(result) == 6


def test_eight_items_passthrough() -> None:
    """8 subtasks: exactly at max valid count, should pass through."""
    names = [f"task_{i}" for i in range(8)]
    result = _parse_subtask_list(_json_subtasks(names))
    assert len(result) == 8


def test_two_items_passthrough() -> None:
    """2 subtasks: minimum valid count."""
    result = _parse_subtask_list(_json_subtasks(["setup", "run"]))
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Name-based depends_on is preserved through Pydantic round-trip
# ---------------------------------------------------------------------------


def test_name_based_deps_preserved() -> None:
    """depends_on strings survive the Subtask / model_dump() round-trip."""
    output = _json_subtasks_with_deps(["parse_input", "transform", "write_output"])
    result = _parse_subtask_list(output)
    dep = result[1]["depends_on"]
    assert isinstance(dep, list)
    assert len(dep) > 0
    assert isinstance(dep[0], str)
    assert dep[0] == "parse_input"


# ---------------------------------------------------------------------------
# Too many items (>8) triggers fallback
# ---------------------------------------------------------------------------


def test_nine_items_triggers_fallback() -> None:
    """9 subtasks exceeds DecomposeResult max=8; fallback is returned."""
    names = [f"subtask_{i}" for i in range(9)]
    result = _parse_subtask_list(_json_subtasks(names))
    assert len(result) == 1
    assert result[0]["name"] == "implementation"


def test_sixteen_items_triggers_fallback() -> None:
    """16 subtasks (thinking-block leakage scenario) triggers fallback."""
    names = [f"leaked_{i}" for i in range(16)]
    output = _json_subtasks(names)
    result = _parse_subtask_list(output)
    assert len(result) == 1
    assert result[0]["name"] == "implementation"
    # Fallback description is the first 200 chars of original output
    assert result[0]["description"] == output[:200].strip()


# ---------------------------------------------------------------------------
# Single item passes validation
# ---------------------------------------------------------------------------


def test_single_item_passes_validation() -> None:
    """1 subtask is valid (min_length=1); parsed name is preserved."""
    output = json.dumps({"subtasks": [{"name": "only_task", "description": "the entire implementation", "depends_on": []}]})
    result = _parse_subtask_list(output)
    assert len(result) == 1
    assert result[0]["name"] == "only_task"


# ---------------------------------------------------------------------------
# Empty / non-JSON output triggers fallback
# ---------------------------------------------------------------------------


def test_empty_output_fallback() -> None:
    """Empty string: invalid JSON triggers fallback."""
    result = _parse_subtask_list("")
    assert len(result) == 1
    assert result[0]["name"] == "implementation"
    assert result[0]["depends_on"] == []


def test_non_list_output_fallback() -> None:
    """Prose output (not JSON): triggers fallback."""
    text = "Here is my analysis of the problem.\nNo structure."
    result = _parse_subtask_list(text)
    assert len(result) == 1
    assert result[0]["name"] == "implementation"


# ---------------------------------------------------------------------------
# Hard-cap at 8 after dedup (belt-and-suspenders, tested via 8 unique names)
# ---------------------------------------------------------------------------


def test_hard_cap_eight_items_in_pipeline_dedup() -> None:
    """Belt-and-suspenders cap: parser accepts exactly 8 items, confirming
    the <=8 guarantee at parse layer. The [:8] slice in run_phased_pipeline
    operates on already-valid output from validation.
    """
    names = [f"task_{i}" for i in range(8)]
    result = _parse_subtask_list(_json_subtasks(names))
    # Parser accepts 8; confirms <=8 guarantee at parse layer
    assert len(result) == 8
    # Confirm all names are present (no silent truncation at parse stage)
    parsed_names = [r["name"] for r in result]
    for n in names:
        assert n in parsed_names


# ----------------------------------------------------------
# validate=False still uses DecomposeResult (JSON parsing)
# ----------------------------------------------------------


def test_validate_false_single_item_passes_through() -> None:
    """validate=False lets single-item JSON output through."""
    output = json.dumps({"subtasks": [{"name": "parse_input", "description": "NameError on line 5", "depends_on": []}]})
    result = _parse_subtask_list(output, validate=False)
    assert len(result) == 1
    assert result[0]["name"] == "parse_input"


def test_validate_false_many_items_triggers_fallback() -> None:
    """validate=False still enforces DecomposeResult max_length=8.

    With JSON parsing, >8 items fail DecomposeResult validation even
    with validate=False, because model_validate_json always enforces
    the schema. The fallback is returned.
    """
    names = [f"diag_{i}" for i in range(12)]
    result = _parse_subtask_list(_json_subtasks(names), validate=False)
    assert len(result) == 1
    assert result[0]["name"] == "implementation"
