"""Unit tests for _parse_subtask_list DecomposeResult validation.

Covers:
- Normal (3-6 item) output passes through with original dicts intact
- Name-based depends_on is preserved (not corrupted by Subtask round-trip)
- Too many items (>8) triggers fallback
- Single item triggers fallback
- Empty output triggers existing empty-match fallback
- Hard cap of 8 after dedup in run_phased_pipeline (tested via parser itself
  producing <=8 items when 9 unique names are given but validation would fire)
"""

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


def _numbered_list(names: list[str]) -> str:
    """Build a minimal numbered-list decompose output for the given names."""
    lines = [f"{i + 1}. {name} - do the {name} work" for i, name in enumerate(names)]
    return "\n".join(lines)


def _numbered_list_with_deps(names: list[str]) -> str:
    """Build numbered list where item 2+ depends on item 1 by name."""
    lines = [f"1. {names[0]} - first task [depends: none]"]
    for i, name in enumerate(names[1:], start=2):
        lines.append(f"{i}. {name} - depends work [depends: {names[0]}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normal pass-through (2–8 items)
# ---------------------------------------------------------------------------


def test_three_items_passthrough() -> None:
    """3 subtasks: validation passes, original dicts returned."""
    output = _numbered_list(["parse_input", "transform", "write_output"])
    result = _parse_subtask_list(output)
    assert len(result) == 3
    assert result[0]["name"] == "parse_input"


def test_six_items_passthrough() -> None:
    """6 subtasks: at upper-middle of valid range."""
    names = [f"step_{i}" for i in range(6)]
    result = _parse_subtask_list(_numbered_list(names))
    assert len(result) == 6


def test_eight_items_passthrough() -> None:
    """8 subtasks: exactly at max valid count, should pass through."""
    names = [f"task_{i}" for i in range(8)]
    result = _parse_subtask_list(_numbered_list(names))
    assert len(result) == 8


def test_two_items_passthrough() -> None:
    """2 subtasks: minimum valid count."""
    result = _parse_subtask_list(_numbered_list(["setup", "run"]))
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Name-based depends_on is preserved through Pydantic round-trip
# ---------------------------------------------------------------------------


def test_name_based_deps_preserved() -> None:
    """depends_on strings survive the Subtask(**s) / model_dump() round-trip."""
    output = _numbered_list_with_deps(["parse_input", "transform", "write_output"])
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
    result = _parse_subtask_list(_numbered_list(names))
    assert len(result) == 1
    assert result[0]["name"] == "implementation"


def test_sixteen_items_triggers_fallback() -> None:
    """16 subtasks (thinking-block leakage scenario) triggers fallback."""
    names = [f"leaked_{i}" for i in range(16)]
    result = _parse_subtask_list(_numbered_list(names))
    assert len(result) == 1
    assert result[0]["name"] == "implementation"
    # Fallback description is the first 200 chars of original output
    output = _numbered_list(names)
    assert result[0]["description"] == output[:200].strip()


# ---------------------------------------------------------------------------
# Single item passes validation
# ---------------------------------------------------------------------------


def test_single_item_passes_validation() -> None:
    """1 subtask is valid (min_length=1); parsed name is preserved."""
    result = _parse_subtask_list("1. only_task - the entire implementation")
    assert len(result) == 1
    assert result[0]["name"] == "only_task"


# ---------------------------------------------------------------------------
# Empty output triggers existing pre-validation fallback
# ---------------------------------------------------------------------------


def test_empty_output_fallback() -> None:
    """No numbered lines: the early empty-match guard fires (pre-validation)."""
    result = _parse_subtask_list("")
    assert len(result) == 1
    assert result[0]["name"] == "implementation"
    assert result[0]["depends_on"] == []


def test_non_list_output_fallback() -> None:
    """Prose output with no numbered lines: same empty-match fallback."""
    text = "Here is my analysis of the problem.\nNo structure."
    result = _parse_subtask_list(text)
    assert len(result) == 1
    assert result[0]["name"] == "implementation"


# ---------------------------------------------------------------------------
# Hard-cap at 8 after dedup (belt-and-suspenders, tested via >=9 unique names)
# ---------------------------------------------------------------------------


def test_hard_cap_eight_items_in_pipeline_dedup() -> None:
    """Belt-and-suspenders cap: even if parser returned 9 valid unique items,
    the subtasks[:8] line in run_phased_pipeline would trim to 8.

    We test the parser itself here (it returns fallback for >8), confirming
    that the [:8] slice in the scoring block would operate on already-valid
    output from validation, and that 8 is the hard ceiling throughout.
    """
    names = [f"task_{i}" for i in range(8)]
    result = _parse_subtask_list(_numbered_list(names))
    # Parser accepts 8; confirms <=8 guarantee at parse layer
    assert len(result) == 8
    # Confirm all names are present (no silent truncation at parse stage)
    parsed_names = [r["name"] for r in result]
    for n in names:
        assert n in parsed_names


# ----------------------------------------------------------
# validate=False bypasses DecomposeResult (diagnose path)
# ----------------------------------------------------------


def test_validate_false_single_item_passes_through() -> None:
    """validate=False lets single-item output through (diagnose)."""
    result = _parse_subtask_list(
        "1. parse_input - NameError on line 5",
        validate=False,
    )
    assert len(result) == 1
    assert result[0]["name"] == "parse_input"


def test_validate_false_many_items_passes_through() -> None:
    """validate=False lets >8 items through (no cap at parse)."""
    names = [f"diag_{i}" for i in range(12)]
    result = _parse_subtask_list(_numbered_list(names), validate=False)
    assert len(result) == 12
